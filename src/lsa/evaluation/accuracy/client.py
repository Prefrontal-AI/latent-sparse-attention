"""Chat requests to an OpenAI-compatible server, keeping everything the record needs.

lm-eval's own client keeps only the answer text. We also need the reasoning, the token counts and the
finish reason of every reply, and we want every sampling setting to be explicit in the request body.
"""

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import Callable
from typing import Any

import aiohttp

from lsa.evaluation.accuracy.sampling import Sampling

# A template that always opens a think block has no switch for it - handing the model a closed one and asking
# the server to continue that turn is how a reply without reasoning is requested, as an earlier harness did
NO_THINK = "<think></think>"

# What a server says of a prompt that does not fit its context, which a retry cannot change
CONTEXT_REFUSALS: tuple[str, ...] = ("context length", "is longer than")

Messages = list[dict[str, str]]


@dataclasses.dataclass
class Reply:
    """One chat completion as the record keeps it, or the error that took its place."""

    content: str = ""
    reasoning: str = ""
    prompt_tokens: int | None = None
    # Prompt tokens read from the server's prefix cache rather than computed - a server launched with
    # `--enable-cache-report` says, any other leaves it None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    seconds: float = 0.0
    error: str | None = None
    # The first generated token - its text, logprob and alternatives - when the request asked for logprobs
    first_token: dict[str, Any] | None = None


def parse_reply(body: dict[str, Any], seconds: float) -> Reply:
    """Read one OpenAI chat completion body, leaving a missing field None rather than guessing it.

    Args:
        body: The completion.
        seconds: How long the request took.

    Returns:
        The reply.
    """
    choice = body["choices"][0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    tokens = (choice.get("logprobs") or {}).get("content") or []
    return Reply(
        content=message.get("content") or "",
        reasoning=message.get("reasoning_content") or "",
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=usage.get("reasoning_tokens"),
        finish_reason=choice.get("finish_reason"),
        seconds=seconds,
        first_token=tokens[0] if tokens else None,
    )


class Client:
    """Sends chat requests with a concurrency cap. No stop strings - the model stops at its end token.

    Stop strings would also match inside the reasoning and cut it, which shows up as an empty answer.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "served",
        concurrency: int = 8,
        timeout: float = 14400,
        retries: int = 3,
        top_logprobs: int | None = None,
    ) -> None:
        """Point a client at a server.

        Args:
            base_url: The server's http root.
            model: The model name every request carries.
            concurrency: Requests in flight at once when a call brings no semaphores of its own.
            timeout: Seconds one request may take.
            retries: Attempts at a request that failed with something a retry could fix.
            top_logprobs: Alternatives every reply's tokens come back with, or None to ask for no logprobs.
        """
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout
        self.retries = retries
        self.top_logprobs = top_logprobs

    def payload(self, messages: Messages, sampling: Sampling) -> dict[str, Any]:
        """Return the request body for one conversation."""
        if not sampling.think:
            messages = [*messages, {"role": "assistant", "content": NO_THINK}]
        body = {"model": self.model, "messages": messages, **sampling.payload()}
        if self.top_logprobs is not None:
            body.update(logprobs=True, top_logprobs=self.top_logprobs)
        return body

    async def _one(
        self,
        session: aiohttp.ClientSession,
        messages: Messages,
        sampling: Sampling,
        timeout: float | None = None,
    ) -> Reply:
        """Send one request, retrying what a retry could fix.

        Args:
            session: The session from `session`.
            messages: The conversation.
            sampling: What it sends.
            timeout: Seconds this request may take, or None for the client's.

        Returns:
            The reply, or one carrying the last error.
        """
        last = None
        start = time.perf_counter()
        for attempt in range(self.retries):
            start = time.perf_counter()
            try:
                # An explicit None is no timeout at all to aiohttp
                # => always name one, the client's own when the call gives none
                limit = aiohttp.ClientTimeout(total=timeout or self.timeout)
                async with session.post(
                    self.url, json=self.payload(messages, sampling), timeout=limit
                ) as response:
                    body = await response.json(content_type=None)
                    if response.status != 200:
                        last = f"http {response.status}: {str(body)[:300]}"
                    else:
                        return parse_reply(body, time.perf_counter() - start)
            except TimeoutError as exc:
                # A retry starts the reply over from its first token and holds a slot for another whole timeout -
                # an AIME25 chain timed out three times over was six hours of nothing. The row keeps the error,
                # and a continued run sends it again
                return Reply(
                    error=f"{type(exc).__name__}: {str(exc)[:300]}", seconds=time.perf_counter() - start
                )
            except (aiohttp.ClientError, KeyError, ValueError) as exc:
                last = f"{type(exc).__name__}: {str(exc)[:300]}"
            if any(refusal in (last or "") for refusal in CONTEXT_REFUSALS):
                break
            await asyncio.sleep(min(2**attempt, 30))
        return Reply(error=last, seconds=time.perf_counter() - start)

    def session(self) -> aiohttp.ClientSession:
        """Open the session requests go through - the semaphores bound them, so the connector does not."""
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout), connector=aiohttp.TCPConnector(limit=0)
        )

    async def gather(
        self,
        session: aiohttp.ClientSession,
        jobs: list[Messages],
        sampling: Sampling,
        on_reply: Callable[[int, Reply], None] | None = None,
        limits: list[asyncio.Semaphore] | None = None,
        timeout: float | None = None,
    ) -> list[Reply]:
        """Send every job, each holding a slot of every one of `limits` while it is in flight.

        Args:
            session: The session from `session`.
            jobs: One message list per request.
            sampling: What every request sends.
            on_reply: Called as on_reply(index, reply) as each reply lands, or None.
            limits: Semaphores other callers may draw from as well, always taken in this order, or None
                for a cap of `concurrency` of this call's own.
            timeout: Seconds each request may take, or None for the client's.

        Returns:
            The replies, in the order of the jobs.
        """
        limits = limits or [asyncio.Semaphore(self.concurrency)]
        replies: list[Reply] = [Reply()] * len(jobs)

        async def run(index: int, messages: Messages) -> None:
            """Send one job under every limit, and hand its reply on as it lands."""
            async with contextlib.AsyncExitStack() as held:
                for limit in limits:
                    await held.enter_async_context(limit)
                replies[index] = await self._one(session, messages, sampling, timeout)
            if on_reply is not None:
                on_reply(index, replies[index])

        await asyncio.gather(*(run(i, messages) for i, messages in enumerate(jobs)))
        return replies


def user_message(text: str) -> Messages:
    """Return a conversation of one user turn."""
    return [{"role": "user", "content": text}]
