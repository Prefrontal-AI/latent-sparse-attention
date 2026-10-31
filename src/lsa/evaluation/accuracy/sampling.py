"""What a request is sampled with, kept free of the client's dependencies so the suites can name it."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class Sampling:
    """What we send with every request - all of it lands in the record.

    A request whose temperature or top-p is None is refused rather than sent with whatever the server
    would default to.
    """

    temperature: float | None = 0.0
    top_p: float | None = 1.0
    max_tokens: int = 4096
    # GLM's chat template takes reasoning_effort low|high, anything else means max - None sends nothing
    reasoning_effort: str | None = None
    # Tokens the model may spend inside the think block before the server forces it closed, which leaves
    # `max_tokens` minus this for the answer. sglang enforces it from `custom_params`, so the server has
    # to be launched with `--enable-strict-thinking` and a grammar backend that filters tokens - a server
    # without the flag ignores this rather than failing
    thinking_budget: int | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    # Passed to the model's chat template - where Qwen3.5's switch for the think block goes
    chat_template_kwargs: dict[str, Any] | None = None
    # False: the reply starts past a closed think block - `NO_THINK` and sglang's `continue_final_message`
    think: bool = True

    def payload(self) -> dict[str, Any]:
        """Return the sampling fields of a request body.

        Raises:
            ValueError: No temperature or no top-p is set.
        """
        if self.temperature is None or self.top_p is None:
            raise ValueError(f"{self} names no temperature or top-p - a run's generation block supplies them")
        out: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        for name in ("top_k", "min_p", "presence_penalty"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        template = dict(self.chat_template_kwargs or {})
        if self.reasoning_effort is not None:
            template["reasoning_effort"] = self.reasoning_effort
        if template:
            out["chat_template_kwargs"] = template
        if self.thinking_budget is not None:
            out["custom_params"] = {"thinking_budget": self.thinking_budget}
        if not self.think:
            out["continue_final_message"] = True
        return out
