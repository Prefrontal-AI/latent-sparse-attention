import asyncio
import unittest

from lsa.evaluation.accuracy.client import Client, parse_reply, user_message
from lsa.evaluation.accuracy.sampling import Sampling
from tests.helpers.evaluation.fakes import FakeServer, echo_target


def send(client, jobs, sampling, on_reply=None):
    """Send jobs the way the runner does - one session, every job gathered under the client's own cap."""

    async def gathered():
        async with client.session() as session:
            return await client.gather(session, jobs, sampling, on_reply=on_reply)

    return asyncio.run(gathered())


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = {"fail_once": 0}

        def respond(messages):
            text = messages[-1]["content"]
            if text.startswith("EMPTY"):
                return {"content": "", "reasoning": "still thinking " * 5, "finish_reason": "length"}
            if text.startswith("FAIL_ONCE"):
                cls.calls["fail_once"] += 1
                if cls.calls["fail_once"] == 1:
                    return {"status": 500, "error": "boom"}
                return {"content": "recovered"}
            if text.startswith("SLOW"):
                return {"content": "too late", "delay": 1.0}
            if text.startswith("TOO_LONG"):
                return {
                    "status": 400,
                    "error": "the input (2000000 tokens) is longer than the model's context length",
                }
            return echo_target(messages)

        cls.server = FakeServer(respond).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_payload_is_explicit_and_has_no_stop_strings(self):
        client = Client(self.server.base_url, model="served")
        sampling = Sampling(temperature=0.7, top_p=0.9, max_tokens=123, reasoning_effort="low")
        payload = client.payload(user_message("hi"), sampling)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["max_tokens"], 123)
        self.assertEqual(payload["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertEqual(payload["model"], "served")
        self.assertNotIn("stop", payload)
        self.assertNotIn("seed", payload)
        # No effort given -> no template kwargs at all, the server default applies
        self.assertNotIn("chat_template_kwargs", client.payload(user_message("hi"), Sampling()))

    def test_no_think_continues_a_closed_think_block(self):
        client = Client(self.server.base_url)
        payload = client.payload(user_message("hi"), Sampling(think=False))
        self.assertEqual(payload["messages"][-1], {"role": "assistant", "content": "<think></think>"})
        self.assertTrue(payload["continue_final_message"])
        thinking = client.payload(user_message("hi"), Sampling())
        self.assertEqual(thinking["messages"], user_message("hi"))
        self.assertNotIn("continue_final_message", thinking)

    def test_thinking_budget_rides_in_custom_params(self):
        # sglang reads the budget from custom_params rather than from a sampling field, and it is what
        # closes a think block the template cannot switch off - see accuracy/glm53_flash/_base.yaml
        client = Client(self.server.base_url, model="served")
        payload = client.payload(user_message("hi"), Sampling(thinking_budget=8192))
        self.assertEqual(payload["custom_params"], {"thinking_budget": 8192})
        # A budget of zero is a value to send, not an absence
        zero = client.payload(user_message("hi"), Sampling(thinking_budget=0))
        self.assertEqual(zero["custom_params"], {"thinking_budget": 0})
        self.assertNotIn("custom_params", client.payload(user_message("hi"), Sampling()))

    def test_logprobs_are_asked_for_only_when_named_and_the_first_token_is_kept(self):
        plain = Client(self.server.base_url)
        self.assertNotIn("logprobs", plain.payload(user_message("hi"), Sampling()))
        self.assertIsNone(send(plain, [user_message("question\nanswer here")], Sampling())[0].first_token)
        client = Client(self.server.base_url, top_logprobs=5)
        payload = client.payload(user_message("hi"), Sampling())
        self.assertEqual((payload["logprobs"], payload["top_logprobs"]), (True, 5))
        reply = send(client, [user_message("question\nanswer here")], Sampling())[0]
        self.assertEqual(reply.first_token["token"], "answer")
        self.assertEqual(reply.first_token["logprob"], -0.25)
        self.assertEqual(len(reply.first_token["top_logprobs"]), 1)

    def test_replies_keep_order_and_carry_reasoning_and_counts(self):
        client = Client(self.server.base_url, concurrency=4)
        jobs = [user_message(f"question {i}\nanswer {i}") for i in range(10)]
        seen = []
        replies = send(client, jobs, Sampling(), on_reply=lambda i, r: seen.append(i))
        self.assertEqual([r.content for r in replies], [f"answer {i}" for i in range(10)])
        self.assertEqual(sorted(seen), list(range(10)))
        reply = replies[3]
        self.assertEqual(reply.reasoning, "thinking about it")
        self.assertEqual(reply.reasoning_tokens, 3)
        self.assertEqual(reply.prompt_tokens, 4)
        self.assertEqual(reply.completion_tokens, 5)
        self.assertEqual(reply.finish_reason, "stop")
        self.assertIsNone(reply.error)
        self.assertGreater(reply.seconds, 0)
        # The server saw exactly our sampling settings on every request
        for body in self.server.requests[-10:]:
            self.assertEqual(body["temperature"], 0.0)
            self.assertEqual(body["max_tokens"], 4096)

    def test_empty_answer_when_reasoning_used_the_budget(self):
        client = Client(self.server.base_url)
        reply = send(client, [user_message("EMPTY please")], Sampling(max_tokens=16))[0]
        self.assertEqual((reply.content, reply.error), ("", None))
        self.assertEqual(reply.finish_reason, "length")
        self.assertEqual(reply.reasoning_tokens, 10)

    def test_retry_then_success(self):
        client = Client(self.server.base_url, retries=3)
        reply = send(client, [user_message("FAIL_ONCE")], Sampling())[0]
        self.assertEqual(reply.content, "recovered")
        self.assertIsNone(reply.error)

    def test_prompt_too_long_is_an_error_not_an_empty_answer(self):
        client = Client(self.server.base_url, retries=3)
        before = len(self.server.requests)
        reply = send(client, [user_message("TOO_LONG")], Sampling())[0]
        self.assertIn("longer than", reply.error)
        # No retry for a prompt that cannot fit
        self.assertEqual(len(self.server.requests) - before, 1)

    def test_a_request_that_times_out_is_an_error_and_is_not_sent_again(self):
        client = Client(self.server.base_url, timeout=0.2, retries=3)
        before = len(self.server.requests)
        reply = send(client, [user_message("SLOW")], Sampling())[0]
        self.assertTrue(reply.error.startswith("TimeoutError"), reply.error)
        self.assertEqual(len(self.server.requests) - before, 1)

    def test_parse_reply_tolerates_missing_usage(self):
        reply = parse_reply({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}, 0.5)
        self.assertEqual(reply.content, "x")
        self.assertIsNone(reply.prompt_tokens)
        self.assertIsNone(reply.reasoning_tokens)
        self.assertIsNone(reply.cached_tokens)
        self.assertEqual(reply.reasoning, "")

    def test_parse_reply_reads_the_prompt_tokens_the_cache_served(self):
        body = {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 400_000, "prompt_tokens_details": {"cached_tokens": 399_744}},
        }
        self.assertEqual(parse_reply(body, 0.5).cached_tokens, 399_744)


class SamplingFieldsTest(unittest.TestCase):
    def test_what_a_card_names_is_sent_and_what_it_leaves_out_is_not(self):
        body = Client("http://unused").payload(
            user_message("hi"),
            Sampling(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, max_tokens=8),
        )
        self.assertEqual((body["top_k"], body["min_p"], body["presence_penalty"]), (20, 0.0, 1.5))
        self.assertNotIn("top_k", Client("http://unused").payload(user_message("hi"), Sampling()))

    def test_template_switches_and_effort_share_one_kwargs_mapping(self):
        sampling = Sampling(chat_template_kwargs={"enable_thinking": False}, reasoning_effort="low")
        body = Client("http://unused").payload(user_message("hi"), sampling)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False, "reasoning_effort": "low"})

    def test_a_request_with_no_temperature_is_refused_rather_than_left_to_the_server(self):
        with self.assertRaisesRegex(ValueError, "generation block"):
            Client("http://unused").payload(user_message("hi"), Sampling(temperature=None))
