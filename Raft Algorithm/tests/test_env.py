import os
import unittest
from pathlib import Path
import uuid

from rlraft.env import load_env


class EnvTests(unittest.TestCase):
    def test_load_env_reads_key_without_overriding_by_default(self) -> None:
        tmp = Path("runs") / "test-artifacts" / f"env-{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        env_path = tmp / ".env"
        env_path.write_text(
            "OPENAI_API_KEY=from_file\nOPENAI_MODEL=\"test-model\"\n",
            encoding="utf-8",
        )
        old_key = os.environ.get("OPENAI_API_KEY")
        old_model = os.environ.get("OPENAI_MODEL")
        try:
            os.environ["OPENAI_API_KEY"] = "already_set"
            load_env(str(env_path))
            self.assertEqual(os.environ["OPENAI_API_KEY"], "already_set")
            self.assertEqual(os.environ["OPENAI_MODEL"], "test-model")
        finally:
            _restore_env("OPENAI_API_KEY", old_key)
            _restore_env("OPENAI_MODEL", old_model)


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


class SecretScrubTests(unittest.TestCase):
    """check_llm_node_policy() output must never contain key material.

    Uses fake sentinel values only -- no real secrets are read, written,
    or asserted on. If this test fails, it prints sentinels, never keys.
    """

    SENTINEL = "sk-sentinel-NOT-A-REAL-KEY-0000"

    def test_status_report_contains_no_key_material(self) -> None:
        import json

        from rlraft.rl import llm_node

        # GROQ_API_KEY is a router-supported name; OPENAI_API_KEY is not,
        # so both are poisoned to prove the scrubber covers known AND
        # unknown *_API_KEY variables.
        old_groq = os.environ.get("GROQ_API_KEY")
        old_openai = os.environ.get("OPENAI_API_KEY")
        real_request = llm_node._request_llm_action
        os.environ["GROQ_API_KEY"] = self.SENTINEL
        os.environ["OPENAI_API_KEY"] = self.SENTINEL
        # no network in tests: force the fallback path; the poisoned error
        # string below is what the scrubber must remove
        llm_node._request_llm_action = lambda observation: None  # noqa: E731
        try:
            # poison every string the report path could echo back
            llm_node.LAST_NODE_LLM_ERROR = f"http_401:bad {self.SENTINEL} body"
            report = llm_node.check_llm_node_policy()
            dumped = json.dumps(report)
            self.assertNotIn(self.SENTINEL, dumped)
        finally:
            llm_node._request_llm_action = real_request
            llm_node.LAST_NODE_LLM_ERROR = None
            _restore_env("GROQ_API_KEY", old_groq)
            _restore_env("OPENAI_API_KEY", old_openai)

    def test_scrub_is_noop_without_secrets(self) -> None:
        from rlraft.rl.llm_node import _scrub_secrets

        report = {"source": "fallback", "llm_error": "none"}
        self.assertEqual(_scrub_secrets(dict(report)), report)


if __name__ == "__main__":
    unittest.main()
