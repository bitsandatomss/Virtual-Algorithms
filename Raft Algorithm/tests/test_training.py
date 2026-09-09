import json
import os
import random
import unittest
from pathlib import Path
import uuid

from rlraft.rl.llm_node import LLMNodePolicyClient
from rlraft.rl.mappo import (
    ActorCritic,
    MAPPOConfig,
    _collect_episode_rollout,
    train_mappo_policy,
)
from rlraft.rl.training import train_policy
from rlraft.sim.sim import generate_conditions


class TrainingTests(unittest.TestCase):
    def test_training_writes_policy(self) -> None:
        tmp = Path("runs") / "test-artifacts" / f"training-{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        path = tmp / "policy.json"
        result = train_policy(episodes=200, nodes=20, output_path=str(path), seed=3)
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["policy_type"], "tabular_marl_timeout_q_policy")
        self.assertEqual(data["algorithm"], "independent-q-learning-with-parameter-sharing")
        self.assertIn("q_values", data)
        self.assertIn("greedy_policy", data)
        self.assertIn("evaluation_metrics", data)
        self.assertGreaterEqual(result.success_rate, 0.0)
        self.assertLessEqual(result.split_vote_rate, 1.0)

    def test_mappo_rollout_has_gae_training_fields(self) -> None:
        rng = random.Random(12)
        model = ActorCritic(local_dim=5, global_dim=7, action_dim=5, hidden_dim=16)
        records, source_counts = _collect_episode_rollout(
            model=model,
            conditions=generate_conditions(6, rng),
            rng=rng,
            config=MAPPOConfig(max_rounds=3),
            llm_client=None,
        )
        self.assertGreater(len(records), 0)
        self.assertEqual(source_counts, {})
        self.assertTrue(all("advantage" in record for record in records))
        self.assertTrue(all("return" in record for record in records))
        self.assertTrue(all("old_log_prob" in record for record in records))

    def test_mappo_training_writes_integrated_artifact(self) -> None:
        tmp = Path("runs") / "test-artifacts" / f"mappo-{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        path = tmp / "llm_mappo.json"
        result = train_mappo_policy(
            episodes=2,
            nodes=5,
            output_path=str(path),
            seed=9,
            batch_episodes=1,
            use_llm_prior=False,
            eval_episodes=5,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["policy_type"], "mappo_timeout_policy")
        self.assertEqual(data["algorithm"], "mappo_ctde_gae")
        self.assertIn("gae_lambda", data)
        self.assertIn("actor_critic", data)
        self.assertGreaterEqual(result.success_rate, 0.0)

    def test_llm_node_fallback_prefers_fast_fresh_nodes(self) -> None:
        old_key = os.environ.get("OPENAI_API_KEY")
        old_key_2 = os.environ.get("OPENAI_API_KEY_2")
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENAI_API_KEY_2"] = ""
        rng = random.Random(2)
        client = LLMNodePolicyClient(require_llm=False)
        try:
            fast = client.decide(
                {
                    "node_id": 0,
                    "mean_rtt_ms": 35.0,
                    "loss_rate": 0.0,
                    "log_gap": 0,
                    "recent_split": False,
                },
                rng,
            )
            slow = client.decide(
                {
                    "node_id": 1,
                    "mean_rtt_ms": 330.0,
                    "loss_rate": 0.25,
                    "log_gap": 6,
                    "recent_split": True,
                },
                rng,
            )
            order = ["very_short", "short", "medium", "long", "very_long"]
            self.assertLess(order.index(fast.action), order.index(slow.action))
        finally:
            _restore_env("OPENAI_API_KEY", old_key)
            _restore_env("OPENAI_API_KEY_2", old_key_2)


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
