import unittest

from rlraft.rl.guidance import AdvisorGuidance, deterministic_guidance, get_guidance


class GuidanceTests(unittest.TestCase):
    def test_deterministic_guidance_reacts_to_bad_metrics(self) -> None:
        guidance = deterministic_guidance({"split_vote_rate": 0.5, "best_node_win_rate": 0.1})
        self.assertGreater(guidance.split_penalty, AdvisorGuidance().split_penalty)
        self.assertGreater(guidance.best_leader_bonus, AdvisorGuidance().best_leader_bonus)

    def test_llm_reward_shaping_is_disabled(self) -> None:
        with self.assertRaises(ValueError):
            get_guidance("llm")


if __name__ == "__main__":
    unittest.main()
