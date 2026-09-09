import unittest

from rlraft.core.raft_rules import decide_vote, is_candidate_log_up_to_date


class RaftVoteRuleTests(unittest.TestCase):
    def test_rejects_stale_term(self) -> None:
        decision = decide_vote(
            current_term=3,
            voted_for=None,
            candidate_term=2,
            candidate_id=1,
            candidate_last_index=0,
            candidate_last_term=1,
            local_last_index=0,
            local_last_term=1,
        )
        self.assertFalse(decision.granted)
        self.assertEqual(decision.reason, "stale_term")
        self.assertEqual(decision.term, 3)

    def test_one_vote_per_term(self) -> None:
        decision = decide_vote(
            current_term=4,
            voted_for=1,
            candidate_term=4,
            candidate_id=2,
            candidate_last_index=3,
            candidate_last_term=4,
            local_last_index=3,
            local_last_term=4,
        )
        self.assertFalse(decision.granted)
        self.assertEqual(decision.reason, "already_voted")
        self.assertEqual(decision.voted_for, 1)

    def test_grants_vote_for_newer_term_and_log(self) -> None:
        decision = decide_vote(
            current_term=4,
            voted_for=1,
            candidate_term=5,
            candidate_id=2,
            candidate_last_index=5,
            candidate_last_term=5,
            local_last_index=3,
            local_last_term=4,
        )
        self.assertTrue(decision.granted)
        self.assertTrue(decision.step_down)
        self.assertEqual(decision.term, 5)
        self.assertEqual(decision.voted_for, 2)

    def test_log_freshness_uses_term_then_index(self) -> None:
        self.assertTrue(is_candidate_log_up_to_date(2, 5, 10, 4))
        self.assertFalse(is_candidate_log_up_to_date(10, 4, 2, 5))
        self.assertTrue(is_candidate_log_up_to_date(5, 5, 4, 5))
        self.assertFalse(is_candidate_log_up_to_date(4, 5, 5, 5))


if __name__ == "__main__":
    unittest.main()
