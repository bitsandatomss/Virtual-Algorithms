"""Exhaustive + property verification of conservation invariants.

Rationale (docs/THEORY.md §1): invariants are the non-negotiable core.
Learning shapes timing; it must never break conservation. These tests
verify the hardcoded rules themselves, independent of any learned model.
"""

import random
import unittest

from rlraft.core.raft_rules import decide_vote, is_candidate_log_up_to_date
from rlraft.sim.sim import generate_conditions, simulate_election_round
from rlraft.virtual.invariants import BackoffInvariants, GossipInvariants, RaftInvariants


class VoteRuleExhaustiveTests(unittest.TestCase):
    def test_granted_implies_all_three_constraints(self) -> None:
        """Exhaustive over a small grid: a granted vote must satisfy
        freshness-of-term, single-vote, and log-up-to-dateness."""
        violations = 0
        checked = granted = 0
        for cur in (1, 2):
            for voted in (None, 1, 2):
                for cterm in (1, 2, 3):
                    for cand in (1, 2):
                        for cli, clt, lli, llt in (
                            (5, 1, 5, 1), (6, 1, 5, 1), (4, 1, 5, 1),
                            (5, 2, 5, 1), (5, 1, 5, 2),
                        ):
                            checked += 1
                            d = decide_vote(cur, voted, cterm, cand, cli, clt, lli, llt)
                            if not d.granted:
                                continue
                            granted += 1
                            if cterm < cur:
                                violations += 1
                            # Subtlety (real Raft, not a bug): a higher-term
                            # candidacy resets voted_for to None (step-down),
                            # so a prior vote in the OLD term does not block
                            # granting in the NEW term. Only same-term double
                            # votes are violations.
                            if cterm <= cur and voted not in (None, cand):
                                violations += 1
                            if not is_candidate_log_up_to_date(cli, clt, lli, llt):
                                violations += 1
        self.assertGreater(granted, 0)  # grid must exercise grants
        self.assertEqual(violations, 0, f"{violations}/{granted} bad grants of {checked}")
        # cross-check the declarative invariant helper agrees
        bad = RaftInvariants.check_vote(2, 1, 2, 2, 5, 1, 5, 1)
        self.assertEqual(bad, [])

    def test_stale_term_never_granted(self) -> None:
        d = decide_vote(3, None, 2, 9, 10, 5, 0, 1)
        self.assertFalse(d.granted)
        self.assertEqual(d.reason, "stale_term")

    def test_double_vote_never_granted(self) -> None:
        d = decide_vote(3, 1, 3, 2, 5, 1, 5, 1)
        self.assertFalse(d.granted)
        self.assertEqual(d.reason, "already_voted")

    def test_stale_log_never_granted(self) -> None:
        d = decide_vote(3, None, 3, 2, 1, 1, 9, 3)
        self.assertFalse(d.granted)
        self.assertEqual(d.reason, "stale_log")


class ElectionConservationFuzzTests(unittest.TestCase):
    def test_simulator_never_breaks_vote_conservation(self) -> None:
        """200 randomized election rounds across cluster sizes: winner
        always holds a majority; no vote overflow."""
        from rlraft.sim.sim import StaticRandomPolicy

        rng = random.Random(1234)
        audited = 0
        for nodes in (3, 5, 11, 25):
            for _ in range(50):
                conds = generate_conditions(nodes, rng)
                res = simulate_election_round(conds, StaticRandomPolicy(), rng)
                violations = RaftInvariants.check_election_outcome(
                    dict(res.votes_by_candidate), res.leader_id, nodes, res.success,
                )
                self.assertEqual(violations, [], f"nodes={nodes} {res}")
                audited += 1
        self.assertEqual(audited, 200)

    def test_surrogate_prediction_bounds_reject_garbage(self) -> None:
        self.assertIn(
            "probability_out_of_range:success_prob",
            RaftInvariants.check_prediction(
                {"success_prob": 1.5, "split_prob": 0.1, "election_time_ms": 100.0}, 5),
        )
        self.assertIn(
            "negative_time",
            RaftInvariants.check_prediction(
                {"success_prob": 0.5, "split_prob": 0.1, "election_time_ms": -3.0}, 5),
        )

    def test_backoff_and_gossip_bounds(self) -> None:
        self.assertIn("negative_delay", BackoffInvariants.check_prediction({"delay_ms": -1.0}))
        self.assertIn("unbounded_delay", BackoffInvariants.check_prediction({"delay_ms": 1e9}))
        self.assertIn("coverage_out_of_range",
                      GossipInvariants.check_prediction({"coverage": 2.0, "infected": 1.0, "rounds": 1.0}, 10))


if __name__ == "__main__":
    unittest.main()
