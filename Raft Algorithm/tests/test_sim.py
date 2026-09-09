import random
import unittest

from rlraft.sim.sim import (
    DynatuneLikePolicy,
    StaticRandomPolicy,
    best_connected_node,
    generate_conditions,
    simulate_failover,
)


class SimulationTests(unittest.TestCase):
    def test_best_connected_node_is_alive_min_score(self) -> None:
        rng = random.Random(1)
        conditions = generate_conditions(50, rng)
        best = best_connected_node(conditions)
        self.assertIn(best, range(50))
        self.assertTrue(conditions[best].alive)

    def test_simulator_returns_election_outcome(self) -> None:
        rng = random.Random(2)
        conditions = generate_conditions(50, rng)
        result = simulate_failover(conditions, StaticRandomPolicy(), rng)
        self.assertGreaterEqual(result.rounds, 1)
        self.assertGreaterEqual(result.candidates_started, 1)
        self.assertGreaterEqual(result.total_time_ms, 0.0)

    def test_dynatune_like_policy_is_executable(self) -> None:
        rng = random.Random(3)
        conditions = generate_conditions(75, rng)
        result = simulate_failover(conditions, DynatuneLikePolicy(), rng)
        self.assertGreaterEqual(result.rounds, 1)


if __name__ == "__main__":
    unittest.main()
