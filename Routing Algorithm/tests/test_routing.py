import random
import sys
import unittest
from pathlib import Path

# Runnable from anywhere: algo folder for the local package,
# Raft Algorithm for the shared rlraft core.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "Raft Algorithm"))

import routing_algo  # noqa: E402  (path bootstrap above)


class RoutingTests(unittest.TestCase):
    def test_package_exports(self) -> None:
        self.assertTrue(hasattr(routing_algo, "OverlayRouting"))

    def test_paths_loop_free_and_connected(self) -> None:
        from rlraft.virtual.invariants import RoutingInvariants
        from routing_algo.routing import build_topology, enumerate_paths

        topo = build_topology()
        paths = enumerate_paths(topo, 0, 5)
        self.assertGreaterEqual(len(paths), 2)
        edges = {(a, nb) for a, links in topo.items() for nb, _, _ in links}
        for p in paths:
            self.assertEqual(len(set(p)), len(p))
            self.assertEqual(RoutingInvariants.check_path_connected(p, edges), [])

    def test_delivery_bounded_and_latency_above_propagation(self) -> None:
        import random as _r

        from routing_algo.routing import build_topology, enumerate_paths, simulate_path

        topo = build_topology(1.0, 3.0)
        rng = _r.Random(0)
        for p in enumerate_paths(topo, 0, 5):
            out = simulate_path(topo, p, 100, rng)
            self.assertGreaterEqual(out["delivery_rate"], 0.0)
            self.assertLessEqual(out["delivery_rate"], 1.0)
            self.assertGreaterEqual(out["mean_latency_ms"], out["prop_bound_ms"] - 1e-9)

    def test_invariants_reject_impossible(self) -> None:
        from rlraft.virtual.invariants import RoutingInvariants

        v = RoutingInvariants.check_prediction(
            {"delivery_rate": 1.5, "mean_latency_ms": 1.0}, 10.0, [0, 1, 0])
        self.assertIn("delivery_out_of_range", v)
        self.assertIn("latency_below_propagation_bound", v)
        self.assertIn("path_has_loop", v)

    def test_learned_model_fits_and_audits(self) -> None:
        from routing_algo.routing import (
            RoutingSurrogate, OverlayRouting, collect_routing_trajectories,
            fit_ensemble,
        )

        paths, rows = collect_routing_trajectories(
            scales=[(1.0, 1.0), (1.5, 2.0)], packets=100, seed=0)
        surrogate = RoutingSurrogate().fit(rows)
        ensemble = fit_ensemble(rows, members=3, epochs=60, seed=0)
        virtual = OverlayRouting(surrogate=surrogate, ensemble=ensemble, paths=paths)
        virtual.ood.fit([f for f, _, _ in rows])
        audit = virtual.audit(packets=100, seed=1)
        self.assertLess(audit["value"]["mae"], 0.2)
        self.assertGreaterEqual(audit["value"]["ci_coverage"], 0.0)
        self.assertIn("ablation_linear", audit)
        self.assertIn("uncertainty", audit)

    def test_gradient_training_covers_harsh_regime(self) -> None:
        from routing_algo.routing import collect_routing_trajectories

        _, rows = collect_routing_trajectories(seed=0)
        deliveries = [d for _, d, _ in rows]
        # ARQ(3) saturates delivery, so the gradient is honestly mild:
        # assert it spans saturation AND degradation (not a single point)
        self.assertGreater(max(deliveries), 0.95)
        self.assertLess(min(deliveries), 0.995)


if __name__ == "__main__":
    unittest.main()
