import random
import sys
import unittest
from pathlib import Path

# Runnable from anywhere: algo folder for the local package,
# Raft Algorithm for the shared rlraft core.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "Raft Algorithm"))

import paging_algo  # noqa: E402  (path bootstrap above)


class PagingTests(unittest.TestCase):
    def test_package_exports(self) -> None:
        self.assertTrue(hasattr(paging_algo, "Paging"))

    def test_belady_is_lower_bound(self) -> None:
        rng = random.Random(0)
        from paging_algo.paging import belady_min_faults, generate_refs, simulate_paging

        for _ in range(5):
            refs = generate_refs(32, 6, 400, 150, rng)
            min_f = belady_min_faults(refs, 6)
            for pol in ("lru", "lfu", "random"):
                faults = int(simulate_paging(refs, 6, pol, rng)["faults"])
                self.assertGreaterEqual(faults, min_f, pol)

    def test_resident_bounded_and_faults_are_misses(self) -> None:
        rng = random.Random(1)
        from rlraft.virtual.invariants import PagingInvariants
        from paging_algo.paging import generate_refs, simulate_paging

        refs = generate_refs(32, 6, 400, 150, rng)
        out = simulate_paging(refs, 6, "lru", rng)
        self.assertEqual(
            PagingInvariants.check_outcome(
                int(out["faults"]), len(refs), int(out["max_resident"]), 6), [])
        self.assertEqual(
            PagingInvariants.check_outcome(10, 100, 99, 6),
            ["resident_exceeds_frames"])

    def test_learned_evictor_matches_lru_on_train_by_construction(self) -> None:
        """Grid contains LRU-equivalent weights, so fitted cost <= LRU cost
        on the training workloads. Guards against argmin/argmax regressions."""
        rng = random.Random(2)
        from paging_algo.paging import (
            fit_learned_evictor, generate_refs, simulate_learned_eviction,
            simulate_paging,
        )

        train = [(generate_refs(32, 6, 300, 150, rng), 6) for _ in range(2)]
        w = fit_learned_evictor(train, seed=0)
        learned_cost = sum(simulate_learned_eviction(r, f, w["w_rec"], w["w_freq"])["faults"]
                           for r, f in train)
        lru_cost = sum(simulate_paging(r, f, "lru", rng)["faults"] for r, f in train)
        self.assertLessEqual(learned_cost, lru_cost)

    def test_policy_ranking_audit(self) -> None:
        from paging_algo.paging import (
            PolicyFaultModel, Paging, collect_paging_trajectories,
            fit_fault_ensemble, fit_learned_evictor,
        )

        rows = collect_paging_trajectories(
            workloads=[{"pages": 32, "hot": 6, "length": 500, "shift_every": 200, "frames": 6}],
            seed=0,
        )
        model = PolicyFaultModel().fit([(f, r) for f, r, _, _ in rows])
        fensemble = fit_fault_ensemble(rows, members=3, epochs=60, seed=0)
        weights = fit_learned_evictor([(refs, fr) for _, _, refs, fr in rows])
        virtual = Paging(model=model, fensemble=fensemble, evictor_weights=weights)
        virtual.ood.fit([f for f, _, _, _ in rows])
        audit = virtual.audit(
            {"pages": 32, "hot": 8, "length": 500, "shift_every": 200, "frames": 6},
            seed=1,
        )
        self.assertGreaterEqual(audit["ranking"]["spearman_rho"], 0.0)
        self.assertGreaterEqual(audit["learned_faults"], audit["belady_min_faults"])
        self.assertIn("ablation_linear", audit)
        self.assertIn("uncertainty", audit)

    def test_evictor_suite_beats_lru_with_significance(self) -> None:
        from paging_algo.paging import (
            PolicyFaultModel, Paging, collect_paging_trajectories,
            fit_learned_evictor,
        )

        rows = collect_paging_trajectories(
            workloads=[{"pages": 32, "hot": 6, "length": 500, "shift_every": 200, "frames": 6},
                       {"pages": 32, "hot": 10, "length": 500, "shift_every": 100, "frames": 8}],
            seed=0,
        )
        weights = fit_learned_evictor([(refs, fr) for _, _, refs, fr in rows])
        virtual = Paging(evictor_weights=weights)
        report = virtual.evaluate_evictor(
            [{"pages": 32, "hot": 6, "length": 500, "shift_every": 200, "frames": 6},
             {"pages": 32, "hot": 10, "length": 500, "shift_every": 100, "frames": 8}],
            seeds=(0, 1),
        )
        paired = report["paired_learned_minus_lru"]
        self.assertLessEqual(paired["mean_diff"], 0.0)
        self.assertGreaterEqual(paired["win_rate_a"], 0.5)


if __name__ == "__main__":
    unittest.main()
