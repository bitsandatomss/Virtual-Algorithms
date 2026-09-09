import random
import sys
import unittest
from pathlib import Path

# Runnable from anywhere: algo folder for the local package,
# Raft Algorithm for the shared rlraft core.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "Raft Algorithm"))

import sched_algo  # noqa: E402  (path bootstrap above)


class SchedTests(unittest.TestCase):
    def test_package_exports(self) -> None:
        self.assertTrue(hasattr(sched_algo, "Scheduling"))

    def test_sjf_beats_fcfs_same_stream(self) -> None:
        rng = random.Random(0)
        from sched_algo.sched import generate_jobs, simulate_sched

        jobs = generate_jobs(100, 0.05, 12.0, rng)
        fcfs = simulate_sched(jobs, "fcfs", rng)["mean_wait_ms"]
        sjf = simulate_sched(jobs, "sjf", rng)["mean_wait_ms"]
        self.assertLessEqual(sjf, fcfs + 1e-9)

    def test_waits_nonnegative_and_util_bounded(self) -> None:
        rng = random.Random(1)
        from rlraft.virtual.invariants import SchedInvariants
        from sched_algo.sched import generate_jobs, simulate_sched

        jobs = generate_jobs(80, 0.06, 10.0, rng)
        for disc in ("fcfs", "sjf", "rr_q1", "rr_q4"):
            out = simulate_sched(jobs, disc, rng)
            self.assertEqual(SchedInvariants.check_prediction(out), [])
            self.assertGreaterEqual(out["mean_wait_ms"], 0.0)

    def test_invariants_reject_garbage(self) -> None:
        from rlraft.virtual.invariants import SchedInvariants

        v = SchedInvariants.check_prediction(
            {"mean_wait_ms": -1.0, "mean_flow_ms": 5.0, "utilization": 0.5})
        self.assertIn("negative_wait", v)

    def test_learned_model_identifies_best_discipline(self) -> None:
        from sched_algo.sched import (
            SchedWaitModel, Scheduling, collect_sched_trajectories, fit_ensemble,
        )

        rows = collect_sched_trajectories(
            loads=[{"arrival_rate": 0.05, "mean_burst_ms": 12.0}], seed=0)
        model = SchedWaitModel().fit([(f, w) for f, w, _ in rows])
        ensemble = fit_ensemble(rows, members=3, epochs=60, seed=0)
        virtual = Scheduling(model=model, ensemble=ensemble)
        virtual.ood.fit([f for f, _, _ in rows])
        audit = virtual.audit({"arrival_rate": 0.05, "mean_burst_ms": 12.0},
                              jobs=60, repeats=4, seed=1)
        self.assertEqual(audit["ranking"]["regret_fraction"], 0.0)
        self.assertIn("ablation_linear", audit)
        self.assertIn("uncertainty", audit)

    def test_burst_variability_regime_exists(self) -> None:
        import random

        from sched_algo.sched import generate_jobs, simulate_sched

        rng = random.Random(0)
        lo = [b for _, b in generate_jobs(200, 0.05, 12.0, rng, burst_cv=0.0)]
        hi = [b for _, b in generate_jobs(200, 0.05, 12.0, rng, burst_cv=1.2)]
        import statistics

        self.assertLess(statistics.pstdev(lo), statistics.pstdev(hi))
        # variability must move the needle on waits or the feature is decorative.
        # paired streams: same seed => identical arrivals, differing bursts.
        w_lo, w_hi = [], []
        for s in (11, 22, 33):
            jobs_lo = generate_jobs(200, 0.06, 10.0, random.Random(s), burst_cv=0.0)
            jobs_hi = generate_jobs(200, 0.06, 10.0, random.Random(s), burst_cv=1.2)
            w_lo.append(simulate_sched(jobs_lo, "fcfs", random.Random(s))["mean_wait_ms"])
            w_hi.append(simulate_sched(jobs_hi, "fcfs", random.Random(s))["mean_wait_ms"])
        self.assertGreater(sum(w_hi) / len(w_hi), sum(w_lo) / len(w_lo))


if __name__ == "__main__":
    unittest.main()
