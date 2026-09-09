import random
import sys
import unittest
from pathlib import Path

# Runnable from anywhere: algo folder for the local package,
# Raft Algorithm for the shared rlraft core.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "Raft Algorithm"))

import tcp_algo  # noqa: E402  (path bootstrap above)


class TCPTests(unittest.TestCase):
    def test_package_exports(self) -> None:
        self.assertTrue(hasattr(tcp_algo, "TCPCongestion"))

    def test_throughput_never_exceeds_bottleneck(self) -> None:
        from tcp_algo.tcp import simulate_aimd

        rng = random.Random(0)
        for bw in (5.0, 25.0, 100.0):
            for cap in (4, 16, 64):
                for loss in (0.0, 0.02):
                    out = simulate_aimd(50.0, bw, 32, loss, cap, 120, rng)
                    self.assertLessEqual(out["throughput_mbps"], bw + 1e-9)
                    self.assertGreaterEqual(out["throughput_mbps"], 0.0)

    def test_cwnd_trace_conservation(self) -> None:
        from rlraft.virtual.invariants import TCPInvariants

        self.assertEqual(TCPInvariants.check_trace([1.0, 4.0, 8.0], 8.0), [])
        self.assertIn("trace_cwnd_below_one_mss",
                      TCPInvariants.check_trace([0.2, 4.0], 8.0))
        self.assertIn("trace_cwnd_exceeds_cap",
                      TCPInvariants.check_trace([1.0, 99.0], 8.0))

    def test_prediction_invariants_reject_garbage(self) -> None:
        from rlraft.virtual.invariants import TCPInvariants

        bad = {"throughput_mbps": 999.0, "cap_pkts": 8.0, "loss_rate": 0.0, "rtt_ms": 10.0}
        self.assertIn("throughput_exceeds_bottleneck",
                      TCPInvariants.check_prediction(bad, 10.0))
        bad2 = dict(bad, throughput_mbps=5.0, loss_rate=2.0)
        self.assertIn("loss_out_of_range", TCPInvariants.check_prediction(bad2, 10.0))

    def test_learned_model_ranks_caps(self) -> None:
        from tcp_algo.tcp import (
            AimdThroughputModel, TCPCongestion, collect_tcp_trajectories,
            fit_ensemble,
        )

        rows = collect_tcp_trajectories(
            grid=[{"bw_mbps": 25.0, "base_rtt_ms": 50.0, "loss_p": 0.005, "buf_pkts": 32}],
            episodes_per_setting=6, rtts=60, seed=0,
        )
        model = AimdThroughputModel().fit([(f, t) for f, t, _ in rows])
        ensemble = fit_ensemble(rows, members=3, epochs=60, seed=0)
        virtual = TCPCongestion(model=model, ensemble=ensemble)
        virtual.ood.fit([f for f, _, _ in rows])
        audit = virtual.audit(
            {"bw_mbps": 25.0, "base_rtt_ms": 50.0, "loss_p": 0.005, "buf_pkts": 32},
            episodes_per_cap=6, rtts=60, seed=1,
        )
        self.assertGreaterEqual(audit["ranking"]["spearman_rho"], 0.5)
        self.assertLessEqual(audit["ranking"]["regret_fraction"], 0.5)
        self.assertIn("ablation_linear", audit)
        self.assertIn("per_episode", audit["uncertainty"])

    def test_ensemble_beats_linear_on_values(self) -> None:
        from tcp_algo.tcp import (
            AimdThroughputModel, TCPCongestion, collect_tcp_trajectories,
            fit_ensemble,
        )

        rows = collect_tcp_trajectories(seed=0)
        model = AimdThroughputModel().fit([(f, t) for f, t, _ in rows])
        ensemble = fit_ensemble(rows, members=3, epochs=80, seed=0)
        virtual = TCPCongestion(model=model, ensemble=ensemble)
        audit = virtual.audit(
            {"bw_mbps": 25.0, "base_rtt_ms": 50.0, "loss_p": 0.005, "buf_pkts": 32},
            episodes_per_cap=6, rtts=60, seed=1,
        )
        self.assertLessEqual(audit["value"]["mae"], audit["ablation_linear"]["value"]["mae"])

    def test_cap_policy_picks_optimal(self) -> None:
        from tcp_algo.tcp import (
            AimdThroughputModel, TCPCongestion, collect_tcp_trajectories,
            fit_ensemble,
        )

        rows = collect_tcp_trajectories(seed=0)
        virtual = TCPCongestion(
            model=AimdThroughputModel().fit([(f, t) for f, t, _ in rows]),
            ensemble=fit_ensemble(rows, members=3, epochs=80, seed=0),
        )
        report = virtual.evaluate_policy(
            [{"bw_mbps": 25.0, "base_rtt_ms": 50.0, "loss_p": 0.005, "buf_pkts": 32}],
            episodes_per_cap=6, rtts=60, seed=1,
        )
        self.assertLessEqual(report["ensemble"]["mean_regret"], 0.05)


if __name__ == "__main__":
    unittest.main()
