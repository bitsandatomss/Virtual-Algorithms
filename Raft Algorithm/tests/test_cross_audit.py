import unittest


class CrossAuditTests(unittest.TestCase):
    def test_matrix_light_backoff_gossip(self) -> None:
        from rlraft.virtual.cross_audit import run_matrix

        report = run_matrix(algos=["backoff", "gossip"], seeds=(7,), light=True)
        self.assertEqual(report["seeds"], [7])
        self.assertIn("backoff", report["matrix"])
        self.assertIn("gossip", report["matrix"])
        backoff = report["matrix"]["backoff"]
        # backoff is capped by construction, whatever the numbers say
        self.assertEqual(backoff["maturity"], "INTERPOLATIVE")
        self.assertIn("no decision semantics", backoff["maturity_reason"])
        gossip = report["matrix"]["gossip"]
        self.assertIn(gossip["maturity"], {"INTERPOLATIVE", "COUNTERFACTUAL", "MECHANISTIC"})
        for entry in (backoff, gossip):
            for key in ("rho", "regret_fraction", "mae", "ci_coverage", "auroc"):
                self.assertIn("mean", entry[key])
            self.assertEqual(len(entry["per_seed"]), 1)

    def test_gossip_audit_fanout_ranking(self) -> None:
        import random

        from rlraft.virtual.algorithms import Gossip
        from rlraft.virtual.base import Intervention

        g = Gossip(nodes=20)
        rng = random.Random(0)
        # interior regime (see cross_audit._gossip): saturated observations
        # leave p unidentifiable from below
        obs = [g.physical_step({}, Intervention(
            "spread", {"fanout": 2, "rounds": 3, "p": 0.7}), rng)["coverage"]
            for _ in range(8)]
        g.fit_p(obs, fanout=2, rounds=3)
        # fitted p must be near the true 0.7 (grid resolution 0.05)
        self.assertLessEqual(abs(g.p - 0.7), 0.10)
        audit = g.audit(episodes_per_fanout=6, seed=0)
        self.assertGreaterEqual(audit["ranking"]["spearman_rho"], 0.5)
        # calibrator is fed by fit_p: gate ECE must be real, not a bypass
        self.assertGreater(len(g.calibrator.pairs), 0)
        self.assertLess(g.calibrator.ece(), 1.0)

    def test_backoff_audit_ood_refusal(self) -> None:
        import random

        from rlraft.virtual.algorithms import Backoff

        vb = Backoff()
        rng = random.Random(0)
        vb.fit([(f, vb.physical_delay(f, rng)) for f in range(9) for _ in range(5)])
        audit = vb.audit(seed=0, samples_per_bucket=10)
        self.assertLess(audit["value"]["mae"], 0.01)
        self.assertEqual(audit["ood_refusal_rate"], 1.0)
        self.assertEqual(audit["maturity"], "INTERPOLATIVE")


if __name__ == "__main__":
    unittest.main()
