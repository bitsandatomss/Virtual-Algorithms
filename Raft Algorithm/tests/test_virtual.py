import random
import unittest

from rlraft.virtual import (
    VIRTUAL_ALGORITHM_REGISTRY,
    BackoffInvariants,
    GossipInvariants,
    RaftInvariants,
    TrustGate,
    Backoff,
    Gossip,
    VirtualLab,
    RaftElection,
    DynamicsSurrogate,
    FeatureDistribution,
    TrajectoryDataset,
)
from rlraft.virtual.base import Intervention, MaturityLevel


class VirtualAlgorithmTests(unittest.TestCase):
    def test_registry_lists_three_algorithms(self) -> None:
        self.assertIn("raft_election", VIRTUAL_ALGORITHM_REGISTRY)
        self.assertIn("backoff", VIRTUAL_ALGORITHM_REGISTRY)
        self.assertIn("gossip", VIRTUAL_ALGORITHM_REGISTRY)

    def test_raft_invariants_conserve_majority(self) -> None:
        violations = RaftInvariants.check_election_outcome(
            {0: 1}, 0, cluster_size=5, success=True,
        )
        self.assertIn("winner_without_majority", violations)
        ok = RaftInvariants.check_election_outcome(
            {0: 3}, 0, cluster_size=5, success=True,
        )
        self.assertEqual(ok, [])

    def test_raft_double_vote_never_granted(self) -> None:
        violations = RaftInvariants.check_vote(
            3, 1, 3, 2, 5, 1, 5, 1,
        )
        self.assertEqual(violations, [])  # decide_vote itself refuses; nothing granted wrongly

    def test_trust_gate_refuses_ood(self) -> None:
        gate = TrustGate(ood_budget=1.6, uncertainty_budget=0.45, ece_budget=0.18)
        trusted, reasons, maturity = gate.decide(4.5, 0.1, 0.05, [], True)
        self.assertFalse(trusted)
        self.assertEqual(maturity, MaturityLevel.EXEMPLAR)
        trusted2, _, m2 = gate.decide(0.2, 0.1, 0.05, [], True)
        self.assertTrue(trusted2)
        self.assertEqual(m2, MaturityLevel.COUNTERFACTUAL)

    def test_backoff_virtualization_preserves_monotonicity(self) -> None:
        import random as _r

        vb = Backoff(base_ms=100.0)
        rows = [(f, vb.physical_delay(f, _r.Random(f))) for f in range(9) for _ in range(5)]
        vb.fit(rows)
        delays = [vb.virtual_step(
            vb.to_virtual_state({"failures": f}),
            Intervention("retry", {"failures": f}),
        ).outcome["delay_ms"] for f in range(9)]
        self.assertEqual(BackoffInvariants.check_monotone(delays), [])

    def test_gossip_fit_and_virtual_step_trusted(self) -> None:
        g = Gossip(nodes=20, p=0.5)
        rng = random.Random(0)
        obs = [g.physical_step({}, Intervention("spread", {"fanout": 3, "rounds": 5, "p": 0.7}), rng)["coverage"]
               for _ in range(10)]
        g.fit_p(obs, fanout=3, rounds=5)
        pred = g.virtual_step(g.to_virtual_state({"fanout": 3}), Intervention("spread", {"fanout": 3, "rounds": 5}))
        self.assertTrue(0.0 <= pred.outcome["coverage"] <= 1.0)
        self.assertEqual(GossipInvariants.check_prediction(pred.outcome, 20), [])
        self.assertTrue(pred.trusted)

    def test_surrogate_trains_and_predicts(self) -> None:
        dataset = TrajectoryDataset().collect(nodes_list=[5], episodes_per_setting=8, seed=1)
        self.assertGreater(len(dataset), 0)
        surrogate = DynamicsSurrogate()
        metrics = surrogate.train(dataset, epochs=30, seed=1)
        self.assertLess(metrics["mae_success_prob"], 0.45)
        pred = surrogate.predict(dataset.rows[0][0])
        self.assertIn("success_prob", pred)

    def test_raft_virtual_vs_physical_with_trust(self) -> None:
        dataset = TrajectoryDataset().collect(nodes_list=[5], episodes_per_setting=8, seed=2)
        surrogate = DynamicsSurrogate()
        surrogate.train(dataset, epochs=30, seed=2)
        virtual = RaftElection(surrogate=surrogate)
        from rlraft.virtual.surrogate import cluster_features

        virtual.ood.fit([cluster_features(5, 120.0, 0.02, 0.2, False, a) for a in range(5)])
        for feats, outcome in dataset.rows:
            virtual.calibrator.add(surrogate.predict(feats)["success_prob"], outcome["success_prob"])
        state = virtual.to_virtual_state({"nodes": 5})
        pred = virtual.virtual_step(state, Intervention("set_timeout_arm", {"arm": "short"}))
        self.assertIn("success_prob", pred.outcome)
        # far-OOD cluster must lose trust (the 'explosion' guard)
        far = virtual.to_virtual_state({"nodes": 200, "mean_rtt_ms": 900.0, "mean_loss": 0.9})
        far_pred = virtual.virtual_step(far, Intervention("set_timeout_arm", {"arm": "short"}))
        self.assertFalse(far_pred.trusted)

    def test_virtual_lab_closed_loop(self) -> None:
        lab = VirtualLab(nodes=5, seed=3)
        result = lab.run(rounds=2, episodes_per_setting=8, epochs=30)
        self.assertEqual(len(result["rounds"]), 2)
        self.assertIn(result["final_maturity"], {"EXEMPLAR", "COUNTERFACTUAL", "SCIENTIFIC"})

    def test_ood_detector_separates_regimes(self) -> None:
        from rlraft.virtual.surrogate import cluster_features

        fd = FeatureDistribution()
        fd.fit([cluster_features(5, 120.0, 0.02, 0.2, False, a) for a in range(5)])
        near = fd.ood_score(cluster_features(5, 130.0, 0.03, 0.2, False, 2))
        far = fd.ood_score(cluster_features(200, 900.0, 0.9, 4.0, True, 0))
        self.assertLess(near, far)


if __name__ == "__main__":
    unittest.main()
