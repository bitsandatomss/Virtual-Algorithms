import unittest

from rlraft.virtual.eval import (
    abstention_gain,
    bootstrap_ci,
    horizon_growth_fit,
    pearson,
    ranking_consistency,
    run_audit,
    spearman,
    sufficiency_probe,
    value_consistency,
)
from rlraft.virtual.trust import MahalanobisOOD, reliability_table, wilson_interval


class EvalMathTests(unittest.TestCase):
    def test_spearman_perfect_and_inverse(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman([1, 1, 1], [1, 2, 3]), 0.0)

    def test_wilson_covers_extremes(self) -> None:
        lo, hi = wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.2)
        lo, hi = wilson_interval(10, 10)
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)

    def test_bootstrap_ci_contains_mean(self) -> None:
        lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4], resamples=200, seed=0)
        self.assertLessEqual(lo, 0.25)
        self.assertGreaterEqual(hi, 0.25)

    def test_ranking_regret_zero_when_best_matches(self) -> None:
        pred = {"a": 0.9, "b": 0.5}
        emp = {"a": {"success_rate": 0.8}, "b": {"success_rate": 0.4}}
        rc = ranking_consistency(pred, emp)
        self.assertAlmostEqual(rc["spearman_rho"], 1.0)
        self.assertAlmostEqual(rc["regret"], 0.0)

    def test_ranking_detects_inversion(self) -> None:
        pred = {"a": 0.9, "b": 0.1}
        emp = {"a": {"success_rate": 0.2}, "b": {"success_rate": 0.8}}
        rc = ranking_consistency(pred, emp)
        self.assertLess(rc["spearman_rho"], 0.0)
        self.assertGreater(rc["regret"], 0.4)

    def test_value_consistency_ci_coverage(self) -> None:
        pred = {"a": 0.5}
        emp = {"a": {"success_rate": 0.5, "success_lo": 0.3, "success_hi": 0.7}}
        vc = value_consistency(pred, emp)
        self.assertAlmostEqual(vc["mae"], 0.0)
        self.assertAlmostEqual(vc["ci_coverage"], 1.0)

    def test_abstention_gain_positive_when_gate_helps(self) -> None:
        gain = abstention_gain([0.5, 0.0, 0.0], [False, True, True])
        self.assertGreater(gain["gain"], 0.15)
        self.assertAlmostEqual(gain["coverage"], 2 / 3)

    def test_horizon_fit_recovers_growth(self) -> None:
        fit = horizon_growth_fit([1, 2, 3, 4], [0.1, 0.135, 0.182, 0.246])
        self.assertGreater(fit["growth"], 1.2)
        self.assertLess(fit["growth"], 1.5)
        self.assertEqual(fit["fitted"], 1.0)

    def test_horizon_fit_falls_back_without_data(self) -> None:
        fit = horizon_growth_fit([1], [0.1])
        self.assertEqual(fit["fitted"], 0.0)

    def test_reliability_table_counts(self) -> None:
        table = reliability_table([(0.1, 0.0), (0.9, 1.0)], bins=2)
        self.assertEqual(table[0]["count"], 1)
        self.assertEqual(table[1]["count"], 1)
        self.assertAlmostEqual(table[1]["empirical"], 1.0)


class MahalanobisTests(unittest.TestCase):
    def test_separates_near_from_far(self) -> None:
        from rlraft.virtual.surrogate import cluster_features

        det = MahalanobisOOD()
        train = [cluster_features(5, 120.0, 0.02, 0.2, False, a) for a in range(5)] * 4
        det.fit(train)
        near = det.ood_score(cluster_features(5, 130.0, 0.03, 0.2, False, 2))
        far = det.ood_score(cluster_features(100, 800.0, 0.8, 4.0, True, 0))
        self.assertLess(near, far)

    def test_budget_is_percentile_not_magic(self) -> None:
        det = MahalanobisOOD()
        train = [[float(i), float(i)] for i in range(20)]
        det.fit(train)
        scores = det.scores(train)
        budget = MahalanobisOOD.budget_from_percentile(scores, 95.0)
        inside = sum(1 for s in scores if s <= budget)
        self.assertGreaterEqual(inside / len(scores), 0.9)

    def test_auroc_random_is_half(self) -> None:
        self.assertAlmostEqual(
            MahalanobisOOD.auroc([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), 0.5
        )
        self.assertAlmostEqual(
            MahalanobisOOD.auroc([1.0, 2.0], [3.0, 4.0]), 1.0
        )


class EnsembleTests(unittest.TestCase):
    def test_ensemble_trains_and_disagrees_honestly(self) -> None:
        from rlraft.virtual.surrogate import EnsembleDynamicsSurrogate, TrajectoryDataset

        ds = TrajectoryDataset().collect_episodes([5], 6, seed=0)
        self.assertEqual(len(ds.episodes), 30)  # 5 arms x 6 episodes
        train, val = ds.train_val_split(0.3, seed=0)
        ens = EnsembleDynamicsSurrogate(members=2, hidden_dim=16)
        ens.train(train, epochs=10, seed=0)
        pred = ens.predict(val[0][0])
        self.assertIn("success_prob", pred)
        self.assertIn("member_std", pred)
        self.assertGreaterEqual(pred["uncertainty"], pred["member_std"])

    def test_sufficiency_probe_ratio_bounded(self) -> None:
        from rlraft.virtual.surrogate import TrajectoryDataset

        ds = TrajectoryDataset().collect_episodes([5], 6, seed=1)
        probe = sufficiency_probe(ds.episodes)
        self.assertGreaterEqual(probe["ratio"], 0.0)
        self.assertGreater(probe["n_groups"], 0)


class AuditSmokeTests(unittest.TestCase):
    def test_audit_smoke_reports_maturity(self) -> None:
        report = run_audit(
            train_nodes=[5], test_nodes=10, ood_nodes=20,
            episodes_per_arm=6, ensemble_members=2, epochs=10, seed=0,
            stress_loss=False,
        )
        self.assertIn(report["maturity"],
                      {"EXEMPLAR", "INTERPOLATIVE", "COUNTERFACTUAL", "MECHANISTIC"})
        self.assertIn("heldout_mae_ci95", report)
        self.assertIn("ranking_consistency", report)
        self.assertIn("ood_benchmark", report)


class EnsembleRegressorTests(unittest.TestCase):
    def test_fits_line_and_reports_disagreement(self) -> None:
        from rlraft.virtual.surrogate import EnsembleRegressor

        rows = [([float(x)], [2.0 * x + 1.0]) for x in range(10)]
        ens = EnsembleRegressor(input_dim=1, output_dim=1, members=3, hidden_dim=16)
        stats = ens.train(rows, epochs=120, seed=0)
        self.assertLess(stats["train_mse"], 0.5)
        mean, std = ens.predict([5.0])
        self.assertAlmostEqual(mean[0], 11.0, delta=1.5)
        self.assertGreaterEqual(std[0], 0.0)

    def test_vector_targets(self) -> None:
        from rlraft.virtual.surrogate import EnsembleRegressor

        rows = [([float(x)], [float(x), -float(x)]) for x in range(8)]
        ens = EnsembleRegressor(input_dim=1, output_dim=2, members=2, hidden_dim=16)
        ens.train(rows, epochs=120, seed=0)
        mean, std = ens.predict([4.0])
        self.assertEqual(len(mean), 2)
        self.assertEqual(len(std), 2)


class SignificanceHelperTests(unittest.TestCase):
    def test_interval_coverage_counts(self) -> None:
        from rlraft.virtual.eval import interval_coverage

        self.assertAlmostEqual(interval_coverage([0.1, 0.5], [0.1, 0.1], k=2.0), 0.5)
        self.assertAlmostEqual(interval_coverage([], []), 0.0)

    def test_paired_compare_detects_winner(self) -> None:
        from rlraft.virtual.eval import paired_compare

        rep = paired_compare([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], seed=0)
        self.assertLess(rep["mean_diff"], 0.0)
        self.assertEqual(rep["win_rate_a"], 1.0)
        self.assertLess(rep["ci_hi"], 0.0)

    def test_paired_compare_rejects_mismatched(self) -> None:
        from rlraft.virtual.eval import paired_compare

        with self.assertRaises(ValueError):
            paired_compare([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
