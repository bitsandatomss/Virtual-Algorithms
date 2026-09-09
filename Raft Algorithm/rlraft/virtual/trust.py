"""Trust machinery: OOD detection, uncertainty, calibration, gating.

context.txt trust requirements:
  1. calibration -- does the surrogate know when it doesn't know?
  2. conservation constraints -- delegated to invariants.py
  3. OOD detection -- recognize unfamiliar regimes
  4. uncertainty propagation -- compounding over multi-step horizons
  5. active selection -- which real experiment maximally reduces uncertainty?
"""

from __future__ import annotations

import math

from rlraft.virtual.base import MaturityLevel


class FeatureDistribution:
    """Diagonal-Gaussian model of the training virtual-state distribution.

    OOD score = mean absolute z-score, clipped to [0, ~5]. Cheap, no
    dependencies, sufficient to catch regime shifts (e.g. 200-node cluster
    after training on 20-node clusters, or loss_rate far outside training).
    """

    def __init__(self) -> None:
        self.mean: list[float] = []
        self.var: list[float] = []
        self.count = 0

    def fit(self, rows: list[list[float]]) -> None:
        if not rows:
            return
        dim = len(rows[0])
        n = len(rows)
        mean = [sum(r[d] for r in rows) / n for d in range(dim)]
        var = [
            sum((r[d] - mean[d]) ** 2 for r in rows) / max(n - 1, 1)
            for d in range(dim)
        ]
        self.mean = mean
        self.var = [max(v, 1e-3) for v in var]
        self.count = n

    def ood_score(self, features: list[float]) -> float:
        if not self.mean:
            return 1.0  # unfitted => unknown regime
        total = 0.0
        for x, m, v in zip(features, self.mean, self.var):
            total += abs(x - m) / math.sqrt(v)
        return min(total / max(len(features), 1), 5.0)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureDistribution":
        fd = cls()
        fd.mean = list(data.get("mean", []))
        fd.var = list(data.get("var", []))
        fd.count = int(data.get("count", 0))
        return fd


class UncertaintyTracker:
    """Ensemble-disagreement + horizon growth model.

    u(h) = base * (growth ** h). Multi-step virtual rollouts compound
    uncertainty geometrically; the TrustGate refuses horizons whose
    propagated uncertainty exceeds the budget.
    """

    def __init__(self, growth: float = 1.35) -> None:
        self.growth = growth

    @staticmethod
    def ensemble_uncertainty(predictions: list[float]) -> float:
        if len(predictions) < 2:
            return 0.5
        mean = sum(predictions) / len(predictions)
        var = sum((p - mean) ** 2 for p in predictions) / len(predictions)
        return math.sqrt(var)

    def propagate(self, base: float, horizon: int) -> float:
        return base * (self.growth ** max(horizon - 1, 0))


class Calibrator:
    """Binning estimator of expected calibration error (ECE).

    Tracks (predicted_prob, empirical_outcome) pairs. A surrogate with
    99.99% accuracy in-distribution but confident + wrong OOD gets a
    large ECE and loses trust -- the 'explosion' scenario from context.txt.
    """

    def __init__(self, bins: int = 10) -> None:
        self.bins = bins
        self.pairs: list[tuple[float, float]] = []

    def add(self, predicted: float, outcome: float) -> None:
        self.pairs.append((min(max(predicted, 0.0), 1.0), min(max(outcome, 0.0), 1.0)))

    def ece(self) -> float:
        if not self.pairs:
            return 1.0
        buckets: list[list[tuple[float, float]]] = [[] for _ in range(self.bins)]
        for p, o in self.pairs:
            idx = min(int(p * self.bins), self.bins - 1)
            buckets[idx].append((p, o))
        total = len(self.pairs)
        err = 0.0
        for b in buckets:
            if not b:
                continue
            mean_p = sum(p for p, _ in b) / len(b)
            mean_o = sum(o for _, o in b) / len(b)
            err += len(b) / total * abs(mean_p - mean_o)
        return err


class TrustGate:
    """Decides: is this virtual experiment trustworthy?
    Refuses when ANY of: invariant violation, OOD beyond budget,
    uncertainty beyond budget, calibration error beyond budget.
    Assigns maturity: EXEMPLAR (default) -> INTERPOLATIVE (in-dist, low
    uncertainty) -> COUNTERFACTUAL (intervention + still trusted).
    Higher levels (MECHANISTIC/SCIENTIFIC) require longitudinal evidence
    and are assigned by VirtualLab, not the gate.
    """

    def __init__(
        self,
        ood_budget: float = 1.6,
        uncertainty_budget: float = 0.45,
        ece_budget: float = 0.18,
    ) -> None:
        self.ood_budget = ood_budget
        self.uncertainty_budget = uncertainty_budget
        self.ece_budget = ece_budget

    def decide(
        self,
        ood_score: float,
        uncertainty: float,
        calibration_error: float,
        invariant_violations: list[str],
        is_intervention: bool = False,
    ) -> tuple[bool, list[str], MaturityLevel]:
        reasons: list[str] = []
        if invariant_violations:
            reasons.append(f"invariant_violation:{','.join(invariant_violations)}")
        if ood_score > self.ood_budget:
            reasons.append(f"ood:{ood_score:.2f}>budget:{self.ood_budget}")
        if uncertainty > self.uncertainty_budget:
            reasons.append(f"uncertain:{uncertainty:.2f}>budget:{self.uncertainty_budget}")
        if calibration_error > self.ece_budget:
            reasons.append(f"miscalibrated:{calibration_error:.2f}>budget:{self.ece_budget}")
        trusted = not reasons
        if not trusted:
            maturity = MaturityLevel.EXEMPLAR
        elif is_intervention:
            maturity = MaturityLevel.COUNTERFACTUAL
            reasons.append("counterfactual:trusted_intervention")
        else:
            maturity = MaturityLevel.INTERPOLATIVE
            reasons.append("interpolative:in_distribution")
        return trusted, reasons, maturity


class MahalanobisOOD:
    """Full-covariance Gaussian OOD detector (numpy, no sklearn needed).

    Score = sqrt Mahalanobis distance to the training feature mean.
    Unlike FeatureDistribution (diagonal), this respects feature
    correlations (e.g. nodes/loss co-vary across regimes). Budgets come
    from the in-distribution score distribution
    (budget_from_percentile), never from magic numbers.
    """

    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self.mean: list[float] = []
        self.precision: list[list[float]] = []
        self.count = 0

    def fit(self, rows: list[list[float]]) -> None:
        import numpy as np

        X = np.asarray(rows, dtype=float)
        self.mean = X.mean(axis=0).tolist()
        cov = np.cov(X, rowvar=False) + self.reg * np.eye(X.shape[1])
        self.precision = np.linalg.inv(cov).tolist()
        self.count = len(rows)

    def scores(self, rows: list[list[float]]) -> list[float]:
        import numpy as np

        if not self.mean:
            return [float("nan")] * len(rows)
        m = np.asarray(self.mean)
        P = np.asarray(self.precision)
        out = []
        for r in rows:
            d = np.asarray(r, dtype=float) - m
            out.append(float(np.sqrt(max(d @ P @ d, 0.0))))
        return out

    def ood_score(self, features: list[float]) -> float:
        return self.scores([features])[0]

    @staticmethod
    def budget_from_percentile(scores: list[float], percentile: float = 95.0) -> float:
        import numpy as np

        return float(np.percentile(np.asarray(scores, dtype=float), percentile))

    @staticmethod
    def auroc(in_scores: list[float], ood_scores: list[float]) -> float:
        """P(random OOD score > random in-dist score). 0.5 = useless."""
        wins = draws = 0
        for o in ood_scores:
            for i in in_scores:
                if o > i:
                    wins += 1
                elif o == i:
                    draws += 1
        total = len(in_scores) * max(len(ood_scores), 1)
        return (wins + 0.5 * draws) / max(total, 1)


def reliability_table(
    pairs: list[tuple[float, float]], bins: int = 10,
) -> list[dict[str, float]]:
    """Per-bin (predicted, empirical, count) for reliability diagrams."""
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for p, o in pairs:
        buckets[min(int(min(max(p, 0.0), 1.0) * bins), bins - 1)].append((p, o))
    table = []
    for b in buckets:
        if not b:
            table.append({"predicted": float("nan"), "empirical": float("nan"), "count": 0})
        else:
            table.append({
                "predicted": sum(p for p, _ in b) / len(b),
                "empirical": sum(o for _, o in b) / len(b),
                "count": len(b),
            })
    return table


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate (no normal approx)."""
    import math

    if total == 0:
        return (0.0, 1.0)
    p = hits / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (max((center - margin) / denom, 0.0), min((center + margin) / denom, 1.0))
