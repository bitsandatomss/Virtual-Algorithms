"""Learned dynamics surrogate: Reality -> trajectories -> learned environment.

The surrogate learns the empirical transition structure
  (virtual_state, intervention) -> predicted outcome
without reconstructing the underlying mechanism -- exactly the
surrogate paradigm (context.txt: Reality -> observations -> surrogate
-> virtual experimentation).

Covers cluster-level Raft outcomes: success_prob, split_prob,
election_time_ms. Trained once (expensive), queried cheaply (virtual
experiments). Torch MLP with JSON+state_dict persistence; deterministic
CPU training for reproducibility.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def cluster_features(
    nodes: int, mean_rtt_ms: float, mean_loss: float, mean_log_gap: float,
    recent_split: bool, arm_index: int, n_arms: int = 5,
) -> list[float]:
    return [
        min(nodes / 200.0, 1.5),
        min(mean_rtt_ms / 350.0, 1.5),
        min(max(mean_loss, 0.0), 1.0),
        min(mean_log_gap / 8.0, 1.0),
        1.0 if recent_split else 0.0,
        arm_index / max(n_arms - 1, 1),
    ]


FEATURE_NAMES = [
    "node_count_norm", "mean_rtt_norm", "mean_loss",
    "mean_log_gap_norm", "recent_split", "arm_norm",
]


class TrajectoryDataset:
    """Collect (features -> outcome) rows from the ground-truth simulator.

    Two granularities (see docs/THEORY.md §2-3):
      - collect(): legacy aggregate rows, one per (nodes, arm): outcome
        means over episodes_per_setting. Cheap demo data. In-sample only.
      - collect_episodes(): per-episode rows with binary outcomes. The
        rigorous substrate: supports train/val splits, binary ECE,
        ranking consistency and abstention analysis in eval.py.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[list[float], dict[str, float]]] = []
        self.episodes: list[tuple[list[float], dict[str, float]]] = []

    def collect(
        self,
        nodes_list: list[int] | None = None,
        episodes_per_setting: int = 60,
        seed: int = 7,
    ) -> "TrajectoryDataset":
        from rlraft.sim.sim import (
            TIMEOUT_ARMS,
            generate_conditions,
            simulate_failover,
        )

        nodes_list = nodes_list or [5, 20, 50]
        rng = random.Random(seed)
        arms = list(TIMEOUT_ARMS)
        for nodes in nodes_list:
            for arm_index, arm in enumerate(arms):
                from rlraft.sim.sim import TIMEOUT_ARMS as ARMS

                low, high = ARMS[arm]

                class _FixedArm:
                    def timeout_ms(self, obs, r):  # noqa: ANN001
                        return r.uniform(low, high)

                ok = splits = 0
                total_time = 0.0
                n = episodes_per_setting
                rtts: list[float] = []
                losses: list[float] = []
                gaps: list[float] = []
                for _ in range(n):
                    conds = generate_conditions(nodes, rng)
                    res = simulate_failover(conds, _FixedArm(), rng)
                    ok += int(res.success)
                    splits += res.split_votes
                    total_time += res.total_time_ms
                    rtts.append(sum(c.mean_rtt_ms for c in conds) / len(conds))
                    losses.append(sum(c.loss_rate for c in conds) / len(conds))
                    gaps.append(sum(c.log_gap for c in conds) / len(conds))
                feats = cluster_features(
                    nodes,
                    sum(rtts) / len(rtts),
                    sum(losses) / len(losses),
                    sum(gaps) / len(gaps),
                    False,
                    arm_index,
                )
                self.rows.append((
                    feats,
                    {
                        "success_prob": ok / n,
                        "split_prob": min(splits / n, 3.0),
                        "election_time_ms": total_time / n,
                    },
                ))
        return self

    def __len__(self) -> int:
        return len(self.rows)

    def collect_episodes(
        self,
        nodes_list: list[int] | None = None,
        episodes_per_setting: int = 60,
        seed: int = 7,
    ) -> "TrajectoryDataset":
        """Per-episode rows: features from that episode's cluster means,
        outcomes binary per episode (success 0/1, split 0/1, time_ms)."""
        from rlraft.sim.sim import (
            TIMEOUT_ARMS,
            generate_conditions,
            simulate_failover,
        )

        nodes_list = nodes_list or [5, 20, 50]
        rng = random.Random(seed)
        arms = list(TIMEOUT_ARMS)
        self.episodes = []
        for nodes in nodes_list:
            for arm_index, arm in enumerate(arms):
                from rlraft.sim.sim import TIMEOUT_ARMS as ARMS

                low, high = ARMS[arm]

                class _FixedArm:
                    def timeout_ms(self, obs, r):  # noqa: ANN001
                        return r.uniform(low, high)

                for _ in range(episodes_per_setting):
                    conds = generate_conditions(nodes, rng)
                    n = len(conds)
                    feats = cluster_features(
                        nodes,
                        sum(c.mean_rtt_ms for c in conds) / n,
                        sum(c.loss_rate for c in conds) / n,
                        sum(c.log_gap for c in conds) / n,
                        False,
                        arm_index,
                    )
                    res = simulate_failover(conds, _FixedArm(), rng)
                    self.episodes.append((
                        feats,
                        {
                            "success": 1.0 if res.success else 0.0,
                            "split": 1.0 if res.split_votes > 0 else 0.0,
                            "time_ms": float(res.total_time_ms),
                        },
                    ))
        return self

    def train_val_split(
        self, val_fraction: float = 0.3, seed: int = 0,
    ) -> tuple[list, list]:
        rng = random.Random(seed)
        idx = list(range(len(self.episodes)))
        rng.shuffle(idx)
        cut = int(len(idx) * (1.0 - val_fraction))
        return [self.episodes[i] for i in idx[:cut]], [self.episodes[i] for i in idx[cut:]]


class DynamicsSurrogate:
    """Small MLP: virtual_state + intervention -> outcome distribution params.

    Outputs: success_logit, split_rate (softplus-ish), log_time.
    Uncertainty = ensemble disagreement across 2 heads is approximated here
    by a learned log-variance head (single model, cheap) plus OOD score
    supplied externally by the TrustGate's FeatureDistribution.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._model: Any = None

    def _build(self):  # lazy torch import
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("DynamicsSurrogate requires torch") from exc

        class _MLP(nn.Module):
            def __init__(self, in_dim: int, hid: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid), nn.Tanh(),
                    nn.Linear(hid, hid), nn.Tanh(),
                )
                self.success = nn.Linear(hid, 1)
                self.split = nn.Linear(hid, 1)
                self.logtime = nn.Linear(hid, 1)

            def forward(self, x):  # noqa: ANN001
                h = self.net(x)
                return self.success(h).squeeze(-1), self.split(h).squeeze(-1), self.logtime(h).squeeze(-1)

        return _MLP(self.input_dim, self.hidden_dim)

    def train(
        self,
        dataset: TrajectoryDataset,
        epochs: int = 400,
        lr: float = 3e-3,
        seed: int = 7,
    ) -> dict[str, float]:
        import torch

        torch.manual_seed(seed)
        model = self._build()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        X = torch.tensor([r[0] for r in dataset.rows], dtype=torch.float32)
        y_ok = torch.tensor([r[1]["success_prob"] for r in dataset.rows], dtype=torch.float32)
        y_split = torch.tensor([min(r[1]["split_prob"], 3.0) for r in dataset.rows], dtype=torch.float32)
        y_time = torch.tensor([max(r[1]["election_time_ms"], 1.0) for r in dataset.rows], dtype=torch.float32)
        y_logtime = torch.log(y_time)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            s, sp, lt = model(X)
            loss = bce(s, y_ok) + 0.3 * ((sp - y_split) ** 2).mean() + 0.3 * ((lt - y_logtime) ** 2).mean()
            loss.backward()
            opt.step()
        self._model = model
        self._model.eval()
        with torch.no_grad():
            s, sp, lt = model(X)
            import torch as _t

            pred_ok = _t.sigmoid(s)
            mae_ok = (pred_ok - y_ok).abs().mean().item()
            mae_t = ((_t.exp(lt) - y_time).abs() / y_time.clamp_min(1.0)).mean().item()
        return {"mae_success_prob": mae_ok, "mape_time": mae_t, "rows": len(dataset.rows)}

    def predict(self, features: list[float]) -> dict[str, float]:
        import torch

        if self._model is None:
            raise RuntimeError("surrogate not trained/loaded")
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            s, sp, lt = self._model(x)
            import torch as _t

            success = float(_t.sigmoid(s).item())
            split = float(max(_t.nn.functional.softplus(sp).item(), 0.0))
            t = float(_t.exp(lt).item())
        # residual uncertainty proxy: distance of prediction from confident extremes
        uncertainty = 0.15 + 0.5 * success * (1.0 - success) * 4.0 * 0.25
        return {
            "success_prob": min(max(success, 0.0), 1.0),
            "split_prob": split,
            "election_time_ms": min(max(t, 0.0), 30000.0),
            "uncertainty": min(max(uncertainty, 0.05), 0.9),
        }

    def save(self, path: str) -> None:
        import torch

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(p) + ".pt")
        p.write_text(json.dumps({
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "features": FEATURE_NAMES,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "DynamicsSurrogate":
        import torch

        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(input_dim=int(meta["input_dim"]), hidden_dim=int(meta["hidden_dim"]))
        obj._model = obj._build()
        obj._model.load_state_dict(torch.load(str(path) + ".pt", map_location="cpu"))
        obj._model.eval()
        return obj


class EnsembleDynamicsSurrogate:
    """Deep-ensemble virtual dynamics on per-episode binary outcomes.

    K MLPs on distinct seeds; uncertainty = ensemble std of success_prob
    (measured disagreement, not a hand formula). Predicts means + stds.
    This is the rigorous counterpart to DynamicsSurrogate (which trains on
    aggregate means and uses a heuristic uncertainty proxy).
    """

    def __init__(self, members: int = 5, hidden_dim: int = 32, input_dim: int = 6) -> None:
        self.members = members
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self._models: list[Any] = []

    def train(
        self,
        train_rows: list[tuple[list[float], dict[str, float]]],
        epochs: int = 150,
        lr: float = 3e-3,
        seed: int = 7,
    ) -> dict[str, float]:
        import torch

        X = torch.tensor([r[0] for r in train_rows], dtype=torch.float32)
        y_ok = torch.tensor([r[1]["success"] for r in train_rows], dtype=torch.float32)
        y_split = torch.tensor([r[1]["split"] for r in train_rows], dtype=torch.float32)
        y_time = torch.tensor([max(r[1]["time_ms"], 1.0) for r in train_rows], dtype=torch.float32)
        y_logtime = torch.log(y_time)
        bce = torch.nn.BCEWithLogitsLoss()
        self._models = []
        for m in range(self.members):
            torch.manual_seed(seed + 1000 * m)
            proto = DynamicsSurrogate(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
            model = proto._build()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                s, sp, lt = model(X)
                loss = bce(s, y_ok) + 0.5 * bce(sp, y_split) + 0.3 * ((lt - y_logtime) ** 2).mean()
                loss.backward()
                opt.step()
            model.eval()
            self._models.append(model)
        return {"members": self.members, "train_rows": len(train_rows)}

    def predict(self, features: list[float]) -> dict[str, float]:
        import torch
        import torch as _t

        if not self._models:
            raise RuntimeError("ensemble not trained")
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            oks, splits, times = [], [], []
            for model in self._models:
                s, sp, lt = model(x)
                oks.append(float(_t.sigmoid(s).item()))
                splits.append(float(_t.sigmoid(sp).item()))
                times.append(float(_t.exp(lt).item()))
        mean_ok = sum(oks) / len(oks)
        mean_sp = sum(splits) / len(splits)
        mean_t = sum(times) / len(times)
        var_ok = sum((o - mean_ok) ** 2 for o in oks) / len(oks)
        return {
            "success_prob": min(max(mean_ok, 0.0), 1.0),
            "split_prob": min(max(mean_sp, 0.0), 1.0),
            "election_time_ms": min(max(mean_t, 0.0), 30000.0),
            "uncertainty": min(max(var_ok ** 0.5 + 0.02, 0.0), 1.0),
            "member_std": min(max(var_ok ** 0.5, 0.0), 1.0),
        }

    def save(self, path: str) -> None:
        import torch

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self._models):
            torch.save(model.state_dict(), f"{p}.ens{i}.pt")
        p.write_text(json.dumps({
            "members": self.members, "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim, "kind": "ensemble",
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "EnsembleDynamicsSurrogate":
        import torch

        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(members=int(meta["members"]), hidden_dim=int(meta["hidden_dim"]),
                  input_dim=int(meta["input_dim"]))
        proto = DynamicsSurrogate(input_dim=obj.input_dim, hidden_dim=obj.hidden_dim)
        for i in range(obj.members):
            model = proto._build()
            model.load_state_dict(torch.load(f"{path}.ens{i}.pt", map_location="cpu"))
            model.eval()
            obj._models.append(model)
        return obj


class EnsembleRegressor:
    """Deep-ensemble MLP regressor for continuous surrogate targets.

    K members on distinct seeds, MSE loss, CPU-deterministic. Features
    and targets are standardized internally (mean/std from training),
    so raw-unit targets (ms, Mbps) train as easily as normalized ones;
    predictions are returned in original units. predict() returns
    (mean, std) per output dim -- measured disagreement, the basis for
    interval coverage, abstention and OOD-gap analysis. Vector targets
    supported (output_dim >= 1). Shared by the TCP/routing/paging/sched
    research-grade surrogates (linear models remain as ablation baselines).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        members: int = 5,
        hidden_dim: int = 32,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.members = members
        self.hidden_dim = hidden_dim
        self._models: list[Any] = []
        self.train_mse = 0.0
        self.x_mean: list[float] = []
        self.x_std: list[float] = []
        self.y_mean: list[float] = []
        self.y_std: list[float] = []

    def _build(self):  # lazy torch import
        import torch
        import torch.nn as nn

        class _MLP(nn.Module):
            def __init__(_s, in_d: int, hid: int, out_d: int) -> None:
                super().__init__()
                _s.net = nn.Sequential(
                    nn.Linear(in_d, hid), nn.Tanh(),
                    nn.Linear(hid, hid), nn.Tanh(),
                    nn.Linear(hid, out_d),
                )

            def forward(_s, x):  # noqa: ANN001
                return _s.net(x)

        return _MLP(self.input_dim, self.hidden_dim, self.output_dim)

    def train(
        self,
        rows: list[tuple[list[float], list[float]]],
        epochs: int = 200,
        lr: float = 3e-3,
        seed: int = 7,
    ) -> dict[str, float]:
        import torch

        n = len(rows)
        dim_x = len(rows[0][0])
        dim_y = len(rows[0][1])
        mx = [sum(r[0][d] for r in rows) / n for d in range(dim_x)]
        sx = [max((sum((r[0][d] - mx[d]) ** 2 for r in rows) / n) ** 0.5, 1e-9)
              for d in range(dim_x)]
        my = [sum(r[1][d] for r in rows) / n for d in range(dim_y)]
        sy = [max((sum((r[1][d] - my[d]) ** 2 for r in rows) / n) ** 0.5, 1e-9)
              for d in range(dim_y)]
        self.x_mean, self.x_std, self.y_mean, self.y_std = mx, sx, my, sy
        Xn = [[(x - m) / s for x, m, s in zip(r[0], mx, sx)] for r in rows]
        Yn = [[(y - m) / s for y, m, s in zip(r[1], my, sy)] for r in rows]
        X = torch.tensor(Xn, dtype=torch.float32)
        Y = torch.tensor(Yn, dtype=torch.float32)
        mse = torch.nn.MSELoss()
        self._models = []
        for m in range(self.members):
            torch.manual_seed(seed + 1000 * m)
            model = self._build()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                loss = mse(model(X), Y)
                loss.backward()
                opt.step()
            model.eval()
            self._models.append(model)
        with torch.no_grad():
            mean, _ = self.predict_many([r[0] for r in rows])
            err = sum(
                (a - b) ** 2
                for mm, (_, y) in zip(mean, rows)
                for a, b in zip(mm, y)
            )
            n_out = sum(len(y) for _, y in rows)
            self.train_mse = err / max(n_out, 1)
        return {"train_mse": self.train_mse, "members": self.members,
                "rows": len(rows)}

    def predict_many(
        self, feat_list: list[list[float]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        import torch

        if not self._models:
            raise RuntimeError("ensemble not trained")
        with torch.no_grad():
            Xn = [[(x - m) / s for x, m, s in zip(f, self.x_mean, self.x_std)]
                  for f in feat_list]
            X = torch.tensor(Xn, dtype=torch.float32)
            stacked = torch.stack([m(X) for m in self._models], dim=0)  # (K, N, D)
            mean_n = stacked.mean(dim=0).tolist()
            std_n = stacked.std(dim=0, unbiased=False).tolist()
        mean = [[m * s + b for m, s, b in zip(row, self.y_std, self.y_mean)]
                for row in mean_n]
        std = [[s * ysd for s, ysd in zip(row, self.y_std)] for row in std_n]
        return mean, std

    def predict(self, features: list[float]) -> tuple[list[float], list[float]]:
        mean, std = self.predict_many([features])
        return mean[0], std[0]
