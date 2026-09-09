"""
run_sweeps.py  –  RL-Raft empirical sweep for paper results
=============================================================
Runs the deterministic Raft election simulator across a grid of:
  • node counts : [5, 20, 50, 100]
  • random seeds: [42, 99, 123, 200, 77]

For each (nodes, seed) pair we evaluate:
  static | dynatune_like | llm_mappo | qlearning | llm_nodes

Policy artefacts (MAPPO / Q-learning) are trained once per node-count
and re-used across seeds.  LLM-node decisions are cached in-process, so
API calls are minimised.

Outputs
-------
  runs/sweeps/<TIMESTAMP>/
      seed<seed>_nodes<n>.csv  – per-run CSV
      combined_results.csv     – all runs merged, ready for paper tables
      sweep_summary.txt        – aggregated mean ± std per (policy, nodes)

Usage
-----
  python scripts/run_sweeps.py [--episodes 1000] [--train-episodes 20000]
                               [--output-dir runs/sweeps] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# ---------------------------------------------------------------------------
# Allow running as   python scripts/run_sweeps.py   from repo root
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from rlraft.sim.sim import (
    DynatuneLikePolicy,
    StaticRandomPolicy,
    FailoverResult,
    generate_conditions,
    simulate_failover,
)
from rlraft.rl.llm_node import LLMNodeTimeoutPolicy
from rlraft.rl.mappo import MAPPOTimeoutPolicy, train_mappo_policy
from rlraft.rl.training import load_learned_policy, train_policy

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

NODE_COUNTS  = [5, 20, 50, 100]
SEEDS        = [42, 99, 123, 200, 77]
EPISODES     = 1000    # overridable via --episodes
TRAIN_EPS    = 20_000  # overridable via --train-episodes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_mappo(nodes: int, train_eps: int, policy_dir: Path, seed: int) -> MAPPOTimeoutPolicy:
    """Train (or load cached) MAPPO policy for a given node count."""
    path = policy_dir / f"llm_mappo_n{nodes}.json"
    if not path.exists():
        print(f"  [train] MAPPO  nodes={nodes}  episodes={train_eps} → {path.name}")
        train_mappo_policy(
            episodes=train_eps,
            nodes=nodes,
            output_path=str(path),
            seed=seed,
            use_llm_prior=True,
        )
    else:
        print(f"  [cache] MAPPO  {path.name}")
    return MAPPOTimeoutPolicy.from_artifact(str(path))


def _train_qlearning(nodes: int, train_eps: int, policy_dir: Path) -> Any | None:
    """Train (or load cached) Q-learning policy for a given node count."""
    path = policy_dir / f"qlearning_n{nodes}.json"
    if not path.exists():
        print(f"  [train] Q-learning  nodes={nodes}  episodes={train_eps // 2} → {path.name}")
        train_policy(
            episodes=max(train_eps // 2, 4000),
            nodes=nodes,
            output_path=str(path),
        )
    else:
        print(f"  [cache] Q-learning  {path.name}")
    try:
        return load_learned_policy(str(path))
    except Exception as exc:
        print(f"  [warn] could not load Q policy: {exc}")
        return None


def _run_single(
    name: str,
    policy: Any,
    nodes: int,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run `episodes` simulated failovers for one policy and return aggregated metrics."""
    rng = random.Random(seed + sum(ord(c) for c in name) * 997)

    successes = splits = best_wins = failures = 0
    total_time = total_rounds = total_candidates = 0.0
    failover_times: list[float] = []
    round_counts:   list[int]   = []

    for _ in range(episodes):
        conditions = generate_conditions(nodes, rng)
        result: FailoverResult = simulate_failover(conditions, policy, rng)
        successes   += int(result.success)
        failures    += int(not result.success)
        splits      += result.split_votes
        best_wins   += int(result.best_node_won)
        total_time  += result.total_time_ms
        total_rounds+= result.rounds
        total_candidates += result.candidates_started
        failover_times.append(result.total_time_ms)
        round_counts.append(result.rounds)

    n = episodes
    return {
        "policy":                    name,
        "nodes":                     nodes,
        "seed":                      seed,
        "episodes":                  n,
        "success_rate":              successes / n,
        "failure_rate":              failures / n,
        "split_vote_rate":           splits / n,
        "best_node_win_rate":        best_wins / n,
        "average_failover_ms":       total_time / n,
        "median_failover_ms":        _median(failover_times),
        "p95_failover_ms":           _percentile(failover_times, 95),
        "p99_failover_ms":           _percentile(failover_times, 99),
        "average_rounds":            total_rounds / n,
        "average_candidates_started": total_candidates / n,
        "stddev_failover_ms":        stdev(failover_times) if len(failover_times) > 1 else 0.0,
        "stddev_rounds":             stdev(float(r) for r in round_counts) if len(round_counts) > 1 else 0.0,
    }


def _median(data: list[float]) -> float:
    s = sorted(data)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


def _percentile(data: list[float], p: int) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweeps(
    episodes: int,
    train_episodes: int,
    output_dir: Path,
    dry_run: bool,
) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Shared LLM-node policy (caches decisions in-process)
    llm_policy = LLMNodeTimeoutPolicy(require_llm=False)

    all_rows: list[dict[str, Any]] = []

    for nodes in NODE_COUNTS:
        print(f"\n{'=' * 60}")
        print(f"NODE COUNT = {nodes}")
        print(f"{'=' * 60}")

        # Build policy map (train once per node-count)
        if dry_run:
            policies: dict[str, Any] = {
                "static":       StaticRandomPolicy(),
                "dynatune_like": DynatuneLikePolicy(),
            }
        else:
            mappo_policy = _train_mappo(nodes, train_episodes, policy_dir, seed=SEEDS[0])
            q_policy     = _train_qlearning(nodes, train_episodes, policy_dir)

            policies = {
                "static":        StaticRandomPolicy(),
                "dynatune_like": DynatuneLikePolicy(),
                "llm_mappo":     mappo_policy,
                "llm_nodes":     llm_policy,
            }
            if q_policy is not None:
                policies["qlearning"] = q_policy

        for seed in SEEDS:
            print(f"\n  seed={seed}", flush=True)
            seed_rows: list[dict[str, Any]] = []

            for policy_name, policy in policies.items():
                eps = 5 if dry_run else episodes
                print(f"    policy={policy_name:20s}  episodes={eps} … ", end="", flush=True)
                t0 = time.perf_counter()
                row = _run_single(policy_name, policy, nodes, eps, seed)
                elapsed = time.perf_counter() - t0
                print(
                    f"success={row['success_rate']:.3f}  "
                    f"split={row['split_vote_rate']:.3f}  "
                    f"best_wins={row['best_node_win_rate']:.3f}  "
                    f"avg_ms={row['average_failover_ms']:.1f}  "
                    f"({elapsed:.1f}s)"
                )
                seed_rows.append(row)
                all_rows.append(row)

            # Write per-(nodes, seed) CSV immediately so partial results are saved
            csv_path = run_dir / f"seed{seed}_nodes{nodes}.csv"
            _write_csv(seed_rows, csv_path)

    # ------------------------------------------------------------------
    # Write combined CSV
    # ------------------------------------------------------------------
    combined_path = run_dir / "combined_results.csv"
    _write_csv(all_rows, combined_path)
    print(f"\n[done] Combined results → {combined_path}")

    # ------------------------------------------------------------------
    # Write aggregated summary  (mean ± std across seeds)
    # ------------------------------------------------------------------
    summary_path = run_dir / "sweep_summary.txt"
    _write_summary(all_rows, summary_path)
    print(f"[done] Sweep summary    → {summary_path}")

    # ------------------------------------------------------------------
    # Write a LaTeX-ready table snippet
    # ------------------------------------------------------------------
    latex_path = run_dir / "table.tex"
    _write_latex_table(all_rows, latex_path)
    print(f"[done] LaTeX table      → {latex_path}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(all_rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Group by (policy, nodes); compute mean and std of metric columns."""
    from collections import defaultdict

    metric_cols = [
        "success_rate", "failure_rate", "split_vote_rate",
        "best_node_win_rate", "average_failover_ms",
        "median_failover_ms", "p95_failover_ms", "p99_failover_ms",
        "average_rounds", "average_candidates_started",
        "stddev_failover_ms",
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_rows:
        key = (row["policy"], row["nodes"])
        groups[key].append(row)

    aggregated: dict[tuple, dict[str, Any]] = {}
    for key, rows in sorted(groups.items()):
        agg: dict[str, Any] = {"policy": key[0], "nodes": key[1], "n_seeds": len(rows)}
        for col in metric_cols:
            vals = [r[col] for r in rows if col in r]
            if vals:
                agg[f"{col}_mean"] = mean(vals)
                agg[f"{col}_std"]  = stdev(vals) if len(vals) > 1 else 0.0
        aggregated[key] = agg
    return aggregated


def _write_summary(all_rows: list[dict[str, Any]], path: Path) -> None:
    agg = _aggregate_rows(all_rows)
    lines: list[str] = [
        "RL-Raft Empirical Sweep Summary",
        "=" * 70,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seeds evaluated: {SEEDS}",
        f"Episodes per run: {EPISODES}",
        "",
    ]
    current_nodes = None
    for (policy, nodes), data in sorted(agg.items(), key=lambda x: (x[0][1], x[0][0])):
        if nodes != current_nodes:
            lines.append(f"\n--- nodes={nodes} ---")
            current_nodes = nodes
        lines.append(
            f"  {policy:22s}  "
            f"success={data.get('success_rate_mean', 0):.4f}±{data.get('success_rate_std', 0):.4f}  "
            f"split={data.get('split_vote_rate_mean', 0):.4f}±{data.get('split_vote_rate_std', 0):.4f}  "
            f"best_wins={data.get('best_node_win_rate_mean', 0):.4f}±{data.get('best_node_win_rate_std', 0):.4f}  "
            f"avg_ms={data.get('average_failover_ms_mean', 0):.1f}±{data.get('average_failover_ms_std', 0):.1f}  "
            f"p95_ms={data.get('p95_failover_ms_mean', 0):.1f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_table(all_rows: list[dict[str, Any]], path: Path) -> None:
    """Emit a LaTeX booktabs table: columns = policies, rows = node counts."""
    agg = _aggregate_rows(all_rows)

    # Collect all unique policies and sort for stable ordering
    policy_order = ["static", "dynatune_like", "qlearning", "llm_nodes", "llm_mappo"]
    policies_present = sorted(
        {p for (p, _) in agg},
        key=lambda p: policy_order.index(p) if p in policy_order else 99,
    )
    display = {
        "static":        "Static",
        "dynatune_like": "Dynatune-like",
        "qlearning":     "Q-learning",
        "llm_nodes":     "LLM-nodes",
        "llm_mappo":     "LLM-MAPPO (ours)",
    }

    col_spec = "l" + "c" * len(policies_present)
    header = " & ".join(["Nodes"] + [display.get(p, p) for p in policies_present])

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Election Success Rate (mean$\pm$std across " + str(len(SEEDS)) + r" seeds)}",
        r"\label{tab:success_rate}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    for nodes in NODE_COUNTS:
        cells = [str(nodes)]
        for p in policies_present:
            data = agg.get((p, nodes))
            if data:
                mu  = data.get("success_rate_mean", float("nan"))
                std = data.get("success_rate_std",  float("nan"))
                cells.append(rf"${mu:.3f}_{{\pm{std:.3f}}}$")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"% ---- Failover latency table (avg ms) ----",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Average Failover Latency (ms, mean$\pm$std)}",
        r"\label{tab:latency}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for nodes in NODE_COUNTS:
        cells = [str(nodes)]
        for p in policies_present:
            data = agg.get((p, nodes))
            if data:
                mu  = data.get("average_failover_ms_mean", float("nan"))
                std = data.get("average_failover_ms_std",  float("nan"))
                cells.append(rf"${mu:.1f}_{{\pm{std:.1f}}}$")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL-Raft empirical sweep – generates results for paper publication"
    )
    parser.add_argument(
        "--episodes", type=int, default=EPISODES,
        help=f"Simulation episodes per (policy, nodes, seed) triple (default: {EPISODES})",
    )
    parser.add_argument(
        "--train-episodes", type=int, default=TRAIN_EPS,
        help=f"Training episodes for MAPPO / Q-learning (default: {TRAIN_EPS})",
    )
    parser.add_argument(
        "--output-dir", default="runs/sweeps",
        help="Parent directory for results (default: runs/sweeps)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick smoke test: 5 episodes, only static+dynatune, no training",
    )
    parser.add_argument(
        "--nodes", type=int, nargs="+", default=None,
        help="Override NODE_COUNTS, e.g. --nodes 5 50",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Override SEEDS, e.g. --seeds 42 99",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global NODE_COUNTS, SEEDS, EPISODES
    args = _parse_args(argv)
    if args.nodes:
        NODE_COUNTS = args.nodes
    if args.seeds:
        SEEDS = args.seeds
    EPISODES = args.episodes

    output_dir = Path(args.output_dir)
    print("=" * 60)
    print("RL-Raft Empirical Sweep")
    print(f"  nodes={NODE_COUNTS}  seeds={SEEDS}  episodes={EPISODES}")
    print(f"  train_episodes={args.train_episodes}  dry_run={args.dry_run}")
    print(f"  output_dir={output_dir}")
    print("=" * 60)

    run_sweeps(
        episodes=EPISODES,
        train_episodes=args.train_episodes,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
