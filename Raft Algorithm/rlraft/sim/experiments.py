from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

from rlraft.config import ClusterConfig
from rlraft.core.supervisor import ClusterSupervisor
from rlraft.sim.sim import DynatuneLikePolicy, StaticRandomPolicy, generate_conditions, simulate_failover
from rlraft.rl.training import load_learned_policy, train_policy
from rlraft.rl.llm_node import LLMNodeTimeoutPolicy
from rlraft.rl.mappo import MAPPOTimeoutPolicy, train_mappo_policy


def run_failover_experiment(
    policy_mode: str,
    cluster_size: int = 5,
    duration_s: float = 8.0,
    output_dir: str = "runs",
) -> dict[str, Any]:
    config = ClusterConfig(cluster_size=cluster_size, policy_mode=policy_mode)
    if policy_mode == "learned":
        train_policy(episodes=6000, nodes=cluster_size, output_path=config.learned_policy_path)
    supervisor = ClusterSupervisor(config)
    supervisor.start()
    try:
        leader = supervisor.wait_for_leader(timeout_s=4.0)
        if leader is not None:
            time.sleep(0.5)
            supervisor.crash_node(leader)
            supervisor.wait_for_leader(timeout_s=4.0, exclude=leader)
        time.sleep(max(0.0, duration_s - 4.5))
        snapshot = supervisor.snapshot()
        result = {
            "policy_mode": policy_mode,
            "cluster_size": cluster_size,
            **snapshot["metrics"],
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        json_path = Path(output_dir) / f"{policy_mode}-{timestamp}.json"
        json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        result["snapshot_path"] = str(json_path)
        return result
    finally:
        supervisor.stop()


def run_policy_comparison(
    policies: list[str] | None = None,
    repetitions: int = 3,
    cluster_size: int = 5,
    output_dir: str = "runs",
) -> Path:
    policies = policies or ["static", "learned"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for rep in range(repetitions):
            result = run_failover_experiment(
                policy_mode=policy,
                cluster_size=cluster_size,
                output_dir=output_dir,
            )
            result["repetition"] = rep + 1
            rows.append(result)

    csv_path = Path(output_dir) / f"comparison-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def run_simulation_comparison(
    nodes: int = 50,
    episodes: int = 1000,
    output_dir: str = "runs",
    seed: int = 7,
    include_llm: bool = False,
    train_episodes: int = 12000,
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mappo_path = Path(output_dir) / "llm_mappo_policy.json"
    if not mappo_path.exists():
        train_mappo_policy(
            episodes=train_episodes,
            nodes=nodes,
            output_path=str(mappo_path),
            seed=seed,
            use_llm_prior=True,
        )
    policies = {
        "static": StaticRandomPolicy(),
        "dynatune_like": DynatuneLikePolicy(),
        "llm_mappo": MAPPOTimeoutPolicy.from_artifact(str(mappo_path)),
    }
    q_path = Path(output_dir) / "learned_policy.json"
    if q_path.exists():
        policies["qlearning"] = load_learned_policy(str(q_path))
    if include_llm:
        policies["llm_nodes"] = LLMNodeTimeoutPolicy(require_llm=False)
    rows: list[dict[str, Any]] = []
    for name, policy in policies.items():
        successes = splits = best_wins = 0
        total_time = total_rounds = total_candidates = 0.0
        import random

        rng = random.Random(seed + sum(ord(char) for char in name) * 997)
        for episode in range(episodes):
            conditions = generate_conditions(nodes, rng)
            result = simulate_failover(conditions, policy, rng)
            successes += int(result.success)
            splits += result.split_votes
            best_wins += int(result.best_node_won)
            total_time += result.total_time_ms
            total_rounds += result.rounds
            total_candidates += result.candidates_started
        rows.append(
            {
                "policy": name,
                "nodes": nodes,
                "episodes": episodes,
                "success_rate": successes / episodes,
                "split_vote_rate": splits / episodes,
                "best_node_win_rate": best_wins / episodes,
                "average_failover_ms": total_time / episodes,
                "average_rounds": total_rounds / episodes,
                "average_candidates_started": total_candidates / episodes,
            }
        )

    csv_path = Path(output_dir) / f"sim-comparison-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
