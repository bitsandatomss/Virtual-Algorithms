from __future__ import annotations

import argparse
import json
import sys
import time

from rlraft.config import ClusterConfig, load_config, save_default_config
from rlraft.web.dashboard import DashboardServer
from rlraft.sim.experiments import run_policy_comparison, run_simulation_comparison
from rlraft.core.supervisor import ClusterSupervisor
from rlraft.rl.training import train_policy
from rlraft.rl.mappo import train_mappo_policy
from rlraft.rl.llm_node import check_llm_node_policy
from rlraft.rl.logger import TrainingLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL-Raft multi-process demo")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start a live cluster and dashboard")
    start.add_argument("--config", help="path to JSON config")
    start.add_argument("--nodes", type=int, default=None, help="cluster size")
    start.add_argument("--policy", choices=["static", "adaptive", "learned", "mappo", "llm", "rl_stub"], default=None)
    start.add_argument("--host", default=None)
    start.add_argument("--port", type=int, default=None)
    start.add_argument("--no-dashboard", action="store_true")
    start.add_argument("--train-episodes", type=int, default=6000)

    exp = sub.add_parser("run-experiment", help="compare static and adaptive failover")
    exp.add_argument("--repetitions", type=int, default=3)
    exp.add_argument("--nodes", type=int, default=50)
    exp.add_argument("--output-dir", default="runs")

    sim = sub.add_parser("sim-compare", help="deterministic simulator comparison")
    sim.add_argument("--nodes", type=int, default=50)
    sim.add_argument("--episodes", type=int, default=1000)
    sim.add_argument("--output-dir", default="runs")
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--include-llm", action="store_true", help="include direct LLM-node policy with deterministic fallback")
    sim.add_argument("--train-episodes", type=int, default=12000, help="MAPPO episodes if a comparison policy must be trained")

    train = sub.add_parser("train", help="run multi-agent practice rounds")
    train.add_argument("--episodes", type=int, default=6000)
    train.add_argument("--nodes", type=int, default=50)
    train.add_argument("--output", default="runs/policies/learned_ppo.json")
    train.add_argument("--algorithm", choices=["llm_mappo", "mappo", "qlearning"], default="llm_mappo")
    train.add_argument("--advisor", choices=["deterministic", "off"], default="off", help="legacy qlearning only; LLM reward shaping is disabled")
    train.add_argument("--advisor-interval", type=int, default=5000)
    train.add_argument("--require-llm", action="store_true", help="fail unless direct LLM-node policy can make a decision")
    train.add_argument("--hidden-dim", type=int, default=128, help="actor-critic hidden layer size (128/256/512)")

    smoke = sub.add_parser("smoke", help="start a cluster briefly and print final metrics")
    smoke.add_argument("--nodes", type=int, default=50)
    smoke.add_argument("--policy", choices=["static", "adaptive", "learned", "mappo", "llm", "rl_stub"], default="llm")
    smoke.add_argument("--seconds", type=float, default=5.0)
    smoke.add_argument("--train-episodes", type=int, default=6000)
    smoke.add_argument("--snapshot", default=None)

    cfg = sub.add_parser("write-config", help="write a default config file")
    cfg.add_argument("path", nargs="?", default="config.json")

    sub.add_parser("check-llm", help="check direct LLM-node policy access without printing secrets")

    vlist = sub.add_parser("virtual-list", help="list registered virtual algorithms")

    vsur = sub.add_parser("virtual-train-surrogate", help="train Raft dynamics surrogate (Reality->trajectories->surrogate)")
    vsur.add_argument("--nodes", type=int, default=20)
    vsur.add_argument("--episodes-per-setting", type=int, default=40)
    vsur.add_argument("--epochs", type=int, default=250)
    vsur.add_argument("--output", default="runs/virtual/surrogate")
    vsur.add_argument("--seed", type=int, default=7)

    veval = sub.add_parser("virtual-eval", help="counterfactual sweep with trust gating (no simulator executed per query)")
    veval.add_argument("--surrogate", default="runs/virtual/surrogate")
    veval.add_argument("--nodes", type=int, default=20)
    veval.add_argument("--horizon", type=int, default=1)

    vlab = sub.add_parser("virtual-lab", help="closed active loop: observe->learn->simulate->experiment->observe")
    vlab.add_argument("--nodes", type=int, default=20)
    vlab.add_argument("--rounds", type=int, default=3)
    vlab.add_argument("--episodes-per-setting", type=int, default=40)
    vlab.add_argument("--epochs", type=int, default=250)
    vlab.add_argument("--seed", type=int, default=7)

    vaudit = sub.add_parser("virtual-audit", help="rigorous audit: held-out MAE+CIs, ranking rho+regret, OOD AUROC, abstention, stress")
    vaudit.add_argument("--train-nodes", type=int, default=5, help="training cluster size (single regime by default)")
    vaudit.add_argument("--test-nodes", type=int, default=20)
    vaudit.add_argument("--ood-nodes", type=int, default=100)
    vaudit.add_argument("--episodes-per-arm", type=int, default=30)
    vaudit.add_argument("--members", type=int, default=3)
    vaudit.add_argument("--epochs", type=int, default=80)
    vaudit.add_argument("--seed", type=int, default=7)
    vaudit.add_argument("--output", default=None, help="optional JSON path for the audit report")

    vsweep = sub.add_parser("virtual-sweep", help="fit surrogate + counterfactual sweep + audit for tcp/routing/paging/scheduling")
    vsweep.add_argument("--algo", choices=["tcp_congestion", "overlay_routing", "paging", "scheduling"],
                        default="tcp_congestion")
    vsweep.add_argument("--seed", type=int, default=7)
    vsweep.add_argument("--output", default=None, help="optional JSON path for the sweep report")

    vmat = sub.add_parser("virtual-matrix", help="unified multi-seed matrix: ranking/value + OOD AUROC + maturity for all algos")
    vmat.add_argument("--seeds", default="7,11,13", help="comma-separated seeds")
    vmat.add_argument("--algos", default=None, help="comma-separated subset (default: all non-raft)")
    vmat.add_argument("--light", action="store_true", help="reduced episodes for smoke runs")
    vmat.add_argument("--include-raft", action="store_true", help="also run raft audits (torch cost)")
    vmat.add_argument("--output", default=None, help="optional JSON path for the matrix report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write-config":
        save_default_config(args.path)
        print(f"Wrote {args.path}")
        return 0
    if args.command == "run-experiment":
        csv_path = run_policy_comparison(
            repetitions=args.repetitions,
            cluster_size=args.nodes,
            output_dir=args.output_dir,
        )
        print(f"Wrote comparison metrics to {csv_path}")
        return 0
    if args.command == "sim-compare":
        from pathlib import Path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_simulation_comparison(
            nodes=args.nodes,
            episodes=args.episodes,
            output_dir=str(output_dir),
            seed=args.seed,
            include_llm=args.include_llm,
            train_episodes=args.train_episodes,
        )
        print(f"Wrote deterministic simulation comparison to {csv_path}")
        return 0
    if args.command == "train":
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if args.require_llm:
            llm_status = check_llm_node_policy()
            if llm_status["source"] not in {"llm", "llm_cache", "groq", "openrouter", "deepseek"}:
                print(json.dumps(llm_status, indent=2))
                return 1
        if args.algorithm in {"llm_mappo", "mappo"}:
            hidden_dim = getattr(args, "hidden_dim", 128)
            logger = TrainingLogger(
                episodes=args.episodes,
                nodes=args.nodes,
                algorithm=args.algorithm,
                config={
                    "hidden_dim": hidden_dim,
                    "use_llm_prior": args.algorithm == "llm_mappo",
                    "require_llm": args.require_llm,
                    "output": args.output,
                },
            )
            try:
                result = train_mappo_policy(
                    episodes=args.episodes,
                    nodes=args.nodes,
                    output_path=args.output,
                    use_llm_prior=args.algorithm == "llm_mappo",
                    require_llm=args.require_llm,
                    hidden_dim=hidden_dim,
                    logger=logger,
                )
            finally:
                logger.close()
        else:
            result = train_policy(
                args.episodes,
                args.nodes,
                args.output,
                advisor=args.advisor,
                advisor_interval=args.advisor_interval,
            )
        print(
            f"Trained {args.algorithm} timeout policy: "
            f"success={result.success_rate:.3f} "
            f"split={result.split_vote_rate:.3f} "
            f"best_node_wins={result.best_node_win_rate:.3f} "
            f"avg_failover={result.average_failover_ms:.1f}ms "
            f"path={result.policy_path}"
        )
        return 0
    if args.command == "check-llm":
        print(json.dumps(check_llm_node_policy(), indent=2))
        return 0
    if args.command == "virtual-list":
        from rlraft.virtual import (
            VIRTUAL_ALGORITHM_REGISTRY,
            OverlayRouting,
            Paging,
            Scheduling,
            TCPCongestion,
        )
        _ = (OverlayRouting, Paging, Scheduling, TCPCongestion)  # force registration

        print(json.dumps(sorted(VIRTUAL_ALGORITHM_REGISTRY), indent=2))
        return 0
    if args.command == "virtual-train-surrogate":
        from rlraft.virtual.lab import VirtualLab

        lab = VirtualLab(nodes=args.nodes, seed=args.seed)
        metrics = lab.bootstrap(
            episodes_per_setting=args.episodes_per_setting, epochs=args.epochs,
        )
        assert lab.virtual.surrogate is not None
        lab.virtual.surrogate.save(args.output)
        print(json.dumps({"surrogate": args.output, **metrics}, indent=2))
        return 0
    if args.command == "virtual-eval":
        from rlraft.virtual import RaftElection
        from rlraft.virtual.base import Intervention
        from rlraft.virtual.surrogate import DynamicsSurrogate, cluster_features

        surrogate = DynamicsSurrogate.load(args.surrogate)
        virtual = RaftElection(surrogate=surrogate)
        virtual.ood.fit([cluster_features(args.nodes, 120.0, 0.02, 0.2, False, a) for a in range(5)])
        # Honest quick-calibration: a few ground-truth episodes per arm so the
        # trust gate has empirical (predicted, observed) pairs instead of
        # refusing everything as uncalibrated.
        import random as _random

        _rng = _random.Random(11)
        for arm in RaftElection.ARMS:
            st = virtual.to_virtual_state({"nodes": args.nodes})
            p = virtual.surrogate.predict(
                cluster_features(args.nodes, 120.0, 0.02, 0.2, False,
                                 RaftElection.ARMS.index(arm)))
            for _ in range(5):
                obs = virtual.physical_step(
                    {"nodes": args.nodes},
                    Intervention("set_timeout_arm", {"arm": arm}), _rng)
                virtual.calibrator.add(p["success_prob"], 1.0 if obs["success"] else 0.0)
        state = virtual.to_virtual_state({"nodes": args.nodes})
        rows = []
        for arm in RaftElection.ARMS:
            pred = virtual.virtual_step(state, Intervention("set_timeout_arm", {"arm": arm, "horizon": args.horizon}))
            rows.append({"arm": arm, **{k: round(v, 4) if isinstance(v, float) else v for k, v in pred.outcome.items()},
                         "uncertainty": round(pred.uncertainty, 4), "ood": round(pred.ood_score, 4), "trusted": pred.trusted})
        print(json.dumps(rows, indent=2))
        return 0
    if args.command == "virtual-lab":
        from rlraft.virtual.lab import VirtualLab

        lab = VirtualLab(nodes=args.nodes, seed=args.seed)
        print(json.dumps(lab.run(rounds=args.rounds, episodes_per_setting=args.episodes_per_setting,
                                 epochs=args.epochs), indent=2))
        return 0
    if args.command == "virtual-audit":
        from rlraft.virtual.eval import run_audit

        report = run_audit(
            train_nodes=[args.train_nodes], test_nodes=args.test_nodes,
            ood_nodes=args.ood_nodes, episodes_per_arm=args.episodes_per_arm,
            ensemble_members=args.members, epochs=args.epochs, seed=args.seed,
        )
        if args.output:
            from pathlib import Path as _P

            _P(args.output).parent.mkdir(parents=True, exist_ok=True)
            _P(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "virtual-sweep":
        from rlraft.virtual.sweep import run_sweep

        report = run_sweep(args.algo, seed=args.seed)
        if args.output:
            from pathlib import Path as _P

            _P(args.output).parent.mkdir(parents=True, exist_ok=True)
            _P(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "virtual-matrix":
        from rlraft.virtual.cross_audit import run_matrix

        seeds = tuple(int(s) for s in args.seeds.split(","))
        algos = args.algos.split(",") if args.algos else None
        report = run_matrix(algos=algos, seeds=seeds, light=args.light,
                            include_raft=args.include_raft)
        if args.output:
            from pathlib import Path as _P

            _P(args.output).parent.mkdir(parents=True, exist_ok=True)
            _P(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "smoke":
        return smoke_cluster(args)
    if args.command == "start":
        return start_cluster(args)
    return 2


def start_cluster(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.nodes is not None:
        config.cluster_size = args.nodes
    if args.policy is not None:
        config.policy_mode = args.policy
    if args.host is not None:
        config.dashboard_host = args.host
    if args.port is not None:
        config.dashboard_port = args.port
    if config.policy_mode in {"learned", "mappo"}:
        from pathlib import Path

        if not Path(config.learned_policy_path).exists():
            result = train_mappo_policy(
                episodes=args.train_episodes,
                nodes=config.cluster_size,
                output_path=config.learned_policy_path,
                seed=config.random_seed,
            )
            print(
                "Trained learned timeout policy before launch: "
                f"success={result.success_rate:.3f}, "
                f"split={result.split_vote_rate:.3f}, "
                f"best_node_wins={result.best_node_win_rate:.3f}"
            )

    supervisor = ClusterSupervisor(config)
    dashboard: DashboardServer | None = None
    supervisor.start()
    try:
        if not args.no_dashboard:
            dashboard = DashboardServer(supervisor, config.dashboard_host, config.dashboard_port)
            dashboard.start()
            print(f"Dashboard: http://{config.dashboard_host}:{config.dashboard_port}")
        print(
            f"Started RL-Raft cluster with {config.cluster_size} nodes "
            f"(policy={config.policy_mode}). Press Ctrl+C to stop."
        )
        while True:
            snapshot = supervisor.snapshot()
            metrics = snapshot["metrics"]
            print(
                f"leader={metrics['leader_id']} term={metrics['current_term']} "
                f"elections={metrics['elections_started']} splits={metrics['split_votes']} "
                f"dropped={metrics['messages_dropped']}",
                end="\r",
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping cluster...")
    finally:
        if dashboard:
            dashboard.stop()
        supervisor.stop()
    return 0


def smoke_cluster(args: argparse.Namespace) -> int:
    config = ClusterConfig(cluster_size=args.nodes, policy_mode=args.policy)
    if config.policy_mode in {"learned", "mappo"}:
        from pathlib import Path

        if not Path(config.learned_policy_path).exists():
            train_mappo_policy(
                episodes=args.train_episodes,
                nodes=config.cluster_size,
                output_path=config.learned_policy_path,
                seed=config.random_seed,
            )
    supervisor = ClusterSupervisor(config)
    supervisor.start()
    try:
        leader = supervisor.wait_for_leader(timeout_s=args.seconds)
        time.sleep(max(0.0, args.seconds - 1.0))
        snapshot = supervisor.snapshot()
        if args.snapshot:
            supervisor.save_snapshot(args.snapshot)
        metrics = snapshot["metrics"]
        print(
            f"nodes={args.nodes} policy={args.policy} leader={leader} "
            f"term={metrics['current_term']} elections={metrics['elections_started']} "
            f"splits={metrics['split_votes']} delivered={metrics['messages_delivered']} "
            f"dropped={metrics['messages_dropped']}"
        )
        return 0 if leader is not None else 1
    finally:
        supervisor.stop()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
