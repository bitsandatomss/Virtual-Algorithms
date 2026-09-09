from __future__ import annotations

"""Training logger for LLM-MAPPO.

Writes structured JSONL metrics and a human-readable log to the logs/
directory. Progress is printed to stdout with an inline progress bar.
"""

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Captures per-batch training metrics and writes them to logs/.

    Usage::

        logger = TrainingLogger(episodes=35000, nodes=50, algorithm="llm_mappo")
        for episode_batch in ...:
            ...
            logger.log_batch(
                episode=current_episode,
                metrics={...},
                llm_sources={...},
                loss=float,
            )
        logger.log_final(eval_metrics={...}, policy_path="runs/...")
        logger.close()
    """

    LOGS_DIR = Path("logs")

    def __init__(
        self,
        episodes: int,
        nodes: int,
        algorithm: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.episodes = episodes
        self.nodes = nodes
        self.algorithm = algorithm
        self.start_time = time.time()
        self._last_print_len = 0

        # Create logs directory
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Timestamped file names
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.LOGS_DIR / f"training_{ts}.jsonl"
        self.log_path = self.LOGS_DIR / f"training_{ts}.log"
        self.latest_log_path = self.LOGS_DIR / "latest.log"
        self.latest_jsonl_path = self.LOGS_DIR / "latest.jsonl"

        self._jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")
        self._log_file = open(self.log_path, "w", encoding="utf-8")

        # Write header
        header = {
            "event": "training_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": algorithm,
            "episodes": episodes,
            "nodes": nodes,
            "config": config or {},
        }
        self._write_jsonl(header)
        self._write_log(
            f"{'='*70}\n"
            f"  RL-Raft LLM-MAPPO Training\n"
            f"  Algorithm : {algorithm}\n"
            f"  Episodes  : {episodes:,}\n"
            f"  Nodes     : {nodes}\n"
            f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Log file  : {self.log_path}\n"
            f"{'='*70}\n"
        )
        self._sync_latest()
        print(
            f"\n[rlraft] Training started -> {self.log_path}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_batch(
        self,
        episode: int,
        metrics: dict[str, float],
        llm_sources: dict[str, int] | None = None,
        loss: float | None = None,
    ) -> None:
        """Called after each PPO batch update."""
        elapsed = time.time() - self.start_time
        progress = episode / max(self.episodes, 1)
        eta_s = (elapsed / max(progress, 1e-6)) * (1.0 - progress) if progress > 0 else 0.0

        record: dict[str, Any] = {
            "event": "batch",
            "episode": episode,
            "progress_pct": round(progress * 100, 2),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta_s, 1),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        if llm_sources:
            record["llm_sources"] = llm_sources
            total_decisions = max(sum(llm_sources.values()), 1)
            llm_real = llm_sources.get("groq", 0) + llm_sources.get("openrouter", 0) + llm_sources.get("deepseek", 0)
            record["llm_real_pct"] = round(100.0 * llm_real / total_decisions, 1)
            record["llm_cache_pct"] = round(100.0 * llm_sources.get("llm_cache", 0) / total_decisions, 1)
            record["fallback_pct"] = round(100.0 * llm_sources.get("deterministic_fallback", 0) / total_decisions, 1)
        if loss is not None:
            record["loss"] = round(loss, 5)

        self._write_jsonl(record)

        # Human-readable line
        success = metrics.get("success_rate", metrics.get("rolling_success_rate", 0.0))
        split = metrics.get("split_vote_rate", metrics.get("rolling_split_rate", 0.0))
        best = metrics.get("best_node_win_rate", metrics.get("rolling_best_win_rate", 0.0))
        t_ms = metrics.get("average_failover_ms", metrics.get("rolling_failover_ms", 0.0))
        llm_info = ""
        if llm_sources:
            llm_real = llm_sources.get("groq", 0) + llm_sources.get("openrouter", 0) + llm_sources.get("deepseek", 0)
            total_d = max(sum(llm_sources.values()), 1)
            llm_info = f"  llm={100*llm_real//total_d}%"
        log_line = (
            f"[ep {episode:>6,}/{self.episodes:,}] "
            f"success={success:.3f}  split={split:.3f}  "
            f"best_win={best:.3f}  t={t_ms:.0f}ms{llm_info}"
        )
        self._write_log(log_line)
        self._sync_latest()
        self._print_progress(episode, progress, elapsed, eta_s, success, split)

    def log_final(
        self,
        eval_metrics: dict[str, float],
        policy_path: str,
    ) -> None:
        """Called once training is complete."""
        elapsed = time.time() - self.start_time
        record = {
            "event": "training_complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "policy_path": policy_path,
            "eval_metrics": {k: round(v, 4) for k, v in eval_metrics.items()},
        }
        self._write_jsonl(record)
        self._write_log(
            f"\n{'='*70}\n"
            f"  TRAINING COMPLETE in {_fmt_duration(elapsed)}\n"
            f"  Policy      : {policy_path}\n"
            f"  Success     : {eval_metrics.get('success_rate', 0):.3f}\n"
            f"  Split votes : {eval_metrics.get('split_vote_rate', 0):.3f}\n"
            f"  Best-node   : {eval_metrics.get('best_node_win_rate', 0):.3f}\n"
            f"  Avg failover: {eval_metrics.get('average_failover_ms', 0):.1f} ms\n"
            f"{'='*70}\n"
        )
        self._sync_latest()
        # Clear progress line
        sys.stdout.write("\r" + " " * self._last_print_len + "\r")
        print(
            f"[rlraft] Training done in {_fmt_duration(elapsed)} - "
            f"success={eval_metrics.get('success_rate', 0):.3f}  "
            f"best_node={eval_metrics.get('best_node_win_rate', 0):.3f}\n"
            f"         Policy  -> {policy_path}\n"
            f"         Log     -> {self.log_path}\n"
            f"         JSONL   -> {self.jsonl_path}",
            flush=True,
        )

    def close(self) -> None:
        self._jsonl_file.close()
        self._log_file.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self._jsonl_file.write(line + "\n")
        self._jsonl_file.flush()

    def _write_log(self, text: str) -> None:
        self._log_file.write(text + "\n")
        self._log_file.flush()

    def _sync_latest(self) -> None:
        """Copy current files to logs/latest.* for easy access."""
        try:
            shutil.copy2(self.log_path, self.latest_log_path)
            shutil.copy2(self.jsonl_path, self.latest_jsonl_path)
        except OSError:
            pass

    def _print_progress(
        self,
        episode: int,
        progress: float,
        elapsed: float,
        eta_s: float,
        success: float,
        split: float,
    ) -> None:
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        line = (
            f"\r[{bar}] {progress*100:.1f}%  "
            f"ep={episode:,}/{self.episodes:,}  "
            f"success={success:.3f}  split={split:.3f}  "
            f"elapsed={_fmt_duration(elapsed)}  ETA={_fmt_duration(eta_s)}"
        )
        # Pad to overwrite previous line
        if len(line) < self._last_print_len:
            line = line + " " * (self._last_print_len - len(line))
        self._last_print_len = len(line)
        sys.stdout.write(line)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"
