from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any

from rlraft.config import ClusterConfig
from rlraft.core.hub import run_hub
from rlraft.core.messages import command
from rlraft.core.node import run_node


@dataclass(slots=True)
class ClusterMetrics:
    started_at: float = field(default_factory=time.time)
    leader_id: int | None = None
    current_term: int = 0
    elections_started: int = 0
    leaders_elected: int = 0
    split_votes: int = 0
    messages_seen: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    last_election_duration_s: float | None = None
    last_failover_time_s: float | None = None
    leader_stability_s: float = 0.0


class ClusterSupervisor:
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.node_inboxes = [Queue() for _ in config.node_ids]
        self.hub_inbox = Queue()
        self.event_outbox = Queue()
        self.hub_thread: threading.Thread | None = None
        self.node_threads: list[threading.Thread] = []
        self.collector_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

        self.nodes: dict[int, dict[str, Any]] = {
            node_id: {
                "node_id": node_id,
                "state": "starting",
                "term": 0,
                "leader_id": None,
                "active": True,
                "log_len": 0,
                "commit_index": -1,
                "estimated_rtt_ms": 0.0,
            }
            for node_id in config.node_ids
        }
        self.network = asdict(config.network)
        self.metrics = ClusterMetrics()
        self.recent_events: list[dict[str, Any]] = []
        self._election_start_by_term: dict[int, float] = {}
        self._candidates_by_term: dict[int, set[int]] = {}
        self._split_terms: set[int] = set()
        self._leader_since: float | None = None
        self._leader_crashed_at: float | None = None
        self._election_history: dict[int, dict[str, Any]] = {}

    def start(self) -> None:
        if self.hub_thread is not None:
            return
        self.hub_thread = threading.Thread(
            target=run_hub,
            args=(self.config, self.hub_inbox, self.node_inboxes, self.event_outbox),
            name="rlraft-hub",
            daemon=True,
        )
        self.hub_thread.start()

        for node_id in self.config.node_ids:
            thread = threading.Thread(
                target=run_node,
                args=(node_id, self.config, self.node_inboxes[node_id], self.hub_inbox),
                name=f"rlraft-node-{node_id}",
                daemon=True,
            )
            thread.start()
            self.node_threads.append(thread)

        self.collector_thread = threading.Thread(target=self._collect_events, daemon=True)
        self.collector_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.hub_inbox.put(command("shutdown"))
        except Exception:
            pass

        deadline = time.time() + 4.0
        for thread in [*self.node_threads, self.hub_thread]:
            if thread is None:
                continue
            remaining = max(0.0, deadline - time.time())
            thread.join(timeout=remaining)

    def crash_node(self, node_id: int) -> None:
        self.hub_inbox.put(command("crash", node_id=node_id))

    def restart_node(self, node_id: int) -> None:
        self.hub_inbox.put(command("restart", node_id=node_id))

    def set_network(
        self,
        base_latency_ms: int | None = None,
        latency_jitter_ms: int | None = None,
        packet_loss: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if base_latency_ms is not None:
            payload["base_latency_ms"] = base_latency_ms
        if latency_jitter_ms is not None:
            payload["latency_jitter_ms"] = latency_jitter_ms
        if packet_loss is not None:
            payload["packet_loss"] = packet_loss
        self.hub_inbox.put(command("set_network", **payload))

    def partition(self, groups: list[list[int]]) -> None:
        self.hub_inbox.put(command("partition", groups=groups))

    def heal(self) -> None:
        self.hub_inbox.put(command("heal"))

    def client_command(self, value: str) -> None:
        leader_id = self.snapshot()["metrics"]["leader_id"]
        self.hub_inbox.put(command("client_command", leader_id=leader_id, command_value=value))

    def wait_for_leader(
        self,
        timeout_s: float = 5.0,
        exclude: int | None = None,
    ) -> int | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            leader_id = self.snapshot()["metrics"]["leader_id"]
            if leader_id is not None and leader_id != exclude:
                node = self.snapshot()["nodes"].get(str(leader_id))
                if node and node.get("active") and node.get("state") == "leader":
                    return leader_id
            time.sleep(0.05)
        return None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            if self._leader_since is not None and self.metrics.leader_id is not None:
                self.metrics.leader_stability_s = now - self._leader_since
                
            sorted_history = sorted(self._election_history.values(), key=lambda e: e["term"], reverse=True)
            latest_elections = sorted_history[:5]
            
            return {
                "config": self.config.to_dict(),
                "nodes": {str(k): dict(v) for k, v in self.nodes.items()},
                "network": dict(self.network),
                "metrics": asdict(self.metrics),
                "recent_events": list(self.recent_events[-80:]),
                "election_history": latest_elections,
                "processes": {
                    "hub_alive": bool(self.hub_thread and self.hub_thread.is_alive()),
                    "nodes_alive": [thread.is_alive() for thread in self.node_threads],
                },
            }

    def save_snapshot(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def _collect_events(self) -> None:
        while not self.stop_event.is_set():
            try:
                message = self.event_outbox.get(timeout=0.1)
            except Empty:
                continue
            self._handle_event(message)

    def _handle_event(self, message: dict[str, Any]) -> None:
        with self.lock:
            event_name = message.get("event", "")
            if "messages_seen" in message:
                self.metrics.messages_seen = message.get("messages_seen", self.metrics.messages_seen)
                self.metrics.messages_delivered = message.get(
                    "messages_delivered", self.metrics.messages_delivered
                )
                self.metrics.messages_dropped = message.get(
                    "messages_dropped", self.metrics.messages_dropped
                )

            if "network" in message:
                self.network = dict(message["network"])
            if "node_latency_penalty_ms" in message:
                self.network["node_latency_penalty_ms"] = message["node_latency_penalty_ms"]
            if "node_loss_penalty" in message:
                self.network["node_loss_penalty"] = message["node_loss_penalty"]

            node_id = message.get("node_id")
            if isinstance(node_id, int) and node_id in self.nodes:
                self.nodes[node_id].update(
                    {
                        key: message[key]
                        for key in (
                            "node_id",
                            "state",
                            "term",
                            "leader_id",
                            "active",
                            "log_len",
                            "commit_index",
                            "estimated_rtt_ms",
                        )
                        if key in message
                    }
                )

            if event_name == "pre_vote_started":
                term = int(message["term"])
                node_id = int(message["node_id"])
                history = self._election_history.setdefault(term, {
                    "term": term,
                    "pre_candidates": [],
                    "candidates": [],
                    "pre_votes": {},
                    "votes": {},
                    "winner": None,
                    "status": "ongoing",
                    "start_time": time.time(),
                    "end_time": None
                })
                if node_id not in history["pre_candidates"]:
                    history["pre_candidates"].append(node_id)

            elif event_name == "pre_vote_decided":
                term = int(message["term"])
                voter_id = int(message["node_id"])
                candidate_id = int(message["candidate_id"])
                granted = bool(message["granted"])
                reason = str(message["reason"])
                history = self._election_history.setdefault(term, {
                    "term": term,
                    "pre_candidates": [],
                    "candidates": [],
                    "pre_votes": {},
                    "votes": {},
                    "winner": None,
                    "status": "ongoing",
                    "start_time": time.time(),
                    "end_time": None
                })
                history["pre_votes"].setdefault(str(candidate_id), {})[str(voter_id)] = {"granted": granted, "reason": reason}

            elif event_name == "election_started":
                self._record_election_started(message)
                term = int(message["term"])
                node_id = int(message["node_id"])
                history = self._election_history.setdefault(term, {
                    "term": term,
                    "pre_candidates": [],
                    "candidates": [],
                    "pre_votes": {},
                    "votes": {},
                    "winner": None,
                    "status": "ongoing",
                    "start_time": time.time(),
                    "end_time": None
                })
                if node_id not in history["candidates"]:
                    history["candidates"].append(node_id)

            elif event_name == "vote_decided":
                term = int(message["term"])
                voter_id = int(message["node_id"])
                candidate_id = int(message["candidate_id"])
                granted = bool(message["granted"])
                reason = str(message["reason"])
                history = self._election_history.setdefault(term, {
                    "term": term,
                    "pre_candidates": [],
                    "candidates": [],
                    "pre_votes": {},
                    "votes": {},
                    "winner": None,
                    "status": "ongoing",
                    "start_time": time.time(),
                    "end_time": None
                })
                history["votes"].setdefault(str(candidate_id), {})[str(voter_id)] = {"granted": granted, "reason": reason}

            elif event_name == "leader_elected":
                self._record_leader_elected(message)
                term = int(message["term"])
                winner = int(message["node_id"])
                if term in self._election_history:
                    self._election_history[term]["winner"] = winner
                    self._election_history[term]["status"] = "success"
                    self._election_history[term]["end_time"] = time.time()

            elif event_name == "election_failed":
                term = int(message["term"])
                if term in self._election_history:
                    self._election_history[term]["status"] = "failed"
                    self._election_history[term]["end_time"] = time.time()

            elif event_name == "node_crashed":
                self._record_node_crashed(message)
            elif event_name == "node_restarted":
                if isinstance(node_id, int):
                    self.nodes[node_id]["active"] = True

            if event_name not in {"message_scheduled", "message_delivered"}:
                clean = {
                    key: value
                    for key, value in message.items()
                    if key not in {"kind", "messages_seen", "messages_delivered", "messages_dropped"}
                }
                clean["wall_time"] = time.strftime("%H:%M:%S")
                self.recent_events.append(clean)
                self.recent_events = self.recent_events[-120:]

    def _record_election_started(self, message: dict[str, Any]) -> None:
        term = int(message["term"])
        node_id = int(message["node_id"])
        now = time.time()
        self.metrics.elections_started += 1
        self.metrics.current_term = max(self.metrics.current_term, term)
        self._election_start_by_term.setdefault(term, now)
        candidates = self._candidates_by_term.setdefault(term, set())
        candidates.add(node_id)
        if len(candidates) > 1 and term not in self._split_terms:
            self._split_terms.add(term)
            self.metrics.split_votes += 1

    def _record_leader_elected(self, message: dict[str, Any]) -> None:
        term = int(message["term"])
        leader_id = int(message["node_id"])
        now = time.time()

        if self.metrics.leader_id is not None and self._leader_since is not None:
            self.metrics.leader_stability_s = now - self._leader_since

        self.metrics.leaders_elected += 1
        self.metrics.leader_id = leader_id
        self.metrics.current_term = term
        self._leader_since = now

        started = self._election_start_by_term.get(term)
        if started is not None:
            self.metrics.last_election_duration_s = now - started
        if self._leader_crashed_at is not None:
            self.metrics.last_failover_time_s = now - self._leader_crashed_at
            self._leader_crashed_at = None

    def _record_node_crashed(self, message: dict[str, Any]) -> None:
        node_id = int(message["node_id"])
        if node_id in self.nodes:
            self.nodes[node_id]["active"] = False
        if self.metrics.leader_id == node_id:
            self.metrics.leader_id = None
            self._leader_crashed_at = time.time()
            self._leader_since = None
