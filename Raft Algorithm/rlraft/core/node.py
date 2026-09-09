from __future__ import annotations

from queue import Empty
import time
from typing import Any

from rlraft.config import ClusterConfig
from rlraft.core.messages import (
    APPEND_ENTRIES,
    APPEND_ENTRIES_RESPONSE,
    CLIENT_COMMAND,
    PRE_VOTE,
    PRE_VOTE_RESPONSE,
    REQUEST_VOTE,
    REQUEST_VOTE_RESPONSE,
    event,
    rpc,
)
from rlraft.rl.policy import ElectionPolicy, Observation
from rlraft.core.raft_rules import decide_vote


FOLLOWER = "follower"
CANDIDATE = "candidate"
LEADER = "leader"
PRE_CANDIDATE = "pre_candidate"


def run_node(
    node_id: int,
    config: ClusterConfig,
    inbox: Any,
    hub_inbox: Any,
) -> None:
    node = RaftNode(node_id, config, inbox, hub_inbox)
    node.run()


class RaftNode:
    def __init__(self, node_id: int, config: ClusterConfig, inbox: Any, hub_inbox: Any):
        self.node_id = node_id
        self.config = config
        self.inbox = inbox
        self.hub_inbox = hub_inbox
        self.policy = ElectionPolicy(config.policy_mode, config, node_id)

        self.active = True
        self.running = True
        self.state = FOLLOWER
        self.current_term = 0
        self.voted_for: int | None = None
        self.log: list[dict[str, Any]] = []
        
        import json
        from pathlib import Path
        self.data_dir = Path("node_data")
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / f"node_{node_id}.json"
        if self.data_file.exists():
            try:
                data = json.loads(self.data_file.read_text(encoding="utf-8"))
                self.current_term = data.get("current_term", 0)
                self.voted_for = data.get("voted_for", None)
                self.log = data.get("log", [])
            except Exception:
                pass

        self.leader_id: int | None = None
        self.votes_received: set[int] = set()
        self.pre_votes_received: set[int] = set()
        self.commit_index = -1
        self.next_index: dict[int, int] = {}
        self.match_index: dict[int, int] = {}

        now = time.monotonic()
        self.last_heartbeat_received = now
        self.last_heartbeat_sent = 0.0
        self.last_status_sent = 0.0
        self.estimated_rtt_ms = 50.0
        self.recent_split_vote = False
        self.recent_election_lost = False
        self.election_deadline = now + self._next_timeout()

    def run(self) -> None:
        self._emit_status(force=True)
        while self.running:
            self._drain_inbox()
            now = time.monotonic()

            if self.active:
                if self.state == LEADER:
                    if now - self.last_heartbeat_sent >= self.config.heartbeat_interval:
                        self._send_heartbeats()
                elif now >= self.election_deadline:
                    if self.state == CANDIDATE:
                        self.recent_split_vote = True
                        self.recent_election_lost = True
                        self._emit(
                            "election_failed",
                            term=self.current_term,
                            reason="timeout_without_majority",
                        )
                    self._start_pre_vote()

            self._emit_status()
            time.sleep(self.config.tick_interval)

        self._emit("node_stopped")

    def _drain_inbox(self) -> None:
        while True:
            try:
                message = self.inbox.get_nowait()
            except Empty:
                return
            self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        kind = message.get("kind")
        if kind == "node_command":
            self._handle_command(message)
            return
        if kind != "rpc" or not self.active:
            return

        rpc_type = message.get("type")
        if rpc_type == REQUEST_VOTE:
            self._handle_request_vote(message)
        elif rpc_type == REQUEST_VOTE_RESPONSE:
            self._handle_request_vote_response(message)
        elif rpc_type == PRE_VOTE:
            self._handle_pre_vote(message)
        elif rpc_type == PRE_VOTE_RESPONSE:
            self._handle_pre_vote_response(message)
        elif rpc_type == APPEND_ENTRIES:
            self._handle_append_entries(message)
        elif rpc_type == APPEND_ENTRIES_RESPONSE:
            self._handle_append_entries_response(message)
        elif rpc_type == CLIENT_COMMAND:
            self._handle_client_command(message)

    def _handle_command(self, message: dict[str, Any]) -> None:
        command = message.get("command")
        if command == "shutdown":
            self.running = False
        elif command == "crash":
            self.active = False
            self.state = FOLLOWER
            self.leader_id = None
            self.votes_received.clear()
            self.pre_votes_received.clear()
            self._emit("node_crashed")
        elif command == "restart":
            self.active = True
            self.state = FOLLOWER
            self.voted_for = None
            self.leader_id = None
            self.votes_received.clear()
            self.pre_votes_received.clear()
            self._reset_election_deadline()
            self._emit("node_restarted")
        elif command == "client_command":
            self._handle_client_command(
                {
                    "kind": "rpc",
                    "type": CLIENT_COMMAND,
                    "src": "dashboard",
                    "command_value": message.get("command_value"),
                }
            )

    def _handle_pre_vote(self, message: dict[str, Any]) -> None:
        heartbeat_fresh = self.state == LEADER or (
            self.leader_id is not None
            and time.monotonic() - self.last_heartbeat_received
            < max(self.config.election_timeout_min, self.config.heartbeat_interval * 3)
        )
        log_ok = decide_vote(
            current_term=self.current_term,
            voted_for=None,
            candidate_term=message["term"],
            candidate_id=message["src"],
            candidate_last_index=message.get("last_log_index", -1),
            candidate_last_term=message.get("last_log_term", 0),
            local_last_index=self._last_log_index(),
            local_last_term=self._last_log_term(),
        )
        granted = (not heartbeat_fresh) and message["term"] >= self.current_term and log_ok.granted
        reason = "granted" if granted else ("leader_alive" if heartbeat_fresh else ("stale_term" if message["term"] < self.current_term else "stale_log"))
        self._emit(
            "pre_vote_decided",
            term=message["term"],
            candidate_id=message["src"],
            granted=granted,
            reason=reason,
        )
        self._send(
            message["src"],
            PRE_VOTE_RESPONSE,
            term=self.current_term,
            vote_granted=granted,
            reason=reason,
            request_sent_at=message.get("sent_at"),
        )

    def _handle_pre_vote_response(self, message: dict[str, Any]) -> None:
        self._update_rtt(message)
        if message["term"] > self.current_term:
            self._become_follower(message["term"])
            return
        if self.state != PRE_CANDIDATE:
            return
        if not message.get("vote_granted"):
            return
        self.pre_votes_received.add(message["src"])
        if len(self.pre_votes_received) >= self.config.majority:
            self._start_election()

    def _handle_request_vote(self, message: dict[str, Any]) -> None:
        decision = decide_vote(
            current_term=self.current_term,
            voted_for=self.voted_for,
            candidate_term=message["term"],
            candidate_id=message["src"],
            candidate_last_index=message.get("last_log_index", -1),
            candidate_last_term=message.get("last_log_term", 0),
            local_last_index=self._last_log_index(),
            local_last_term=self._last_log_term(),
        )

        if decision.term > self.current_term or decision.step_down:
            self._become_follower(decision.term)

        self.current_term = decision.term
        self.voted_for = decision.voted_for
        if decision.granted:
            self._reset_election_deadline()
            
        if decision.term > self.current_term or decision.granted:
            self._persist()

        self._emit(
            "vote_decided",
            term=message["term"],
            candidate_id=message["src"],
            granted=decision.granted,
            reason=decision.reason,
        )

        self._send(
            message["src"],
            REQUEST_VOTE_RESPONSE,
            term=self.current_term,
            vote_granted=decision.granted,
            reason=decision.reason,
            request_sent_at=message.get("sent_at"),
        )

    def _handle_request_vote_response(self, message: dict[str, Any]) -> None:
        self._update_rtt(message)

        if message["term"] > self.current_term:
            self._become_follower(message["term"])
            return
        if self.state != CANDIDATE or message["term"] != self.current_term:
            return
        if not message.get("vote_granted"):
            return

        self.votes_received.add(message["src"])
        if len(self.votes_received) >= self.config.majority:
            self._become_leader()

    def _handle_append_entries(self, message: dict[str, Any]) -> None:
        term = message["term"]
        success = False
        if term < self.current_term:
            reason = "stale_term"
        else:
            if term > self.current_term or self.state != FOLLOWER:
                self._become_follower(term)
            self.leader_id = message["src"]
            self.last_heartbeat_received = time.monotonic()
            self._reset_election_deadline()
            incoming_log = message.get("log")
            if incoming_log is not None:
                self.log = list(incoming_log)
                self._persist()
            self.commit_index = min(message.get("leader_commit", -1), len(self.log) - 1)
            success = True
            reason = "accepted"

        self._send(
            message["src"],
            APPEND_ENTRIES_RESPONSE,
            term=self.current_term,
            success=success,
            reason=reason,
            match_index=self._last_log_index() if success else -1,
            request_sent_at=message.get("sent_at"),
        )

    def _handle_append_entries_response(self, message: dict[str, Any]) -> None:
        self._update_rtt(message)

        if message["term"] > self.current_term:
            self._become_follower(message["term"])
            return
        if self.state != LEADER or message["term"] != self.current_term:
            return
        if not message.get("success"):
            return

        peer = message["src"]
        self.match_index[peer] = message.get("match_index", -1)
        self.next_index[peer] = self.match_index[peer] + 1
        self._advance_commit_index()

    def _handle_client_command(self, message: dict[str, Any]) -> None:
        if not self.active:
            return
        if self.state != LEADER:
            self._emit(
                "client_command_rejected",
                leader_id=self.leader_id,
                reason="not_leader",
            )
            return

        entry = {
            "term": self.current_term,
            "index": len(self.log),
            "value": message.get("command_value", f"cmd-{len(self.log)}"),
        }
        self.log.append(entry)
        self._persist()
        self.match_index[self.node_id] = self._last_log_index()
        self._emit("log_appended", entry=entry)
        self._send_heartbeats()

    def _start_election(self) -> None:
        self.state = CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self._persist()
        self.leader_id = None
        self.votes_received = {self.node_id}
        self._reset_election_deadline()
        self._emit("election_started", term=self.current_term)

        for peer in self._peers():
            self._send(
                peer,
                REQUEST_VOTE,
                term=self.current_term,
                last_log_index=self._last_log_index(),
                last_log_term=self._last_log_term(),
            )

        if len(self.votes_received) >= self.config.majority:
            self._become_leader()

    def _start_pre_vote(self) -> None:
        self.state = PRE_CANDIDATE
        self.leader_id = None
        self.pre_votes_received = {self.node_id}
        self._reset_election_deadline()
        self._emit("pre_vote_started", term=self.current_term + 1)

        for peer in self._peers():
            self._send(
                peer,
                PRE_VOTE,
                term=self.current_term + 1,
                last_log_index=self._last_log_index(),
                last_log_term=self._last_log_term(),
            )

        if len(self.pre_votes_received) >= self.config.majority:
            self._start_election()

    def _become_leader(self) -> None:
        self.state = LEADER
        self.leader_id = self.node_id
        self.next_index = {peer: len(self.log) for peer in self._peers()}
        self.match_index = {peer: -1 for peer in self._peers()}
        self.match_index[self.node_id] = self._last_log_index()
        self.recent_election_lost = False
        self.recent_split_vote = False
        self._emit("leader_elected", term=self.current_term)
        self._send_heartbeats()

    def _become_follower(self, term: int) -> None:
        was_leader = self.state == LEADER
        self.state = FOLLOWER
        self.current_term = term
        self.voted_for = None
        self._persist()
        self.votes_received.clear()
        self.pre_votes_received.clear()
        self.leader_id = None
        self._reset_election_deadline()
        if was_leader:
            self._emit("leader_stepped_down", term=term)

    def _send_heartbeats(self) -> None:
        self.last_heartbeat_sent = time.monotonic()
        for peer in self._peers():
            self._send(
                peer,
                APPEND_ENTRIES,
                term=self.current_term,
                leader_commit=self.commit_index,
                log=self.log,
            )

    def _advance_commit_index(self) -> None:
        for idx in range(self.commit_index + 1, len(self.log)):
            replicas = sum(1 for value in self.match_index.values() if value >= idx)
            if replicas >= self.config.majority and self.log[idx]["term"] == self.current_term:
                self.commit_index = idx
                self._emit("commit_advanced", commit_index=self.commit_index)

    def _send(self, dst: int, rpc_type: str, **payload: Any) -> None:
        self.hub_inbox.put(
            rpc(
                self.node_id,
                dst,
                rpc_type,
                sent_at=time.monotonic(),
                **payload,
            )
        )

    def _emit(self, name: str, **payload: Any) -> None:
        base = {
            "node_id": self.node_id,
            "state": self.state,
            "term": self.current_term,
            "leader_id": self.leader_id,
            "active": self.active,
            "log_len": len(self.log),
            "commit_index": self.commit_index,
            "monotonic_time": time.monotonic(),
        }
        base.update(payload)
        self.hub_inbox.put(event(name, **base))

    def _emit_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_status_sent < self.config.status_interval:
            return
        self.last_status_sent = now
        self._emit("status", estimated_rtt_ms=self.estimated_rtt_ms)

    def _update_rtt(self, message: dict[str, Any]) -> None:
        sent_at = message.get("request_sent_at")
        if not sent_at:
            return
        sample = max((time.monotonic() - sent_at) * 1000.0, 0.0)
        self.estimated_rtt_ms = 0.8 * self.estimated_rtt_ms + 0.2 * sample

    def _reset_election_deadline(self) -> None:
        self.election_deadline = time.monotonic() + self._next_timeout()

    def _persist(self) -> None:
        import json
        data = {
            "current_term": self.current_term,
            "voted_for": self.voted_for,
            "log": self.log,
        }
        self.data_file.write_text(json.dumps(data), encoding="utf-8")

    def _next_timeout(self) -> float:
        observation = Observation(
            node_id=self.node_id,
            estimated_rtt_ms=self.estimated_rtt_ms,
            heartbeat_gap_s=time.monotonic() - self.last_heartbeat_received,
            log_gap=0,
            recent_split_vote=self.recent_split_vote,
            recent_election_lost=self.recent_election_lost,
        )
        return self.policy.next_timeout(observation)

    def _last_log_index(self) -> int:
        return len(self.log) - 1

    def _last_log_term(self) -> int:
        return self.log[-1]["term"] if self.log else 0

    def _peers(self) -> list[int]:
        return [peer for peer in self.config.node_ids if peer != self.node_id]
