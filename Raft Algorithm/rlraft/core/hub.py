from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
from queue import Empty
import random
import time
from typing import Any

from rlraft.config import ClusterConfig
from rlraft.core.messages import event


@dataclass(slots=True)
class HubNetworkState:
    base_latency_ms: int
    latency_jitter_ms: int
    packet_loss: float
    partitions: list[list[int]] = field(default_factory=list)


def run_hub(
    config: ClusterConfig,
    inbox: Any,
    node_inboxes: list[Any],
    event_outbox: Any,
) -> None:
    hub = NetworkHub(config, inbox, node_inboxes, event_outbox)
    hub.run()


class NetworkHub:
    def __init__(
        self,
        config: ClusterConfig,
        inbox: Any,
        node_inboxes: list[Any],
        event_outbox: Any,
    ):
        self.config = config
        self.inbox = inbox
        self.node_inboxes = node_inboxes
        self.event_outbox = event_outbox
        self.rng = random.Random(config.random_seed * 37)
        self.running = True
        self.sequence = 0
        self.pending: list[tuple[float, int, dict[str, Any]]] = []
        self.network = HubNetworkState(
            base_latency_ms=config.network.base_latency_ms,
            latency_jitter_ms=config.network.latency_jitter_ms,
            packet_loss=config.network.packet_loss,
            partitions=[list(group) for group in config.network.partitions],
        )
        self.messages_seen = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.node_latency_penalty_ms, self.node_loss_penalty = self._build_node_conditions()

    def run(self) -> None:
        self._emit("hub_started", network=asdict(self.network))
        while self.running:
            self._drain_inbox()
            self._deliver_due_messages()
            time.sleep(0.005)
        self._emit("hub_stopped")

    def _drain_inbox(self) -> None:
        while True:
            try:
                message = self.inbox.get_nowait()
            except Empty:
                return
            self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        kind = message.get("kind")
        if kind == "rpc":
            self._route_rpc(message)
        elif kind == "event":
            self.event_outbox.put(message)
        elif kind == "control":
            self._handle_control(message)

    def _route_rpc(self, message: dict[str, Any]) -> None:
        self.messages_seen += 1
        src = message["src"]
        dst = message["dst"]

        if self._is_partitioned(src, dst):
            self.messages_dropped += 1
            self._emit("message_dropped", reason="partition", src=src, dst=dst, type=message["type"])
            return
        if self.rng.random() < self.network.packet_loss:
            self.messages_dropped += 1
            self._emit("message_dropped", reason="packet_loss", src=src, dst=dst, type=message["type"])
            return

        drop_probability = min(
            0.95,
            self.network.packet_loss
            + max(self.node_loss_penalty[src], self.node_loss_penalty[dst]),
        )
        if self.rng.random() < drop_probability:
            self.messages_dropped += 1
            self._emit("message_dropped", reason="node_loss", src=src, dst=dst, type=message["type"])
            return

        latency = max(
            0.0,
            self.network.base_latency_ms
            + self.rng.uniform(0.0, float(self.network.latency_jitter_ms)),
            self.network.base_latency_ms,
        )
        latency += self.node_latency_penalty_ms[src] + self.node_latency_penalty_ms[dst]
        deliver_at = time.monotonic() + latency / 1000.0
        self.sequence += 1
        heapq.heappush(self.pending, (deliver_at, self.sequence, message))
        self._emit("message_scheduled", src=src, dst=dst, type=message["type"], latency_ms=latency)

    def _deliver_due_messages(self) -> None:
        now = time.monotonic()
        while self.pending and self.pending[0][0] <= now:
            _, _, message = heapq.heappop(self.pending)
            dst = message["dst"]
            if isinstance(dst, int) and 0 <= dst < len(self.node_inboxes):
                self.node_inboxes[dst].put(message)
                self.messages_delivered += 1
                self._emit("message_delivered", src=message["src"], dst=dst, type=message["type"])

    def _handle_control(self, message: dict[str, Any]) -> None:
        command = message.get("command")
        if command == "shutdown":
            self.running = False
            for inbox in self.node_inboxes:
                inbox.put({"kind": "node_command", "command": "shutdown"})
        elif command in {"crash", "restart"}:
            node_id = int(message["node_id"])
            self.node_inboxes[node_id].put({"kind": "node_command", "command": command})
            self._emit(f"node_{command}_requested", node_id=node_id)
        elif command == "set_network":
            self._set_network(message)
        elif command == "client_command":
            leader_id = message.get("leader_id")
            if leader_id is not None:
                self.node_inboxes[int(leader_id)].put(
                    {
                        "kind": "node_command",
                        "command": "client_command",
                        "command_value": message.get("command_value"),
                    }
                )
        elif command == "partition":
            self.network.partitions = [list(map(int, group)) for group in message.get("groups", [])]
            self._emit("network_changed", network=asdict(self.network))
        elif command == "heal":
            self.network.partitions = []
            self._emit("network_changed", network=asdict(self.network))

    def _set_network(self, message: dict[str, Any]) -> None:
        if "base_latency_ms" in message:
            self.network.base_latency_ms = max(0, int(message["base_latency_ms"]))
        if "latency_jitter_ms" in message:
            self.network.latency_jitter_ms = max(0, int(message["latency_jitter_ms"]))
        if "packet_loss" in message:
            self.network.packet_loss = min(max(float(message["packet_loss"]), 0.0), 1.0)
        self._emit("network_changed", network=asdict(self.network))

    def _is_partitioned(self, src: int, dst: int) -> bool:
        if not self.network.partitions:
            return False

        def group_of(node: int) -> int | None:
            for idx, group in enumerate(self.network.partitions):
                if node in group:
                    return idx
            return None

        src_group = group_of(src)
        dst_group = group_of(dst)
        return src_group != dst_group

    def _build_node_conditions(self) -> tuple[list[float], list[float]]:
        if not self.config.graded_node_conditions:
            return [0.0 for _ in self.config.node_ids], [0.0 for _ in self.config.node_ids]

        latency: list[float] = []
        loss: list[float] = []
        for node_id in self.config.node_ids:
            rank = node_id / max(self.config.cluster_size - 1, 1)
            latency.append(rank * 80.0)
            loss.append(rank * 0.08)
        return latency, loss

    def _emit(self, name: str, **payload: Any) -> None:
        self.event_outbox.put(
            event(
                name,
                monotonic_time=time.monotonic(),
                messages_seen=self.messages_seen,
                messages_delivered=self.messages_delivered,
                messages_dropped=self.messages_dropped,
                node_latency_penalty_ms=self.node_latency_penalty_ms,
                node_loss_penalty=self.node_loss_penalty,
                **payload,
            )
        )
