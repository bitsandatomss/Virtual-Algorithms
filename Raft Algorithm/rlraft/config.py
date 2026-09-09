from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class NetworkConfig:
    base_latency_ms: int = 30
    latency_jitter_ms: int = 20
    packet_loss: float = 0.0
    partitions: list[list[int]] = field(default_factory=list)


@dataclass(slots=True)
class ClusterConfig:
    cluster_size: int = 50
    policy_mode: str = "llm"
    learned_policy_path: str = "runs/policies/learned_ppo.json"
    heartbeat_interval: float = 0.12
    tick_interval: float = 0.02
    election_timeout_min: float = 0.35
    election_timeout_max: float = 0.75
    adaptive_timeout_min: float = 0.18
    adaptive_timeout_max: float = 0.95
    status_interval: float = 0.20
    random_seed: int = 7
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000
    metrics_dir: str = "runs"
    graded_node_conditions: bool = True
    network: NetworkConfig = field(default_factory=NetworkConfig)

    @property
    def node_ids(self) -> list[int]:
        return list(range(self.cluster_size))

    @property
    def majority(self) -> int:
        return self.cluster_size // 2 + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | None = None) -> ClusterConfig:
    if not path:
        return ClusterConfig()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    network_data = data.pop("network", {})
    return ClusterConfig(network=NetworkConfig(**network_data), **data)


def save_default_config(path: str) -> None:
    Path(path).write_text(
        json.dumps(ClusterConfig().to_dict(), indent=2),
        encoding="utf-8",
    )
