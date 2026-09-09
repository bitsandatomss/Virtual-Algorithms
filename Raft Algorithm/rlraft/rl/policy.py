from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import threading

from rlraft.config import ClusterConfig
from rlraft.sim.sim import observation_bucket as sim_observation_bucket
from rlraft.sim.sim import NodeObservation
from rlraft.rl.llm_node import LLMNodePolicyClient
from rlraft.rl.mappo import MAPPOTimeoutPolicy

_PPO_MODELS = {}
_MAPPO_POLICIES = {}
_POLICY_LOCK = threading.Lock()



@dataclass(slots=True)
class Observation:
    node_id: int
    estimated_rtt_ms: float = 50.0
    heartbeat_gap_s: float = 0.0
    log_gap: int = 0
    recent_split_vote: bool = False
    recent_election_lost: bool = False


class ElectionPolicy:
    """Timeout policy hook.

    The MAPPO phase can replace this class with a policy that loads a neural
    network. The node only depends on `next_timeout`, which keeps execution
    decentralized.
    """

    def __init__(self, mode: str, config: ClusterConfig, node_id: int):
        self.mode = mode
        self.config = config
        self.node_id = node_id
        self.rng = random.Random(config.random_seed + node_id * 997)
        self.learned_policy = self._load_learned_policy()
        self.llm_client = LLMNodePolicyClient(require_llm=False) if mode == "llm" else None

    def next_timeout(self, observation: Observation) -> float:
        if self.mode == "static":
            return self.rng.uniform(
                self.config.election_timeout_min,
                self.config.election_timeout_max,
            )
        if self.mode in {"adaptive", "rl_stub"}:
            return self._adaptive_timeout(observation)
        if self.mode == "llm":
            return self._llm_timeout(observation)
        if self.mode in {"learned", "mappo"}:
            return self._learned_timeout(observation)
        raise ValueError(f"unknown policy mode: {self.mode}")

    def _adaptive_timeout(self, observation: Observation) -> float:
        cfg = self.config
        rtt_badness = min(observation.estimated_rtt_ms / 350.0, 1.0)
        heartbeat_badness = min(observation.heartbeat_gap_s / 1.2, 1.0)
        log_badness = min(observation.log_gap / 8.0, 1.0)
        penalty = 0.55 * rtt_badness + 0.25 * heartbeat_badness + 0.20 * log_badness

        if observation.recent_split_vote:
            penalty += 0.15
        if observation.recent_election_lost:
            penalty += 0.10

        penalty = min(max(penalty, 0.0), 1.0)
        span = cfg.adaptive_timeout_max - cfg.adaptive_timeout_min
        jitter = self.rng.uniform(0.0, 0.08)
        return cfg.adaptive_timeout_min + penalty * span + jitter

    def _learned_timeout(self, observation: Observation) -> float:
        if not self.learned_policy:
            return self._adaptive_timeout(observation)

        if self.learned_policy.get("policy_type") == "softmax_timeout_policy":
            return self._softmax_timeout(observation)
        if self.learned_policy.get("policy_type") == "linear_timeout_policy":
            return self._linear_timeout(observation)
        if self.learned_policy.get("policy_type") == "tabular_marl_timeout_q_policy":
            return self._tabular_marl_timeout(observation)
        if self.learned_policy.get("policy_type") == "mappo_timeout_policy":
            return self._mappo_timeout(observation)
        if self.learned_policy.get("policy_type") == "ppo_timeout_policy":
            return self._ppo_timeout(observation)

        node_policy = self.learned_policy.get("node_policy", {})
        action_name = node_policy.get(str(self.node_id))
        actions = self.learned_policy.get("actions", {})
        if action_name in actions:
            low, high = actions[action_name]
            return self.rng.uniform(float(low), float(high))

        bucket = observation_bucket(observation)
        action_name = self.learned_policy.get("bucket_policy", {}).get(bucket)
        if action_name not in actions:
            return self._adaptive_timeout(observation)
        low, high = actions[action_name]
        return self.rng.uniform(float(low), float(high))

    def _softmax_timeout(self, observation: Observation) -> float:
        features = [
            1.0,
            min(observation.estimated_rtt_ms / 350.0, 1.5),
            0.0,
            min(observation.log_gap / 8.0, 1.0),
            1.0 if observation.recent_split_vote else 0.0,
        ]
        weights = self.learned_policy["weights"]
        logits = [
            sum(weight * value for weight, value in zip(row, features))
            for row in weights
        ]
        best = max(range(len(logits)), key=logits.__getitem__)
        arms = self.learned_policy["arms"]
        arm_name = list(arms)[best]
        low, high = arms[arm_name]
        return self.rng.uniform(float(low) / 1000.0, float(high) / 1000.0)

    def _linear_timeout(self, observation: Observation) -> float:
        features = {
            "bias": 1.0,
            "rtt_norm": min(observation.estimated_rtt_ms / 350.0, 1.5),
            "loss_rate": 0.0,
            "log_gap_norm": min(observation.log_gap / 8.0, 1.0),
            "recent_split": 1.0 if observation.recent_split_vote else 0.0,
        }
        value = float(self.learned_policy.get("min_ms", 145.0))
        for name, coefficient in self.learned_policy["coefficients"].items():
            value += float(coefficient) * features.get(name, 0.0)
        jitter = float(self.learned_policy.get("jitter_ms", 45.0))
        value += self.rng.uniform(0.0, jitter)
        value = min(max(value, float(self.learned_policy.get("min_ms", 145.0))), float(self.learned_policy.get("max_ms", 2400.0)))
        return value / 1000.0

    def _tabular_marl_timeout(self, observation: Observation) -> float:
        sim_obs = NodeObservation(
            node_id=observation.node_id,
            mean_rtt_ms=observation.estimated_rtt_ms,
            loss_rate=0.0,
            log_gap=observation.log_gap,
            recent_split=observation.recent_split_vote,
        )
        bucket = sim_observation_bucket(sim_obs)
        q_values = self.learned_policy["q_values"]
        values = q_values.get(bucket) or q_values.get("default")
        action = max(values, key=values.get)
        low, high = self.learned_policy["arms"][action]
        return self.rng.uniform(float(low), float(high)) / 1000.0

    def _mappo_timeout(self, observation: Observation) -> float:
        path = str(Path(self.config.learned_policy_path))
        if path not in _MAPPO_POLICIES:
            with _POLICY_LOCK:
                if path not in _MAPPO_POLICIES:
                    _MAPPO_POLICIES[path] = MAPPOTimeoutPolicy.from_artifact(path)
        sim_obs = NodeObservation(
            node_id=observation.node_id,
            mean_rtt_ms=observation.estimated_rtt_ms,
            loss_rate=0.0,
            log_gap=observation.log_gap,
            recent_split=observation.recent_split_vote,
        )
        return _MAPPO_POLICIES[path].timeout_ms(sim_obs, self.rng) / 1000.0

    def _llm_timeout(self, observation: Observation) -> float:
        if self.llm_client is None:
            return self._adaptive_timeout(observation)
        decision = self.llm_client.decide(
            {
                "node_id": observation.node_id,
                "estimated_rtt_ms": observation.estimated_rtt_ms,
                "loss_rate": 0.0,
                "log_gap": observation.log_gap,
                "recent_split_vote": observation.recent_split_vote,
                "recent_election_lost": observation.recent_election_lost,
            },
            self.rng,
        )
        return decision.timeout_ms / 1000.0

    def _ppo_timeout(self, observation: Observation) -> float:
        import numpy as np
        from stable_baselines3 import PPO

        model_path = self.learned_policy.get("model_path")
        if not model_path:
            return self._adaptive_timeout(observation)
            
        # Cache the model to avoid reloading on every timeout
        if model_path not in _PPO_MODELS:
            with _POLICY_LOCK:
                if model_path not in _PPO_MODELS:
                    _PPO_MODELS[model_path] = PPO.load(model_path, device="cpu")
        model = _PPO_MODELS[model_path]
        
        features = [
            1.0,
            min(observation.estimated_rtt_ms / 350.0, 1.5),
            0.0,
            min(observation.log_gap / 8.0, 1.0),
            1.0 if observation.recent_split_vote else 0.0,
        ]
        obs_array = np.array(features, dtype=np.float32)
        action, _ = model.predict(obs_array, deterministic=True)
        
        value = 145.0 + (float(action[0]) + 1.0) / 2.0 * 2255.0
        return value / 1000.0


    def _load_learned_policy(self) -> dict | None:
        if self.mode not in {"learned", "mappo"}:
            return None
        path = Path(self.config.learned_policy_path)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def observation_bucket(observation: Observation) -> str:
    rtt = observation.estimated_rtt_ms
    if rtt < 90:
        quality = "fast"
    elif rtt < 190:
        quality = "medium"
    else:
        quality = "slow"

    suffix = []
    if observation.recent_split_vote:
        suffix.append("split")
    if observation.recent_election_lost:
        suffix.append("lost")
    return quality if not suffix else quality + ":" + "+".join(suffix)
