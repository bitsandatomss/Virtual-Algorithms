import gymnasium as gym
import numpy as np

from rlraft.sim import (
    NodeCondition,
    TimeoutPolicy,
    generate_conditions,
    simulate_failover,
    observations_from_conditions,
)

class PpoTimeoutPolicy(TimeoutPolicy):
    def __init__(self, timeouts: dict[int, float]):
        self.timeouts = timeouts

    def timeout_ms(self, observation, rng) -> float:
        return self.timeouts[observation.node_id]

class RaftEnv(gym.Env):
    """
    A single-agent sequential Environment for Multi-Agent Raft timeout policy training.
    Treats one Raft failover as an episode of N steps. 
    At step i, requests the timeout action for Node i.
    At step N-1, simulates the failover and returns the global reward.
    """
    def __init__(self, max_nodes: int = 100):
        super().__init__()
        self.max_nodes = max_nodes
        self.nodes = max_nodes
        
        # State: bias, rtt_norm, loss_rate, log_gap_norm, recent_split
        self.observation_space = gym.spaces.Box(low=0.0, high=2.0, shape=(5,), dtype=np.float32)
        
        # Action: single continuous value representing timeout
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.current_step = 0
        self.actions = []
        self.conditions = []
        self.observations = {}
        import random
        self.rng = random.Random()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            
        # Domain Randomization: Randomize cluster size between 5 and max_nodes
        self.nodes = self.rng.randint(5, self.max_nodes)
        
        # Domain Randomization: Randomize adversarial network conditions
        failure_rate = self.rng.uniform(0.0, 0.15)
        log_lag_rate = self.rng.uniform(0.0, 0.20)
        
        self.conditions = generate_conditions(
            self.nodes, 
            self.rng, 
            failure_rate=failure_rate, 
            log_lag_rate=log_lag_rate
        )
        self.observations = observations_from_conditions(self.conditions, recent_split=False)
        self.current_step = 0
        self.actions = []
        
        return self._get_obs(self.current_step), {}

    def _get_obs(self, index: int) -> np.ndarray:
        obs = self.observations[index]
        return np.array(obs.features(), dtype=np.float32)

    def step(self, action: np.ndarray):
        # Record the action for the current node
        self.actions.append(float(action[0]))
        self.current_step += 1
        
        # If we haven't collected actions for all nodes, continue the episode
        if self.current_step < self.nodes:
            return self._get_obs(self.current_step), 0.0, False, False, {}
            
        # All actions collected, run the Raft simulation!
        timeouts = {
            i: 145.0 + (self.actions[i] + 1.0) / 2.0 * 2255.0
            for i in range(self.nodes)
        }
        policy = PpoTimeoutPolicy(timeouts)
        result = simulate_failover(self.conditions, policy, self.rng)
        
        reward = self._compute_reward(result)
        
        # In Gymnasium, we must return the next observation even if done=True.
        # Domain Randomization for next episode
        self.nodes = self.rng.randint(5, self.max_nodes)
        failure_rate = self.rng.uniform(0.0, 0.15)
        log_lag_rate = self.rng.uniform(0.0, 0.20)
        self.conditions = generate_conditions(
            self.nodes, 
            self.rng, 
            failure_rate=failure_rate, 
            log_lag_rate=log_lag_rate
        )
        self.observations = observations_from_conditions(self.conditions, recent_split=False)
        self.current_step = 0
        self.actions = []

        
        # We pass the failover metrics in the info dict so Stable-Baselines can log them if needed
        info = {
            "success": result.success,
            "best_node_won": result.best_node_won,
            "total_time_ms": result.total_time_ms,
            "split_votes": result.split_votes
        }
        
        return self._get_obs(0), reward, True, False, info

    def _compute_reward(self, result) -> float:
        # Matches the scoring function from the original CEM training script
        if not result.success:
            return -10.0
            
        reward = 5.0
        if result.best_node_won:
            reward += 4.8
        reward -= (result.total_time_ms / 1450.0)
        reward -= (result.split_votes * 4.0)
        reward -= (result.candidates_started * 0.03)
        return reward
