import time
import unittest

from rlraft.config import ClusterConfig
from rlraft.core.supervisor import ClusterSupervisor


class ClusterIntegrationTests(unittest.TestCase):
    def test_cluster_elects_leader_and_fails_over(self) -> None:
        config = ClusterConfig(
            cluster_size=5,
            policy_mode="static",
            election_timeout_min=0.20,
            election_timeout_max=0.40,
            heartbeat_interval=0.08,
            status_interval=0.10,
            random_seed=11,
            graded_node_conditions=False,
        )
        supervisor = ClusterSupervisor(config)
        supervisor.start()
        try:
            leader = supervisor.wait_for_leader(timeout_s=5.0)
            self.assertIsNotNone(leader)
            supervisor.client_command("test-entry")
            time.sleep(0.8)
            snapshot = supervisor.snapshot()
            committed = [
                node for node in snapshot["nodes"].values() if node["commit_index"] >= 0
            ]
            self.assertGreaterEqual(len(committed), config.majority)

            supervisor.crash_node(leader)
            new_leader = supervisor.wait_for_leader(timeout_s=6.0, exclude=leader)
            self.assertIsNotNone(new_leader)
            self.assertNotEqual(new_leader, leader)
        finally:
            supervisor.stop()

    def test_minority_partition_cannot_elect_new_leader(self) -> None:
        config = ClusterConfig(
            cluster_size=5,
            policy_mode="static",
            election_timeout_min=0.20,
            election_timeout_max=0.40,
            heartbeat_interval=0.08,
            status_interval=0.10,
            random_seed=23,
            graded_node_conditions=False,
        )
        supervisor = ClusterSupervisor(config)
        supervisor.start()
        try:
            leader = supervisor.wait_for_leader(timeout_s=5.0)
            self.assertIsNotNone(leader)
            minority = [leader, (leader + 1) % 5]
            majority = [node for node in range(5) if node not in minority]
            supervisor.partition([minority, majority])
            supervisor.crash_node(leader)
            new_leader = supervisor.wait_for_leader(timeout_s=6.0, exclude=leader)
            self.assertIsNotNone(new_leader)
            self.assertIn(new_leader, majority)
            supervisor.heal()
        finally:
            supervisor.stop()


if __name__ == "__main__":
    unittest.main()
