import unittest

from rlraft.virtual import VIRTUAL_ALGORITHM_REGISTRY


class RegistryTests(unittest.TestCase):
    def test_all_algorithms_registered(self) -> None:
        # Touch the lazily-loaded sibling packages so registration runs
        # (also exercises the PEP 562 path in rlraft.virtual.__init__).
        from rlraft.virtual import OverlayRouting, Paging, Scheduling, TCPCongestion  # noqa: F401

        for name in ("raft_election", "backoff", "gossip", "tcp_congestion",
                     "overlay_routing", "paging", "scheduling"):
            self.assertIn(name, VIRTUAL_ALGORITHM_REGISTRY)


if __name__ == "__main__":
    unittest.main()
