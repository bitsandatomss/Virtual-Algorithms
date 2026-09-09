# Routing Algorithm (overlay path selection)

Learned virtualization of overlay routing: an agent picks overlay paths
via a learned delivery/latency model instead of per-packet simulation.
(In these chats we still call it "virtual"; the artifact itself carries
no Virtual prefix by convention.)

- Grounding: context.txt "physical links / packets / routers -> virtual
  network / logical circuit / overlay"; "the application doesn't operate
  at packet-routing level".
- Physical (`routing_algo/routing.py`): 6-node topology, per-link (delay,
  loss), per-link ARQ(3), enumerated loop-free paths.
- Learned (`OverlayRouting`, vector ensemble MLP primary +
  ridge-linear ablation): delivery + latency on loss-concentration
  features, trained over a loss gradient.
- Invariants: latency â‰¥ propagation bound, delivery in [0,1],
  loop-free + connected paths.
- Class: `OverlayRouting`, registry key `overlay_routing`.

## Run

```powershell
python -m unittest discover -s "Routing Algorithm/tests"
python -m pytest "Routing Algorithm/tests" -q
python -m rlraft.cli virtual-sweep --algo overlay_routing --seed 7
```

Shared framework: `../Raft Algorithm/rlraft/virtual/`.
Latest audit: ensemble rho mean 0.73 (range 0.40-1.00), regret 0.0
on all seeds (best path always right); MAE ~0.01. Linear ablation
still orders the 2%-level differences better — capacity is not the
bottleneck, effect size is. Details: `../Raft Algorithm/docs/CROSS_ALGO_ANALYSIS.md`.
