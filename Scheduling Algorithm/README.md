# Scheduling Algorithm (CPU/job scheduling)

Learned virtualization of queueing disciplines: an agent ranks FCFS /
SJF / round-robin via a learned waiting-time model. (In these chats we
still call it "virtual"; the artifact itself carries no Virtual prefix
by convention.)

- Grounding: context.txt "processes" as hidden computer-environment
  structure; OS/hardware machinery maintaining the relationship.
- Physical (`sched_algo/sched.py`): stochastic job stream, single
  server; SJF assumes burst knowledge (oracle bound); waiting measured
  exactly as flow − burst.
- Learned (`Scheduling`, wait ensemble MLP primary + ridge-linear
  ablation): waits on (utilization, burst, quantum, burst_info,
  burst_cv, util×cv) — the interaction term encodes
  Pollaczek-Khinchine variability effects.
- Invariants: waits/flows non-negative, utilization in [0,1].
- Class: `Scheduling`, registry key `scheduling`.

## Run

```powershell
python -m unittest discover -s "Scheduling Algorithm/tests"
python -m pytest "Scheduling Algorithm/tests" -q
python -m rlraft.cli virtual-sweep --algo scheduling --seed 7
```

Shared framework: `../Raft Algorithm/rlraft/virtual/`.
Latest audit: seed-7 rho 1.0, regret 0 with PK features + ensemble,
but multi-seed rho 0-1.0 -- still the most seed-fragile entry;
RR-variant effects (~3ms) sit below resolution. Details:
`../Raft Algorithm/docs/CROSS_ALGO_ANALYSIS.md`.
