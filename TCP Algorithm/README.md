# TCP Algorithm (AIMD congestion control)

Learned virtualization of AIMD congestion control: an agent picks cwnd-cap
interventions against a learned throughput model instead of the packet-level
mechanism. (In these chats we still call it "virtual"; the artifact itself
carries no Virtual prefix by convention.)

- Grounding: context.txt's packets/routers/socket boundary
  ("an application can manipulate a socket despite the physical network
  being packets moving through routers"). TCP naming is an extension of
  that boundary, not a quote — see `../Raft Algorithm/docs/ALGORITHMS.md`.
- Physical (`tcp_algo/tcp.py::simulate_aimd`): per-RTT AIMD (cwnd += 1
  clean, halve on drop-tail/random loss) over (base RTT, bottleneck Mbps,
  buffer, loss).
- Learned (`TCPCongestion`, ensemble MLP primary + ridge-linear
  ablation): throughput on dimensionless BDP-ratio features; predicts
  throughput only -- cwnd traces are audited physically, never
  hallucinated. Uncertainty = measured ensemble disagreement.
- Invariants: cwnd in [1, cap], throughput ≤ bottleneck, loss in [0,1].
- Class: `TCPCongestion`, registry key `tcp_congestion`.

## Run

```powershell
# from the workspace root (or anywhere; paths self-bootstrap)
python -m unittest discover -s "TCP Algorithm/tests"
python -m pytest "TCP Algorithm/tests" -q
# full sweep + audit (run from Raft Algorithm):
python -m rlraft.cli virtual-sweep --algo tcp_congestion --seed 7
```

Shared framework (trust, eval harness, registry): `../Raft Algorithm/rlraft/virtual/`.
Latest audit: ensemble rho 1.0 stable over seeds 7/11/13, regret 0,
MAE 0.03 (linear ablation 0.31); per-episode uncertainty
error-correlation +0.90, coverage 0.62; cap-policy regret 0.0 on 3/3
nets (linear 0.13). Details: `../Raft Algorithm/docs/CROSS_ALGO_ANALYSIS.md`.
