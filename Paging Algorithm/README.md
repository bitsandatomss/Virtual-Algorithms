# Paging Algorithm (demand paging)

Learned virtualization of demand paging: an agent ranks replacement
policies via learned fault-rate models and can deploy a learned evictor.
(In these chats we still call it "virtual"; the artifact itself carries
no Virtual prefix by convention.)

- Grounding: context.txt "physical memory -> virtual address space";
  "the physical RAM has pages, addresses, contention, caches, DMA".
- Physical (`paging_algo/paging.py`): locality reference strings, F
  frames, LRU/LFU/RANDOM + Belady MIN oracle (lower bound only, never
  an intervention).
- Learned: one-hot ensemble MLP fault model (primary) + per-policy
  ridge-linear (ablation) on reuse-distance features; grid-fit
  victim-score evictor (w_rec=1 recovers LRU, w_freq=1 recovers LFU).
- Invariants: resident ≤ frames, faults ≤ refs, nothing beats MIN.
- Class: `Paging`, registry key `paging`.

## Run

```powershell
python -m unittest discover -s "Paging Algorithm/tests"
python -m pytest "Paging Algorithm/tests" -q
python -m rlraft.cli virtual-sweep --algo paging --seed 7
```

Shared framework: `../Raft Algorithm/rlraft/virtual/`.
Latest audit: policy ranking rho 1.0, regret 0; evictor suite
(8 paired runs): learned beats LRU by 44 faults, CI [-63,-27], win
rate 8/8; MIN gaps 248 vs 275. Uncertainty error-correlation +0.72,
coverage 0.0 (direction without scale). Details:
`../Raft Algorithm/docs/CROSS_ALGO_ANALYSIS.md`.
