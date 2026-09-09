# RL-Raft

RL-Raft is a Raft leader-election research prototype. Nodes try to learn or
infer election timeout behavior so fast, well-connected, fresh-log nodes start
elections earlier, while slow, lossy, stale nodes back off.

Raft safety rules are not learned or weakened. Policies only choose when a node
starts an election. Majority voting, one vote per term, and log freshness remain
normal Raft rules.

## What Is Implemented

- `llm`: direct LLM node agents. Each node sends its local observation to an LLM
  and receives a timeout action. If the API is unavailable, the node falls back
  to a deterministic local policy.
- `llm_mappo`: integrated LLM-prior MAPPO. Each node gets an LLM/fallback
  timeout prior during MAPPO rollouts, and the neural actor learns over repeated
  election rounds.
- `mappo`: ablation without LLM priors.
- `qlearning`: legacy tabular independent Q-learning baseline with parameter
  sharing. LLM reward shaping is disabled by default.
- `static`: vanilla randomized Raft election timeout baseline.
- `adaptive`: deterministic measurement-based heuristic baseline.

The old hand-coded quality offset in learned tabular execution has been removed.
Learned policies now choose a timeout arm, then sample inside that arm without an
extra hidden quality bonus.

## API Keys

LLM node agents load keys from `.env` automatically:

```text
OPENAI_API_KEY=...
OPENAI_API_KEY_2=...
OPENAI_MODEL=gpt-4o-mini
```

The LLM node policy tries `OPENAI_API_KEY`, then `OPENAI_API_KEY_2`, and etc.

```powershell
python -m rlraft.cli check-llm
```

## Training

Train the default deep MARL policy:

```powershell
python -m rlraft.cli train --algorithm llm_mappo --episodes 35000 --nodes 50 --output runs/policies/llm_mappo_policy.json
```

Fail fast unless the direct LLM node policy can call the API:

```powershell
python -m rlraft.cli train --algorithm llm_mappo --episodes 35000 --nodes 50 --require-llm
```

Train the legacy tabular baseline:

```powershell
python -m rlraft.cli train --algorithm qlearning --episodes 12000 --nodes 50 --output runs/policies/qlearning_policy.json
```

## Experiments

```powershell
python -m rlraft.cli sim-compare --nodes 50 --episodes 1000 --seed 92 --output-dir runs/simulations
```

Include direct LLM node agents in the comparison:

```powershell
python -m rlraft.cli sim-compare --nodes 50 --episodes 100 --include-llm --output-dir runs/llm-sim
```

## Live Demo

Start 50 direct LLM node agents with fallback:

```powershell
python -m rlraft.cli start --nodes 50 --policy llm --port 8000
```

Start the trained MAPPO policy:

```powershell
python -m rlraft.cli start --nodes 50 --policy mappo --port 8000
```

Open `http://127.0.0.1:8000`.

## Project Structure

- `rlraft/core/`: process-based Raft demo engine, network hub, vote rules.
- `rlraft/sim/`: deterministic event-driven Raft election simulator and experiments.
- `rlraft/rl/mappo.py`: integrated LLM-MAPPO deep MARL trainer and policy loader.
- `rlraft/rl/llm_node.py`: direct LLM node-agent timeout policy.
- `rlraft/rl/training.py`: legacy tabular Q-learning baseline.
- `rlraft/web/`: dashboard server.
- `frontend/`: dashboard assets.
- `tests/`: Raft rules, simulator, trainer, env, and fallback tests.

## Research Caveat

The integrated LLM-MAPPO path is now real code, but strong research claims
still require long runs, multi-seed ablations, and real LLM-vs-fallback
comparisons. Short smoke runs only prove plumbing.

The current local API diagnostic reaches OpenAI with the keys in `.env`, but
both configured keys return `insufficient_quota`. Until billing/quota is fixed,
`llm_mappo` uses recorded deterministic fallback priors unless `--require-llm`
is passed, in which case training fails loudly.
