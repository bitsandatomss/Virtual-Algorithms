import os
import shutil
import re
from pathlib import Path

base = Path("rlraft")
dirs = ["core", "rl", "sim", "web"]
for d in dirs:
    (base / d).mkdir(exist_ok=True)
    (base / d / "__init__.py").touch()

mapping = {
    "dashboard.py": "web",
    "experiments.py": "sim",
    "sim.py": "sim",
    "env.py": "rl",
    "training.py": "rl",
    "policy.py": "rl",
    "hub.py": "core",
    "messages.py": "core",
    "node.py": "core",
    "raft_rules.py": "core",
    "supervisor.py": "core",
}

for file, module in mapping.items():
    src = base / file
    dst = base / module / file
    if src.exists():
        shutil.move(str(src), str(dst))

def update_imports(file_path):
    content = file_path.read_text(encoding="utf-8")
    
    replacements = {
        r"from \.dashboard": r"from rlraft.web.dashboard",
        r"from \.experiments": r"from rlraft.sim.experiments",
        r"from \.sim ": r"from rlraft.sim.sim ",
        r"from \.sim\n": r"from rlraft.sim.sim\n",
        r"from \.sim import": r"from rlraft.sim.sim import",
        r"from \.env": r"from rlraft.rl.env",
        r"from \.training": r"from rlraft.rl.training",
        r"from \.policy": r"from rlraft.rl.policy",
        r"from \.hub": r"from rlraft.core.hub",
        r"from \.messages": r"from rlraft.core.messages",
        r"from \.node": r"from rlraft.core.node",
        r"from \.raft_rules": r"from rlraft.core.raft_rules",
        r"from \.supervisor": r"from rlraft.core.supervisor",
        r"from \.config": r"from rlraft.config",
    }
    
    for old, new in replacements.items():
        content = re.sub(old, new, content)
        
    file_path.write_text(content, encoding="utf-8")

for py_file in base.rglob("*.py"):
    update_imports(py_file)
    
print("Refactoring complete.")
