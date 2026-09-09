import os
import shutil
from pathlib import Path

runs_dir = Path("runs")
if not runs_dir.exists():
    runs_dir.mkdir()

folders = ["policies", "simulations", "smoke_tests", "experiments"]
for folder in folders:
    (runs_dir / folder).mkdir(exist_ok=True)

for file in runs_dir.iterdir():
    if file.is_dir():
        continue
    
    name = file.name
    if "sim-comparison" in name or "comparison" in name:
        target = "simulations"
    elif "policy" in name:
        target = "policies"
    elif "smoke" in name:
        target = "smoke_tests"
    else:
        target = "experiments"
        
    shutil.move(str(file), str(runs_dir / target / name))

print("Runs folder organized successfully.")
