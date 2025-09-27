# run_rounds.py -- replacement to use simulation instead of agent processes
import sys
import subprocess
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent

def run_simulation(rounds=3):
    cmd = [PY, str(ROOT / "simulate_agents.py"), "--rounds", str(rounds)]
    print("Starting simulation:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_simulation(rounds=3)
