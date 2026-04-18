from pathlib import Path
from src.spoof.train_eval import run_anti_spoof
project_root = Path(__file__).resolve().parents[1]
print(run_anti_spoof(project_root))
