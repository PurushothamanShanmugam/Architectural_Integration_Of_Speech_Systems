from pathlib import Path
from src.adversarial.fgsm_attack import run_fgsm_attack
project_root = Path(__file__).resolve().parents[1]
print(run_fgsm_attack(project_root))
