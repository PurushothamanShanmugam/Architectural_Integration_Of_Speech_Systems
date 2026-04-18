from pathlib import Path
from src.lid.infer import run_lid_inference
project_root = Path(__file__).resolve().parents[1]
print(run_lid_inference(project_root).shape)
