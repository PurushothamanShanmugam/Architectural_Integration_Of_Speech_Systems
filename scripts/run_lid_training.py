from pathlib import Path
from src.lid.train import train_lid_model
project_root = Path(__file__).resolve().parents[1]
print(train_lid_model(project_root))
