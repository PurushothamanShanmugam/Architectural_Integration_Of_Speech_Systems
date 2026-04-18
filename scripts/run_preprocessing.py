from pathlib import Path
from src.preprocessing.prepare_inputs import run_input_preparation
from src.preprocessing.denoise import run_denoising
project_root = Path(__file__).resolve().parents[1]
print(run_input_preparation(project_root))
print(run_denoising(project_root))
