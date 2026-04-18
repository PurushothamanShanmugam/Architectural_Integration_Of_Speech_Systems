from pathlib import Path
from src.prosody.dtw_warp import run_prosody_alignment
project_root = Path(__file__).resolve().parents[1]
print(run_prosody_alignment(project_root))
