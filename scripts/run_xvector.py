from pathlib import Path
from src.speaker.xvector_extract import run_xvector_extraction
project_root = Path(__file__).resolve().parents[1]
print(run_xvector_extraction(project_root))
