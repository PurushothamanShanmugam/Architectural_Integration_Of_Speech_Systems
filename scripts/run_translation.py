from pathlib import Path
from src.translation.santali_translation import run_translation
project_root = Path(__file__).resolve().parents[1]
print(run_translation(project_root)[:500])
