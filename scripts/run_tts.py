from pathlib import Path
from src.tts.zero_shot_tts import run_tts
project_root = Path(__file__).resolve().parents[1]
print(run_tts(project_root))
