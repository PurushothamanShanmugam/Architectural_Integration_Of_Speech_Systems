from pathlib import Path
from src.stt.transcribe import run_transcription
from src.stt.wer_eval import run_wer_and_boundary_eval
project_root = Path(__file__).resolve().parents[1]
result = run_transcription(project_root)
print(result)
print(run_wer_and_boundary_eval(project_root))
