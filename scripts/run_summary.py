from pathlib import Path
from src.evaluation.summary import write_evaluation_summary
project_root = Path(__file__).resolve().parents[1]
print(write_evaluation_summary(project_root))
