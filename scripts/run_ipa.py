from pathlib import Path
from src.phonetics.ipa_converter import run_ipa_conversion
project_root = Path(__file__).resolve().parents[1]
print(run_ipa_conversion(project_root)[:500])
