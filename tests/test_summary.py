from pathlib import Path
from src.config.settings import get_settings

def test_settings_paths(tmp_path: Path):
    cfg = get_settings(tmp_path)
    assert cfg.data_raw.name == "raw"
    assert cfg.outputs.name == "outputs"
