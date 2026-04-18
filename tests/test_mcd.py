import numpy as np
import soundfile as sf
from pathlib import Path
from src.evaluation.mcd import compute_mcd

def test_mcd_returns_small_value_for_identical_signals(tmp_path: Path):
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = 0.1 * np.sin(2 * np.pi * 220 * t)
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(a, x, sr)
    sf.write(b, x, sr)
    assert compute_mcd(str(a), str(b), sr=sr) < 1.0
