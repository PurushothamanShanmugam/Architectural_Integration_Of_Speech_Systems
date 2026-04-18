import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.evaluation.mcd import compute_mcd


pytestmark = pytest.mark.skipif(
    os.getenv("CI", "").lower() == "true",
    reason="Skipping MCD test in CI because librosa/MFCC can crash on GitHub Actions Windows runners.",
)


def test_mcd_returns_small_value_for_identical_signals(tmp_path: Path):
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    x = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"

    sf.write(a, x, sr)
    sf.write(b, x, sr)

    mcd = compute_mcd(str(a), str(b), sr=sr)
    assert np.isfinite(mcd)
    assert mcd < 1.0
