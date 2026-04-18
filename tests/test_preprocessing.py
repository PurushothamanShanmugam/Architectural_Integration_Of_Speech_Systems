import numpy as np
from src.preprocessing.denoise import spectral_subtraction

def test_spectral_subtraction_length_preserved():
    sr = 16000
    x = np.random.randn(sr).astype(np.float32) * 0.01
    y = spectral_subtraction(x, sr)
    assert len(y) == len(x)
