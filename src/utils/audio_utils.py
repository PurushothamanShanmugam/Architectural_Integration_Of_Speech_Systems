from pathlib import Path
import numpy as np
import soundfile as sf
import torchaudio

def load_mono_resampled(path: Path, target_sr: int = 16000):
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform, sr

def save_wave(path: Path, wave: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sr)
