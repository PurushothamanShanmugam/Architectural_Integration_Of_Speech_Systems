from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
from src.config.settings import get_settings

def spectral_subtraction(audio: np.ndarray, sr: int, noise_frames: int = 30, alpha: float = 1.8, beta: float = 0.002) -> np.ndarray:
    n_fft = 512
    hop = n_fft // 4
    window = np.hanning(n_fft)
    _, _, S = signal.stft(audio, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop)
    S_mag = np.abs(S)
    S_phase = np.angle(S)
    noise_psd = np.mean(S_mag[:, :noise_frames] ** 2, axis=1, keepdims=True)
    S_mag_sq_clean = np.maximum(S_mag ** 2 - alpha * noise_psd, beta * S_mag ** 2)
    S_mag_clean = np.sqrt(S_mag_sq_clean)
    S_clean = S_mag_clean * np.exp(1j * S_phase)
    _, audio_clean = signal.istft(S_clean, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop)
    return audio_clean[:len(audio)].astype(np.float32)

def run_denoising(project_root: Path) -> Path:
    cfg = get_settings(project_root)
    noisy_path = cfg.outputs / "original_segment.wav"
    audio_noisy, sr = sf.read(str(noisy_path))
    audio_clean = spectral_subtraction(audio_noisy, sr, noise_frames=30, alpha=1.8, beta=0.002)
    audio_clean = audio_clean / (np.max(np.abs(audio_clean)) + 1e-8)
    clean_path = cfg.data_processed / "segment_denoised.wav"
    sf.write(str(clean_path), audio_clean, sr)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    t_axis = np.linspace(0, len(audio_noisy) / sr, len(audio_noisy))
    axes[0].plot(t_axis[:sr * 5], audio_noisy[:sr * 5], linewidth=0.4)
    axes[0].set_title("Noisy Signal (first 5 s)")
    axes[1].plot(t_axis[:sr * 5], audio_clean[:sr * 5], linewidth=0.4)
    axes[1].set_title("After Spectral Subtraction")
    plt.tight_layout()
    plt.savefig(cfg.outputs / "denoising_comparison.png", dpi=120)
    plt.close(fig)
    return clean_path
