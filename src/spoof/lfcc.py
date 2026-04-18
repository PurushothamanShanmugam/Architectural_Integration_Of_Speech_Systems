from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import get_window, resample_poly


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _linear_filterbank(
    sr: int,
    n_fft: int,
    n_filters: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Build a linearly spaced triangular filterbank over frequency bins.
    Returns shape: [n_filters, n_fft // 2 + 1]
    """
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_freqs)

    # Linear spacing in Hz
    edges = np.linspace(fmin, fmax, n_filters + 2)

    fb = np.zeros((n_filters, n_freqs), dtype=np.float32)

    for i in range(n_filters):
        left = edges[i]
        center = edges[i + 1]
        right = edges[i + 2]

        # Rising slope
        left_mask = (freqs >= left) & (freqs <= center)
        if center > left:
            fb[i, left_mask] = (freqs[left_mask] - left) / (center - left)

        # Falling slope
        right_mask = (freqs >= center) & (freqs <= right)
        if right > center:
            fb[i, right_mask] = (right - freqs[right_mask]) / (right - center)

    return fb


def _stft_power(
    y: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute power spectrogram using a simple NumPy STFT.
    Returns shape: [n_fft // 2 + 1, num_frames]
    """
    if len(y) < win_length:
        pad = win_length - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    window_vals = get_window(window, win_length, fftbins=True).astype(np.float32)

    num_frames = 1 + max(0, (len(y) - win_length) // hop_length)
    frames = []

    for start in range(0, num_frames * hop_length, hop_length):
        frame = y[start:start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)), mode="constant")

        frame = frame * window_vals

        if win_length < n_fft:
            frame = np.pad(frame, (0, n_fft - win_length), mode="constant")

        spectrum = np.fft.rfft(frame, n=n_fft)
        power = np.abs(spectrum) ** 2
        frames.append(power.astype(np.float32))

    if not frames:
        # Fallback for extremely short signals
        frame = np.pad(y[:win_length], (0, max(0, win_length - len(y[:win_length]))), mode="constant")
        frame = frame * window_vals
        if win_length < n_fft:
            frame = np.pad(frame, (0, n_fft - win_length), mode="constant")
        spectrum = np.fft.rfft(frame, n=n_fft)
        power = np.abs(spectrum) ** 2
        frames = [power.astype(np.float32)]

    return np.stack(frames, axis=1)  # [freq_bins, frames]


def extract_lfcc(
    audio_path,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_lfcc: int = 20,
    n_filters: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
    include_energy: bool = True,
) -> np.ndarray:
    """
    Extract LFCC features from an audio file.

    Returns:
        np.ndarray with shape [num_frames, feature_dim]
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    y, file_sr = sf.read(str(audio_path))

    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float32)

    # Resample to target SR if needed
    if file_sr != sr:
        y = resample_poly(y, sr, file_sr).astype(np.float32)

    # Basic normalization
    peak = np.max(np.abs(y)) if len(y) > 0 else 0.0
    if peak > 0:
        y = y / peak

    if fmax is None:
        fmax = sr / 2.0

    power_spec = _stft_power(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    )  # [freq_bins, frames]

    filterbank = _linear_filterbank(
        sr=sr,
        n_fft=n_fft,
        n_filters=n_filters,
        fmin=fmin,
        fmax=fmax,
    )  # [n_filters, freq_bins]

    # Filterbank energies: [n_filters, frames]
    fb_energies = np.matmul(filterbank, power_spec)

    # Numerical stability
    fb_energies = np.maximum(fb_energies, 1e-10)

    # Log energies
    log_fb = np.log(fb_energies)

    # DCT across filter dimension -> LFCCs
    coeffs = dct(log_fb, type=2, axis=0, norm="ortho")[:n_lfcc, :]  # [n_lfcc, frames]

    if include_energy:
        frame_energy = np.log(np.maximum(np.sum(power_spec, axis=0, keepdims=True), 1e-10))
        coeffs[0:1, :] = frame_energy

    # Return [frames, features]
    return coeffs.T.astype(np.float32)