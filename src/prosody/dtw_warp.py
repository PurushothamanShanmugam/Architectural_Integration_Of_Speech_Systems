from pathlib import Path
import json

import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

from src.config.settings import get_settings


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    num_frames = 1 + max(0, (len(y) - frame_length) // hop_length)
    frames = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frames.append(frame)
    return np.stack(frames, axis=0) if frames else np.zeros((1, frame_length), dtype=np.float32)


def _compute_energy(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-8).astype(np.float32)


def _compute_f0_autocorr(frames: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 400.0) -> np.ndarray:
    """
    Simple autocorrelation-based F0 estimation.
    Returns 0.0 for unvoiced frames.
    """
    min_lag = max(1, int(sr / fmax))
    max_lag = max(min_lag + 1, int(sr / fmin))

    f0_values = []
    window = np.hanning(frames.shape[1]).astype(np.float32)

    for frame in frames:
        x = frame.astype(np.float32)
        x = x - np.mean(x)
        x = x * window

        energy = np.sum(x * x)
        if energy < 1e-6:
            f0_values.append(0.0)
            continue

        ac = np.correlate(x, x, mode="full")
        ac = ac[len(ac) // 2:]

        if max_lag >= len(ac):
            max_lag_eff = len(ac) - 1
        else:
            max_lag_eff = max_lag

        if min_lag >= max_lag_eff:
            f0_values.append(0.0)
            continue

        region = ac[min_lag:max_lag_eff]
        if len(region) == 0:
            f0_values.append(0.0)
            continue

        peak_idx = np.argmax(region) + min_lag
        peak_val = ac[peak_idx]
        norm = ac[0] + 1e-8

        # Simple voiced/unvoiced threshold
        if peak_val / norm < 0.3:
            f0_values.append(0.0)
        else:
            f0_values.append(float(sr / peak_idx))

    return np.array(f0_values, dtype=np.float32)


def extract_f0_energy(audio_path: str, sr: int = 16000, frame_ms: float = 25.0, hop_ms: float = 10.0):
    y, file_sr = sf.read(audio_path)
    y = _to_mono(np.asarray(y))

    if file_sr != sr:
        # lightweight resampling by interpolation
        old_t = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        new_len = int(len(y) * sr / file_sr)
        new_t = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        y = np.interp(new_t, old_t, y).astype(np.float32)

    frame_length = int(sr * frame_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)

    frames = _frame_signal(y, frame_length=frame_length, hop_length=hop_length)
    energy = _compute_energy(frames)
    f0 = _compute_f0_autocorr(frames, sr=sr)

    return f0, energy


def _simple_dtw_path(cost_matrix: np.ndarray):
    n, m = cost_matrix.shape
    acc = np.full((n, m), np.inf, dtype=np.float64)
    acc[0, 0] = cost_matrix[0, 0]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0:
                candidates.append(acc[i - 1, j])
            if j > 0:
                candidates.append(acc[i, j - 1])
            if i > 0 and j > 0:
                candidates.append(acc[i - 1, j - 1])
            acc[i, j] = cost_matrix[i, j] + min(candidates)

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        moves = []
        if i > 0 and j > 0:
            moves.append((acc[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            moves.append((acc[i - 1, j], i - 1, j))
        if j > 0:
            moves.append((acc[i, j - 1], i, j - 1))
        _, i, j = min(moves, key=lambda x: x[0])
        path.append((i, j))

    path.reverse()
    return path


def run_prosody_alignment(project_root: Path) -> dict:
    cfg = get_settings(project_root)

    src_audio = cfg.data_processed / "segment_denoised.wav"
    ref_audio = cfg.data_raw / "student_voice_ref.wav"

    if not src_audio.exists():
        raise FileNotFoundError(f"Missing source audio: {src_audio}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"Missing reference audio: {ref_audio}")

    src_f0, src_energy = extract_f0_energy(str(src_audio))
    ref_f0, ref_energy = extract_f0_energy(str(ref_audio))

    # Downsample feature sequences for tractable DTW
    src_feat = np.stack([src_f0, src_energy], axis=1)
    ref_feat = np.stack([ref_f0, ref_energy], axis=1)

    src_idx = np.arange(0, len(src_feat), 5)
    ref_idx = np.arange(0, len(ref_feat), 5)

    if len(src_idx) == 0:
        src_idx = np.array([0])
    if len(ref_idx) == 0:
        ref_idx = np.array([0])

    src_ds = src_feat[src_idx]
    ref_ds = ref_feat[ref_idx]

    cost = cdist(src_ds, ref_ds, metric="euclidean")
    path = _simple_dtw_path(cost)

    out_path = cfg.outputs / "prosody_alignment.json"
    cfg.outputs.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_num_frames": int(len(src_feat)),
        "reference_num_frames": int(len(ref_feat)),
        "source_downsampled_frames": int(len(src_ds)),
        "reference_downsampled_frames": int(len(ref_ds)),
        "dtw_path_length": int(len(path)),
        "dtw_path": [[int(i), int(j)] for i, j in path[:5000]],  # avoid huge JSON
        "source_f0_mean": float(np.mean(src_f0[src_f0 > 0])) if np.any(src_f0 > 0) else 0.0,
        "reference_f0_mean": float(np.mean(ref_f0[ref_f0 > 0])) if np.any(ref_f0 > 0) else 0.0,
        "source_energy_mean": float(np.mean(src_energy)),
        "reference_energy_mean": float(np.mean(ref_energy)),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "prosody_alignment_path": str(out_path),
        "dtw_path_length": int(len(path)),
        "source_frames": int(len(src_feat)),
        "reference_frames": int(len(ref_feat)),
    }