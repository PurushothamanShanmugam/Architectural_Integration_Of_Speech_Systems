from pathlib import Path
import librosa
import numpy as np
from src.config.settings import get_settings

def compute_mcd(ref_path: str, synth_path: str, sr: int = 16000, n_mfcc: int = 13) -> float:
    ref, _ = librosa.load(ref_path, sr=sr)
    synth, _ = librosa.load(synth_path, sr=sr)
    mc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc + 1)[1:]
    mc_synth = librosa.feature.mfcc(y=synth, sr=sr, n_mfcc=n_mfcc + 1)[1:]
    min_t = min(mc_ref.shape[1], mc_synth.shape[1])
    mc_ref = mc_ref[:, :min_t]
    mc_synth = mc_synth[:, :min_t]
    diff = mc_ref - mc_synth
    mcd_frames = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=0))
    return float(np.mean(mcd_frames))

def run_mcd(project_root: Path) -> float:
    cfg = get_settings(project_root)
    return compute_mcd(str(cfg.data_processed / "student_voice_16k.wav"), str(cfg.outputs / "output_LRL_cloned.wav"), sr=16000)
