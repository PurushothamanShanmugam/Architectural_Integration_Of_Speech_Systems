from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from src.config.settings import get_settings
from src.spoof.lfcc import extract_lfcc
from src.spoof.model import AntiSpoofCM

WIN_FRAMES = 200
HOP_FRAMES = 50

def audio_to_lfcc_windows(path, label, win_frames=WIN_FRAMES, hop_frames=HOP_FRAMES):
    feat = extract_lfcc(path)
    T = feat.shape[0]
    if T < win_frames:
        feat = np.concatenate([feat, np.zeros((win_frames - T, feat.shape[1]), dtype=feat.dtype)], axis=0)
        T = feat.shape[0]
    X, y = [], []
    for start in range(0, T - win_frames + 1, hop_frames):
        X.append(feat[start:start + win_frames])
        y.append(label)
    return np.array(X), np.array(y)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return float(eer), fpr, fnr

def run_anti_spoof(project_root: Path) -> dict:
    cfg = get_settings(project_root)
    bona_1 = str(cfg.data_processed / "student_voice_16k.wav")
    bona_2 = str(cfg.data_processed / "segment_denoised.wav")
    spoof = str(cfg.outputs / "output_LRL_cloned.wav")
    X1, y1 = audio_to_lfcc_windows(bona_1, 1)
    X2, y2 = audio_to_lfcc_windows(bona_2, 1)
    X3, y3 = audio_to_lfcc_windows(spoof, 0)
    X = np.concatenate([X1, X2, X3], axis=0)
    y = np.concatenate([y1, y2, y3], axis=0)

    device = cfg.device
    model = AntiSpoofCM().to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        logits = model(X_tensor)
        loss = F.cross_entropy(logits, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = torch.softmax(model(X_tensor), dim=-1)[:, 1].detach().cpu().numpy()
    eer, fpr, fnr = compute_eer(y, scores)
    torch.save(model.state_dict(), cfg.models / "cm_weights.pt")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, fnr, label=f"EER={eer:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("DET-like curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.outputs / "det_curve.png", dpi=120)
    plt.close(fig)
    return {"eer": eer}
