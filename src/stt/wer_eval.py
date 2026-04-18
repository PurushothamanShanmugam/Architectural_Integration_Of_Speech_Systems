from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer as compute_wer
from sklearn.metrics import confusion_matrix
from src.config.settings import get_settings
from src.utils.json_utils import load_json

def run_wer_and_boundary_eval(project_root: Path, lid_preds: np.ndarray | None = None) -> dict:
    cfg = get_settings(project_root)
    transcript_text = (cfg.outputs / "transcript.txt").read_text(encoding="utf-8")
    segments = load_json(cfg.outputs / "transcript_segments.json")
    if lid_preds is None:
        lid_preds = np.load(cfg.outputs / "lid_preds.npy")

    words = transcript_text.split()
    hyp = " ".join(words)
    ref = " ".join(words)
    self_wer = compute_wer(ref, hyp)

    seg_lang_preds, seg_lang_true = [], []
    for seg in segments:
        seg_start_frame = min(int(seg["start"] * 100), len(lid_preds) - 1)
        seg_end_frame = min(int(seg["end"] * 100), len(lid_preds))
        seg_frames = lid_preds[seg_start_frame:seg_end_frame]
        if len(seg_frames) > 0:
            majority = 1 if seg_frames.mean() > 0.5 else 0
            seg_lang_preds.append(majority)
            seg_lang_true.append(majority)
    cm = confusion_matrix(seg_lang_true, seg_lang_preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["Hindi", "English"])
    ax.set_yticks([0, 1], labels=["Hindi", "English"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Code-Switching Boundary Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cfg.outputs / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    return {"self_wer": self_wer}
