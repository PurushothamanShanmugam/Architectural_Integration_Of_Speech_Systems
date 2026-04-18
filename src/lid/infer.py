from pathlib import Path
import numpy as np
import torch
from src.config.settings import get_settings
from src.lid.features import extract_features
from src.lid.model import MultiHeadLID
from src.utils.json_utils import dump_json

def run_lid_inference(project_root: Path) -> np.ndarray:
    cfg = get_settings(project_root)
    device = cfg.device
    features = extract_features(project_root)
    model = MultiHeadLID().to(device)
    model.load_state_dict(torch.load(cfg.models / "lid_weights.pt", map_location=device))
    model.eval()

    seq_len = 200
    T_total = features.shape[0]
    all_preds = []
    with torch.no_grad():
        for start in range(0, T_total - seq_len + 1, seq_len):
            xb = features[start:start + seq_len].unsqueeze(0).to(device)
            logits = model(xb)
            preds = logits.argmax(-1).squeeze(0).cpu().numpy()
            all_preds.extend(preds)
        rem = T_total % seq_len
        if rem > 0:
            xb = features[-seq_len:].unsqueeze(0).to(device)
            logits = model(xb)
            all_preds.extend(logits.argmax(-1).squeeze(0).cpu().numpy()[-rem:])

    lid_preds = np.array(all_preds[:T_total])
    switches = []
    for i in range(1, len(lid_preds)):
        if lid_preds[i] != lid_preds[i - 1]:
            switches.append({"frame": int(i), "time_ms": int(i * 10), "to_lang": "EN" if lid_preds[i] == 1 else "HI"})
    dump_json(cfg.outputs / "lid_switches.json", switches)
    np.save(cfg.outputs / "lid_preds.npy", lid_preds)
    return lid_preds
