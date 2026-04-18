from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from src.config.settings import get_settings
from src.lid.features import extract_features
from src.lid.model import MultiHeadLID
from src.lid.synthetic_labels import generate_markov_labels

def train_lid_model(project_root: Path) -> dict:
    cfg = get_settings(project_root)
    device = cfg.device
    features = extract_features(project_root)
    labels = generate_markov_labels(features.shape[0])
    split = int(0.8 * features.shape[0])
    X_train, X_val = features[:split].unsqueeze(0), features[split:].unsqueeze(0)
    y_train, y_val = labels[:split].unsqueeze(0), labels[split:].unsqueeze(0)

    model = MultiHeadLID().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    epochs = 20
    seq_len = 200
    train_losses, val_f1s = [], []

    for _epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        T_tr = X_train.shape[1]
        for start in range(0, T_tr - seq_len, seq_len):
            xb = X_train[:, start:start + seq_len, :].to(device)
            yb = y_train[:, start:start + seq_len].to(device)
            logits = model(xb)
            loss = criterion(logits.reshape(-1, 2), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            T_val = X_val.shape[1]
            preds_all, labels_all = [], []
            for start in range(0, T_val - seq_len, seq_len):
                xb = X_val[:, start:start + seq_len, :].to(device)
                yb = y_val[:, start:start + seq_len].to(device)
                logits = model(xb)
                preds = logits.argmax(-1).reshape(-1).cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(yb.reshape(-1).cpu().numpy())
            f1 = f1_score(labels_all, preds_all, average="macro") if labels_all else 0.0
            val_f1s.append(f1)
        avg_loss = epoch_loss / max(1, (T_tr // seq_len))
        train_losses.append(avg_loss)

    torch.save(model.state_dict(), cfg.models / "lid_weights.pt")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, marker="o")
    ax1.set_title("Training Loss")
    ax2.plot(val_f1s, marker="s")
    ax2.axhline(0.85, color="red", linestyle="--")
    ax2.set_title("Validation Macro F1")
    plt.tight_layout()
    plt.savefig(cfg.outputs / "lid_training.png", dpi=120)
    plt.close(fig)
    return {"val_f1s": val_f1s, "device": device}
