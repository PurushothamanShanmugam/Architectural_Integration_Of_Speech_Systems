from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TAF
from src.config.settings import get_settings
from src.lid.model import MultiHeadLID

def set_lstm_train(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            m.train()

def wav_to_mel_feats(wav: torch.Tensor, sr: int = 16000, n_fft: int = 400, hop: int = 160, n_mels: int = 40) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(wav.squeeze(0), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    power = stft.abs() ** 2
    fb = TAF.melscale_fbanks(
        n_freqs=n_fft // 2 + 1, f_min=20.0, f_max=float(sr // 2),
        n_mels=n_mels, sample_rate=sr, norm="slaney", mel_scale="htk"
    ).to(wav.device)
    mel = torch.matmul(power.T, fb)
    log_mel = torch.log(mel + 1e-9)
    mu = log_mel.mean(0, keepdim=True)
    std = log_mel.std(0, keepdim=True) + 1e-8
    return ((log_mel - mu) / std).unsqueeze(0)

def lid_forward_differentiable(wav: torch.Tensor, lid_model: nn.Module) -> torch.Tensor:
    feats = wav_to_mel_feats(wav)
    logits = lid_model(feats)
    return logits.mean(dim=1)

def run_fgsm_attack(project_root: Path) -> dict:
    cfg = get_settings(project_root)
    device = cfg.device
    lid_model = MultiHeadLID().to(device)
    lid_model.load_state_dict(torch.load(cfg.models / "lid_weights.pt", map_location=device))
    lid_model.eval()
    set_lstm_train(lid_model)
    for p in lid_model.parameters():
        p.requires_grad = False

    wav_np, sr = sf.read(str(cfg.data_processed / "segment_denoised.wav"))
    wav_np = wav_np[: 5 * sr].astype(np.float32)
    wav = torch.tensor(wav_np, dtype=torch.float32, device=device).unsqueeze(0)
    base_logits = lid_forward_differentiable(wav, lid_model)
    target = torch.tensor([1 - int(base_logits.argmax(dim=-1).item())], device=device)

    eps_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    success = []
    for eps in eps_values:
        x = wav.clone().detach().requires_grad_(True)
        logits = lid_forward_differentiable(x, lid_model)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        adv = torch.clamp(x + eps * x.grad.sign(), -1.0, 1.0)
        adv_logits = lid_forward_differentiable(adv.detach(), lid_model)
        success.append(int(adv_logits.argmax(dim=-1).item() == int(target.item())))

    best_eps = eps_values[int(np.argmax(success))]
    x = wav.clone().detach().requires_grad_(True)
    logits = lid_forward_differentiable(x, lid_model)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    adv = torch.clamp(x + best_eps * x.grad.sign(), -1.0, 1.0)
    adv_np = adv.detach().cpu().squeeze(0).numpy()
    sf.write(str(cfg.outputs / "adversarial_5s_sample.wav"), adv_np, sr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps_values, success, marker="o")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("attack success")
    ax.set_title("FGSM sweep")
    plt.tight_layout()
    plt.savefig(cfg.outputs / "fgsm_sweep.png", dpi=120)
    plt.close(fig)
    return {"best_eps": best_eps}
