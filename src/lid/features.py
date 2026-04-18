from pathlib import Path
import torch
import torchaudio
import soundfile as sf
from src.config.settings import get_settings

class MelFeatureExtractor:
    def __init__(self, device: str, sr: int = 16000, n_mels: int = 40, win_ms: int = 25, hop_ms: int = 10):
        self.device = device
        self.mel_fb = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(win_ms / 1000 * sr),
            hop_length=int(hop_ms / 1000 * sr),
            n_mels=n_mels,
            f_min=20,
            f_max=sr // 2,
        ).to(device)

    def __call__(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        mel = self.mel_fb(wav_tensor)
        log_mel = torch.log(mel + 1e-9)
        mu = log_mel.mean(-1, keepdim=True)
        std = log_mel.std(-1, keepdim=True) + 1e-8
        return ((log_mel - mu) / std).squeeze(0).T

def extract_features(project_root: Path) -> torch.Tensor:
    cfg = get_settings(project_root)
    device = cfg.device
    audio_array, sr = sf.read(str(cfg.data_processed / "segment_denoised.wav"))
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(device)
    feat_extractor = MelFeatureExtractor(device=device, sr=sr)
    chunk_samples = int(30 * sr)
    parts = []
    for start in range(0, audio_tensor.shape[-1], chunk_samples):
        chunk = audio_tensor[:, start:start + chunk_samples]
        parts.append(feat_extractor(chunk))
    return torch.cat(parts, dim=0)
