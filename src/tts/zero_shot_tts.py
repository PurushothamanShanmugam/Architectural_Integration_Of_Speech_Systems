from pathlib import Path
import os
import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel
from src.config.settings import get_settings

def chunk_text(text: str, max_chars: int = 150):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) >= max_chars:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def run_tts(project_root: Path) -> Path:
    cfg = get_settings(project_root)
    device = cfg.device
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
    mms_sr = model.config.sampling_rate
    out_sr = 22050

    file_path = cfg.outputs / "transcript_santali.txt"
    if not file_path.exists():
        file_path.write_text("This is a placeholder text for TTS generation.", encoding="utf-8")
    full_text = file_path.read_text(encoding="utf-8").strip()
    chunks = chunk_text(full_text, max_chars=150)

    audio_parts = []
    silence = np.zeros(int(0.3 * out_sr), dtype=np.float32)
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            inputs = tokenizer(chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            wav = out.waveform.squeeze().cpu().numpy().astype(np.float32)
            n = int(len(wav) * out_sr / mms_sr)
            wav = ss.resample(wav, n)
            audio_parts.extend([wav, silence])
        except Exception:
            audio_parts.append(silence)
    full_audio = np.concatenate(audio_parts) if audio_parts else silence
    full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-8) * 0.95
    out_path = cfg.outputs / "output_LRL_cloned.wav"
    sf.write(str(out_path), full_audio, out_sr)
    return out_path
