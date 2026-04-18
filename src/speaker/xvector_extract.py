from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

from src.config.settings import get_settings


def _ensure_mono_16k(input_path: Path, output_path: Path) -> Path:
    """
    Load an audio file, convert to mono, resample to 16 kHz, and save it.
    Returns the output path.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Missing audio file: {input_path}")

    waveform, sample_rate = torchaudio.load(str(input_path))

    # Convert to mono if needed
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Ensure shape is [1, num_samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to 16k if needed
    target_sr = 16000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, sample_rate)

    return output_path


def run_xvector_extraction(project_root: Path) -> dict:
    """
    Extract a speaker embedding from the student's reference voice using
    SpeechBrain ECAPA-TDNN and save it to models/student_xvector.pt.
    """
    cfg = get_settings(project_root)

    raw_student_audio = cfg.data_raw / "student_voice_ref.wav"
    processed_student_audio = cfg.data_processed / "student_voice_16k_for_xvector.wav"

    model_dir = cfg.models / "speechbrain_spkrec"
    embedding_path = cfg.models / "student_xvector.pt"

    # Prepare audio
    prepared_audio = _ensure_mono_16k(raw_student_audio, processed_student_audio)

    # Load pretrained SpeechBrain speaker recognizer.
    # COPY avoids Windows symlink privilege issues.
    classifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(model_dir),
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": cfg.device},
    )

    # Load prepared waveform
    waveform, sample_rate = torchaudio.load(str(prepared_audio))

    # Safety: SpeechBrain expects mono input for this use case
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Batch shape: [batch, time]
    if waveform.dim() == 2:
        waveform_batch = waveform
    else:
        waveform_batch = waveform.squeeze(0)

    # Convert [1, time] explicitly
    if waveform_batch.dim() == 2 and waveform_batch.size(0) != 1:
        waveform_batch = waveform_batch[:1, :]

    # Relative length tensor
    rel_length = torch.tensor([1.0], device=waveform_batch.device)

    # Extract embedding
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform_batch, rel_length)

    embedding = embedding.detach().cpu()
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding, embedding_path)

    return {
        "input_audio": str(prepared_audio),
        "sample_rate": sample_rate,
        "embedding_shape": list(embedding.shape),
        "embedding_path": str(embedding_path),
        "model_dir": str(model_dir),
        "local_strategy": "COPY",
    }