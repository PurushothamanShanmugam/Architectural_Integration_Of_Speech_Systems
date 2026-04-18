from pathlib import Path
import json
import torch
from transformers import pipeline
from src.config.settings import get_settings


def _segments_from_chunks(chunks):
    segments = []
    for i, chunk in enumerate(chunks or []):
        ts = chunk.get("timestamp", None)
        if ts is None:
            start, end = None, None
        else:
            start, end = ts

        segments.append(
            {
                "id": i,
                "start": None if start is None else round(float(start), 2),
                "end": None if end is None else round(float(end), 2),
                "text": chunk.get("text", "").strip(),
            }
        )
    return segments


def run_transcription(project_root: Path) -> dict:
    cfg = get_settings(project_root)

    cfg.outputs.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.models / "hf_whisper_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_path = cfg.data_processed / "segment_denoised.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing denoised audio file: {audio_path}")

    device = 0 if torch.cuda.is_available() else -1

    # Real Whisper model via Hugging Face Transformers
    model_name = "openai/whisper-tiny"

    print(f"Loading Whisper model from Hugging Face: {model_name}")
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        batch_size=4,
        device=device,
        model_kwargs={"cache_dir": str(cache_dir)},
    )

    print("Starting transcription with Whisper...")
    result = asr(
        str(audio_path),
        return_timestamps=True,
        generate_kwargs={
            "task": "transcribe"
            # Optional:
            # "language": "en"
            # "language": "hi"
        },
    )

    transcript_text = result.get("text", "").strip()
    segments = _segments_from_chunks(result.get("chunks", []))

    if not transcript_text:
        raise RuntimeError("Whisper returned an empty transcript.")

    transcript_path = cfg.outputs / "transcript.txt"
    segments_path = cfg.outputs / "transcript_segments.json"

    transcript_path.write_text(transcript_text, encoding="utf-8")

    with segments_path.open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to: {transcript_path}")
    print(f"Segments saved to: {segments_path}")

    return {
        "model_name": model_name,
        "device": "cuda" if device == 0 else "cpu",
        "transcript_path": str(transcript_path),
        "segments_path": str(segments_path),
        "num_segments": len(segments),
    }