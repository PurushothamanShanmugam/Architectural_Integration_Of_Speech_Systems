from pathlib import Path
import soundfile as sf
from src.config.settings import get_settings
from src.utils.io_utils import ensure_dirs
from src.utils.audio_utils import load_mono_resampled

def run_input_preparation(project_root: Path) -> dict:
    cfg = get_settings(project_root)
    ensure_dirs(cfg.data_raw, cfg.data_processed, cfg.data_intermediate, cfg.outputs, cfg.models)

    lecture_path = cfg.data_raw / "lecture.wav"
    student_path = cfg.data_raw / "student_voice_ref.wav"
    if not lecture_path.exists():
        raise FileNotFoundError(f"Missing input file: {lecture_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Missing input file: {student_path}")

    waveform, sr = load_mono_resampled(lecture_path, cfg.target_sr)
    start_sample = int(cfg.segment_start_sec * sr)
    end_sample = start_sample + int(cfg.segment_length_sec * sr)
    segment = waveform[:, start_sample:end_sample]
    sf.write(str(cfg.outputs / "original_segment.wav"), segment.squeeze().numpy(), sr)

    student_wav, student_sr = load_mono_resampled(student_path, cfg.target_sr)
    sf.write(str(cfg.data_processed / "student_voice_16k.wav"), student_wav.squeeze().numpy(), student_sr)

    return {
        "lecture_sr": sr,
        "lecture_duration_sec": segment.shape[-1] / sr,
        "student_duration_sec": student_wav.shape[-1] / student_sr,
    }
