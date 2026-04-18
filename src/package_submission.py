from pathlib import Path
import zipfile
from src.config.settings import get_settings

def package_outputs(project_root: Path, roll_no: str = "ROLLNO") -> Path:
    cfg = get_settings(project_root)
    zip_path = cfg.project_root / f"{roll_no}_PA2.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "outputs/original_segment.wav",
            "outputs/output_LRL_cloned.wav",
            "data/processed/student_voice_16k.wav",
            "outputs/transcript.txt",
            "outputs/transcript_ipa.txt",
            "outputs/transcript_santali.txt",
            "outputs/transcript_segments.json",
            "outputs/lid_switches.json",
            "outputs/evaluation_summary.txt",
            "outputs/denoising_comparison.png",
            "outputs/lid_training.png",
            "outputs/prosody_warping.png",
            "outputs/det_curve.png",
            "outputs/fgsm_sweep.png",
            "outputs/confusion_matrix.png",
            "outputs/xvector_plot.png",
            "models/lid_weights.pt",
            "models/cm_weights.pt",
            "models/student_xvector.pt",
        ]:
            path = cfg.project_root / rel
            if path.exists():
                zf.write(path, rel)
    return zip_path
