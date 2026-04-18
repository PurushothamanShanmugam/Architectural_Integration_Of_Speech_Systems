from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass(frozen=True)
class Settings:
    project_root: Path
    target_sr: int = 16000
    segment_start_sec: int = 0
    segment_length_sec: int = 600
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def data_intermediate(self) -> Path:
        return self.project_root / "data" / "intermediate"

    @property
    def outputs(self) -> Path:
        return self.project_root / "outputs"

    @property
    def models(self) -> Path:
        return self.project_root / "models"

def get_settings(project_root: Path) -> Settings:
    return Settings(project_root=project_root)
