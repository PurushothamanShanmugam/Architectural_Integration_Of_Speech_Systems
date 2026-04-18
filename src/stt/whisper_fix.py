import importlib
import subprocess
import sys

def check_openai_whisper() -> bool:
    try:
        import whisper
        return hasattr(whisper, "load_model")
    except ImportError:
        return False

def ensure_whisper_available() -> None:
    if check_openai_whisper():
        return
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "whisper"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
    importlib.invalidate_caches()
