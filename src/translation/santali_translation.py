from pathlib import Path
import re
from src.config.settings import get_settings

SANTALI_DICT = {
    "speech": "boli", "language": "bhasa", "voice": "aawaj", "sound": "sur", "noise": "ghol",
    "frequency": "dhorom leka", "signal": "ishar", "acoustic": "leka dhon", "audio": "aawaj",
    "recording": "rekord", "microphone": "mikrofon", "model": "naman", "learning": "gian",
    "training": "sikha", "network": "jaal", "neural": "niyurel", "deep": "gahir",
    "feature": "khashiyat", "input": "aadar", "output": "nirgam", "classification": "bhed",
    "prediction": "anumanit", "accuracy": "thik", "error": "galti", "loss": "harani",
    "gradient": "dhalan", "optimizer": "sudharak", "phoneme": "dhoni ekak", "vowel": "swar",
    "consonant": "byanjan", "syllable": "akshor", "stress": "jor", "pitch": "sur uchata",
    "tone": "aahat", "prosody": "boli dhara", "algorithm": "niyam kram", "data": "tattha",
    "computer": "ganakjan", "program": "karjokram", "code": "sanket", "function": "karya",
    "system": "prabandha", "process": "kriya", "method": "tarika", "analysis": "visleshan",
    "result": "phal", "test": "pariksha", "lecture": "pathsala", "class": "kaksha",
    "student": "chhatra", "teacher": "guru", "university": "mahavidyalaya", "assignment": "karya",
    "cepstrum": "cepstrum", "spectrogram": "dhwani chitr", "transform": "rupantar",
}

def translate_to_santali(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    translated = []
    for tok in tokens:
        if re.fullmatch(r"\w+", tok):
            translated.append(SANTALI_DICT.get(tok, tok))
        else:
            translated.append(tok)
    return " ".join(translated).replace(" ,", ",").replace(" .", ".")

def run_translation(project_root: Path) -> str:
    cfg = get_settings(project_root)
    transcript = (cfg.outputs / "transcript.txt").read_text(encoding="utf-8")
    translated = translate_to_santali(transcript)
    (cfg.outputs / "transcript_santali.txt").write_text(translated, encoding="utf-8")
    return translated
