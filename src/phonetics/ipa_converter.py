from pathlib import Path
import numpy as np
import re
from phonemizer import phonemize
from src.config.settings import get_settings

HINGLISH_IPA_MAP = {
    "aa": "aː", "ee": "iː", "oo": "uː", "ai": "ɛː", "au": "ɔː", "ae": "eː",
    "a": "ə", "i": "ɪ", "u": "ʊ", "e": "e", "o": "o",
    "kh": "kʰ", "gh": "ɡʱ", "ch": "tʃ", "chh": "tʃʰ", "jh": "dʒʱ", "th": "tʰ", "dh": "dʱ",
    "ph": "pʰ", "bh": "bʱ", "mh": "mʱ", "sh": "ʃ", "nh": "ɲ", "ng": "ŋ",
    "rr": "ɽ", "rrh": "ɽʱ", "tt": "ʈ", "dd": "ɖ", "tth": "ʈʰ", "ddh": "ɖʱ", "nn": "ɳ",
    "ya": "jə", "wa": "wə", "hai": "hɛː", "hain": "hɛ̃ː", "ka": "kə", "ki": "kɪ", "ko": "ko",
    "ke": "keː", "se": "seː", "me": "meː", "mein": "mẽː", "kya": "kjɑː", "yeh": "jeː",
    "woh": "woː", "aur": "ɔːr", "nahi": "nəhɪː", "toh": "toː", "matlab": "mətlɐb",
    "seedha": "siːdʰɑː", "samajhna": "sɐmɐdʒʰnɑː", "bolna": "bolnɑː", "accha": "ɐtʃʰɑː",
    "theek": "tʰiːk",
}
HINGLISH_IPA_MAP_SORTED = sorted(HINGLISH_IPA_MAP.items(), key=lambda x: -len(x[0]))

def hinglish_to_ipa(word: str) -> str:
    word_lower = word.lower()
    ipa = ""
    i = 0
    while i < len(word_lower):
        matched = False
        for pattern, ipa_sym in HINGLISH_IPA_MAP_SORTED:
            if word_lower[i:].startswith(pattern):
                ipa += ipa_sym
                i += len(pattern)
                matched = True
                break
        if not matched:
            ipa += word_lower[i]
            i += 1
    return ipa

def text_to_unified_ipa(text: str, frame_preds: np.ndarray, words_list: list[str]) -> str:
    ipa_tokens = []
    frame_cursor = 0
    for word in words_list:
        span_frames = max(1, int(len(word) * 6))
        segment = frame_preds[frame_cursor:frame_cursor + span_frames]
        lang = 1 if len(segment) and segment.mean() > 0.5 else 0
        clean = re.sub(r"[^\w\-']", "", word)
        if not clean:
            continue
        if lang == 1:
            try:
                ipa = phonemize(clean, language="en-us", backend="espeak", strip=True, preserve_punctuation=False)
            except Exception:
                ipa = clean
        else:
            ipa = hinglish_to_ipa(clean)
        ipa_tokens.append(ipa)
        frame_cursor += span_frames
    return " ".join(ipa_tokens)

def run_ipa_conversion(project_root: Path) -> str:
    cfg = get_settings(project_root)
    text = (cfg.outputs / "transcript.txt").read_text(encoding="utf-8")
    words = text.split()
    frame_preds = np.load(cfg.outputs / "lid_preds.npy")
    ipa_text = text_to_unified_ipa(text, frame_preds, words)
    (cfg.outputs / "transcript_ipa.txt").write_text(ipa_text, encoding="utf-8")
    return ipa_text
