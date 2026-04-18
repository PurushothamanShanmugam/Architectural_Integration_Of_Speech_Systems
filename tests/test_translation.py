from src.translation.santali_translation import translate_to_santali

def test_translation_uses_dictionary_when_available():
    text = "speech model test"
    out = translate_to_santali(text)
    assert "boli" in out
    assert "naman" in out
