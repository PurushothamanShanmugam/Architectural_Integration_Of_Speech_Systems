from src.phonetics.ipa_converter import hinglish_to_ipa

def test_hinglish_mapping_prefers_longest_match():
    assert hinglish_to_ipa("samajhna").startswith("s")
    assert "ɐ" in hinglish_to_ipa("matlab")
