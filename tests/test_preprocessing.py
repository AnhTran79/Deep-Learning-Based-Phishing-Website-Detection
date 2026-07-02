from src.preprocessing.text_cleaning import html_to_visible_text, normalize_text
from src.preprocessing.tokenizers import CharTokenizerConfig, encode_char_sequence


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  Login\n\tSecure  ") == "login secure"


def test_html_to_visible_text_keeps_tag_tokens():
    text = html_to_visible_text("<form><input type='password'></form>")
    assert "tag_form" in text


def test_html_to_visible_text_keeps_phishing_signals():
    text = html_to_visible_text(
        "<form action='https://evil.example/login'>"
        "<input name='email'><input type='password' placeholder='Password'>"
        "<a href='https://bank.example/reset'>Reset</a></form>"
    )
    assert "attr_type_password" in text
    assert "attr_name_email" in text
    assert "action_domain_evil.example" in text
    assert "href_domain_bank.example" in text


def test_encode_char_sequence_pads_to_max_len():
    encoded = encode_char_sequence("abc", CharTokenizerConfig(max_len=5, vocab_size=128))
    assert len(encoded) == 5
    assert encoded[-2:] == [0, 0]
