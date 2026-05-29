from src.features.handcrafted_features import extract_handcrafted_features


def test_extract_handcrafted_features_detects_url_and_html_signals():
    features = extract_handcrafted_features(
        "https://secure-login.example.com/update",
        "<html><form action='https://bad.example/collect'><input type='password'></form></html>",
    )
    assert features.contains_login == 1
    assert features.contains_secure == 1
    assert features.contains_update == 1
    assert features.num_forms == 1
    assert features.num_password_inputs == 1
    assert features.num_suspicious_form_actions == 1
