from llm_lb.eval.extract import extract_regex, normalize


def test_normalize_strips_punct_and_case():
    assert normalize("  Paris.") == "paris"
    assert normalize('"William Shakespeare"!') == "william shakespeare"
    assert normalize("Pacific  Ocean") == "pacific ocean"


def test_extract_regex_first_group():
    assert extract_regex("Answer: 42 done.", r"Answer:\s*(\d+)") == "42"


def test_extract_regex_no_group_returns_match():
    assert extract_regex("hello world", r"world") == "world"


def test_extract_regex_no_match_returns_stripped_input():
    assert extract_regex("  garbage  ", r"\d+") == "garbage"
