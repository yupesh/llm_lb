from llm_lb.eval.extract import extract_regex, normalize, strip_reasoning


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


def test_strip_reasoning_removes_think_blocks():
    assert strip_reasoning("<think>a or b?</think>b") == "b"
    assert strip_reasoning("<think>x</think>\n\nParis") == "Paris"
    assert strip_reasoning("<THINK>upper</THINK>x") == "x"
    assert strip_reasoning("<think>a</think><think>b</think>final") == "final"


def test_strip_reasoning_noop_on_plain_text():
    assert strip_reasoning("plain answer") == "plain answer"
    assert strip_reasoning("unclosed <think>trace still open") == "unclosed <think>trace still open"
