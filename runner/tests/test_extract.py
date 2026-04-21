from llm_lb.eval.extract import extract_label, extract_regex, normalize, strip_reasoning


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


def test_extract_label_plain_match():
    assert extract_label("benign", ["benign", "jailbreak"]) == "benign"
    assert extract_label("The answer is jailbreak.", ["benign", "jailbreak"]) == "jailbreak"


def test_extract_label_substring_ambiguity():
    # "safe" is a substring of "unsafe" — the earlier iteration-order
    # implementation would match "safe" inside every "unsafe" output,
    # pinning `safety_classification` to 0.500 for every model.
    assert extract_label("unsafe", ["safe", "unsafe"]) == "unsafe"
    assert extract_label("Label: unsafe", ["safe", "unsafe"]) == "unsafe"
    assert extract_label("safe", ["safe", "unsafe"]) == "safe"
    # Earliest occurrence wins when the candidates are disjoint.
    assert extract_label("unsafe then safe", ["safe", "unsafe"]) == "unsafe"
    assert extract_label("It's safe, not unsafe", ["safe", "unsafe"]) == "safe"


def test_extract_label_case_insensitive():
    assert extract_label("UNSAFE", ["safe", "unsafe"]) == "unsafe"


def test_extract_label_no_match_returns_stripped_raw():
    assert extract_label("  maybe  ", ["yes", "no"]) == "maybe"


def test_extract_label_aliases_map_synonym_to_canonical():
    """Llama-Guard outputs `unsafe`/`safe`; `jailbreak_detection` uses
    `jailbreak`/`safe`. The alias map lets Guard score on this task without
    adapter-level hacks."""
    labels = ["jailbreak", "safe"]
    aliases = {"unsafe": "jailbreak"}
    assert extract_label("unsafe", labels, aliases) == "jailbreak"
    assert extract_label("safe", labels, aliases) == "safe"
    # `safe` is a substring of `unsafe`; longest-match still wins so `unsafe`
    # resolves to `jailbreak` rather than being read as an inner `safe`.
    assert extract_label("Label: unsafe", labels, aliases) == "jailbreak"


def test_extract_label_aliases_full_remap():
    """beavertails uses `benign`/`jailbreak`; Guard outputs `safe`/`unsafe`.
    Both sides need remapping."""
    labels = ["benign", "jailbreak"]
    aliases = {"unsafe": "jailbreak", "safe": "benign"}
    assert extract_label("unsafe", labels, aliases) == "jailbreak"
    assert extract_label("safe", labels, aliases) == "benign"
    assert extract_label("benign", labels, aliases) == "benign"  # canonical still works
    assert extract_label("jailbreak", labels, aliases) == "jailbreak"


def test_extract_label_aliases_none_is_default():
    """No aliases argument keeps the current behaviour unchanged."""
    assert extract_label("unsafe", ["jailbreak", "safe"]) == "safe"  # substring fallback
    assert extract_label("unsafe", ["jailbreak", "safe"], None) == "safe"


def test_extract_label_labels_with_commas():
    labels = [
        "animal_abuse",
        "controversial_topics,politics",
        "violence,aiding_and_abetting,incitement",
    ]
    assert (
        extract_label("controversial_topics,politics", labels)
        == "controversial_topics,politics"
    )
    assert (
        extract_label(
            "violence,aiding_and_abetting,incitement",
            labels,
        )
        == "violence,aiding_and_abetting,incitement"
    )
