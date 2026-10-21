from src.features.code_search.domain.rules.code_search_rules import (
    build_bad_character_table,
    boyer_moore_horspool_search,
    AhoCorasickAutomaton,
)


def test_boyer_moore_horspool_search():
    text = "import httpx from python_shared.http import client"
    pattern = "httpx"
    matches = boyer_moore_horspool_search(text, pattern)
    assert matches == [7]


def test_aho_corasick_automaton():
    patterns = ["httpx", "client", "python_shared"]
    automaton = AhoCorasickAutomaton(patterns)
    text = "from python_shared.http import httpx, client"
    results = automaton.search(text)
    matched_patterns = {pat for _, pat in results}
    assert "python_shared" in matched_patterns
    assert "httpx" in matched_patterns
    assert "client" in matched_patterns
