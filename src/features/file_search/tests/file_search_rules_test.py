from src.features.file_search.domain.rules.file_search_rules import (
    is_valid_algorithm,
    should_prune_directory,
    matches_pattern,
)


def test_is_valid_algorithm():
    assert is_valid_algorithm("DFS") is True
    assert is_valid_algorithm("BFS") is True
    assert is_valid_algorithm("dfs") is True
    assert is_valid_algorithm("INVALID") is False


def test_should_prune_directory():
    assert should_prune_directory(".git", set()) is True
    assert should_prune_directory("__pycache__", set()) is True
    assert should_prune_directory("node_modules", set()) is True
    assert should_prune_directory("custom_temp", {"custom_temp"}) is True
    assert should_prune_directory("src", set()) is False


def test_matches_pattern():
    assert matches_pattern("main.py", ".py") is True
    assert matches_pattern("main.py", "*") is True
    assert matches_pattern("main.py", "") is True
    assert matches_pattern("test_file.rs", ".py") is False
