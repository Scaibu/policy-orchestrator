from pathlib import Path
from src.features.git_verifier.domain.rules.git_verifier_rules import (
    read_git_head,
    parse_gitmodules,
    resolve_submodule_commit,
)


def test_read_git_head():
    repo_root = Path(__file__).resolve().parents[4]
    git_dir = repo_root / ".git"
    if git_dir.exists():
        branch, commit = read_git_head(git_dir)
        assert branch is not None


def test_parse_gitmodules():
    repo_root = Path(__file__).resolve().parents[4]
    submodules = parse_gitmodules(repo_root)
    assert isinstance(submodules, list)
