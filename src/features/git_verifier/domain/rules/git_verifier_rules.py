"""
================================================================================
BLUEPRINT: GIT VERIFIER DECISION RULES & MERKLE TREE PARSER
================================================================================
1. OVERVIEW:
   Implements inspection algorithms for Git Merkle tree repositories:
   - HEAD Reference Resolution: Reads .git/HEAD to identify symbolic ref or
     detached commit hash. Handles git submodule gitdir pointers.
   - Submodule Parsing (.gitmodules & gitlink mode 160000): Extracts submodule
     name, path, url, and checks whether child repo commit matches gitlink commit.
   - Clean State Determination: Checks for untracked/modified entries.
================================================================================
"""
import configparser
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def resolve_git_directory(git_entry: Path) -> Optional[Path]:
    if not git_entry.exists():
        return None
    if git_entry.is_file():
        content = git_entry.read_text(encoding="utf-8", errors="ignore").strip()
        if content.startswith("gitdir:"):
            target_rel = content.split(":", 1)[1].strip()
            return (git_entry.parent / target_rel).resolve()
        return None
    return git_entry.resolve()


def read_git_head(git_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    real_git_dir = resolve_git_directory(git_dir) if git_dir.is_file() else git_dir
    if real_git_dir is None:
        return None, None

    head_file = real_git_dir / "HEAD"
    if not head_file.is_file():
        return None, None

    content = head_file.read_text(encoding="utf-8", errors="ignore").strip()
    if content.startswith("ref:"):
        ref_path = content.split(":", 1)[1].strip()
        branch_name = ref_path.split("/")[-1]
        target_ref = real_git_dir / ref_path
        commit_hash = (
            target_ref.read_text(encoding="utf-8", errors="ignore").strip()
            if target_ref.is_file()
            else None
        )
        return branch_name, commit_hash
    else:
        return "DETACHED_HEAD", content


def parse_gitmodules(repo_root: Path) -> List[Dict[str, str]]:
    modules_file = repo_root / ".gitmodules"
    if not modules_file.is_file():
        return []

    config = configparser.ConfigParser()
    try:
        config.read(str(modules_file))
    except Exception:
        return []

    submodules: List[Dict[str, str]] = []
    for section in config.sections():
        if section.startswith('submodule "') and section.endswith('"'):
            name = section[11:-1]
            sub_path = config.get(section, "path", fallback="")
            sub_url = config.get(section, "url", fallback="")
            if sub_path:
                submodules.append({"name": name, "path": sub_path, "url": sub_url})
    return submodules


def resolve_submodule_commit(repo_root: Path, sub_path: str) -> Optional[str]:
    sub_dir = repo_root / sub_path
    if not sub_dir.is_dir():
        return None

    git_entry = sub_dir / ".git"
    if not git_entry.exists():
        return None

    real_dir = resolve_git_directory(git_entry)
    if real_dir:
        _, commit = read_git_head(real_dir)
        return commit

    return None
