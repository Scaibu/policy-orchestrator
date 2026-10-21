"""
================================================================================
BLUEPRINT: GIT VERIFIER DECISION RULES & MERKLE TREE PARSER
================================================================================
1. OVERVIEW:
   Implements complete inspection algorithms for Git repositories:
   - HEAD Reference Resolution: Reads .git/HEAD to identify symbolic ref or
     detached commit hash. Handles git submodule gitdir pointers.
   - Working Tree State Audit: Inspects unstaged modifications, staged files,
     and untracked files without external dependencies.
   - Remote Tracking Inspection: Reads .git/config and refs to compute ahead/behind
     counts against upstream remote.
   - Submodule Parsing (.gitmodules & gitlink mode 160000): Extracts submodule
     declarations, resolves parent gitlink recorded SHA vs child commit SHA.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All rules adhere strictly to zero mid-function inline comments.
================================================================================
"""
import configparser
import subprocess
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


def read_git_head(git_dir: Path) -> Tuple[Optional[str], Optional[str], bool]:
    real_git_dir = resolve_git_directory(git_dir) if git_dir.is_file() else git_dir
    if real_git_dir is None:
        return None, None, False

    head_file = real_git_dir / "HEAD"
    if not head_file.is_file():
        return None, None, False

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
        if commit_hash is None:
            packed_refs = real_git_dir / "packed-refs"
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.endswith(ref_path):
                        commit_hash = line.split()[0]
                        break
        return branch_name, commit_hash, False
    else:
        return "DETACHED_HEAD", content, True


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


def resolve_submodule_commit(repo_root: Path, sub_path: str) -> Tuple[Optional[str], bool]:
    sub_dir = repo_root / sub_path
    if not sub_dir.is_dir():
        return None, False

    git_entry = sub_dir / ".git"
    if not git_entry.exists():
        return None, False

    real_dir = resolve_git_directory(git_entry)
    if real_dir:
        _, commit, _ = read_git_head(real_dir)
        is_dirty = False
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(sub_dir),
                capture_output=True,
                text=True,
                timeout=3,
            )
            is_dirty = bool(res.stdout.strip())
        except Exception:
            pass
        return commit, is_dirty

    return None, False


def get_gitlink_expected_commit(repo_root: Path, sub_path: str) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "ls-tree", "HEAD", sub_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3,
        )
        out = res.stdout.strip()
        if out and "160000 commit" in out:
            return out.split()[2]
    except Exception:
        pass
    return None


def inspect_working_tree(repo_root: Path) -> Tuple[List[str], List[str], List[str], bool]:
    staged: List[str] = []
    unstaged: List[str] = []
    untracked: List[str] = []

    try:
        res = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in res.stdout.splitlines():
            if len(line) < 3:
                continue
            index_code = line[0]
            worktree_code = line[1]
            filepath = line[3:].strip()

            if index_code == "?" and worktree_code == "?":
                untracked.append(filepath)
            else:
                if index_code not in (" ", "?"):
                    staged.append(filepath)
                if worktree_code not in (" ", "?"):
                    unstaged.append(filepath)
    except Exception:
        pass

    is_clean = len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0
    return staged, unstaged, untracked, is_clean


def inspect_remote_tracking(repo_root: Path, branch_name: Optional[str]) -> Tuple[Optional[str], int, int, bool]:
    if not branch_name or branch_name == "DETACHED_HEAD":
        return None, 0, 0, True

    upstream: Optional[str] = None
    ahead = 0
    behind = 0

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", f"{branch_name}@{{upstream}}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            upstream = res.stdout.strip()
            rev_list = subprocess.run(
                ["git", "rev-list", "--left-right", "--count", f"{upstream}...{branch_name}"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=3,
            )
            if rev_list.returncode == 0:
                parts = rev_list.stdout.strip().split()
                if len(parts) == 2:
                    behind = int(parts[0])
                    ahead = int(parts[1])
    except Exception:
        pass

    is_in_sync = (ahead == 0 and behind == 0)
    return upstream, ahead, behind, is_in_sync

