"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: TASK LIFECYCLE PORT CONTRACT
================================================================================
Abstract boundary interface decoupling the domain orchestrator from external
operating system processes, Git binaries, and the GitHub CLI (gh).

ZERO-INLINE-COMMENT DOCTRINE: Method bodies remain 100% comment-free and pure.
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class GitTaskExecutionPort(ABC):

    @abstractmethod
    def create_github_issue(self, repo: str, title: str, body: str) -> Tuple[int, str]:
        pass

    @abstractmethod
    def create_and_checkout_branch(self, branch_name: str, base_branch: str) -> None:
        pass

    @abstractmethod
    def run_verification_tests(self, test_cmd: str) -> bool:
        pass

    @abstractmethod
    def create_pull_request(self, repo: str, base: str, head: str, title: str, body: str) -> Tuple[int, str]:
        pass

    @abstractmethod
    def merge_pull_request(self, repo: str, pr_number: int, delete_branch: bool) -> bool:
        pass

    @abstractmethod
    def checkout_and_pull(self, branch: str) -> None:
        pass

    @abstractmethod
    def close_github_issue(self, repo: str, issue_number: int, reason: str) -> None:
        pass

    @abstractmethod
    def sync_parent_submodule(self, submodule_path: str, parent_path: str, commit_msg: str) -> Optional[str]:
        pass

