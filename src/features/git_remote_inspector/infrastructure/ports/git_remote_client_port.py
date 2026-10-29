from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class GitRemoteClientPort(ABC):
    @abstractmethod
    def fetch_tree(
        self,
        ctx: TraceContext,
        repo: str,
        ref: str,
        recursive: bool = True,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        pass

    @abstractmethod
    def fetch_blob(
        self,
        ctx: TraceContext,
        repo: str,
        blob_sha: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        pass

    @abstractmethod
    def fetch_file_content(
        self,
        ctx: TraceContext,
        repo: str,
        path: str,
        ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        pass

    @abstractmethod
    def fetch_diff(
        self,
        ctx: TraceContext,
        repo: str,
        base_ref: str,
        head_ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        pass

    @abstractmethod
    def fetch_branches(
        self,
        ctx: TraceContext,
        repo: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        pass

