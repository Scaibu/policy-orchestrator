from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class GitRemoteInspectorRepositoryPort(ABC):
    @abstractmethod
    def save_inspection_record(
        self,
        ctx: TraceContext,
        record: Dict[str, Any],
    ) -> Result[bool, Exception]:
        pass

    @abstractmethod
    def get_inspection_record(
        self,
        ctx: TraceContext,
        record_id: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        pass
