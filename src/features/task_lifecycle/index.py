"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: TASK LIFECYCLE PUBLIC FACADE
================================================================================
Public facade interface exporting the domain orchestrator service, commands,
and Result DTOs for external package and delivery callers.

ZERO-INLINE-COMMENT DOCTRINE: Method bodies remain 100% comment-free and pure.
================================================================================
"""

from src.features.task_lifecycle.contracts.task_lifecycle_types import (
    StartTaskCommand,
    StartTaskResult,
    FinishTaskCommand,
    FinishTaskResult,
    SyncPointerCommand,
    SyncPointerResult,
)
from src.features.task_lifecycle.service.task_lifecycle_service import (
    TaskLifecycleDomainService,
)
from src.features.task_lifecycle.infrastructure.adapters.task_lifecycle_adapter import (
    SubprocessGitTaskAdapter,
)

__all__ = [
    "StartTaskCommand",
    "StartTaskResult",
    "FinishTaskCommand",
    "FinishTaskResult",
    "SyncPointerCommand",
    "SyncPointerResult",
    "TaskLifecycleDomainService",
    "SubprocessGitTaskAdapter",
]

