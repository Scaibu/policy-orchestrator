from abc import ABC, abstractmethod
from typing import Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.infra.observability.tracer import get_tracer

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


class AbstractTracedComponent(ABC):
    def __init__(self, component_name: str = "policy_orchestrator") -> None:
        self._tracer = get_tracer(component_name)


    @abstractmethod
    def mapFailure(self, cause: Exception) -> Exception:
        pass
