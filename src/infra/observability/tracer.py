import functools
import inspect
from typing import Any, Callable

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

def get_tracer(instrumenting_module_name: str = "policy_orchestrator"):
    if _HAS_OTEL:
        return trace.get_tracer(instrumenting_module_name)
    return None

def with_span(span_name: str):
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if _HAS_OTEL:
                    tracer = trace.get_tracer("policy_orchestrator")
                    with tracer.start_as_current_span(span_name) as span:
                        try:
                            return await func(*args, **kwargs)
                        except Exception as exc:
                            span.record_exception(exc)
                            span.set_status(Status(StatusCode.ERROR, str(exc)))
                            raise
                else:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if _HAS_OTEL:
                    tracer = trace.get_tracer("policy_orchestrator")
                    with tracer.start_as_current_span(span_name) as span:
                        try:
                            return func(*args, **kwargs)
                        except Exception as exc:
                            span.record_exception(exc)
                            span.set_status(Status(StatusCode.ERROR, str(exc)))
                            raise
                else:
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator
