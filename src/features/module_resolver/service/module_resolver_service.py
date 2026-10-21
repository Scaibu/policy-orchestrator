"""
================================================================================
BLUEPRINT: MODULE RESOLVER DOMAIN SERVICE (HEXAGONAL ORCHESTRATOR)
================================================================================
1. OVERVIEW:
   Coordinates Python module resolution following PEP 302 and PEP 451.
   Inspects sequential sys.path entries, detects shadowing / collisions, and
   persists telemetry under OpenTelemetry trace context.
================================================================================
"""
import time
import uuid
from typing import Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.module_resolver.contracts.module_resolver_types import (
    ResolveModuleCommand,
    ModuleResolutionResult,
)
from src.features.module_resolver.domain.rules.module_resolver_rules import (
    resolve_module_cascade,
    PKG_TYPE_NOT_FOUND,
)
from src.features.module_resolver.domain.machines.module_resolver_machine import (
    STATE_PENDING,
    STATE_RESOLVING,
    STATE_CHECKING,
    STATE_COMPLETED,
    transition_to,
)
from src.features.module_resolver.infrastructure.ports.module_resolver_repository_port import (
    ModuleResolverRepositoryPort,
)


class ModuleResolverDomainService(AbstractTracedComponent):
    def __init__(self, repository: ModuleResolverRepositoryPort) -> None:
        super().__init__()
        self._repository = repository

    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def resolve_module(
        self, ctx: TraceContext, cmd: ResolveModuleCommand
    ) -> Result[ModuleResolutionResult, Exception]:
        start = time.perf_counter()
        span = None
        if self._tracer:
            span = self._tracer.start_span("module_resolver.resolve_module")

        try:
            state = transition_to(STATE_PENDING, STATE_RESOLVING)
            (
                resolved_path,
                pkg_type,
                shadowed,
                has_collision,
                dist_name,
                version,
                is_builtin,
                is_stdlib,
            ) = resolve_module_cascade(
                module_name=cmd.module_name,
                search_paths=cmd.search_paths,
            )

            state = transition_to(state, STATE_CHECKING)
            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start) * 1000

            is_valid = (resolved_path is not None or is_builtin) and pkg_type != PKG_TYPE_NOT_FOUND

            record_id = f"modres_{uuid.uuid4().hex[:12]}"
            self._repository.record_resolution(
                record_id=record_id,
                module_name=cmd.module_name,
                is_valid=is_valid,
                has_collision=has_collision,
                duration_ms=duration_ms,
            )

            result = ModuleResolutionResult(
                module_name=cmd.module_name,
                is_valid=is_valid,
                resolved_path=resolved_path,
                package_type=pkg_type,
                distribution_name=dist_name,
                version=version,
                is_builtin=is_builtin,
                is_stdlib=is_stdlib,
                shadowed_paths=shadowed,
                has_collision=has_collision,
                duration_ms=round(duration_ms, 2),
                status=state,
            )
            return Result.ok(result)
        except Exception as exc:
            mapped = self.mapFailure(exc)
            return Result.err(mapped)
        finally:
            if span:
                span.end()

