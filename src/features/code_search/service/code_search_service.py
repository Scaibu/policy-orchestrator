"""
================================================================================
BLUEPRINT: CODE SEARCH DOMAIN SERVICE (HEXAGONAL ORCHESTRATOR)
================================================================================
1. OVERVIEW:
   Coordinates text search execution using Boyer-Moore-Horspool or Aho-Corasick.
   Processes target files, computes line numbers and character offsets, and
   persists telemetry under OpenTelemetry trace context.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Workflow logic and metrics tracking are captured in this top header block.
================================================================================
"""
import time
import uuid
from pathlib import Path
from typing import List, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.code_search.contracts.code_search_types import (
    CodeSearchCommand,
    CodeSearchResult,
    CodeSearchMatch,
)
from src.features.code_search.domain.rules.code_search_rules import (
    boyer_moore_horspool_search,
    AhoCorasickAutomaton,
)
from src.features.code_search.domain.machines.code_search_machine import (
    STATE_PENDING,
    STATE_COMPILING,
    STATE_SCANNING,
    STATE_COMPLETED,
    transition_to,
)
from src.features.code_search.infrastructure.ports.code_search_repository_port import CodeSearchRepositoryPort


class CodeSearchDomainService(AbstractTracedComponent):
    def __init__(self, repository: CodeSearchRepositoryPort) -> None:
        super().__init__()
        self._repository = repository

    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def search_code(
        self, ctx: TraceContext, cmd: CodeSearchCommand
    ) -> Result[CodeSearchResult, Exception]:
        start = time.perf_counter()
        span = None
        if self._tracer:
            span = self._tracer.start_span("code_search.search_code")

        try:
            state = transition_to(STATE_PENDING, STATE_COMPILING)
            patterns = cmd.patterns if cmd.case_sensitive else [p.lower() for p in cmd.patterns]
            automaton = AhoCorasickAutomaton(patterns) if cmd.algorithm.upper() == "AHO_CORASICK" else None

            state = transition_to(state, STATE_SCANNING)
            all_matches: List[CodeSearchMatch] = []
            scanned_count = 0

            for file_str in cmd.target_files:
                file_path = Path(file_str)
                if not file_path.is_file():
                    continue

                scanned_count += 1
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                lines = content.splitlines(keepends=True)
                file_matches = 0
                current_offset = 0

                for line_idx, line in enumerate(lines, start=1):
                    search_line = line if cmd.case_sensitive else line.lower()
                    if automaton is not None:
                        found = automaton.search(search_line)
                        for col_offset, matched_pat in found:
                            all_matches.append(
                                CodeSearchMatch(
                                    file_path=str(file_path),
                                    line_number=line_idx,
                                    char_offset=current_offset + col_offset,
                                    pattern=matched_pat,
                                    line_content=line.rstrip("\r\n"),
                                )
                            )
                            file_matches += 1
                            if file_matches >= cmd.max_matches_per_file:
                                break
                    else:
                        for pat in patterns:
                            offsets = boyer_moore_horspool_search(search_line, pat)
                            for col_offset in offsets:
                                all_matches.append(
                                    CodeSearchMatch(
                                        file_path=str(file_path),
                                        line_number=line_idx,
                                        char_offset=current_offset + col_offset,
                                        pattern=pat,
                                        line_content=line.rstrip("\r\n"),
                                    )
                                )
                                file_matches += 1
                                if file_matches >= cmd.max_matches_per_file:
                                    break
                            if file_matches >= cmd.max_matches_per_file:
                                break

                    current_offset += len(line)

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start) * 1000

            record_id = f"codesearch_{uuid.uuid4().hex[:12]}"
            self._repository.record_search(
                record_id=record_id,
                pattern_count=len(cmd.patterns),
                matches_found=len(all_matches),
                duration_ms=duration_ms,
            )

            result = CodeSearchResult(
                total_matches=len(all_matches),
                scanned_files_count=scanned_count,
                matches=all_matches,
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
