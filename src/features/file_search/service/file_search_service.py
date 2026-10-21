"""
================================================================================
BLUEPRINT: FILE SEARCH DOMAIN SERVICE (HEXAGONAL ORCHESTRATOR)
================================================================================
1. OVERVIEW:
   Coordinates directory traversal algorithms (DFS/BFS) using in-memory queues,
   evaluates pruning rules, applies pattern filters, and records search metrics.
2. ALGORITHMS:
   - DFS (Depth-First Search): Explores deepest directory branches first using
     LIFO stack.
   - BFS (Breadth-First Search): Explores shallow directories layer-by-layer using
     FIFO queue (collections.deque).
   - Inode Caching & Branch Pruning: Skips blacklisted directories (.git, node_modules)
     in O(1) time using hash set lookups.
3. ZERO-INLINE-COMMENT DOCTRINE:
   Execution flow, span management, and algorithm invariants are documented here.
================================================================================
"""
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.file_search.contracts.file_search_types import (
    FileSearchCommand,
    FileSearchResult,
    FileNodeItem,
)
from src.features.file_search.domain.rules.file_search_rules import (
    is_valid_algorithm,
    should_prune_directory,
    matches_pattern,
)
from src.features.file_search.domain.machines.file_search_machine import (
    STATE_PENDING,
    STATE_SCANNING,
    STATE_FILTERED,
    STATE_COMPLETED,
    transition_to,
)
from src.features.file_search.infrastructure.ports.file_search_repository_port import FileSearchRepositoryPort


class FileSearchDomainService(AbstractTracedComponent):
    def __init__(self, repository: FileSearchRepositoryPort) -> None:
        super().__init__()
        self._repository = repository

    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def search_files(
        self, ctx: TraceContext, cmd: FileSearchCommand
    ) -> Result[FileSearchResult, Exception]:
        start = time.perf_counter()
        span = None
        if self._tracer:
            span = self._tracer.start_span("file_search.search_files")

        try:
            if not is_valid_algorithm(cmd.algorithm):
                raise ValueError(f"Invalid traversal algorithm: {cmd.algorithm}. Must be DFS or BFS")

            root = Path(cmd.root_path).resolve()
            if not root.exists():
                raise FileNotFoundError(f"Root path does not exist: {cmd.root_path}")

            state = transition_to(STATE_PENDING, STATE_SCANNING)
            exclude_set = set(cmd.exclude_dirs)
            found_items: List[FileNodeItem] = []

            if cmd.algorithm.upper() == "BFS":
                queue = deque([(root, 0)])
                while queue:
                    current_path, depth = queue.popleft()
                    if depth > cmd.max_depth:
                        continue
                    try:
                        with os.scandir(current_path) as it:
                            for entry in it:
                                if entry.is_dir(follow_symlinks=False):
                                    if not should_prune_directory(entry.name, exclude_set):
                                        queue.append((Path(entry.path), depth + 1))
                                else:
                                    if matches_pattern(entry.name, cmd.pattern):
                                        stat = entry.stat(follow_symlinks=False)
                                        found_items.append(
                                            FileNodeItem(
                                                path=entry.path,
                                                name=entry.name,
                                                size_bytes=stat.st_size,
                                                is_dir=False,
                                                depth=depth + 1,
                                            )
                                        )
                    except PermissionError:
                        continue
            else:
                stack = [(root, 0)]
                while stack:
                    current_path, depth = stack.pop()
                    if depth > cmd.max_depth:
                        continue
                    try:
                        with os.scandir(current_path) as it:
                            for entry in it:
                                if entry.is_dir(follow_symlinks=False):
                                    if not should_prune_directory(entry.name, exclude_set):
                                        stack.append((Path(entry.path), depth + 1))
                                else:
                                    if matches_pattern(entry.name, cmd.pattern):
                                        stat = entry.stat(follow_symlinks=False)
                                        found_items.append(
                                            FileNodeItem(
                                                path=entry.path,
                                                name=entry.name,
                                                size_bytes=stat.st_size,
                                                is_dir=False,
                                                depth=depth + 1,
                                            )
                                        )
                    except PermissionError:
                        continue

            state = transition_to(state, STATE_FILTERED)
            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start) * 1000

            record_id = f"search_{uuid.uuid4().hex[:12]}"
            self._repository.record_search(
                record_id=record_id,
                root_path=str(root),
                total_found=len(found_items),
                duration_ms=duration_ms,
            )

            result = FileSearchResult(
                root_path=str(root),
                total_found=len(found_items),
                files=found_items,
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
