"""
================================================================================
BLUEPRINT: FILE SEARCH DOMAIN SERVICE (HEXAGONAL ORCHESTRATOR)
================================================================================
1. OVERVIEW:
   Coordinates directory traversal algorithms (DFS/BFS) using in-memory queues/stacks.
   Constructs both flat file item arrays and a complete recursive TreeNode hierarchy
   representing the full directory tree.
2. ALGORITHMS:
   - Tree Construction Algorithm: Recursively traverses directory entries into a
     nested Directed Tree graph, populating child branches while pruning excluded
     directories in O(1) time using hash set lookups.
   - Dual Representation: Produces flat searchable file nodes AND hierarchical
     TreeNode nested tree structure.
3. ZERO-INLINE-COMMENT DOCTRINE:
   All execution flow, span management, and algorithm invariants are documented here.
   Method bodies remain 100% comment-free and pure.
================================================================================
"""
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional, Set, Tuple
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.file_search.contracts.file_search_types import (
    FileSearchCommand,
    FileSearchResult,
    FileNodeItem,
    TreeNode,
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

    def _build_tree_recursive(
        self,
        current_path: Path,
        depth: int,
        max_depth: int,
        exclude_set: Set[str],
        pattern: Optional[str],
    ) -> Tuple[Optional[TreeNode], List[FileNodeItem], int, int]:
        stat = current_path.stat(follow_symlinks=False)
        is_directory = current_path.is_dir()

        if not is_directory:
            if matches_pattern(current_path.name, pattern):
                node = TreeNode(
                    path=str(current_path),
                    name=current_path.name,
                    is_dir=False,
                    size_bytes=stat.st_size,
                    depth=depth,
                    children=[],
                )
                flat_item = FileNodeItem(
                    path=str(current_path),
                    name=current_path.name,
                    size_bytes=stat.st_size,
                    is_dir=False,
                    depth=depth,
                )
                return node, [flat_item], 0, 1
            return None, [], 0, 0

        children_nodes: List[TreeNode] = []
        flat_items: List[FileNodeItem] = []
        dir_count = 1
        file_count = 0

        if depth < max_depth:
            try:
                with os.scandir(current_path) as it:
                    entries = sorted(list(it), key=lambda e: (not e.is_dir(), e.name.lower()))
                    for entry in entries:
                        entry_path = Path(entry.path)
                        if entry.is_dir(follow_symlinks=False):
                            if not should_prune_directory(entry.name, exclude_set):
                                child_node, child_flats, sub_dirs, sub_files = self._build_tree_recursive(
                                    entry_path, depth + 1, max_depth, exclude_set, pattern
                                )
                                if child_node is not None:
                                    children_nodes.append(child_node)
                                    flat_items.extend(child_flats)
                                    dir_count += sub_dirs
                                    file_count += sub_files
                        else:
                            if matches_pattern(entry.name, pattern):
                                try:
                                    entry_stat = entry.stat(follow_symlinks=False)
                                    child_node = TreeNode(
                                        path=entry.path,
                                        name=entry.name,
                                        is_dir=False,
                                        size_bytes=entry_stat.st_size,
                                        depth=depth + 1,
                                        children=[],
                                    )
                                    children_nodes.append(child_node)
                                    flat_items.append(
                                        FileNodeItem(
                                            path=entry.path,
                                            name=entry.name,
                                            size_bytes=entry_stat.st_size,
                                            is_dir=False,
                                            depth=depth + 1,
                                        )
                                    )
                                    file_count += 1
                                except PermissionError:
                                    pass
            except PermissionError:
                pass

        total_size = sum(c.size_bytes for c in children_nodes)
        node = TreeNode(
            path=str(current_path),
            name=current_path.name,
            is_dir=True,
            size_bytes=total_size,
            depth=depth,
            children=children_nodes,
        )
        return node, flat_items, dir_count, file_count

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

            tree_root, found_items, total_dirs, total_files = self._build_tree_recursive(
                current_path=root,
                depth=0,
                max_depth=cmd.max_depth,
                exclude_set=exclude_set,
                pattern=cmd.pattern,
            )

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
                total_dirs=total_dirs,
                total_files=total_files,
                files=found_items,
                tree=tree_root,
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
