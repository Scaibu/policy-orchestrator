"""
================================================================================
BLUEPRINT: FOLDER SCAFFOLD KNOWLEDGE GRAPH ANALYZER PORT (HEXAGONAL)
================================================================================
Defines analytical contracts to inspect, traverse, and score architectural file
graphs stored across AlloyDB and Neo4j. Supports blast-radius impact analysis,
bidirectional lineage resolution, and topological traversal.
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.folder_scaffold.types.folder_scaffold_types import (
    BlastRadiusReport,
    FileLineageReport,
)


class FolderScaffoldAnalyzerServicePort(ABC):
    @abstractmethod
    def analyze_impact(
        self,
        ctx: TraceContext,
        target_id: str,
        direction: str = "UPSTREAM",
        max_depth: int = 5,
    ) -> Result[BlastRadiusReport, Exception]:
        pass

    @abstractmethod
    def get_file_lineage(
        self,
        ctx: TraceContext,
        file_path: str,
        max_depth: int = 5,
    ) -> Result[FileLineageReport, Exception]:
        pass
