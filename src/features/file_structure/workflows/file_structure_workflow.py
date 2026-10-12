
from typing import List
from dataclasses import dataclass

@dataclass(frozen=True)
class WorkflowStepDefinition:
    step_id: str
    name: str
    action: str
    timeout_seconds: int
    retry_count: int

FILE_STRUCTURE_SCAFFOLD_STEPS: List[WorkflowStepDefinition] = [
    WorkflowStepDefinition(
        step_id="step_validate_naming",
        name="Validate Feature / Package Naming Conventions",
        action="validate_naming",
        timeout_seconds=5,
        retry_count=0,
    ),
    WorkflowStepDefinition(
        step_id="step_create_directories",
        name="Create Mandatory Subdirectories and .gitkeep files",
        action="create_directories",
        timeout_seconds=10,
        retry_count=1,
    ),
    WorkflowStepDefinition(
        step_id="step_create_canonical_files",
        name="Create 10 Canonical Feature Files",
        action="create_files",
        timeout_seconds=15,
        retry_count=1,
    ),
    WorkflowStepDefinition(
        step_id="step_create_ingress_routes",
        name="Generate Dedicated REST Router and Handler",
        action="generate_ingress",
        timeout_seconds=15,
        retry_count=1,
    ),
    WorkflowStepDefinition(
        step_id="step_sync_knowledge_graph",
        name="Synchronize Graph Nodes and Dependency Edges",
        action="sync_graph",
        timeout_seconds=30,
        retry_count=2,
    ),
]

class FileStructureScaffoldWorkflow:
    @staticmethod
    def get_steps() -> List[WorkflowStepDefinition]:
        return FILE_STRUCTURE_SCAFFOLD_STEPS

KnowledgeGraphSyncWorkflow = FileStructureScaffoldWorkflow
