from typing import Final, Set, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_TRAVERSAL_DEPTH: Final[int] = 6

WEIGHT_CONTRACT_NODE: Final[float] = 3.0
WEIGHT_HANDLER_NODE: Final[float] = 2.0
WEIGHT_DB_MIGRATION_NODE: Final[float] = 2.0
WEIGHT_SERVICE_NODE: Final[float] = 1.5
WEIGHT_PORT_NODE: Final[float] = 1.5
WEIGHT_DEFAULT_NODE: Final[float] = 1.0

RISK_SCORE_THRESHOLD_HIGH: Final[float] = 4.0
RISK_SCORE_THRESHOLD_MEDIUM: Final[float] = 2.0

RISK_LEVEL_LEVEL_0: Final[str] = "Level 0: Isolated Business Logic Scope"
RISK_LEVEL_LEVEL_1: Final[str] = "Level 1: Local Feature Scope"
RISK_LEVEL_LEVEL_2: Final[str] = "Level 2: Storage & Persistence Migration Risk"
RISK_LEVEL_LEVEL_3: Final[str] = "Level 3: Public Contract Boundary Risk"

CRITICAL_STORAGE_LABELS: Final[Set[str]] = {"DatabaseMigration", "NamedQuery"}
HIGH_RISK_INGRESS_LABELS: Final[Set[str]] = {"Contract", "IngressHandler", "IngressRouter"}
ISOLATED_LOGIC_LABELS: Final[Set[str]] = {"RuleSet", "StateMachine", "WorkflowEngine"}
LOCAL_FEATURE_LABELS: Final[Set[str]] = {"SchemaACL", "RepositoryPort", "DomainModel", "RepositoryAdapter"}

@dataclass(frozen=True)
class FeatureGraphNodeIdsModel:
    context_id: str
    index_id: str
    types_id: str
    acl_id: str
    query_id: str
    port_id: str
    adapter_id: str
    rules_id: str
    machine_id: str
    workflow_id: str
    service_id: str
    contract_openapi_id: str
    contract_graphql_id: str
    contract_grpc_id: str
    contract_asyncapi_id: str
    router_id: str
    handler_id: str
    route_rules_id: str
    resolver_id: str
    grpc_handler_id: str
    consumer_id: str
    migration_id: str
    schema_lock_id: str
    db_pool_id: str
    producer_id: str
    topic_id: str
    topics_lock_id: str
    dlq_id: str
    shared_lib_id: str
    config_schema_id: str
    tracer_id: str
    test_suite_id: str

@dataclass(frozen=True)
class FeatureGraphContextModel:
    feature_name: str
    package_name: str
    package_root: str
    package_prefix: str
    status: str
    owner: str
    contract: str
    contract_version: str
    flags: str
    migrations: str
    node_ids: FeatureGraphNodeIdsModel

    def to_properties_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "package_name": self.package_name,
            "status": self.status,
            "owner": self.owner,
            "contract": self.contract,
            "contract_version": self.contract_version,
            "flags": self.flags,
            "migrations": self.migrations,
        }

@dataclass(frozen=True)
class FeatureGraphNodeSpec:
    id: str
    label: str
    file_path: str
    layer: str

@dataclass(frozen=True)
class ScaffoldTargetPathsModel:
    target_dir: Path
    router_dir: Path
    handler_dir: Path

@dataclass(frozen=True)
class DeliveryArtifactSpec:
    artifact_type: str
    target_dir: Path
    file_name: str
    enabled: bool
