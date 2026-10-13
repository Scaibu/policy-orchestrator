TEMPLATE_TYPE_FEATURE = "FEATURE"
TEMPLATE_TYPE_PACKAGE = "PACKAGE"
DEFAULT_TEMPLATE_VERSION = "v1"

STATUS_ACTIVE = "active"
STATUS_SCAFFOLDING = "SCAFFOLDING"
STATUS_INDEXED = "INDEXED"
STATUS_ERROR = "ERROR"
STATUS_COMPLETED = "COMPLETED"
STATUS_SUCCESS = "SUCCESS"

PATH_SRC = "src"
PATH_FEATURES = "features"
PATH_API = "api"
PATH_REST = "rest"
PATH_V1 = "v1"
PATH_ROUTERS = "routers"
PATH_HANDLERS = "handlers"
DEFAULT_FEATURES_DIR = "src/features"
DEFAULT_ROUTERS_DIR = "src/api/rest/v1/routers"
DEFAULT_HANDLERS_DIR = "src/api/rest/v1/handlers"

REL_CALLS = "CALLS"
REL_DECLARES_STATE = "DECLARES_STATE"
REL_CONTROLS_STATE = "CONTROLS_STATE"
REL_DEFINES_MODEL = "DEFINES_MODEL"
REL_RENDERS_STATE = "RENDERS_STATE"
REL_DISPATCHES_ACTION = "DISPATCHES_ACTION"
REL_IMPLEMENTS_PORT = "IMPLEMENTS_PORT"
REL_CONSUMES_PORT = "CONSUMES_PORT"
REL_EXECUTES_QUERY = "EXECUTES_QUERY"
REL_EVALUATES_RULES = "EVALUATES_RULES"
REL_TRANSITIONS_MACHINE = "TRANSITIONS_MACHINE"
REL_RUNS_WORKFLOW = "RUNS_WORKFLOW"
REL_DEFINES_SCHEMA = "DEFINES_SCHEMA"
REL_CONTAINS_FEATURE = "CONTAINS_FEATURE"
REL_DEPENDS_ON = "DEPENDS_ON"
REL_RESOLVES_CONFIG = "RESOLVES_CONFIG"
REL_CONFORMS_TO = "CONFORMS_TO"
REL_MAPS_CONTRACT = "MAPS_CONTRACT"
REL_GENERATED_FROM = "GENERATED_FROM"
REL_GOVERNS = "GOVERNS"
REL_OPTIMIZES = "OPTIMIZES"
REL_CHECKSUMS = "CHECKSUMS"
REL_STREAMS_CHANGES_FROM = "STREAMS_CHANGES_FROM"
REL_PUBLISHES_TO = "PUBLISHES_TO"
REL_BINDS_TO = "BINDS_TO"
REL_HANDLES_FAILURES_FROM = "HANDLES_FAILURES_FROM"
REL_VALIDATES_AGAINST = "VALIDATES_AGAINST"
REL_ENFORCES = "ENFORCES"
REL_ROUTES_FAILURE_TO = "ROUTES_FAILURE_TO"
REL_DISPATCHES_TO = "DISPATCHES_TO"
REL_DEPLOYS = "DEPLOYS"
REL_PACKAGES = "PACKAGES"
REL_EXPORTS = "EXPORTS"
REL_VERIFIES = "VERIFIES"

ROLE_ROUTER = "Router"
ROLE_HANDLER = "Handler"
ROLE_SCHEMA = "Schema"
ROLE_TYPES = "Types"
ROLE_CONTEXT = "Context"
ROLE_SERVICE = "Service"
ROLE_RULES = "Rules"
ROLE_MACHINE = "Machine"
ROLE_WORKFLOW = "Workflow"
ROLE_REPOSITORY_PORT = "RepositoryPort"
ROLE_REPOSITORY_ADAPTER = "RepositoryAdapter"
ROLE_QUERIES = "Queries"
ROLE_INDEX = "Index"
ROLE_TEST = "Test"
ROLE_FEATURE_STATE = "FeatureState"
ROLE_FEATURE_CONTROLLER = "FeatureController"
ROLE_DOMAIN_MODEL = "DomainModel"
ROLE_FEATURE_VIEW = "FeatureView"
ROLE_CONTRACT_SPEC = "ContractSpec"
ROLE_PACKAGE_ROOT = "PackageRoot"
ROLE_FEATURE = "Feature"

LAYER_INGRESS = "ingress"
LAYER_CONTRACT = "contract"
LAYER_DOMAIN_MODEL = "domain_model"
LAYER_DOMAIN_CORE = "domain_core"
LAYER_BUSINESS_LOGIC = "business_logic"
LAYER_STATE_MACHINE = "state_machine"
LAYER_AUTOMATION_DAG = "automation_dag"
LAYER_PERSISTENCE_PORT = "persistence_port"
LAYER_PERSISTENCE_ADAPTER = "persistence_adapter"
LAYER_PERSISTENCE_SQL = "persistence_sql"
LAYER_FACADE = "facade"
LAYER_VERIFICATION = "verification"
LAYER_CLIENT_STATE = "client_state"
LAYER_CLIENT_CONTROLLER = "client_controller"
LAYER_CLIENT_MODEL = "client_model"
LAYER_CLIENT_UI = "client_ui"
LAYER_GENERAL = "general"

DIRECTION_OUTGOING = "outgoing"
DIRECTION_INCOMING = "incoming"

GRAPH_LABEL_FILE_NODE = "FileNode"
GRAPH_LABEL_FEATURE = "Feature"
GRAPH_LABEL_PACKAGE_ROOT = "PackageRoot"

CANONICAL_FEATURE_SUBDIRECTORIES = (
    "schema",
    "types",
    "service",
    "rules",
    "machines",
    "workflows",
    "repository",
    "queries",
    "tests",
)

CANONICAL_PACKAGE_SUBDIRECTORIES = (
    "contracts",
    "config",
    "database",
    "messaging",
    "deploy",
    "src/shared",
    "src/infra",
    "src/features",
    "src/api",
    "scripts",
)

CLI_CMD_SCAFFOLD_TREE = "scaffold-tree"
CLI_CMD_ANALYZE_IMPACT = "analyze-impact"
CLI_CMD_ANALYZE_LINEAGE = "analyze-lineage"

STORE_MEMORY = "memory"
STORE_NEO4J = "neo4j"

DIRECTION_UPSTREAM = "upstream"
DIRECTION_DOWNSTREAM = "downstream"

DEFAULT_TRAVERSAL_DEPTH = 5

VAR_PACKAGE_NAME = "PACKAGE_NAME"
VAR_FEATURE_NAME = "FEATURE_NAME"

TRACE_ID_CLI_SCAFFOLD_TREE = "cli-scaffold-tree"
REQ_ID_CLI_SCAFFOLD_TREE = "cli-req-scaffold-tree"
CORR_ID_CLI_SCAFFOLD_TREE = "cli-corr-scaffold-tree"

TRACE_ID_CLI_ANALYZE_IMPACT = "cli-analyze-impact"
REQ_ID_CLI_ANALYZE_IMPACT = "cli-req-analyze-impact"
CORR_ID_CLI_ANALYZE_IMPACT = "cli-corr-analyze-impact"

TRACE_ID_CLI_ANALYZE_LINEAGE = "cli-analyze-lineage"
REQ_ID_CLI_ANALYZE_LINEAGE = "cli-req-analyze-lineage"
CORR_ID_CLI_ANALYZE_LINEAGE = "cli-corr-analyze-lineage"

CFG_KEY_URI = "uri"
CFG_KEY_USER = "user"
CFG_KEY_PASSWORD = "password"
CFG_KEY_DATABASE = "database"
DEFAULT_GRAPH_DATABASE = "neo4j"

ENCODING_UTF8 = "utf-8"

KEY_NODES = "nodes"
KEY_RELATIONSHIPS = "relationships"
KEY_PATH = "path"
KEY_TARGET = "target"
KEY_SOURCE = "source"
KEY_PACKAGE_NAME = "package_name"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

CLI_PROG_FOLDER_SCAFFOLD = "folder-scaffold-cli"
CLI_DESC_FOLDER_SCAFFOLD = "Declarative Folder Scaffold & Graph Analysis CLI"

CLI_HELP_SCAFFOLD_TREE = "Declaratively scaffold a tree of folders, files, and relationships from a JSON manifest"
CLI_HELP_ANALYZE_IMPACT = "Analyze blast radius / impacted nodes for a target file or node in the knowledge graph"
CLI_HELP_ANALYZE_LINEAGE = "Trace architectural dependency lineage for a given file in the knowledge graph"

CLI_ERR_MANIFEST_REQUIRED = "Error: --manifest <path.json> is required for scaffold-tree."
CLI_ERR_MANIFEST_NOT_FOUND = "Error: Manifest file '{path}' does not exist."
CLI_ERR_SCAFFOLD_TREE_FAILED = "Error scaffolding tree: {error}"
CLI_ERR_TARGET_REQUIRED = "Error: --target <node_id> is required for impact analysis."
CLI_ERR_ANALYZE_IMPACT_FAILED = "Error analyzing impact: {error}"
CLI_ERR_FILE_REQUIRED = "Error: --file <file_path> is required for file lineage tracing."
CLI_ERR_ANALYZE_LINEAGE_FAILED = "Error tracing lineage: {error}"

HEADER_TREE_SCAFFOLDED = "TREE SCAFFOLDED & KNOWLEDGE GRAPH SYNCHRONIZED"
HEADER_BLAST_RADIUS = "BLAST RADIUS & GRAPH IMPACT ANALYSIS"
HEADER_FILE_LINEAGE = "FILE KNOWLEDGE GRAPH LINEAGE"

CLI_CMD_SCAFFOLD_PACKAGE = "scaffold-package"
CLI_CMD_SCAFFOLD_FEATURE = "scaffold-feature"

CLI_HELP_SCAFFOLD_PACKAGE = "Scaffold complete polyglot sub-package structure per 0-package-folder-writing-rule.md"
CLI_HELP_SCAFFOLD_FEATURE = "Scaffold hexagonal backend or frontend feature in a package per 1-feature-folder-writing-rule.md"

CLI_ERR_PACKAGE_NAME_REQUIRED = "Error: --name <package_name> is required for scaffold-package."
CLI_ERR_FEATURE_NAME_REQUIRED = "Error: --name <feature_name> and --package <package_name> are required for scaffold-feature."

TEMPLATE_FILE_PACKAGE_STRUCTURE = "0-rule-package-structure.template.json"
TEMPLATE_FILE_BACKEND_FEATURE = "1-rule-backend-feature-structure.template.json"
TEMPLATE_FILE_FRONTEND_FEATURE = "1-rule-frontend-feature-structure.template.json"

FEATURE_TYPE_BACKEND = "backend"
FEATURE_TYPE_FRONTEND = "frontend"
