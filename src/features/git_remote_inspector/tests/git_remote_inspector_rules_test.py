from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
    sanitize_file_path,
    matches_glob_pattern,
    filter_tree_nodes,
    extract_diff_patch,
)


def test_sanitize_file_path():
    assert sanitize_file_path("src/index.ts") == "src/index.ts"
    assert sanitize_file_path("/src/index.ts") == "src/index.ts"
    assert sanitize_file_path("src/../index.ts") == "index.ts"
    assert sanitize_file_path("a/b/c/../../d.ts") == "a/d.ts"


def test_matches_glob_pattern():
    assert matches_glob_pattern("src/main.ts", None) is True
    assert matches_glob_pattern("src/main.ts", "*") is True
    assert matches_glob_pattern("src/main.ts", "*.ts") is True
    assert matches_glob_pattern("src/main.ts", "*.py") is False
    assert matches_glob_pattern("src/features/auth/index.ts", "src/features/**") is True


def test_filter_tree_nodes():
    tree = [
        {"path": "src/main.ts", "type": "blob", "sha": "sha1", "size": 100},
        {"path": "src/utils.py", "type": "blob", "sha": "sha2", "size": 200},
        {"path": "src", "type": "tree", "sha": "sha3"},
    ]
    matched = filter_tree_nodes(tree, "*.ts")
    assert len(matched) == 1
    assert matched[0].path == "src/main.ts"
    assert matched[0].sha == "sha1"


def test_extract_diff_patch():
    compare_payload = {
        "files": [
            {
                "filename": "src/app.py",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n-old\n+new",
            }
        ]
    }
    count, patch = extract_diff_patch(compare_payload)
    assert count == 1
    assert "diff --git a/src/app.py" in patch
    assert "+new" in patch


def test_map_network_exception():
    import requests
    from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
        map_network_exception,
    )
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        SSLError,
        ConnectionTimeoutError,
        ReadTimeoutError,
    )
    ssl_err = requests.exceptions.SSLError("bad cert")
    assert isinstance(map_network_exception(ssl_err), SSLError)
    conn_err = requests.exceptions.ConnectionError("failed to connect")
    assert isinstance(map_network_exception(conn_err), ConnectionTimeoutError)
    timeout_err = requests.exceptions.Timeout("timed out")
    assert isinstance(map_network_exception(timeout_err), ReadTimeoutError)


def test_map_http_status_error():
    from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
        map_http_status_error,
    )
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        AuthenticationError,
        RateLimitExceededError,
        RefNotFoundError,
        FileNotFoundError,
        ProviderServerError,
    )
    assert isinstance(map_http_status_error(401, "unauthorized", "GitHub"), AuthenticationError)
    assert isinstance(map_http_status_error(403, "rate limit exceeded", "GitLab"), RateLimitExceededError)
    assert isinstance(map_http_status_error(404, "not found", "Bitbucket", "ref"), RefNotFoundError)
    assert isinstance(map_http_status_error(404, "not found", "GitHub", "file"), FileNotFoundError)
    assert isinstance(map_http_status_error(503, "unavailable", "GitLab"), ProviderServerError)


def test_categorize_contract_nodes():
    from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
        categorize_contract_nodes,
    )
    tree = [
        {"path": "contracts/openapi/v1.yaml", "type": "blob", "sha": "sha-oas", "size": 1000},
        {"path": "proto/service.proto", "type": "blob", "sha": "sha-proto", "size": 500},
        {"path": "src/features/auth/types.ts", "type": "blob", "sha": "sha-types", "size": 300},
        {"path": "src/features/auth/router.ts", "type": "blob", "sha": "sha-router", "size": 400},
        {"path": "src/events/kafka_consumer.go", "type": "blob", "sha": "sha-event", "size": 600},
        {"path": "lib/domain/models.dart", "type": "blob", "sha": "sha-dart", "size": 250},
        {"path": "src/models/user.rs", "type": "blob", "sha": "sha-rust", "size": 350},
        {"path": "migrations/001_init.sql", "type": "blob", "sha": "sha-sql", "size": 800},
        {"path": "README.md", "type": "blob", "sha": "sha-md", "size": 100},
    ]
    lang, contracts = categorize_contract_nodes(tree)
    assert len(contracts) == 8
    assert contracts[0]["category"] == "API_SPECIFICATION"
    assert contracts[0]["priority"] == 1
    assert any(c["category"] == "RPC_GRAPHQL_SCHEMA" for c in contracts)
    assert any(c["path"] == "lib/domain/models.dart" for c in contracts)
    assert any(c["path"] == "src/models/user.rs" for c in contracts)

