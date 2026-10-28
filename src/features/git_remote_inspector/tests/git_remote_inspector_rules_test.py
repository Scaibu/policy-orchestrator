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
