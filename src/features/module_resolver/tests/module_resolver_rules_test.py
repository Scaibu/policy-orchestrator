from pathlib import Path
from src.features.module_resolver.domain.rules.module_resolver_rules import (
    resolve_module_cascade,
    inspect_search_path_entry,
    PKG_TYPE_REGULAR,
    PKG_TYPE_MODULE,
    PKG_TYPE_NAMESPACE,
    PKG_TYPE_NOT_FOUND,
)


def test_resolve_existing_module():
    repo_root = str(Path(__file__).resolve().parents[4])
    resolved, pkg_type, shadowed, has_collision = resolve_module_cascade(
        module_name="src.features.module_resolver.domain.rules.module_resolver_rules",
        search_paths=[repo_root],
    )
    assert resolved is not None
    assert pkg_type == PKG_TYPE_MODULE
    assert has_collision is False


def test_resolve_package_init():
    repo_root = str(Path(__file__).resolve().parents[4])
    resolved, pkg_type, shadowed, has_collision = resolve_module_cascade(
        module_name="src.features.module_resolver",
        search_paths=[repo_root],
    )
    assert resolved is not None
    assert pkg_type in (PKG_TYPE_NAMESPACE, PKG_TYPE_REGULAR)


def test_resolve_nonexistent():
    resolved, pkg_type, shadowed, has_collision = resolve_module_cascade(
        module_name="non_existent.package.path",
        search_paths=["/tmp"],
    )
    assert resolved is None
    assert pkg_type == PKG_TYPE_NOT_FOUND
