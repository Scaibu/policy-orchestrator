import unittest
from src.features.git_submodule_manager.domain.rules.git_submodule_manager_rules import (
    sanitize_submodule_path,
    parse_gitmodules_content,
    serialize_gitmodules_content,
    upsert_submodule_in_gitmodules,
    build_gitlink_tree_nodes,
    is_valid_sha,
    calculate_commit_drift,
)
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    InvalidSubmoduleSpecifierError,
)


class TestGitSubmoduleRules(unittest.TestCase):
    def test_sanitize_submodule_path(self):
        self.assertEqual(sanitize_submodule_path("src/plugins/"), "src/plugins")
        self.assertEqual(sanitize_submodule_path("/libs/core"), "libs/core")
        self.assertEqual(sanitize_submodule_path("a/b/../c"), "a/c")
        with self.assertRaises(InvalidSubmoduleSpecifierError):
            sanitize_submodule_path("")

    def test_parse_and_serialize_gitmodules(self):
        content = (
            '[submodule "vendor/lib"]\n'
            "\tpath = vendor/lib\n"
            "\turl = https://github.com/org/lib.git\n"
            "\tbranch = main\n"
        )
        parsed = parse_gitmodules_content(content)
        self.assertIn("vendor/lib", parsed)
        self.assertEqual(parsed["vendor/lib"]["url"], "https://github.com/org/lib.git")

        updated = upsert_submodule_in_gitmodules(
            content, "plugins/auth", "https://github.com/org/auth.git", "develop"
        )
        re_parsed = parse_gitmodules_content(updated)
        self.assertIn("plugins/auth", re_parsed)
        self.assertIn("vendor/lib", re_parsed)
        self.assertEqual(re_parsed["plugins/auth"]["branch"], "develop")

    def test_build_gitlink_tree_nodes(self):
        shas = {"vendor/lib": "a" * 40}
        nodes = build_gitlink_tree_nodes(shas, gitmodules_blob_sha="b" * 40)
        self.assertEqual(len(nodes), 2)
        sub_node = next(n for n in nodes if n["path"] == "vendor/lib")
        self.assertEqual(sub_node["mode"], "160000")
        self.assertEqual(sub_node["type"], "commit")
        self.assertEqual(sub_node["sha"], "a" * 40)

        mod_node = next(n for n in nodes if n["path"] == ".gitmodules")
        self.assertEqual(mod_node["mode"], "100644")
        self.assertEqual(mod_node["type"], "blob")

    def test_sha_and_drift_calculation(self):
        self.assertTrue(is_valid_sha("a" * 40))
        self.assertFalse(is_valid_sha("invalid"))
        in_sync, drift = calculate_commit_drift("a" * 40, "a" * 40)
        self.assertTrue(in_sync)
        self.assertEqual(drift, 0)

        in_sync2, drift2 = calculate_commit_drift("a" * 40, "b" * 40)
        self.assertFalse(in_sync2)
        self.assertEqual(drift2, 1)


if __name__ == "__main__":
    unittest.main()
