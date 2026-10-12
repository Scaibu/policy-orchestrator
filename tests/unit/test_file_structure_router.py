"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: FILE STRUCTURE REST ROUTER INTEGRATION TESTS
================================================================================
"""

import unittest
from fastapi.testclient import TestClient
from src.api.rest.app import app

class TestFileStructureRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_scaffold_feature_endpoint(self) -> None:
        payload = {
            "featureName": "billing_v2",
            "baseDir": "/tmp/test_routes_src",
            "withRouter": True,
            "withHandler": True,
            "portType": "local",
        }
        res = self.client.post("/api/v1/file-structure/scaffold/feature", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["feature"], "billing_v2")

    def test_get_feature_map_endpoint(self) -> None:
        res = self.client.get("/api/v1/file-structure/feature/billing_v2")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["feature"], "billing_v2")
        self.assertIn("context", data["data"])
        self.assertIn("types", data["data"])
        self.assertIn("schema", data["data"])
        self.assertIn("router", data["data"])
        self.assertIn("handler", data["data"])
        self.assertIn("service", data["data"])

    def test_impact_analysis_endpoint(self) -> None:
        payload = {
            "targetId": "billing_v2",
            "direction": "UPSTREAM",
            "maxDepth": 4,
        }
        res = self.client.post("/api/v1/file-structure/impact-analysis", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["target_id"], "billing_v2")

    def test_scaffold_package_endpoint(self) -> None:
        payload = {
            "packageName": "test_pkg_delivery",
            "baseDir": "/tmp/test_pkg_dest",
        }
        res = self.client.post("/api/v1/file-structure/scaffold/package", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["package"], "test_pkg_delivery")

    def test_link_files_endpoint(self) -> None:
        payload = {
            "source": "src/features/billing/service/billing_service.py",
            "relType": "IMPORTS_SHARED",
            "target": "src/shared/utils/formatters.py",
        }
        res = self.client.post("/api/v1/file-structure/link-files", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["data"]["linked"])

    def test_create_file_endpoint(self) -> None:
        payload = {
            "filePath": "/tmp/test_custom_rules.py",
            "role": "RuleSet",
            "relType": "ENFORCES_RULES",
            "targetFileOrNode": "billing_v2",
            "direction": "incoming",
        }
        res = self.client.post("/api/v1/file-structure/create-file", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["role"], "RuleSet")

    def test_template_crud_endpoints(self) -> None:
        create_payload = {
            "templateType": "FEATURE",
            "version": "v99_test",
            "directories": ["src/custom", "tests/custom"],
            "files": ["src/custom/app.py", "tests/custom/test_app.py"],
            "metadata": {"test": True},
        }
        res_create = self.client.post("/api/v1/file-structure/scaffold/templates", json=create_payload)
        self.assertEqual(res_create.status_code, 201)
        data_create = res_create.json()
        self.assertTrue(data_create["success"])
        tpl_id = data_create["data"]["id"]

        res_list = self.client.get("/api/v1/file-structure/scaffold/templates")
        self.assertEqual(res_list.status_code, 200)
        self.assertTrue(res_list.json()["success"])

        res_get = self.client.get("/api/v1/file-structure/scaffold/templates/FEATURE?version=v99_test")
        self.assertEqual(res_get.status_code, 200)
        self.assertTrue(res_get.json()["success"])

        update_payload = {
            "directories": ["src/custom_updated"],
            "files": ["src/custom_updated/app.py"],
            "metadata": {"updated": True},
        }
        res_update = self.client.put(f"/api/v1/file-structure/scaffold/templates/{tpl_id}", json=update_payload)
        self.assertEqual(res_update.status_code, 200)
        self.assertTrue(res_update.json()["success"])

        res_delete = self.client.delete(f"/api/v1/file-structure/scaffold/templates/{tpl_id}")
        self.assertEqual(res_delete.status_code, 200)
        self.assertTrue(res_delete.json()["success"])

if __name__ == "__main__":
    unittest.main()

