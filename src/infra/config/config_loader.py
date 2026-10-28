import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

class AppConfigLoader:
    _cached_config: Optional[Dict[str, Any]] = None
    _cached_endpoints: Optional[Dict[str, Any]] = None
    _cached_flags: Optional[Dict[str, Any]] = None
    _cached_theme: Optional[Dict[str, Any]] = None
    _cached_branding: Optional[Dict[str, Any]] = None

    @classmethod
    def _get_config_dir(cls) -> Path:
        return Path(__file__).resolve().parents[3] / "config"

    @classmethod
    def _read_yaml_file(cls, filename: str) -> Dict[str, Any]:
        file_path = cls._get_config_dir() / filename
        if file_path.exists() and yaml is not None:
            try:
                content = file_path.read_text(encoding="utf-8")
                parsed = yaml.safe_load(content)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}

    @classmethod
    def _read_json_file(cls, filename: str) -> Dict[str, Any]:
        file_path = cls._get_config_dir() / filename
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, val in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                merged[key] = cls._deep_merge(merged[key], val)
            else:
                merged[key] = val
        return merged

    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        if cls._cached_config is not None:
            return cls._cached_config

        data = cls._read_yaml_file("default.yaml")
        app_env = os.environ.get("APP_ENV", os.environ.get("ENVIRONMENT", "development")).lower()
        if app_env in ("production", "prod"):
            prod_overrides = cls._read_yaml_file("production.yaml")
            data = cls._deep_merge(data, prod_overrides)
        elif app_env in ("test", "testing"):
            test_overrides = cls._read_yaml_file("test.yaml")
            data = cls._deep_merge(data, test_overrides)

        db_section = data.setdefault("database", {})
        if "DATABASE_URL" in os.environ:
            db_section["url"] = os.environ["DATABASE_URL"]
        elif "url" not in db_section:
            db_section["url"] = "postgresql://alloy_admin:alloysecret@localhost:5432/policy_orchestration"

        graph_section = data.setdefault("knowledge_graph", {})
        if "NEO4J_URI" in os.environ:
            graph_section["uri"] = os.environ["NEO4J_URI"]
        if "NEO4J_USER" in os.environ:
            graph_section["user"] = os.environ["NEO4J_USER"]
        if "NEO4J_PASSWORD" in os.environ:
            graph_section["password"] = os.environ["NEO4J_PASSWORD"]

        github_section = data.setdefault("github", {})
        if "GITHUB_TOKEN" in os.environ:
            github_section["token"] = os.environ["GITHUB_TOKEN"]
        if "GITHUB_API_URL" in os.environ:
            github_section["api_base_url"] = os.environ["GITHUB_API_URL"]

        cls._cached_config = data
        return data

    @classmethod
    def load_endpoints(cls) -> Dict[str, Any]:
        if cls._cached_endpoints is not None:
            return cls._cached_endpoints
        data = cls._read_yaml_file("endpoints.yaml")
        cls._cached_endpoints = data
        return data

    @classmethod
    def load_feature_flags(cls) -> Dict[str, Any]:
        if cls._cached_flags is not None:
            return cls._cached_flags
        data = cls._read_yaml_file("feature-flags.yaml")
        cls._cached_flags = data
        return data

    @classmethod
    def load_theme(cls) -> Dict[str, Any]:
        if cls._cached_theme is not None:
            return cls._cached_theme
        data = cls._read_json_file("theme.json")
        cls._cached_theme = data
        return data

    @classmethod
    def load_branding(cls) -> Dict[str, Any]:
        if cls._cached_branding is not None:
            return cls._cached_branding
        data = cls._read_yaml_file("branding.yaml")
        cls._cached_branding = data
        return data

    @classmethod
    def get_database_url(cls) -> str:
        cfg = cls.load_config()
        return str(cfg.get("database", {}).get("url", ""))

    @classmethod
    def get_knowledge_graph_config(cls) -> Dict[str, Any]:
        cfg = cls.load_config()
        return dict(cfg.get("knowledge_graph", {}))

    @classmethod
    def get_endpoint(cls, service_name: str, key: str) -> str:
        endpoints_data = cls.load_endpoints()
        endpoints = endpoints_data.get("endpoints", {})
        service_endpoints = endpoints.get(service_name, {})
        return str(service_endpoints.get(key, ""))

    @classmethod
    def get_full_endpoint(cls, service_name: str, key: str) -> str:
        endpoints_data = cls.load_endpoints()
        base_prefix = str(endpoints_data.get("base_prefix", "/api/v1"))
        service_base = cls.get_endpoint(service_name, "base")
        sub_endpoint = cls.get_endpoint(service_name, key)
        if not sub_endpoint:
            return ""
        if sub_endpoint.startswith(base_prefix):
            return sub_endpoint
        if service_base and not sub_endpoint.startswith(service_base):
            return f"{base_prefix}{service_base}{sub_endpoint}"
        return f"{base_prefix}{sub_endpoint}"

    @classmethod
    def is_feature_flag_enabled(cls, flag_name: str) -> bool:
        flags_data = cls.load_feature_flags()
        flags = flags_data.get("flags", {})
        flag = flags.get(flag_name, {})
        if isinstance(flag, dict):
            return bool(flag.get("enabled", False))
        return bool(flag)

    @classmethod
    def get_scaffold_config(cls) -> Dict[str, Any]:
        cfg = cls.load_config()
        return dict(cfg.get("scaffold", {}))

    @classmethod
    def get_github_config(cls) -> Dict[str, Any]:
        cfg = cls.load_config()
        return dict(cfg.get("github", {}))

