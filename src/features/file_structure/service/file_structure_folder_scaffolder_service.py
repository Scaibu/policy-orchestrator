from typing import Dict, Any, List, Optional
from pathlib import Path

class FileStructureFolderScaffolderService:
    def scaffold_feature_tree(
        self,
        feature_name: str,
        base_dir: str = "src/features",
        package_root: Optional[str] = None,
        with_router: bool = True,
        with_handler: bool = True,
        with_route_rules: bool = False,
        port_type: str = "local",
    ) -> Dict[str, Any]:
        feature_slug = feature_name.lower().replace("-", "_")

        if package_root:
            pkg_path = Path(package_root)
            target_dir = pkg_path / "src" / "features" / feature_slug
            router_dir = pkg_path / "src" / "api" / "rest" / "v1" / "routers"
            handler_dir = pkg_path / "src" / "api" / "rest" / "v1" / "handlers"
        else:
            target_dir = Path(base_dir) / feature_slug
            base_str = str(base_dir)
            if base_str.startswith("/tmp") or ("src/features" not in base_str and base_str != "."):
                router_dir = Path(base_dir) / "api" / "rest" / "v1" / "routers"
                handler_dir = Path(base_dir) / "api" / "rest" / "v1" / "handlers"
            else:
                router_dir = Path("src/api/rest/v1/routers")
                handler_dir = Path("src/api/rest/v1/handlers")

        subdirs = [
            "schema",
            "types",
            "service",
            "rules",
            "machines",
            "workflows",
            "repository",
            "queries",
            "tests",
        ]

        for subdir in subdirs:
            (target_dir / subdir).mkdir(parents=True, exist_ok=True)

        canonical_feature_files = [
            "context.yml",
            "index.py",
            f"schema/{feature_slug}_schema.py",
            f"types/{feature_slug}_types.py",
            f"service/{feature_slug}_service.py",
            f"rules/{feature_slug}_rules.py",
            f"machines/{feature_slug}_machine.py",
            f"workflows/{feature_slug}_workflow.py",
            f"repository/{feature_slug}_repository_port.py",
            f"repository/{feature_slug}_repository.py",
            f"queries/{feature_slug}.queries.sql",
            f"tests/{feature_slug}_rules_test.py",
            f"tests/{feature_slug}_machine_test.py",
        ]

        created_files: List[str] = []
        for rel_path in canonical_feature_files:
            file_path = target_dir / rel_path
            if not file_path.exists():
                file_path.write_text("", encoding="utf-8")
            created_files.append(str(file_path))

        router_path = ""
        if with_router:
            router_dir.mkdir(parents=True, exist_ok=True)
            r_file = router_dir / f"{feature_slug}_router.py"
            if not r_file.exists():
                r_file.write_text("", encoding="utf-8")
            router_path = str(r_file)
            created_files.append(router_path)

        handler_path = ""
        if with_handler:
            handler_dir.mkdir(parents=True, exist_ok=True)
            h_file = handler_dir / f"{feature_slug}_handler.py"
            if not h_file.exists():
                h_file.write_text("", encoding="utf-8")
            handler_path = str(h_file)
            created_files.append(handler_path)

        return {
            "status": "success",
            "feature": feature_slug,
            "package_root": package_root or "",
            "target_dir": str(target_dir),
            "files_created": created_files,
            "total_files": len(created_files),
            "router_path": router_path,
            "handler_path": handler_path,
            "port_type": port_type,
        }

    def create_file(self, file_path: str, role: str, content: Optional[str] = None) -> str:
        fp = Path(file_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        if not fp.exists():
            body = content if content is not None else ""
            fp.write_text(body, encoding="utf-8")
        return str(fp)
