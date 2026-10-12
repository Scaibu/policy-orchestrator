from typing import Dict, Any, List, Optional
from pathlib import Path
from src.features.file_structure.repository.file_structure_repository_port import FileStructureRepositoryPort
from src.features.file_structure.repository.file_structure_repository import FileStructureRepositoryAdapter
from src.shared.constants.scaffold_constants import (
    TEMPLATE_TYPE_FEATURE,
    DEFAULT_TEMPLATE_VERSION,
    DEFAULT_FEATURES_DIR,
)
from src.features.file_structure.rules.file_structure_scaffold_path_rules import (
    ScaffoldPathRuleEngine,
    ScaffoldDeliveryRuleEngine,
)
from src.infra.observability.tracer import with_span

class FileStructureFolderScaffolderService:
    def __init__(
        self,
        repository: Optional[FileStructureRepositoryPort] = None,
        graph_sync: Optional[Any] = None,
    ) -> None:
        self.repository = repository or FileStructureRepositoryAdapter()
        self.graph_sync = graph_sync

    @with_span("file_structure.feature.scaffold")
    def scaffold_feature_tree(
        self,
        feature_name: str,
        base_dir: str = DEFAULT_FEATURES_DIR,
        package_root: Optional[str] = None,
        with_router: bool = True,
        with_handler: bool = True,
        with_route_rules: bool = False,
        port_type: str = "local",
    ) -> Dict[str, Any]:
        feature_slug = feature_name.lower().replace("-", "_")

        paths = ScaffoldPathRuleEngine.resolve_paths({
            "package_root": package_root,
            "base_dir": base_dir,
            "feature_slug": feature_slug,
        })
        target_dir = paths.target_dir

        subdirs: List[str] = []
        canonical_feature_files: List[str] = []

        template = self.repository.get_scaffold_template(
            TEMPLATE_TYPE_FEATURE,
            DEFAULT_TEMPLATE_VERSION,
        )
        if template:
            subdirs = template.get("directories", [])
            file_templates = template.get("files", [])
            canonical_feature_files = [f.format(feature_slug=feature_slug) for f in file_templates]

        for subdir in subdirs:
            (target_dir / subdir).mkdir(parents=True, exist_ok=True)

        created_files: List[str] = []
        for rel_path in canonical_feature_files:
            file_path = target_dir / rel_path
            if not file_path.exists():
                file_path.write_text("", encoding="utf-8")
            created_files.append(str(file_path))

        delivery_specs = ScaffoldDeliveryRuleEngine.resolve_delivery_artifacts(
            paths=paths,
            feature_slug=feature_slug,
            with_router=with_router,
            with_handler=with_handler,
        )

        delivery_paths: Dict[str, str] = {}
        for spec in delivery_specs:
            if spec.enabled:
                spec.target_dir.mkdir(parents=True, exist_ok=True)
                artifact_file = spec.target_dir / spec.file_name
                if not artifact_file.exists():
                    artifact_file.write_text("", encoding="utf-8")
                path_str = str(artifact_file)
                delivery_paths[spec.artifact_type] = path_str
                created_files.append(path_str)

        router_path = delivery_paths.get("router", "")
        handler_path = delivery_paths.get("handler", "")

        self.repository.save_scaffolded_feature(
            feature_name=feature_slug,
            target_dir=str(target_dir),
            port_type=port_type,
            files=created_files,
        )

        if self.graph_sync:
            self.graph_sync.sync_feature_dag(
                feature_name=feature_name,
                with_router=with_router,
                with_handler=with_handler,
                with_route_rules=with_route_rules,
                port_type=port_type,
                package_root=package_root,
            )

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
