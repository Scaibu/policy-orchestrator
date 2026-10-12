from typing import Dict, Any, List, Optional
from pathlib import Path
from src.features.file_structure.repository.file_structure_repository_port import FileStructureRepositoryPort
from src.features.file_structure.repository.file_structure_repository import FileStructureRepositoryAdapter
from src.shared.constants.scaffold_constants import (
    TEMPLATE_TYPE_PACKAGE,
    DEFAULT_TEMPLATE_VERSION,
)
from src.infra.observability.tracer import with_span

class FileStructurePackageScaffolderService:
    def __init__(
        self,
        repository: Optional[FileStructureRepositoryPort] = None,
        graph_sync: Optional[Any] = None,
    ) -> None:
        self.repository = repository or FileStructureRepositoryAdapter()
        self.graph_sync = graph_sync

    @with_span("file_structure.package.scaffold")
    def scaffold_package_tree(
        self,
        package_name: str,
        base_dir: str = ".",
    ) -> Dict[str, Any]:
        pkg_slug = package_name.lower().replace("-", "_")
        target_root = Path(base_dir) / pkg_slug

        package_directories: List[str] = []
        pkg_empty_files: List[str] = []

        template = self.repository.get_scaffold_template(
            TEMPLATE_TYPE_PACKAGE,
            DEFAULT_TEMPLATE_VERSION,
        )
        if template:
            package_directories = template.get("directories", [])
            pkg_empty_files = template.get("files", [])

        created_dirs: List[str] = []
        for d in package_directories:
            dir_path = target_root / d
            dir_path.mkdir(parents=True, exist_ok=True)
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.write_text("", encoding="utf-8")
            created_dirs.append(str(dir_path))

        created_files: List[str] = []
        for rel_path in pkg_empty_files:
            file_path = target_root / rel_path
            if not file_path.exists():
                file_path.write_text("", encoding="utf-8")
            created_files.append(str(file_path))

        self.repository.save_scaffolded_package(
            package_name=pkg_slug,
            target_root=str(target_root),
            directories=created_dirs,
            files=created_files,
        )

        if self.graph_sync:
            self.graph_sync.sync_package_dag(
                package_name=package_name,
                target_root=str(target_root),
            )

        return {
            "status": "success",
            "package": pkg_slug,
            "target_root": str(target_root),
            "directories_scaffolded": len(created_dirs),
            "files_created": len(created_files),
        }
