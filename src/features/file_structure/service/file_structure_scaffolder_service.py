from typing import Dict, Any, Optional
from src.features.file_structure.service.file_structure_folder_scaffolder_service import (
    FileStructureFolderScaffolderService,
)
from src.features.file_structure.service.file_structure_package_scaffolder_service import (
    FileStructurePackageScaffolderService,
)

class FileStructureScaffolderService:
    def __init__(
        self,
        folder_scaffolder: Optional[FileStructureFolderScaffolderService] = None,
        package_scaffolder: Optional[FileStructurePackageScaffolderService] = None,
    ) -> None:
        self.folder_scaffolder = folder_scaffolder or FileStructureFolderScaffolderService()
        self.package_scaffolder = package_scaffolder or FileStructurePackageScaffolderService()

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
        return self.folder_scaffolder.scaffold_feature_tree(
            feature_name=feature_name,
            base_dir=base_dir,
            package_root=package_root,
            with_router=with_router,
            with_handler=with_handler,
            with_route_rules=with_route_rules,
            port_type=port_type,
        )

    def scaffold_package_tree(
        self,
        package_name: str,
        base_dir: str = ".",
    ) -> Dict[str, Any]:
        return self.package_scaffolder.scaffold_package_tree(
            package_name=package_name,
            base_dir=base_dir,
        )

    def create_file(self, file_path: str, role: str, content: Optional[str] = None) -> str:
        return self.folder_scaffolder.create_file(file_path=file_path, role=role, content=content)
