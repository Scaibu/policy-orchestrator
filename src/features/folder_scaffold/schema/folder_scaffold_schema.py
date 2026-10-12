from typing import Dict, Any
from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldFeatureCommand,
    ScaffoldPackageCommand,
    CreateFileResult,
    LinkFilesResult,
    ScaffoldExecutionResult,
)


class FolderScaffoldSchema:
    @staticmethod
    def from_api_create_file(payload: Dict[str, Any]) -> CreateFileCommand:
        return CreateFileCommand(
            path=str(payload["path"]).strip(),
            role=str(payload["role"]).strip(),
            rel_type=str(payload["rel_type"]).strip(),
            target=str(payload["target"]).strip(),
            direction=str(payload.get("direction", "outgoing")).strip().lower(),
            feature_name=payload.get("feature_name"),
            package_name=payload.get("package_name"),
            content=payload.get("content"),
        )

    @staticmethod
    def to_api_create_file(result: CreateFileResult) -> Dict[str, Any]:
        return {
            "node": {
                "id": result.node.id,
                "filePath": result.node.file_path,
                "role": result.node.role,
                "layer": result.node.layer,
                "featureName": result.node.feature_name,
                "packageName": result.node.package_name,
                "sizeBytes": result.node.size_bytes,
            },
            "relationship": {
                "id": result.relationship.id,
                "sourceId": result.relationship.source_id,
                "targetId": result.relationship.target_id,
                "relType": result.relationship.rel_type,
                "properties": result.relationship.properties,
            },
            "createdOnDisk": result.created_on_disk,
            "status": result.status,
        }

    @staticmethod
    def from_api_link_files(payload: Dict[str, Any]) -> LinkFilesCommand:
        return LinkFilesCommand(
            source=str(payload["source"]).strip(),
            rel_type=str(payload["rel_type"]).strip(),
            target=str(payload["target"]).strip(),
            properties=payload.get("properties") or {},
        )

    @staticmethod
    def to_api_link_files(result: LinkFilesResult) -> Dict[str, Any]:
        return {
            "relationship": {
                "id": result.relationship.id,
                "sourceId": result.relationship.source_id,
                "targetId": result.relationship.target_id,
                "relType": result.relationship.rel_type,
                "properties": result.relationship.properties,
            },
            "status": result.status,
        }

    @staticmethod
    def from_api_scaffold_feature(payload: Dict[str, Any]) -> ScaffoldFeatureCommand:
        return ScaffoldFeatureCommand(
            feature_name=str(payload["feature_name"]).strip(),
            base_dir=str(payload.get("base_dir", "src/features")).strip(),
            package_root=payload.get("package_root"),
        )

    @staticmethod
    def from_api_scaffold_package(payload: Dict[str, Any]) -> ScaffoldPackageCommand:
        return ScaffoldPackageCommand(
            package_name=str(payload["package_name"]).strip(),
            base_dir=str(payload.get("base_dir", ".")).strip(),
        )

    @staticmethod
    def to_api_scaffold_result(result: ScaffoldExecutionResult) -> Dict[str, Any]:
        return {
            "targetName": result.target_name,
            "targetPath": result.target_path,
            "totalDirectories": result.total_directories,
            "totalFiles": result.total_files,
            "status": result.status,
        }
