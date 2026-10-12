from typing import Dict, Any, List
from pathlib import Path
from src.features.file_structure.types.file_structure_types import (
    ScaffoldTargetPathsModel,
    DeliveryArtifactSpec,
)
from src.shared.constants.scaffold_constants import (
    PATH_SRC,
    PATH_FEATURES,
    PATH_API,
    PATH_REST,
    PATH_V1,
    PATH_ROUTERS,
    PATH_HANDLERS,
    DEFAULT_FEATURES_DIR,
    DEFAULT_ROUTERS_DIR,
    DEFAULT_HANDLERS_DIR,
)

class ScaffoldPathRuleEngine:
    _rules = [
        (
            lambda ctx: bool(ctx.get("package_root")),
            lambda ctx: ScaffoldTargetPathsModel(
                target_dir=Path(ctx["package_root"]) / PATH_SRC / PATH_FEATURES / ctx["feature_slug"],
                router_dir=Path(ctx["package_root"]) / PATH_SRC / PATH_API / PATH_REST / PATH_V1 / PATH_ROUTERS,
                handler_dir=Path(ctx["package_root"]) / PATH_SRC / PATH_API / PATH_REST / PATH_V1 / PATH_HANDLERS,
            )
        ),
        (
            lambda ctx: str(ctx.get("base_dir", "")).startswith("/tmp") or (
                DEFAULT_FEATURES_DIR not in str(ctx.get("base_dir", "")) and str(ctx.get("base_dir", "")) != "."
            ),
            lambda ctx: ScaffoldTargetPathsModel(
                target_dir=Path(ctx["base_dir"]) / ctx["feature_slug"],
                router_dir=Path(ctx["base_dir"]) / PATH_API / PATH_REST / PATH_V1 / PATH_ROUTERS,
                handler_dir=Path(ctx["base_dir"]) / PATH_API / PATH_REST / PATH_V1 / PATH_HANDLERS,
            )
        ),
        (
            lambda ctx: True,
            lambda ctx: ScaffoldTargetPathsModel(
                target_dir=Path(ctx.get("base_dir") or DEFAULT_FEATURES_DIR) / ctx["feature_slug"],
                router_dir=Path(DEFAULT_ROUTERS_DIR),
                handler_dir=Path(DEFAULT_HANDLERS_DIR),
            )
        ),
    ]

    @classmethod
    def resolve_paths(cls, context: Dict[str, Any]) -> ScaffoldTargetPathsModel:
        for condition, resolver in cls._rules:
            if condition(context):
                return resolver(context)
        return cls._rules[-1][1](context)

class ScaffoldDeliveryRuleEngine:
    @staticmethod
    def resolve_delivery_artifacts(
        paths: ScaffoldTargetPathsModel,
        feature_slug: str,
        with_router: bool,
        with_handler: bool,
    ) -> List[DeliveryArtifactSpec]:
        return [
            DeliveryArtifactSpec(
                artifact_type="router",
                target_dir=paths.router_dir,
                file_name=f"{feature_slug}_router.py",
                enabled=with_router,
            ),
            DeliveryArtifactSpec(
                artifact_type="handler",
                target_dir=paths.handler_dir,
                file_name=f"{feature_slug}_handler.py",
                enabled=with_handler,
            ),
        ]
