from typing import Optional
from pathlib import Path

class FeatureSlugPipeline:
    @staticmethod
    def resolve_slug(feature_name: Optional[str], file_path: str) -> str:
        if feature_name:
            return feature_name.lower().replace("-", "_")
        parts = Path(file_path).parts
        if "features" in parts:
            idx = parts.index("features")
            if idx + 1 < len(parts):
                return parts[idx + 1].lower().replace("-", "_")
        return ""
