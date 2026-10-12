
from pathlib import Path
from typing import Dict, Optional

class FileStructureQueryLoader:
    _cypher_cache: Dict[str, str] = {}
    _sql_cache: Dict[str, str] = {}
    _loaded: bool = False

    @classmethod
    def load_queries(cls) -> None:
        if cls._loaded:
            return

        queries_dir = Path(__file__).resolve().parent.parent / "queries"
        if not queries_dir.exists():
            return

        for fpath in queries_dir.glob("*.cypher"):
            cls._parse_file(fpath, cls._cypher_cache)

        for fpath in queries_dir.glob("*.sql"):
            cls._parse_file(fpath, cls._sql_cache)

        cls._loaded = True

    @classmethod
    def get_cypher(cls, query_name: str) -> str:
        cls.load_queries()
        if query_name not in cls._cypher_cache:
            raise KeyError(f"Named Cypher query '{query_name}' not found in queries directory")
        return cls._cypher_cache[query_name]

    @classmethod
    def get_sql(cls, query_name: str) -> str:
        cls.load_queries()
        if query_name not in cls._sql_cache:
            raise KeyError(f"Named SQL query '{query_name}' not found in queries directory")
        return cls._sql_cache[query_name]

    @classmethod
    def _parse_file(cls, fpath: Path, target_dict: Dict[str, str]) -> None:
        content = fpath.read_text(encoding="utf-8")
        current_name: Optional[str] = None
        current_lines: list = []

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("// name:") or stripped.startswith("-- name:"):
                if current_name and current_lines:
                    target_dict[current_name] = "\n".join(current_lines).strip()
                current_name = stripped.split("name:")[1].strip()
                current_lines = []
            elif current_name is not None:
                current_lines.append(line)

        if current_name and current_lines:
            target_dict[current_name] = "\n".join(current_lines).strip()
