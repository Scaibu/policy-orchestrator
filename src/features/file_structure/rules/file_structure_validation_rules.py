from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass(frozen=True)
class ValidationRule:
    field: str
    check: Callable[[Any], bool]
    error_message: str

class FileStructureValidationRuleEngine:
    _create_file_rules: List[ValidationRule] = [
        ValidationRule(field="file_path", check=lambda v: bool(v and str(v).strip()), error_message="file_path must not be empty"),
        ValidationRule(field="role", check=lambda v: bool(v and str(v).strip()), error_message="role/label must not be empty"),
        ValidationRule(field="rel_type", check=lambda v: bool(v and str(v).strip()), error_message="rel_type is a required parameter when creating/registering a file"),
        ValidationRule(field="target_file_or_node", check=lambda v: bool(v and str(v).strip()), error_message="target_file_or_node is a required parameter to link the new file"),
    ]

    _link_files_rules: List[ValidationRule] = [
        ValidationRule(field="source_file_or_node", check=lambda v: bool(v and str(v).strip()), error_message="source, target, and rel_type are required to link files"),
        ValidationRule(field="target_file_or_node", check=lambda v: bool(v and str(v).strip()), error_message="source, target, and rel_type are required to link files"),
        ValidationRule(field="rel_type", check=lambda v: bool(v and str(v).strip()), error_message="source, target, and rel_type are required to link files"),
    ]

    @classmethod
    def validate_create_file(cls, context: Dict[str, Any]) -> None:
        for rule in cls._create_file_rules:
            val = context.get(rule.field)
            if not rule.check(val):
                raise ValueError(rule.error_message)

    @classmethod
    def validate_link_files(cls, context: Dict[str, Any]) -> None:
        for rule in cls._link_files_rules:
            val = context.get(rule.field)
            if not rule.check(val):
                raise ValueError(rule.error_message)
