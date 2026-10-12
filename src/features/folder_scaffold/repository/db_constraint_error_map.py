from typing import Dict, Tuple

CONSTRAINT_ERROR_MAP: Dict[str, Tuple[str, str]] = {
    "scaffolded_nodes_id_pk": ("CONFLICT.DUPLICATE_NODE_ID", "File node with this ID already exists"),
    "scaffolded_nodes_id_trimmed_ck": ("VALIDATION.INVALID_ID", "Node ID contains invalid whitespace padding"),
    "scaffolded_nodes_path_nonempty_ck": ("VALIDATION.EMPTY_PATH", "File path must be non-empty and trimmed"),
    "scaffolded_nodes_role_nonempty_ck": ("VALIDATION.EMPTY_ROLE", "Role must be non-empty and trimmed"),
    "scaffolded_nodes_layer_nonempty_ck": ("VALIDATION.EMPTY_LAYER", "Layer must be non-empty and trimmed"),
    "scaffolded_nodes_size_nonnegative_ck": ("VALIDATION.NEGATIVE_SIZE", "File size cannot be negative"),
    "scaffolded_rel_id_pk": ("CONFLICT.DUPLICATE_RELATIONSHIP", "Relationship with this ID already exists"),
    "scaffolded_rel_source_fk": ("VALIDATION.ORPHAN_SOURCE", "Source node does not exist in scaffolded_nodes"),
    "scaffolded_rel_target_fk": ("VALIDATION.ORPHAN_TARGET", "Target node does not exist in scaffolded_nodes"),
    "scaffolded_rel_type_nonempty_ck": ("VALIDATION.EMPTY_REL_TYPE", "Relationship type must be non-empty and trimmed"),
    "scaffolded_rel_props_obj_ck": ("VALIDATION.INVALID_PROPERTIES", "Properties payload must be a JSON object"),
}


def map_db_constraint_error(exc_message: str) -> Tuple[str, str]:
    for constraint, (code, msg) in CONSTRAINT_ERROR_MAP.items():
        if constraint in exc_message:
            return code, msg
    return "DATABASE_ERROR", exc_message
