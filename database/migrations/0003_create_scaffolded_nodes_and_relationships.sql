-- migration:      0003
-- description:    create scaffolded files, folders, and relationships ledger with explicit constraints
-- author:         Antigravity / Lead Software Engineer
-- date:           2026-10-12
-- depends_on:     0001
-- reversible:     YES
-- lock_risk:      LOW
-- rows_affected:  0 (schema creation)
-- reason:         AlloyDB storage gate integrity per 05-db-writing-rules.md

CREATE TABLE IF NOT EXISTS scaffolded_nodes (
    id VARCHAR(512) NOT NULL,
    file_path TEXT NOT NULL,
    role VARCHAR(64) NOT NULL,
    layer VARCHAR(64) NOT NULL,
    feature_name VARCHAR(128) NULL,
    package_name VARCHAR(128) NULL,
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    version BIGINT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT scaffolded_nodes_id_pk PRIMARY KEY (id),
    CONSTRAINT scaffolded_nodes_id_trimmed_ck CHECK (id = btrim(id)),
    CONSTRAINT scaffolded_nodes_path_nonempty_ck CHECK (file_path = btrim(file_path) AND file_path <> ''),
    CONSTRAINT scaffolded_nodes_role_nonempty_ck CHECK (role = btrim(role) AND role <> ''),
    CONSTRAINT scaffolded_nodes_layer_nonempty_ck CHECK (layer = btrim(layer) AND layer <> ''),
    CONSTRAINT scaffolded_nodes_feature_trimmed_ck CHECK (feature_name IS NULL OR feature_name = btrim(feature_name)),
    CONSTRAINT scaffolded_nodes_package_trimmed_ck CHECK (package_name IS NULL OR package_name = btrim(package_name)),
    CONSTRAINT scaffolded_nodes_size_nonnegative_ck CHECK (file_size_bytes >= 0)
);

CREATE INDEX IF NOT EXISTS idx_scaffolded_nodes_path ON scaffolded_nodes (file_path);
CREATE INDEX IF NOT EXISTS idx_scaffolded_nodes_feature ON scaffolded_nodes (feature_name);
CREATE INDEX IF NOT EXISTS idx_scaffolded_nodes_role ON scaffolded_nodes (role);

CREATE TABLE IF NOT EXISTS scaffolded_relationships (
    id VARCHAR(512) NOT NULL,
    source_id VARCHAR(512) NOT NULL,
    target_id VARCHAR(512) NOT NULL,
    rel_type VARCHAR(64) NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT scaffolded_rel_id_pk PRIMARY KEY (id),
    CONSTRAINT scaffolded_rel_id_trimmed_ck CHECK (id = btrim(id)),
    CONSTRAINT scaffolded_rel_source_fk FOREIGN KEY (source_id) REFERENCES scaffolded_nodes(id) ON DELETE RESTRICT,
    CONSTRAINT scaffolded_rel_target_fk FOREIGN KEY (target_id) REFERENCES scaffolded_nodes(id) ON DELETE RESTRICT,
    CONSTRAINT scaffolded_rel_type_nonempty_ck CHECK (rel_type = btrim(rel_type) AND rel_type <> ''),
    CONSTRAINT scaffolded_rel_props_obj_ck CHECK (jsonb_typeof(properties) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_scaffolded_rel_source ON scaffolded_relationships (source_id);
CREATE INDEX IF NOT EXISTS idx_scaffolded_rel_target ON scaffolded_relationships (target_id);
CREATE INDEX IF NOT EXISTS idx_scaffolded_rel_type ON scaffolded_relationships (rel_type);
