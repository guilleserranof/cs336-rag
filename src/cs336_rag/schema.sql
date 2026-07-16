-- Knowledge base schema. {embedding_dim} is substituted by db.init_schema().
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL,
    title       TEXT NOT NULL,
    position    INT NOT NULL,
    chunk_index INT NOT NULL,
    start_s     DOUBLE PRECISION NOT NULL,
    end_s       DOUBLE PRECISION NOT NULL,
    content     TEXT NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding   vector({embedding_dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
