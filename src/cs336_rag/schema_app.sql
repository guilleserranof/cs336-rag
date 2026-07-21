-- Application telemetry, independent of the knowledge base.
-- Created by the API at startup; re-ingesting never touches these tables.
-- Application telemetry. Independent of the knowledge base: re-ingesting
-- rebuilds `chunks` but never touches the history the dashboards read.
CREATE TABLE IF NOT EXISTS conversations (
    id                UUID PRIMARY KEY,
    question          TEXT NOT NULL,
    answer            TEXT NOT NULL,
    variant           TEXT NOT NULL,
    retrieval_method  TEXT NOT NULL,
    source_ids        TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    num_sources       INT NOT NULL DEFAULT 0,
    retrieval_ms      DOUBLE PRECISION,
    generation_ms     DOUBLE PRECISION,
    total_ms          DOUBLE PRECISION,
    prompt_tokens     INT,
    completion_tokens INT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_created_at_idx ON conversations (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    conversation_id UUID PRIMARY KEY REFERENCES conversations (id) ON DELETE CASCADE,
    rating          SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_created_at_idx ON feedback (created_at);
