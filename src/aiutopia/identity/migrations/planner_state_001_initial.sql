-- §3.4 + §6.7 — planner_state.db initial schema:
--   planner_state, plan_cache, chat_failures, planner_failures.
-- This DB intentionally co-locates operational tables that change frequently
-- (vs identity.db which holds the slower-moving identity facts).

CREATE TABLE planner_state (
    plan_id              TEXT PRIMARY KEY,
    agent_uuid           TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN
                           ('active', 'completed', 'failed', 'paused',
                            'failed_migration')),
    dag_json             TEXT NOT NULL,
    current_subgoal_id   TEXT,
    pending_events_jsonl TEXT,
    llm_call_log_jsonl   TEXT,
    schema_version       TEXT NOT NULL,
    created_at           INTEGER NOT NULL,
    last_updated         INTEGER NOT NULL
);

CREATE INDEX idx_planner_active
    ON planner_state(status, agent_uuid)
    WHERE status IN ('active', 'paused');

CREATE TABLE plan_cache (
    cache_key            TEXT PRIMARY KEY,
    context_json         TEXT NOT NULL,
    prompt_text          TEXT NOT NULL,
    plan_json            TEXT NOT NULL,
    llm_model            TEXT NOT NULL,
    llm_call_latency_ms  INTEGER NOT NULL,
    llm_call_cost_usd    REAL,
    hit_count            INTEGER NOT NULL DEFAULT 1,
    created_at           INTEGER NOT NULL,
    last_hit_at          INTEGER NOT NULL,
    ttl_seconds          INTEGER NOT NULL DEFAULT 3600,
    invalidated          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_plan_cache_last_hit
    ON plan_cache(last_hit_at) WHERE invalidated = 0;

CREATE TABLE chat_failures (
    failure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_uuid    TEXT NOT NULL,
    player_uuid   TEXT NOT NULL,
    text          TEXT NOT NULL,
    error_type    TEXT NOT NULL CHECK (error_type IN
                    ('timeout', 'api_error', 'qwen_unavail')),
    occurred_at   INTEGER NOT NULL
);

CREATE TABLE planner_failures (
    failure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id       TEXT NOT NULL,
    failure_type  TEXT NOT NULL,
    detail_json   TEXT,
    occurred_at   INTEGER NOT NULL
);
