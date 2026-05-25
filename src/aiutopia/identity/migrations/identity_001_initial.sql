-- §3.5 — Identity DB initial schema (5 tables: roles, agents, agent_lives,
-- funerals, policy_deployments).
-- Note: chat_failures and planner_failures from §3.5 live in planner_state.db
-- per M0 scope (see IMPLEMENTATION_PLAN.md "Spec inconsistency surfaced").

CREATE TABLE roles (
    role_id                    TEXT PRIMARY KEY,
    display_name               TEXT NOT NULL,
    policy_weights_path        TEXT NOT NULL,
    policy_version             INTEGER NOT NULL,
    observation_schema_version INTEGER NOT NULL,
    action_schema_version      INTEGER NOT NULL,
    max_lives                  INTEGER NOT NULL DEFAULT 1,
    default_skin_pool          TEXT
);

CREATE TABLE agents (
    agent_uuid          TEXT PRIMARY KEY,
    role_id             TEXT NOT NULL REFERENCES roles(role_id),
    agent_name          TEXT NOT NULL,
    skill_library_id    TEXT NOT NULL,
    memory_id           TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('alive', 'dead')),
    born_at             INTEGER NOT NULL,
    died_at             INTEGER,
    spawn_position_json TEXT,
    current_skin        TEXT
);

CREATE UNIQUE INDEX idx_agent_name_alive
    ON agents(agent_name) WHERE status = 'alive';

CREATE TABLE agent_lives (
    life_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_uuid     TEXT NOT NULL REFERENCES agents(agent_uuid),
    role_id        TEXT NOT NULL REFERENCES roles(role_id),
    born_at        INTEGER NOT NULL,
    died_at        INTEGER,
    cause_of_death TEXT
);

CREATE TABLE funerals (
    funeral_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    deceased_agent_uuid     TEXT NOT NULL REFERENCES agents(agent_uuid),
    witness_agent_uuids_json TEXT NOT NULL,
    event_summary           TEXT NOT NULL,
    written_to_memory_at    INTEGER NOT NULL
);

CREATE TABLE policy_deployments (
    deployment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    role_id          TEXT NOT NULL REFERENCES roles(role_id),
    from_version     INTEGER,
    to_version       INTEGER NOT NULL,
    deployed_at      INTEGER NOT NULL,
    deployed_by      TEXT NOT NULL,
    notes            TEXT
);

-- Seed the 4 roles with stub policy paths (filled by promote-weights CLI later).
INSERT INTO roles (role_id, display_name, policy_weights_path, policy_version,
                   observation_schema_version, action_schema_version,
                   max_lives, default_skin_pool) VALUES
  ('gatherer', 'Gatherer', '', 0, 1, 1, 1,
   json('["Bjorn","Gunnar","Sigrid","Eirik","Astrid","Halvor","Frida","Magnus","Ingrid","Knut","Solveig","Olav"]')),
  ('builder',  'Builder',  '', 0, 1, 1, 1,
   json('["Bram","Lisa","Hugo","Maeve","Cedric","Nora","Oscar","Petra","Rolf","Saga","Tomas","Vera"]')),
  ('farmer',   'Farmer',   '', 0, 1, 1, 1,
   json('["Hannah","Idris","Jorah","Kara","Linus","Mira","Niko","Otto","Pia","Quinn","Reza","Sten"]')),
  ('defender', 'Defender', '', 0, 1, 1, 1,
   json('["Thora","Ulf","Vidar","Wilma","Xander","Yara","Zane","Anja","Bodvar","Cara","Dag","Eira"]'));
