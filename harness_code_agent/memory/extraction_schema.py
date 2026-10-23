EXTRACTION_SCHEMA = """
                CREATE TABLE IF NOT EXISTS memory_metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS extraction_sessions (
                    namespace TEXT NOT NULL, session_id TEXT NOT NULL, journal_path TEXT NOT NULL,
                    enabled INTEGER NOT NULL, active_until REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(namespace,session_id));
                CREATE TABLE IF NOT EXISTS extraction_checkpoints (
                    namespace TEXT NOT NULL, session_id TEXT NOT NULL,
                    boundary INTEGER NOT NULL, text_offset INTEGER NOT NULL,
                    PRIMARY KEY(namespace,session_id));
                CREATE TABLE IF NOT EXISTS extraction_tasks (
                    id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, session_id TEXT NOT NULL,
                    boundary INTEGER NOT NULL, prefix_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', failures INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL, lease_until REAL, lease_token TEXT,
                    chunk_tokens INTEGER NOT NULL, error_type TEXT,
                    UNIQUE(namespace,session_id,boundary));
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id TEXT PRIMARY KEY, owner_namespace TEXT NOT NULL, target_namespace TEXT NOT NULL,
                    session_id TEXT NOT NULL, task_id INTEGER NOT NULL REFERENCES extraction_tasks(id),
                    sequence INTEGER NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending');
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,
                    summary TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS consolidation_tasks (
                    id INTEGER PRIMARY KEY, namespace TEXT NOT NULL,
                    extraction_task INTEGER NOT NULL REFERENCES extraction_tasks(id),
                    source_boundary INTEGER NOT NULL, source_offset INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', failures INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL DEFAULT 0, lease_until REAL, lease_token TEXT, error_type TEXT,
                    batch_size INTEGER NOT NULL DEFAULT 8,
                    UNIQUE(extraction_task,source_boundary,source_offset));
                CREATE TABLE IF NOT EXISTS memory_model_calls (
                    id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES extraction_tasks(id),
                    provider TEXT NOT NULL, model TEXT NOT NULL, input_hash TEXT NOT NULL,
                    output_hash TEXT, usage TEXT, elapsed_ms INTEGER NOT NULL, error_type TEXT);
                CREATE TABLE IF NOT EXISTS consolidation_members (
                    task_id INTEGER NOT NULL REFERENCES consolidation_tasks(id),
                    candidate_id TEXT NOT NULL UNIQUE REFERENCES memory_candidates(id) ON DELETE CASCADE,
                    PRIMARY KEY(task_id,candidate_id));
                CREATE TABLE IF NOT EXISTS memory_holds (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    candidate_id TEXT NOT NULL REFERENCES memory_candidates(id) ON DELETE CASCADE,
                    PRIMARY KEY(memory_id,candidate_id));
                CREATE TABLE IF NOT EXISTS memory_decisions (
                    candidate_id TEXT PRIMARY KEY REFERENCES memory_candidates(id) ON DELETE CASCADE,
                    memory_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                    action TEXT NOT NULL, previous_version INTEGER, result_version INTEGER NOT NULL,
                    timestamp TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS consolidation_model_calls (
                    id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES consolidation_tasks(id),
                    provider TEXT NOT NULL, model TEXT NOT NULL, input_hash TEXT NOT NULL,
                    output_hash TEXT, usage TEXT, elapsed_ms INTEGER NOT NULL DEFAULT 0, error_type TEXT);
"""
