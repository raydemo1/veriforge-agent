import sqlite_vec

VECTOR_SCHEMA = """
    CREATE TABLE IF NOT EXISTS vector_entries (
        id INTEGER PRIMARY KEY, memory_id TEXT NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,
        version INTEGER NOT NULL, contract TEXT NOT NULL, input_hash TEXT NOT NULL);
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
        embedding float[1024] distance_metric=cosine, namespace text);
    CREATE TABLE IF NOT EXISTS vector_outbox (
        memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
        failures INTEGER NOT NULL DEFAULT 0, available_at REAL NOT NULL DEFAULT 0, error_type TEXT);
    CREATE TABLE IF NOT EXISTS embedding_calls (
        id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, purpose TEXT NOT NULL,
        contract TEXT NOT NULL, input_hash TEXT NOT NULL, input_count INTEGER NOT NULL,
        usage TEXT, elapsed_ms INTEGER NOT NULL DEFAULT 0, error_type TEXT, trace_id TEXT);
"""


def load_vector_extension(db):
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    finally:
        db.enable_load_extension(False)
