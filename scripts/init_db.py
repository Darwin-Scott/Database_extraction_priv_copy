import sqlite3
from pathlib import Path
from dbx.paths import DB, SCHEMA

DB_PATH = DB
SCHEMA_PATH = SCHEMA

def column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r[1] == column for r in rows)

def add_column_if_missing(conn, table: str, column: str, coltype: str):
    if not column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype};")
        print(f"✅ Added column: {table}.{column}")

def init_db(db_path=DB_PATH, schema_path=SCHEMA_PATH):
    db_path = Path(db_path)
    schema_path = Path(schema_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(schema_path.read_text(encoding="utf-8"))

        # --- lightweight migrations ---
        add_column_if_missing(conn, "candidate", "source_file", "TEXT")
        add_column_if_missing(conn, "candidate", "source_imported_at", "TEXT")
        add_column_if_missing(conn, "candidate", "primary_email", "TEXT")
        add_column_if_missing(conn, "candidate", "primary_phone", "TEXT")
        add_column_if_missing(conn, "candidate", "languages_raw", "TEXT")
        add_column_if_missing(conn, "candidate", "languages_json", "TEXT")

        add_column_if_missing(conn, "candidate_profile_text", "location_name", "TEXT")
        add_column_if_missing(conn, "candidate_profile_text", "industry", "TEXT")
        add_column_if_missing(conn, "candidate_profile_text", "current_company", "TEXT")
        add_column_if_missing(conn, "candidate_profile_text", "current_position", "TEXT")
        add_column_if_missing(conn, "candidate_profile_text", "badges_job_seeker", "INTEGER")
        add_column_if_missing(conn, "candidate_profile_text", "badges_open_link", "INTEGER")
        add_column_if_missing(conn, "candidate_profile_text", "profile_snapshot_at", "TEXT")
        add_column_if_missing(conn, "candidate_rank_features", "listed_role_months_sum", "INTEGER")
        add_column_if_missing(conn, "candidate_rank_features", "current_listed_role_months", "INTEGER")
        add_column_if_missing(conn, "candidate_rank_features", "longest_listed_role_months", "INTEGER")
        add_column_if_missing(conn, "candidate_rank_features", "iam_role_months", "INTEGER")
        add_column_if_missing(conn, "candidate_rank_features", "iam_role_count", "INTEGER")
        add_column_if_missing(conn, "candidate_rank_features", "current_role_is_iam", "INTEGER")

        conn.commit()

    print(f"✅ DB initialized: {db_path.resolve()}")

if __name__ == "__main__":
    init_db()
