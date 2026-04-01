import sqlite3
from pathlib import Path

from dbx.paths import DB, SCHEMA


DB_PATH = DB
SCHEMA_PATH = SCHEMA

# Backward-compatible migrations for DB files created before the current schema.
# Fresh DB files get all columns directly from sql/schema.sql.
LEGACY_COLUMN_MIGRATIONS = [
    ("candidate", "source_file", "TEXT"),
    ("candidate", "source_imported_at", "TEXT"),
    ("candidate", "primary_email", "TEXT"),
    ("candidate", "primary_phone", "TEXT"),
    ("candidate", "languages_raw", "TEXT"),
    ("candidate", "languages_json", "TEXT"),
    ("candidate_profile_text", "location_name", "TEXT"),
    ("candidate_profile_text", "industry", "TEXT"),
    ("candidate_profile_text", "current_company", "TEXT"),
    ("candidate_profile_text", "current_position", "TEXT"),
    ("candidate_profile_text", "badges_job_seeker", "INTEGER"),
    ("candidate_profile_text", "badges_open_link", "INTEGER"),
    ("candidate_profile_text", "profile_snapshot_at", "TEXT"),
    ("candidate_rank_features", "listed_role_months_sum", "INTEGER"),
    ("candidate_rank_features", "current_listed_role_months", "INTEGER"),
    ("candidate_rank_features", "longest_listed_role_months", "INTEGER"),
    ("candidate_rank_features", "iam_role_months", "INTEGER"),
    ("candidate_rank_features", "iam_role_count", "INTEGER"),
    ("candidate_rank_features", "current_role_is_iam", "INTEGER"),
]


def column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(row[1] == column for row in rows)


def add_column_if_missing(conn, table: str, column: str, coltype: str):
    if not column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype};")
        print(f"Added legacy migration column: {table}.{column}")


def init_db(db_path=DB_PATH, schema_path=SCHEMA_PATH):
    db_path = Path(db_path)
    schema_path = Path(schema_path)
    db_already_existed = db_path.exists()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(schema_path.read_text(encoding="utf-8"))

        if db_already_existed:
            for table, column, coltype in LEGACY_COLUMN_MIGRATIONS:
                add_column_if_missing(conn, table, column, coltype)

        conn.commit()

    print(f"DB initialized: {db_path.resolve()}")


if __name__ == "__main__":
    init_db()
