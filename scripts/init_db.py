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

        conn.commit()

    print(f"✅ DB initialized: {db_path.resolve()}")

if __name__ == "__main__":
    init_db()