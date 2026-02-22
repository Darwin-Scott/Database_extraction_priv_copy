import sqlite3
from pathlib import Path

def init_db(db_path="candidates.db", schema_path="Database_extraction\schema.sql"):
    db_path = Path(db_path)
    schema_path = Path(schema_path)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        conn.commit()

    print(f"✅ DB initialized: {db_path.resolve()}")

if __name__ == "__main__":
    init_db()