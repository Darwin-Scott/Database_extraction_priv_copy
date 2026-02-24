from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root (src/dbx/.. -> root)

DATA  = ROOT / "data"
SQL   = ROOT / "sql"
LOCAL = ROOT / "local"

RAW_DATA = DATA / "raw"
OUT      = LOCAL / "out"
CHROMA   = LOCAL / "chroma_db"
DB       = LOCAL / "candidates.db"
SCHEMA   = SQL / "schema.sql"