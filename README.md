# database_project — Local-first Recruiting Database (CSV → SQLite)

This repository contains scripts to transform LinkedHelper / LinkedIn CSV exports into a **local SQLite database** (`candidates.db`).

The pipeline is designed to:
- keep **PII and messaging** strictly local (SQLite only),
- store **matching-relevant text** in a normalized structure,
- enable later steps like vector search (Chroma/Qdrant) and LLM ranking.

---

## Files

- `schema.sql` — SQLite schema (tables + indexes)
- `config.yml` — CSV → normalized mapping config
- `requirements.txt` — Python dependencies
- `init_db.py` — creates/initializes the SQLite DB by applying `schema.sql`
- `import_csv.py` — imports one CSV into SQLite (dedupe by `profile_url`)
- `check_db.py` — prints DB tables, columns, indexes, counts, and sample rows
- `extraction.py` — helper script to inspect CSV columns/examples (data exploration)

---

## Setup

### 1) Clone the repo
```bash
git clone https://github.com/IAMHiringDarwin/database_project.git
cd database_project
