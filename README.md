# database_project — Local-first Recruiting Database (CSV → SQLite)

This repo contains a small ingestion pipeline to transform LinkedHelper/LinkedIn CSV exports into a local SQLite database.

## Goals
- Local-first data processing (no cloud DB required)
- Strict separation of:
  - **PII / sensitive data** (stored locally in SQLite only)
  - **matching text** (stored locally and later used for embeddings / LLM prompts)
  - **messaging history** (stored locally only; never used for embeddings/LLM)

## Repository Contents
- `schema.sql` — SQLite schema (tables)
- `config.yml` — mapping configuration from CSV → normalized structure
- `init_db.py` — creates the SQLite DB and applies schema
- `import_csv.py` — imports a CSV into SQLite (dedupe by `profile_url`)
- `check_db.py` — quick sanity checks
- `requirements.txt` — Python dependencies

> Note: CSV files and `.db` files are intentionally excluded from git via `.gitignore`.

---

## Setup

### 1) Clone the repo
```bash
git clone https://github.com/IAMHiringDarwin/database_project.git
cd database_project
