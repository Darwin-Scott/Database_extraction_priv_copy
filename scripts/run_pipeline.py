# scripts/run_pipeline.py
"""
End-to-end pipeline runner.

Examples:

# Full rebuild (DANGER: resets local DB/Chroma/out) + run everything
python scripts/run_pipeline.py --all --reset-local --search "One Identity IAM Consultant Active Directory" --job "One Identity IAM Consultant. Requirements: One Identity Manager, AD integration, SQL, scripting, troubleshooting, German/English."

# Run only search + prepare batch + render (assuming chroma + db already exist)
python scripts/run_pipeline.py --search "One Identity IAM Consultant Active Directory" --job "..."

# Run import pipeline only
python scripts/run_pipeline.py --init-db --import-csv

Notes:
- This runner calls your existing scripts as subprocesses to avoid import/path refactors.
- It assumes your project root contains: scripts/, src/dbx/paths.py, local/, data/raw/, sql/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    # scripts/run_pipeline.py -> repo root
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
SCRIPTS = ROOT / "scripts"
LOCAL = ROOT / "local"
OUT = LOCAL / "out"
CHROMA = LOCAL / "chroma_db"
DB = LOCAL / "candidates.db"
RAW = ROOT / "data" / "raw"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command, fail fast with useful output."""
    print("\n" + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def newest_csv(raw_dir: Path) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw data folder: {raw_dir}")
    candidates = sorted(
        [p for p in raw_dir.glob("*.csv") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return candidates[0]


def reset_local(local_dir: Path) -> None:
    """Delete local db/chroma/out safely (DANGER)."""
    if local_dir.exists():
        print(f"⚠️ RESET: deleting {local_dir}")
        for name in ["candidates.db", "chroma_db", "out"]:
            target = local_dir / name
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)


def ensure_dirs() -> None:
    LOCAL.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end database → documents → chroma → gemini → results pipeline.")
    ap.add_argument("--all", action="store_true", help="Run the full pipeline in order.")
    ap.add_argument("--reset-local", action="store_true", help="Delete local/candidates.db, local/chroma_db, local/out before running.")

    ap.add_argument("--init-db", action="store_true", help="Initialize SQLite DB from sql/schema.sql (scripts/init_db.py).")
    ap.add_argument("--import-csv", action="store_true", help="Import latest CSV from data/raw into SQLite (scripts/import_csv.py).")
    ap.add_argument("--csv", type=str, default="", help="Optional: explicit CSV path. Otherwise uses newest in data/raw.")

    ap.add_argument("--build-docs", action="store_true", help="Build candidates.jsonl from SQLite (scripts/build_documents.py).")
    ap.add_argument("--index", action="store_true", help="Index candidates.jsonl into Chroma (scripts/index_chroma.py).")
    ap.add_argument("--search", type=str, default="", help="Semantic search query (scripts/search_chroma.py). Writes top500_ids.txt.")
    ap.add_argument("--job", type=str, default="", help="Job description text for gemini batch prompt.")
    ap.add_argument("--prepare-batch", action="store_true", help="Prepare gemini_batch.txt using job + top500 ids.")
    ap.add_argument("--rank", action="store_true", help="Run Gemini ranking (scripts/gemini_rank.py). Requires API key.")
    ap.add_argument("--render", action="store_true", help="Render top50_results.csv/md (scripts/render_results.py).")

    args = ap.parse_args()

    # Expand --all into the canonical sequence
    if args.all:
        args.init_db = True
        args.import_csv = True
        args.build_docs = True
        args.index = True
        if not args.search:
            # default query if user didn't pass one
            args.search = "IAM Consultant One Identity Active Directory"
        args.prepare_batch = True
        # rank is optional (depends on API key), but we include it if a key exists
        if os.getenv("GEMINI_API_KEY"):
            args.rank = True
        args.render = True

    if args.reset_local:
        reset_local(LOCAL)

    ensure_dirs()

    # Resolve CSV path if needed
    csv_path: Path | None = None
    if args.import_csv:
        if args.csv:
            csv_path = Path(args.csv).expanduser().resolve()
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")
        else:
            csv_path = newest_csv(RAW)
        print(f"📄 Using CSV: {csv_path}")

    # --- Run steps ---
    py = sys.executable  # uses current venv python

    if args.init_db:
        run([py, str(SCRIPTS / "init_db.py")], cwd=ROOT)

    if args.import_csv:
        # Your import_csv.py currently has csv_file_path hardcoded at bottom.
        # Better: allow passing CSV path as argv.
        # This runner supports both:
        # - If import_csv.py supports argv -> pass it
        # - Otherwise, it will still run without argv and you must set path in file.
        import_script = SCRIPTS / "import_csv.py"
        if csv_path:
            run([py, str(import_script), str(csv_path)], cwd=ROOT)
        else:
            run([py, str(import_script)], cwd=ROOT)

    if args.build_docs:
        run([py, str(SCRIPTS / "build_documents.py")], cwd=ROOT)

    if args.index:
        run([py, str(SCRIPTS / "index_chroma.py")], cwd=ROOT)

    if args.search:
        run([py, str(SCRIPTS / "search_chroma.py"), args.search], cwd=ROOT)

    if args.prepare_batch:
        if not args.job:
            raise SystemExit("Missing --job. Example: --job \"One Identity IAM Consultant...\"")
        run([py, str(SCRIPTS / "prepare_gemini_batch.py"), args.job], cwd=ROOT)

    if args.rank:
        run([py, str(SCRIPTS / "gemini_rank.py")], cwd=ROOT)

    if args.render:
        run([py, str(SCRIPTS / "render_results.py")], cwd=ROOT)

    print("\n✅ Pipeline complete.")
    print(f"DB:     {DB}")
    print(f"OUT:    {OUT}")
    print(f"CHROMA: {CHROMA}")


if __name__ == "__main__":
    main()