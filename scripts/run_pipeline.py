# scripts/run_pipeline.py
"""
End-to-end pipeline runner.

Examples:

# Full rebuild (DANGER: resets local DB/Chroma/out) + run everything
python scripts/run_pipeline.py --all --reset-local --search "One Identity IAM Consultant Active Directory" --job "One Identity IAM Consultant. Requirements: One Identity Manager, AD integration, SQL, scripting, troubleshooting, German/English."

# Run only search + prepare batch + render (assumes db+chroma exist; no gemini call)
python scripts/run_pipeline.py --search "One Identity IAM Consultant Active Directory" --job "..." --prepare-batch --gemini skip --render

# Run with mock gemini outputs (no credits)
python scripts/run_pipeline.py --search "One Identity IAM Consultant Active Directory" --job "..." --prepare-batch --gemini mock --render

# Run with a specific mock JSON file (no credits)
python scripts/run_pipeline.py --gemini mock --mock-json local/out/mock/my_mock_ranked.json --render

# Only check whether Gemini can be called (billing/quota sanity)
python scripts/run_pipeline.py --ping-gemini

Notes:
- This runner calls existing scripts as subprocesses.
- Folder layout assumed:
  scripts/, src/dbx/paths.py, local/, data/raw/, sql/
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

MOCK_DIR = OUT / "mock"
MOCK_JSON_DEFAULT = MOCK_DIR / "gemini_ranked.json"
MOCK_RAW_DEFAULT = MOCK_DIR / "gemini_ranked_raw.txt"

REAL_JSON = OUT / "gemini_ranked.json"
REAL_RAW = OUT / "gemini_ranked_raw.txt"


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
        print(f"⚠️ RESET: deleting selected items under {local_dir}")
        for name in ["candidates.db", "chroma_db", "out"]:
            target = local_dir / name
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)


def ensure_dirs() -> None:
    LOCAL.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    MOCK_DIR.mkdir(parents=True, exist_ok=True)


def have_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def apply_mock_outputs(mock_json: Path | None = None, mock_raw: Path | None = None) -> None:
    """
    Copy mock gemini outputs into the 'real' output locations.

    Priority:
    1) If mock_json is provided: copy it -> REAL_JSON and mirror to REAL_RAW
    2) Else: use default mock folder files (MOCK_JSON_DEFAULT + MOCK_RAW_DEFAULT)
    """
    if mock_json:
        if not mock_json.exists():
            raise FileNotFoundError(f"Mock JSON not found: {mock_json}")
        REAL_JSON.write_text(mock_json.read_text(encoding="utf-8"), encoding="utf-8")
        # mirror raw as the same JSON content (good enough for pipeline)
        REAL_RAW.write_text(mock_json.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✅ Using MOCK Gemini JSON:\n  {mock_json}\n→ {REAL_JSON}")
        return

    # Default mock directory mode:
    if not MOCK_JSON_DEFAULT.exists() or not MOCK_RAW_DEFAULT.exists():
        raise FileNotFoundError(
            "Missing mock outputs. Expected:\n"
            f"  {MOCK_JSON_DEFAULT}\n"
            f"  {MOCK_RAW_DEFAULT}\n"
            "Create them by saving one successful Gemini run into local/out/mock/.\n"
            "Or pass --mock-json <path_to_mock.json>."
        )

    REAL_JSON.write_text(MOCK_JSON_DEFAULT.read_text(encoding="utf-8"), encoding="utf-8")
    REAL_RAW.write_text(MOCK_RAW_DEFAULT.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"✅ Using MOCK Gemini outputs:\n  {REAL_JSON}\n  {REAL_RAW}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end DB → docs → chroma → gemini → results pipeline.")
    ap.add_argument("--all", action="store_true", help="Run the full pipeline in order.")
    ap.add_argument("--reset-local", action="store_true", help="Delete local/candidates.db, local/chroma_db, local/out before running.")

    ap.add_argument("--init-db", action="store_true", help="Initialize SQLite DB (scripts/init_db.py).")
    ap.add_argument("--import-csv", action="store_true", help="Import latest CSV from data/raw into SQLite (scripts/import_csv.py).")
    ap.add_argument("--csv", type=str, default="", help="Optional explicit CSV path. Otherwise newest in data/raw.")

    ap.add_argument("--build-docs", action="store_true", help="Build candidates.jsonl from SQLite (scripts/build_documents.py).")
    ap.add_argument("--index", action="store_true", help="Index candidates.jsonl into Chroma (scripts/index_chroma.py).")
    ap.add_argument("--search", type=str, default="", help="Semantic search query (scripts/search_chroma.py). Writes top500_ids.txt.")
    ap.add_argument("--job", type=str, default="", help="Job description text for gemini batch prompt.")
    ap.add_argument("--prepare-batch", action="store_true", help="Prepare gemini_batch.txt using job + top500 ids.")
    ap.add_argument("--render", action="store_true", help="Render top50_results.csv/md (scripts/render_results.py).")

    ap.add_argument(
        "--gemini",
        choices=["auto", "real", "mock", "skip"],
        default="auto",
        help=(
            "Gemini stage mode:\n"
            " auto = run real only if API key exists, else skip\n"
            " real = always call Gemini (requires API key)\n"
            " mock = copy local/out/mock/* into local/out/* (no API call)\n"
            " skip = do not call Gemini (expects local/out/gemini_ranked.json already exists)"
        ),
    )
    ap.add_argument("--mock-json", type=str, default="", help="When --gemini mock: use this JSON file as gemini_ranked.json.")
    ap.add_argument("--ping-gemini", action="store_true", help="Run scripts/gemini_rank.py --ping and exit.")

    args = ap.parse_args()

    # Expand --all into the canonical sequence
    if args.all:
        args.init_db = True
        args.import_csv = True
        args.build_docs = True
        args.index = True
        if not args.search:
            args.search = "IAM Consultant One Identity Active Directory"
        args.prepare_batch = True
        args.render = True
        # Gemini behavior handled by --gemini (default auto)

    if args.reset_local:
        reset_local(LOCAL)

    ensure_dirs()

    py = sys.executable  # uses current venv python

    # Optional: just test gemini connectivity/billing and exit
    if args.ping_gemini:
        run([py, str(SCRIPTS / "gemini_rank.py"), "--ping"], cwd=ROOT)
        print("\n✅ Gemini ping completed.")
        return

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
    if args.init_db:
        run([py, str(SCRIPTS / "init_db.py")], cwd=ROOT)

    if args.import_csv:
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
            raise SystemExit('Missing --job. Example: --job "One Identity IAM Consultant..."')
        run([py, str(SCRIPTS / "prepare_gemini_batch.py"), args.job], cwd=ROOT)

    # --- Gemini stage ---
    gemini_mode = args.gemini
    if gemini_mode == "auto":
        gemini_mode = "real" if have_api_key() else "skip"

    if gemini_mode == "real":
        if not have_api_key():
            raise SystemExit("Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it or use --gemini mock/skip.")
        run([py, str(SCRIPTS / "gemini_rank.py")], cwd=ROOT)

    elif gemini_mode == "mock":
        mock_json_path = Path(args.mock_json).expanduser().resolve() if args.mock_json else None
        apply_mock_outputs(mock_json=mock_json_path)

    elif gemini_mode == "skip":
        if not REAL_JSON.exists():
            print("⚠️ Skipping Gemini, but no gemini_ranked.json found.")
            print(f"Expected: {REAL_JSON}")
            print("Either run Gemini once, or use --gemini mock with mock files present.")
            raise SystemExit(2)

    # --- Render ---
    if args.render:
        run([py, str(SCRIPTS / "render_results.py")], cwd=ROOT)

    print("\n✅ Pipeline complete.")
    print(f"DB:     {DB}")
    print(f"OUT:    {OUT}")
    print(f"CHROMA: {CHROMA}")
    print(f"Gemini: {args.gemini} (effective mode: {gemini_mode})")


if __name__ == "__main__":
    main()