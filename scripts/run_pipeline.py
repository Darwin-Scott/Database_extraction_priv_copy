# scripts/run_pipeline.py
"""
End-to-end pipeline runner.

Now supports configurable:
- Chroma search top-k (how many IDs to write)
- Gemini prompt input top-n (how many candidate lines to include)
- Gemini output rank-n + explain-n (how many results in JSON + how many reasons)

Examples:

# Full rebuild + run everything, 800 IDs, include 800 in prompt, ask Gemini for top100
python scripts/run_pipeline.py --all --reset-local --import-all-csvs \
  --top-k 800 --top-n 800 --rank-n 100 --explain-n 30 \
  --search "One Identity IAM Consultant Active Directory" \
  --job "..."

# Use existing DB/Chroma, just search+prepare+gemini+render
python scripts/run_pipeline.py --search "..." --prepare-batch --gemini real --render \
  --top-k 500 --top-n 300 --rank-n 50 --explain-n 20 --job "..."
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
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


def all_csvs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw data folder: {raw_dir}")
    files = [p for p in raw_dir.glob("*.csv") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)  # oldest -> newest
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return files


def reset_local(local_dir: Path) -> None:
    if not local_dir.exists():
        return

    print(f"⚠️ RESET: deleting selected items under {local_dir}")

    for name in ["candidates.db", "chroma_db"]:
        target = local_dir / name
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    out_dir = local_dir / "out"
    if out_dir.exists() and out_dir.is_dir():
        keep_dirs = {"model_comparison", "model_comparison_bundle", "recruiter_packets", "mock"}
        for p in out_dir.iterdir():
            if p.is_dir() and p.name in keep_dirs:
                continue
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)


def ensure_dirs() -> None:
    LOCAL.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    MOCK_DIR.mkdir(parents=True, exist_ok=True)


def have_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def apply_mock_outputs(mock_json: Path | None = None) -> None:
    if mock_json:
        if not mock_json.exists():
            raise FileNotFoundError(f"Mock JSON not found: {mock_json}")
        REAL_JSON.write_text(mock_json.read_text(encoding="utf-8"), encoding="utf-8")
        REAL_RAW.write_text(mock_json.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✅ Using MOCK Gemini JSON:\n  {mock_json}\n→ {REAL_JSON}")
        return

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
    ap.add_argument("--import-all-csvs", action="store_true", help="Import ALL CSVs from data/raw (oldest->newest).")

    ap.add_argument("--build-docs", action="store_true", help="Build candidates.jsonl from SQLite (scripts/build_documents.py).")
    ap.add_argument("--index", action="store_true", help="Index candidates.jsonl into Chroma (scripts/index_chroma.py).")
    ap.add_argument("--feature-engineering", action="store_true", help="Build candidate_profile_text and candidate_rank_features from raw imports.")

    ap.add_argument("--search", type=str, default="", help="Semantic search query (scripts/search_chroma.py). Writes top{K}_ids.txt.")
    ap.add_argument("--job", type=str, default="", help="Job description text for gemini batch prompt.")
    ap.add_argument("--prepare-batch", action="store_true", help="Prepare gemini_batch.txt using job + ids.")
    ap.add_argument("--render", action="store_true", help="Render topN_results.csv/md (scripts/render_results.py).")
    # add near other args
    ap.add_argument(
        "--search-mode",
        choices=["subprocess", "inproc"],
        default="subprocess",
        help="How to run the Chroma search step. inproc avoids spawning a new Python process (minor win).",
    )

    # NEW knobs
    ap.add_argument("--top-k", type=int, default=500, help="How many IDs search_chroma writes (default 500).")
    ap.add_argument("--print-k", type=int, default=20, help="How many results search_chroma prints (default 20).")
    ap.add_argument("--ids-path", type=str, default="", help="Optional path to ids file to use for prepare step.")
    ap.add_argument("--top-n", type=int, default=500, help="How many IDs from the ids file to include in Gemini prompt.")
    ap.add_argument("--rank-n", type=int, default=50, help="How many results Gemini should output in JSON (top{N}).")
    ap.add_argument("--explain-n", type=int, default=20, help="How many top candidates should include explanations.")

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

    if args.all:
        args.init_db = True
        args.import_csv = True
        args.feature_engineering = True
        args.build_docs = True
        args.index = True
        if not args.search:
            args.search = "IAM Consultant One Identity Active Directory"
        args.prepare_batch = True
        args.render = True

    if args.reset_local:
        reset_local(LOCAL)

    ensure_dirs()

    py = sys.executable

    if args.ping_gemini:
        run([py, str(SCRIPTS / "gemini_rank.py"), "--ping"], cwd=ROOT)
        print("\n✅ Gemini ping completed.")
        return

    # CSV paths
    csv_paths: list[Path] = []
    if args.import_csv:
        if args.import_all_csvs:
            csv_paths = all_csvs(RAW)
            print(f"📄 Using ALL CSVs ({len(csv_paths)}):")
            for p in csv_paths:
                print(f"  - {p.name}")
        else:
            if args.csv:
                p = Path(args.csv).expanduser().resolve()
                if not p.exists():
                    raise FileNotFoundError(f"CSV not found: {p}")
                csv_paths = [p]
            else:
                csv_paths = [newest_csv(RAW)]
            print(f"📄 Using CSV: {csv_paths[0]}")

    # --- Run steps ---
    if args.init_db:
        run([py, str(SCRIPTS / "init_db.py")], cwd=ROOT)

    if args.import_csv:
        import_script = SCRIPTS / "import_csv.py"
        for p in csv_paths:
            run([py, str(import_script), str(p)], cwd=ROOT)

    if args.feature_engineering:
        run([py, str(SCRIPTS / "feature_engineering.py")], cwd=ROOT)

    if args.build_docs:
        run([py, str(SCRIPTS / "build_documents.py")], cwd=ROOT)

    if args.index:
        run([py, str(SCRIPTS / "index_chroma.py")], cwd=ROOT)

    # Search step: write top{K}_ids.txt unless ids-path is provided and you want to skip search
    ids_path_effective = ""
    if args.search:
        out_name = f"top{int(args.top_k)}_ids.txt"
        out_path = OUT / out_name

        if args.search_mode == "inproc":
            # Run in-process (won't persist across separate CLI invocations, but avoids extra subprocess)
            from scripts.search_chroma import search_chroma_inproc

            search_chroma_inproc(
                query=args.search,
                persist_dir=CHROMA,              # local/chroma_db
                top_k=int(args.top_k),
                print_k=int(args.print_k),
                out_path=out_path,
                log=print,
            )
        else:
            # Run as subprocess (current behavior)
            run(
                [
                    py,
                    str(SCRIPTS / "search_chroma.py"),
                    "--top-k",
                    str(int(args.top_k)),
                    "--print-k",
                    str(int(args.print_k)),
                    "--out",
                    str(out_path),
                    args.search,
                ],
                cwd=ROOT,
            )

        ids_path_effective = str(out_path)

    elif args.ids_path:
        ids_path_effective = args.ids_path

    if args.prepare_batch:
        if not args.job:
            raise SystemExit('Missing --job. Example: --job "One Identity IAM Consultant..."')

        cmd = [py, str(SCRIPTS / "prepare_gemini_batch.py"), "--top-n", str(int(args.top_n)), "--rank-n", str(int(args.rank_n)), "--explain-n", str(int(args.explain_n)), "--job", args.job]
        if ids_path_effective:
            cmd += ["--ids-path", ids_path_effective]
        run(cmd, cwd=ROOT)

    # --- Gemini stage ---
    gemini_mode = args.gemini
    if gemini_mode == "auto":
        gemini_mode = "real" if have_api_key() else "skip"

    if gemini_mode == "real":
        if not have_api_key():
            raise SystemExit("Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it or use --gemini mock/skip.")
        run([py, str(SCRIPTS / "gemini_rank.py"), "--rank-n", str(int(args.rank_n))], cwd=ROOT)

    elif gemini_mode == "mock":
        mock_json_path = Path(args.mock_json).expanduser().resolve() if args.mock_json else None
        apply_mock_outputs(mock_json=mock_json_path)

    elif gemini_mode == "skip":
        if not REAL_JSON.exists():
            print("⚠️ Skipping Gemini, but no gemini_ranked.json found.")
            print(f"Expected: {REAL_JSON}")
            print("Either run Gemini once, or use --gemini mock with mock files present.")
            raise SystemExit(2)

    if args.render:
        run([py, str(SCRIPTS / "render_results.py")], cwd=ROOT)

    print("\n✅ Pipeline complete.")
    print(f"DB:     {DB}")
    print(f"OUT:    {OUT}")
    print(f"CHROMA: {CHROMA}")
    print(f"Gemini: {args.gemini} (effective mode: {gemini_mode})")


if __name__ == "__main__":
    main()
