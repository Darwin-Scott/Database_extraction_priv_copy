# scripts/app.py
"""
Minimal Streamlit UI for the pipeline.

Flow:
1) Put CSVs into data/raw -> optionally reset DB -> click "Import CSVs into DB"
2) Enter semantic query + job description + knobs -> click "Run Matching Pipeline"
3) See output table + downloads

Run:
  streamlit run scripts/app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RAW_DIR = ROOT / "data" / "raw"
LOCAL = ROOT / "local"
OUT = LOCAL / "out"
CHROMA_DIR = LOCAL / "chroma_db"

DB_PATH = LOCAL / "candidates.db"
GEMINI_JSON = OUT / "gemini_ranked.json"
GEMINI_RAW = OUT / "gemini_ranked_raw.txt"

# IMPORTANT: your populated collection is candidates_v1 (per inspect_chroma.py)
DEFAULT_COLLECTION = "candidates_v1"

# Ensure repo root is importable so we can `import scripts.search_chroma` in-process.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Gemini model options (curated, + custom) ---
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "Custom…",
]


def _env_utf8() -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    """Run a subprocess command from repo root. Returns (code, combined_output)."""
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_env_utf8(),
    )
    out = (p.stdout or "") + ("\n" if p.stdout and p.stderr else "") + (p.stderr or "")
    return p.returncode, out


def list_csvs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    files = [p for p in raw_dir.glob("*.csv") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)  # oldest -> newest
    return files


def fmt_ts(p: Path) -> str:
    if not p.exists():
        return "—"
    dt = datetime.fromtimestamp(p.stat().st_mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def have_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_resource
def _get_cached_search_handles() -> tuple[Callable[..., object] | None, str]:
    """
    Cache the heavy embedder + chroma client/collection INSIDE the streamlit process.

    This requires `scripts/search_chroma.py` to provide:
      - get_embedder()
      - get_collection(persist_dir=..., collection_name=... OR collection=...)
      - AND either:
          (A) search_chroma_inproc(...) helper, OR
          (B) we fall back to using collection.query(...) directly (best-effort).

    If import fails, we return (None, reason) and the UI will use subprocess mode.
    """
    try:
        import importlib

        sc = importlib.import_module("scripts.search_chroma")

        # If your module offers an explicit inproc function, use it.
        if hasattr(sc, "search_chroma_inproc"):
            # Warm it up by calling get_* if present
            if hasattr(sc, "get_embedder"):
                _ = sc.get_embedder()
            if hasattr(sc, "get_collection"):
                try:
                    _ = sc.get_collection(persist_dir=CHROMA_DIR, collection=DEFAULT_COLLECTION)
                except TypeError:
                    _ = sc.get_collection(persist_dir=CHROMA_DIR, collection_name=DEFAULT_COLLECTION)
            return sc.search_chroma_inproc, "ok"

        # Otherwise, try to build a small wrapper using get_embedder/get_collection
        if hasattr(sc, "get_embedder") and hasattr(sc, "get_collection"):
            embedder = sc.get_embedder()
            try:
                collection = sc.get_collection(persist_dir=CHROMA_DIR, collection=DEFAULT_COLLECTION)
            except TypeError:
                collection = sc.get_collection(persist_dir=CHROMA_DIR, collection_name=DEFAULT_COLLECTION)

            def _fallback_inproc_search(
                *,
                query: str,
                top_k: int,
                print_k: int,
                out_path: Path,
                log: Callable[[str], None] = print,
            ) -> object:
                # Compute embedding in-process
                log("🧠 Using cached embedding model (in-process).")
                q_emb = embedder.encode([query]).tolist()

                log(f"🗃️ Using Chroma collection: {DEFAULT_COLLECTION}")
                res = collection.query(
                    query_embeddings=q_emb,
                    n_results=int(top_k),
                    include=["metadatas", "documents", "distances"],
                )

                ids = (res.get("ids") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]

                # Print preview
                log("\n🔎 Query:")
                log(query)
                log(f"\n✅ Top {min(len(ids), int(print_k))} results:\n")
                for i, (cid, dist) in enumerate(list(zip(ids, dists))[: int(print_k)], start=1):
                    log(f"{i:02d}. {cid} | distance={dist:.4f}")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
                log(f"\n✅ Wrote {len(ids)} IDs to {out_path}")
                return res

            return _fallback_inproc_search, "ok"

        return None, "search_chroma.py missing expected helpers (get_embedder/get_collection or search_chroma_inproc)."
    except Exception as e:
        return None, f"import failed: {e}"


st.set_page_config(page_title="Recruiting Pipeline", layout="wide")
st.title("Recruiting Pipeline")
st.caption("Simple UI: import CSVs → run matching pipeline → view results.")

# --- Sidebar status ---
with st.sidebar:
    st.header("Status")
    st.write(f"Repo root: `{ROOT}`")
    st.write(f"DB exists: {'✅' if DB_PATH.exists() else '❌'}")
    st.write(f"Chroma dir: `{CHROMA_DIR}` {'✅' if CHROMA_DIR.exists() else '❌'}")
    st.write(f"Default collection: `{DEFAULT_COLLECTION}`")
    st.write(f"Output dir: `{OUT}`")
    st.write(f"Gemini JSON exists: {'✅' if GEMINI_JSON.exists() else '❌'}")
    st.write(f"Gemini RAW exists: {'✅' if GEMINI_RAW.exists() else '❌'}")
    st.write(f"API key present: {'✅' if have_api_key() else '❌'}")

st.divider()

# Log area
log = st.empty()
if "log" not in st.session_state:
    st.session_state["log"] = ""


def append_log(s: str) -> None:
    st.session_state["log"] += s.rstrip() + "\n"
    log.code(st.session_state["log"], language="text")


py = sys.executable

# ========== STEP 1 ==========
st.subheader("1) Import CSVs into DB")
st.write("Put one or more `.csv` files into `data/raw/`, then click the button below.")

csvs = list_csvs(RAW_DIR)
if not csvs:
    st.warning(f"No CSVs found in {RAW_DIR}")
else:
    newest = max(csvs, key=lambda p: p.stat().st_mtime)
    st.info(f"Found {len(csvs)} CSV(s) in {RAW_DIR}. Newest: `{newest.name}`")

c_reset, c_btn = st.columns([1, 2], vertical_alignment="center")
with c_reset:
    reset_db = st.checkbox("Reset DB before import (DANGER)", value=False)
with c_btn:
    import_btn = st.button("📥 Import CSVs into DB", use_container_width=True)

if import_btn:
    if not csvs:
        st.error("No CSV files to import.")
        st.stop()

    OUT.mkdir(parents=True, exist_ok=True)

    # Optional DB reset
    if reset_db:
        append_log("=== RESET DB (delete local/candidates.db) ===")
        if DB_PATH.exists():
            try:
                DB_PATH.unlink()
                append_log(f"Deleted: {DB_PATH}")
            except Exception as e:
                append_log(f"Failed deleting DB: {e}")
                st.error("Could not delete DB. See logs.")
                st.stop()
        else:
            append_log("DB did not exist; nothing to delete.")

    # Ensure DB exists: init_db if missing
    if not DB_PATH.exists():
        append_log("=== init_db.py ===")
        rc, out = run_cmd([py, str(SCRIPTS / "init_db.py")])
        append_log(out)
        if rc != 0:
            st.error("init_db.py failed. See logs.")
            st.stop()

    # Import ALL CSVs (oldest -> newest so newer updates can overwrite)
    for p in csvs:
        append_log(f"=== import_csv.py {p.name} ===")
        rc, out = run_cmd([py, str(SCRIPTS / "import_csv.py"), str(p)])
        append_log(out)
        if rc != 0:
            st.error(f"Import failed for {p.name}. See logs.")
            st.stop()

    append_log("=== feature_engineering.py ===")
    rc, out = run_cmd([py, str(SCRIPTS / "feature_engineering.py")])
    append_log(out)
    if rc != 0:
        st.error("feature_engineering.py failed. See logs.")
        st.stop()

    st.success("CSV import + feature engineering finished.")

st.divider()

# ========== STEP 2 ==========
st.subheader("2) Run Matching Pipeline")

search_query = st.text_area(
    "Semantic search query",
    value="One Identity IAM Consultant Active Directory",
    height=80,
)

job_desc = st.text_area(
    "Job description",
    value="One Identity IAM Consultant. Requirements: One Identity Manager, AD integration, SQL, scripting, troubleshooting, German/English.",
    height=140,
)

st.markdown("#### Search + prompt knobs")

kcol1, kcol2, kcol3, kcol4 = st.columns(4)
with kcol1:
    top_k = st.number_input("Chroma n_results (top_k)", min_value=10, max_value=5000, value=800, step=50)
with kcol2:
    print_k = st.number_input("Print preview (print_k)", min_value=0, max_value=200, value=20, step=5)
with kcol3:
    top_n = st.number_input("IDs into prompt (top_n)", min_value=10, max_value=5000, value=300, step=50)
with kcol4:
    rank_n = st.number_input("Rank output size (rank_n)", min_value=10, max_value=200, value=50, step=10)

ecol1, ecol2, ecol3 = st.columns(3)
with ecol1:
    explain_n = st.number_input("Explain count (explain_n)", min_value=0, max_value=200, value=30, step=5)
with ecol2:
    ids_filename = st.text_input("IDs filename", value=f"top{int(top_k)}_ids.txt")
with ecol3:
    collection_name = st.text_input("Chroma collection", value=DEFAULT_COLLECTION)

ids_path = OUT / ids_filename

mcol1, mcol2 = st.columns([1, 2], vertical_alignment="center")
with mcol1:
    use_gemini = st.checkbox("Use Gemini ranking (costs credits)", value=False)
with mcol2:
    selected_model = st.selectbox("Gemini model", options=GEMINI_MODELS, index=6)

cache_search = st.checkbox("Cache embedding model (in-process search, no reload)", value=True)

custom_model = ""
if selected_model == "Custom…":
    custom_model = st.text_input("Custom model name", value="gemini-2.5-flash-lite-preview-09-2025")

run_btn = st.button("▶️ Run Matching Pipeline", use_container_width=True)

if run_btn:
    if not search_query.strip() or not job_desc.strip():
        st.error("Please fill in both the search query and the job description.")
        st.stop()

    OUT.mkdir(parents=True, exist_ok=True)

    model_name = custom_model.strip() if selected_model == "Custom…" else selected_model

    # --- SEARCH STEP ---
    append_log("=== search_chroma ===")
    append_log(f"cache_search={cache_search} | top_k={int(top_k)} | print_k={int(print_k)}")
    append_log(f"persist_dir={CHROMA_DIR}")
    append_log(f"collection={collection_name}")
    append_log(f"out={ids_path}")
    append_log(f"query={search_query.strip()}")

    ran_inproc = False
    if cache_search:
        inproc_fn, status = _get_cached_search_handles()
        if inproc_fn is None:
            append_log(f"⚠️ in-proc search unavailable ({status}); using subprocess.")
        else:
            try:
                buf: list[str] = []

                def _log(s: str) -> None:
                    buf.append(s)

                # If module provides search_chroma_inproc, it likely accepts collection; try both common names.
                try:
                    inproc_fn(
                        query=search_query.strip(),
                        persist_dir=CHROMA_DIR,
                        collection=collection_name,
                        top_k=int(top_k),
                        print_k=int(print_k),
                        out_path=ids_path,
                        log=_log,
                    )
                except TypeError:
                    # Wrapper variant might not accept collection (already bound to DEFAULT_COLLECTION)
                    inproc_fn(
                        query=search_query.strip(),
                        top_k=int(top_k),
                        print_k=int(print_k),
                        out_path=ids_path,
                        log=_log,
                    )

                append_log("\n".join(buf))
                ran_inproc = True
            except Exception as e:
                append_log(f"⚠️ in-proc search failed (falling back to subprocess): {e}")

    if not ran_inproc:
        # Subprocess MUST pass correct collection, otherwise you hit the empty `candidates` collection.
        cmd = [
            py,
            str(SCRIPTS / "search_chroma.py"),
            "--persist-dir",
            str(CHROMA_DIR),
            "--collection",
            collection_name,
            "--top-k",
            str(int(top_k)),
            "--print-k",
            str(int(print_k)),
            "--out",
            str(ids_path),
            search_query.strip(),
        ]
        append_log(" ".join(cmd))
        rc, out = run_cmd(cmd)
        append_log(out)
        if rc != 0:
            st.error("search_chroma.py failed. See logs.")
            st.stop()

    # --- REST OF STEPS AS SUBPROCESSES ---
    steps: list[tuple[str, list[str]]] = [
        (
            "prepare_gemini_batch.py",
            [
                py,
                str(SCRIPTS / "prepare_gemini_batch.py"),
                "--ids-path",
                str(ids_path),
                "--top-n",
                str(int(top_n)),
                "--rank-n",
                str(int(rank_n)),
                "--explain-n",
                str(int(explain_n)),
                "--job",
                job_desc.strip(),
            ],
        ),
    ]

    if use_gemini:
        if not have_api_key():
            st.error("Gemini is enabled, but no GEMINI_API_KEY / GOOGLE_API_KEY found in env.")
            st.stop()

        steps.append(
            (
                "gemini_rank.py",
                [
                    py,
                    str(SCRIPTS / "gemini_rank.py"),
                    "--model",
                    model_name,
                    "--rank-n",
                    str(int(rank_n)),
                ],
            )
        )
    else:
        if not GEMINI_JSON.exists():
            st.warning(
                "Gemini is OFF, but `local/out/gemini_ranked.json` is missing. "
                "Either enable Gemini once, or provide a mock gemini_ranked.json."
            )

    steps.append(
        (
            "render_results.py",
            [
                py,
                str(SCRIPTS / "render_results.py"),
                "--rank-key",
                f"top{rank_n}",
            ],
        )
    )

    for name, cmd in steps:
        append_log(f"=== {name} ===")
        append_log(" ".join(cmd))
        rc, out = run_cmd(cmd)
        append_log(out)
        if rc != 0:
            st.error(f"{name} failed. See logs.")
            st.stop()

    st.success("Pipeline completed.")

st.divider()

# ========== STEP 3 ==========
st.subheader("3) Results")

candidate_csvs = sorted(OUT.glob("top*_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
latest_csv = candidate_csvs[0] if candidate_csvs else (OUT / "top50_results.csv")

df = safe_read_csv(latest_csv)
if df is None:
    st.info("No results yet. Run the pipeline to generate a `top*_results.csv` in `local/out/`.")
else:
    st.info(f"Showing latest results: `{latest_csv.name}` (modified: {fmt_ts(latest_csv)})")
    st.dataframe(df, use_container_width=True, hide_index=True)

    latest_md = latest_csv.with_suffix(".md")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=latest_csv.read_bytes(),
            file_name=latest_csv.name,
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        if latest_md.exists():
            st.download_button(
                "⬇️ Download Markdown",
                data=latest_md.read_bytes(),
                file_name=latest_md.name,
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("Markdown missing", disabled=True, use_container_width=True)
    with c3:
        if GEMINI_JSON.exists():
            st.download_button(
                "⬇️ Download gemini_ranked.json",
                data=GEMINI_JSON.read_bytes(),
                file_name=GEMINI_JSON.name,
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.button("gemini_ranked.json missing", disabled=True, use_container_width=True)
    with c4:
        if GEMINI_RAW.exists():
            st.download_button(
                "⬇️ Download gemini_ranked_raw.txt",
                data=GEMINI_RAW.read_bytes(),
                file_name=GEMINI_RAW.name,
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("gemini_ranked_raw.txt missing", disabled=True, use_container_width=True)
