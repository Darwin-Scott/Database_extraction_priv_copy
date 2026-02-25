# scripts/app.py
"""
Minimal Streamlit UI for the pipeline.

Flow:
1) Put CSVs into data/raw -> optionally reset DB -> click "Import CSVs into DB"
2) Enter semantic query + job description + gemini model -> click "Run Matching Pipeline"
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

import pandas as pd
import streamlit as st

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RAW_DIR = ROOT / "data" / "raw"
LOCAL = ROOT / "local"
OUT = LOCAL / "out"

DB_PATH = LOCAL / "candidates.db"
CSV_OUT = OUT / "top50_results.csv"
MD_OUT = OUT / "top50_results.md"
GEMINI_JSON = OUT / "gemini_ranked.json"


# --- Gemini model options (curated, + custom) ---
# These names may evolve; "Custom…" lets you paste any model name.
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


def newest_csvs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    files = [p for p in raw_dir.glob("*.csv") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def fmt_ts(p: Path) -> str:
    if not p.exists():
        return "—"
    dt = datetime.fromtimestamp(p.stat().st_mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def load_results() -> pd.DataFrame | None:
    if not CSV_OUT.exists():
        return None
    try:
        return pd.read_csv(CSV_OUT)
    except Exception:
        return None


def have_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


st.set_page_config(page_title="Recruiting Pipeline", layout="wide")
st.title("Recruiting Pipeline")
st.caption("Simple UI: import CSVs → run matching pipeline → view results.")

# --- Sidebar status ---
with st.sidebar:
    st.header("Status")
    st.write(f"Repo root: `{ROOT}`")
    st.write(f"DB exists: {'✅' if DB_PATH.exists() else '❌'}")
    st.write(f"Last results CSV: `{fmt_ts(CSV_OUT)}`")
    st.write(f"Gemini JSON exists: {'✅' if GEMINI_JSON.exists() else '❌'}")
    st.write(f"Output dir: `{OUT}`")
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

csvs = newest_csvs(RAW_DIR)
if not csvs:
    st.warning(f"No CSVs found in {RAW_DIR}")
else:
    st.info(f"Found {len(csvs)} CSV(s) in {RAW_DIR}. Newest: `{csvs[0].name}`")

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

    # Import ALL CSVs (oldest first so newer updates can overwrite)
    # Assumes: python scripts/import_csv.py <path>
    for p in reversed(csvs):
        append_log(f"=== import_csv.py {p.name} ===")
        rc, out = run_cmd([py, str(SCRIPTS / "import_csv.py"), str(p)])
        append_log(out)
        if rc != 0:
            st.error(f"Import failed for {p.name}. See logs.")
            st.stop()

    st.success("CSV import finished.")

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
    height=120,
)

mcol1, mcol2 = st.columns([1, 2], vertical_alignment="center")
with mcol1:
    use_gemini = st.checkbox("Use Gemini ranking (costs credits)", value=False)
with mcol2:
    selected_model = st.selectbox("Gemini model", options=GEMINI_MODELS, index=6)

custom_model = ""
if selected_model == "Custom…":
    custom_model = st.text_input("Custom model name", value="gemini-2.5-flash-lite")

run_btn = st.button("▶️ Run Matching Pipeline", use_container_width=True)

if run_btn:
    if not search_query.strip() or not job_desc.strip():
        st.error("Please fill in both the search query and the job description.")
        st.stop()

    OUT.mkdir(parents=True, exist_ok=True)

    # Decide model name
    model_name = custom_model.strip() if selected_model == "Custom…" else selected_model

    # Search -> batch -> (optional gemini) -> render
    steps: list[tuple[str, list[str]]] = [
        ("search_chroma.py", [py, str(SCRIPTS / "search_chroma.py"), search_query.strip()]),
        ("prepare_gemini_batch.py", [py, str(SCRIPTS / "prepare_gemini_batch.py"), job_desc.strip()]),
    ]

    if use_gemini:
        if not have_api_key():
            st.error("Gemini is enabled, but no GEMINI_API_KEY / GOOGLE_API_KEY found in env.")
            st.stop()
        steps.append(("gemini_rank.py", [py, str(SCRIPTS / "gemini_rank.py"), "--model", model_name]))
    else:
        if not GEMINI_JSON.exists():
            st.warning(
                "Gemini is OFF, but `local/out/gemini_ranked.json` is missing. "
                "Either enable Gemini once, or provide a mock gemini_ranked.json."
            )

    steps.append(("render_results.py", [py, str(SCRIPTS / "render_results.py")]))

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
df = load_results()
if df is None:
    st.info("No results yet. Run the pipeline to generate `local/out/top50_results.csv`.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=CSV_OUT.read_bytes(),
            file_name=CSV_OUT.name,
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        if MD_OUT.exists():
            st.download_button(
                "⬇️ Download Markdown",
                data=MD_OUT.read_bytes(),
                file_name=MD_OUT.name,
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