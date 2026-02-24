# scripts/prepare_gemini_batch.py
"""
Prepare a compact, anonymized batch payload for Gemini deep matching (Stage 2).

Inputs:
- local/candidates.db
- local/out/top500_ids.txt (from search_chroma.py)
- job description text (CLI arg or prompted)

Outputs:
- local/out/gemini_batch.txt                (prompt-ready text)
- local/out/gemini_candidates_compact.jsonl (one candidate per line for debugging)

Usage:
  python scripts/prepare_gemini_batch.py --job "JOB DESCRIPTION HERE..."
  python scripts/prepare_gemini_batch.py "JOB DESCRIPTION HERE..."
  python scripts/prepare_gemini_batch.py   (will prompt for a job description)

Notes:
- No PII is included (no names, no URLs, no emails/phones, no messages).
- Uses candidate_profile_text only.
- Cleans skills like "Skill : null" or "Skill : 3".
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dbx.paths import DB, OUT

DB_PATH = DB
TOP_IDS_PATH = OUT / "top500_ids.txt"

OUT_DIR = OUT
OUT_PROMPT_PATH = OUT_DIR / "gemini_batch.txt"
OUT_JSONL_PATH = OUT_DIR / "gemini_candidates_compact.jsonl"

# compactness knobs
MAX_SUMMARY_CHARS = 400
MAX_WORK_ITEMS = 4
MAX_WORK_DESC_CHARS = 200
MAX_EDU_ITEMS = 2
MAX_SKILLS = 18
MAX_INFERRED_SKILLS = 6

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
CONTACT_HINT_RE = re.compile(r"\b(kontakt|contact|mail|e-mail|email|telefon|phone|mobil|whatsapp)\b", re.IGNORECASE)


def scrub_pii(text: str) -> str:
    text = URL_RE.sub("[URL_REMOVED]", text)
    text = EMAIL_RE.sub("[EMAIL_REMOVED]", text)
    text = CONTACT_HINT_RE.sub("[CONTACT_REMOVED]", text)
    return text


def norm_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clip(s: str, max_chars: int) -> str:
    s = norm_ws(s)
    return s if len(s) <= max_chars else (s[: max_chars - 1].rstrip() + "…")


def safe_load_json(s: Optional[str], default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def clean_skill_token(tok: str) -> Optional[str]:
    tok = tok.strip()
    if not tok:
        return None
    # remove trailing ": null" / ": none" / ": 3"
    tok = re.sub(r"\s*:\s*(null|none|\d+)\s*$", "", tok, flags=re.IGNORECASE)
    tok = tok.strip(" -–•\t")
    tok = norm_ws(tok)
    return tok if tok else None


def parse_skills(skills_raw: Optional[str], limit: int) -> List[str]:
    if not skills_raw:
        return []
    parts = skills_raw.split(",")
    out: List[str] = []
    seen = set()
    for p in parts:
        c = clean_skill_token(p)
        if not c:
            continue
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def parse_inferred_skills(inferred: Optional[str], limit: int) -> List[str]:
    if not inferred:
        return []
    parts = re.split(r"[,\n;]+", inferred)
    out: List[str] = []
    seen = set()
    for p in parts:
        c = clean_skill_token(p)
        if not c:
            continue
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def fmt_languages(languages_json: Optional[str]) -> str:
    langs = safe_load_json(languages_json, [])
    if not langs:
        return ""
    out = []
    for entry in langs:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        prof = entry.get("proficiency")
        if not name:
            continue
        if prof:
            out.append(f"{norm_ws(str(name))}({norm_ws(str(prof))})")
        else:
            out.append(norm_ws(str(name)))
    return ", ".join(out)


def fmt_work(work_json: Optional[str], limit: int) -> str:
    work = safe_load_json(work_json, [])
    if not work:
        return ""
    items = []
    for w in work[:limit]:
        if not isinstance(w, dict):
            continue
        org = w.get("organization")
        title = w.get("title")
        desc = w.get("description")

        parts = []
        if org:
            parts.append(norm_ws(str(org)))
        if title:
            parts.append(norm_ws(str(title)))

        base = "/".join(parts) if parts else ""
        if desc:
            d = clip(str(desc), MAX_WORK_DESC_CHARS)
            if d:
                base = f"{base}:{d}" if base else d

        if base:
            items.append(base)
    return " ; ".join(items)


def fmt_edu(edu_json: Optional[str], limit: int) -> str:
    edu = safe_load_json(edu_json, [])
    if not edu:
        return ""
    items = []
    for e in edu[:limit]:
        if not isinstance(e, dict):
            continue
        degree = e.get("degree")
        fos = e.get("fos")
        school = e.get("school")

        parts = []
        if degree:
            parts.append(norm_ws(str(degree)))
        if fos:
            parts.append(norm_ws(str(fos)))
        if school:
            parts.append(norm_ws(str(school)))

        item = "/".join(parts)
        if item:
            items.append(item)
    return " ; ".join(items)


def load_top_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/search_chroma.py first.")
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(line)

    # de-dupe while keeping order
    seen = set()
    out = []
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def fetch_profiles(conn: sqlite3.Connection, cand_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping cand_id -> row dict from candidate_profile_text
    """
    if not cand_ids:
        return {}

    placeholders = ",".join(["?"] * len(cand_ids))
    query = f"""
        SELECT cand_id, headline, summary, skills_raw, languages_json, work_history_json, education_json, inferred_skills
        FROM candidate_profile_text
        WHERE cand_id IN ({placeholders})
    """
    rows = conn.execute(query, cand_ids).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cand_id = r[0]
        out[cand_id] = {
            "cand_id": r[0],
            "headline": r[1],
            "summary": r[2],
            "skills_raw": r[3],
            "languages_json": r[4],
            "work_history_json": r[5],
            "education_json": r[6],
            "inferred_skills": r[7],
        }
    return out


def build_compact_line(row: Dict[str, Any]) -> str:
    cid = row["cand_id"]

    headline = norm_ws(row["headline"]) if row.get("headline") else ""
    summary = clip(row["summary"], MAX_SUMMARY_CHARS) if row.get("summary") else ""

    skills = parse_skills(row.get("skills_raw"), MAX_SKILLS)
    inferred = parse_inferred_skills(row.get("inferred_skills"), MAX_INFERRED_SKILLS)

    langs = fmt_languages(row.get("languages_json"))
    work = fmt_work(row.get("work_history_json"), MAX_WORK_ITEMS)
    edu = fmt_edu(row.get("education_json"), MAX_EDU_ITEMS)

    # Pipe-separated; short labels to reduce tokens.
    parts = [f"{cid}"]
    if headline:
        parts.append(f"H:{headline}")
    if skills:
        parts.append(f"S:{', '.join(skills)}")
    if inferred:
        parts.append(f"IS:{', '.join(inferred)}")
    if work:
        parts.append(f"X:{work}")
    if edu:
        parts.append(f"E:{edu}")
    if langs:
        parts.append(f"L:{langs}")
    if summary:
        parts.append(f"Y:{summary}")

    line = " | ".join(parts)
    line = scrub_pii(line)
    return line


def read_job_description_from_prompt() -> str:
    print("Paste job description (finish with an empty line):")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def build_prompt(job_description: str, compact_lines: List[str]) -> str:
    prompt: List[str] = []
    prompt.append("TASK: You are a recruiting matching engine.")
    prompt.append("You will receive a job description and a list of anonymized candidate mini-profiles.")
    prompt.append("Each candidate is identified ONLY by CAND_ID.")
    prompt.append("")
    prompt.append("INSTRUCTIONS:")
    prompt.append("1) Rank the BEST 50 candidates for this job.")
    prompt.append("2) For the TOP 20, provide 1-2 short sentences explaining fit and missing requirements.")
    prompt.append("3) Output STRICT JSON only, no markdown.")
    prompt.append('4) JSON format: {"top50":[{"cand_id":"CAND_...","score":0-100,"reason":"..."},...]}')
    prompt.append("")
    prompt.append("JOB_DESCRIPTION:")
    prompt.append(job_description.strip())
    prompt.append("")
    prompt.append("CANDIDATES:")
    prompt.extend(compact_lines)
    return "\n".join(prompt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare anonymized Gemini batch prompt from top500 Chroma IDs.")
    ap.add_argument(
        "--job",
        type=str,
        default="",
        help="Job description text. If omitted, will use positional args or prompt interactively.",
    )
    ap.add_argument(
        "job_positional",
        nargs="*",
        help="Optional job description as positional args (alternative to --job).",
    )
    args = ap.parse_args()

    job_description = (args.job or " ".join(args.job_positional)).strip()
    if not job_description:
        job_description = read_job_description_from_prompt()
    if not job_description:
        raise SystemExit("Job description is empty. Provide --job \"...\" or paste it when prompted.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cand_ids = load_top_ids(TOP_IDS_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        profiles = fetch_profiles(conn, cand_ids)

    missing = 0
    compact_rows: List[Dict[str, Any]] = []
    compact_lines: List[str] = []

    for cid in cand_ids:
        row = profiles.get(cid)
        if not row:
            missing += 1
            continue
        line = build_compact_line(row)
        compact_lines.append(line)
        compact_rows.append({"cand_id": cid, "line": line})

    # JSONL for debugging
    with open(OUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for obj in compact_rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    prompt_text = build_prompt(job_description, compact_lines)
    OUT_PROMPT_PATH.write_text(prompt_text, encoding="utf-8")

    # Summary
    print(f"✅ Loaded top IDs: {len(cand_ids)}")
    print(f"✅ Candidates found in DB: {len(compact_lines)} (missing {missing})")
    print(f"✅ Wrote JSONL: {OUT_JSONL_PATH}")
    print(f"✅ Wrote prompt: {OUT_PROMPT_PATH}")
    print(f"\nPrompt size (chars): {len(prompt_text):,}")
    print("\nFirst 3 candidate lines:")
    for l in compact_lines[:3]:
        print("  " + l)


if __name__ == "__main__":
    main()