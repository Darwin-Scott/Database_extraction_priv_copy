# build_documents.py
"""
Build anonymized, compact "matching documents" from the local SQLite DB.

Output:
- local/out/candidates.jsonl   (one JSON per line: {"cand_id": "...", "text": "...", "meta": {...}})
- local/out/candidates_preview.txt (human-readable preview of the first N docs)

Design goals (from briefing):
- No PII (no names, emails, phones, addresses, profile_url, messages)
- Compact token-efficient text (suitable for embeddings + LLM batch prompts)
- Include only Category C-ish info: headline/summary/skills/languages/work/education + inferred_skills (if present)
- Clean noisy skills format like "Skill : null"
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dbx.paths import DB, OUT, CHROMA, SCHEMA, RAW_DATA


DB_PATH = DB
OUT_DIR = OUT
JSONL_PATH = OUT_DIR / "candidates.jsonl"
PREVIEW_PATH = OUT_DIR / "candidates_preview.txt"

# Tuning knobs (token/size control)
MAX_SUMMARY_CHARS = 800
MAX_WORK_ITEMS = 6
MAX_EDU_ITEMS = 3
MAX_SKILLS = 30
MAX_INFERRED_SKILLS = 15
INCLUDE_WORK_DESCRIPTIONS = True
MAX_WORK_DESC_CHARS = 220


def safe_load_json(s: Optional[str], default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def norm_ws(s: str) -> str:
    """Normalize whitespace and remove weird bullets/spaces."""
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clip(s: str, max_chars: int) -> str:
    s = norm_ws(s)
    return s if len(s) <= max_chars else (s[: max_chars - 1].rstrip() + "…")


def clean_skill_token(tok: str) -> Optional[str]:
    """
    Clean a single skill token like:
      "Identity and Access Management (IAM) : null"
    -> "Identity and Access Management (IAM)"
    """
    tok = tok.strip()
    if not tok:
        return None
    # remove ": null" or ":NULL" or ": None" patterns
    tok = re.sub(r"\s*:\s*(null|none)\s*$", "", tok, flags=re.IGNORECASE)
    tok = tok.strip(" -–•\t")
    tok = norm_ws(tok)
    return tok if tok else None


def parse_skills(skills_raw: Optional[str], limit: int = MAX_SKILLS) -> List[str]:
    if not skills_raw:
        return []
    # split by comma, but keep simple
    parts = [p for p in skills_raw.split(",")]
    cleaned: List[str] = []
    seen = set()
    for p in parts:
        c = clean_skill_token(p)
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(c)
        if len(cleaned) >= limit:
            break
    return cleaned


def parse_inferred_skills(inferred: Optional[str], limit: int = MAX_INFERRED_SKILLS) -> List[str]:
    if not inferred:
        return []
    # allow either comma-separated list or newline list
    parts = re.split(r"[,\n;]+", inferred)
    cleaned: List[str] = []
    seen = set()
    for p in parts:
        c = clean_skill_token(p)
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(c)
        if len(cleaned) >= limit:
            break
    return cleaned


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
            out.append(f"{norm_ws(str(name))} ({norm_ws(str(prof))})")
        else:
            out.append(norm_ws(str(name)))
    return ", ".join(out)


def fmt_work(work_json: Optional[str], limit: int = MAX_WORK_ITEMS) -> str:
    work = safe_load_json(work_json, [])
    if not work:
        return ""
    items = []
    for w in work[:limit]:
        if not isinstance(w, dict):
            continue
        org = w.get("organization")
        title = w.get("title")
        desc = w.get("description") if INCLUDE_WORK_DESCRIPTIONS else None

        piece = []
        if org:
            piece.append(norm_ws(str(org)))
        if title:
            piece.append(norm_ws(str(title)))
        base = " — ".join(piece) if piece else ""

        if desc:
            d = clip(str(desc), MAX_WORK_DESC_CHARS)
            if d:
                base = f"{base} | {d}" if base else d

        if base:
            items.append(base)

    return " ; ".join(items)


def fmt_education(edu_json: Optional[str], limit: int = MAX_EDU_ITEMS) -> str:
    edu = safe_load_json(edu_json, [])
    if not edu:
        return ""
    items = []
    for e in edu[:limit]:
        if not isinstance(e, dict):
            continue
        school = e.get("school")
        degree = e.get("degree")
        fos = e.get("fos")

        parts = []
        if degree:
            parts.append(norm_ws(str(degree)))
        if fos:
            parts.append(norm_ws(str(fos)))
        if school:
            parts.append(norm_ws(str(school)))

        item = " — ".join(parts)
        if item:
            items.append(item)
    return " ; ".join(items)


def build_document(
    cand_id: str,
    headline: Optional[str],
    summary: Optional[str],
    skills_raw: Optional[str],
    languages_json: Optional[str],
    work_history_json: Optional[str],
    education_json: Optional[str],
    inferred_skills: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns:
      text: compact string
      meta: helpful metadata (non-PII)
    """
    headline_s = norm_ws(headline) if headline else ""
    summary_s = clip(summary, MAX_SUMMARY_CHARS) if summary else ""

    skills_list = parse_skills(skills_raw)
    inferred_list = parse_inferred_skills(inferred_skills)

    languages_s = fmt_languages(languages_json)
    work_s = fmt_work(work_history_json)
    edu_s = fmt_education(education_json)

    # Compact, line-based format (token-efficient)
    parts = [f"ID: {cand_id}"]

    if headline_s:
        parts.append(f"Headline: {headline_s}")
    if summary_s:
        parts.append(f"Summary: {summary_s}")
    if skills_list:
        parts.append(f"Skills: {', '.join(skills_list)}")
    if inferred_list:
        # Mark as "inferred" so you can downweight later (e.g., repeat less or tag)
        parts.append(f"InferredSkills: {', '.join(inferred_list)}")
    if work_s:
        parts.append(f"Experience: {work_s}")
    if edu_s:
        parts.append(f"Education: {edu_s}")
    if languages_s:
        parts.append(f"Languages: {languages_s}")

    text = " | ".join(parts)

    meta = {
        "has_headline": bool(headline_s),
        "has_summary": bool(summary_s),
        "n_skills": len(skills_list),
        "n_inferred_skills": len(inferred_list),
        "has_work": bool(work_s),
        "has_education": bool(edu_s),
        "has_languages": bool(languages_s),
    }

    return text, meta


def fetch_candidates(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT
          cand_id,
          headline,
          summary,
          skills_raw,
          languages_json,
          work_history_json,
          education_json,
          inferred_skills
        FROM candidate_profile_text
        ORDER BY cand_id
        """
    ).fetchall()
    return rows


def main(db_path: str = DB_PATH):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        rows = fetch_candidates(conn)

    # Build docs + write JSONL
    n = 0
    examples: List[str] = []

    with open(JSONL_PATH, "w", encoding="utf-8") as f_jsonl:
        for r in rows:
            cand_id = r["cand_id"]
            text, meta = build_document(
                cand_id=cand_id,
                headline=r["headline"],
                summary=r["summary"],
                skills_raw=r["skills_raw"],
                languages_json=r["languages_json"],
                work_history_json=r["work_history_json"],
                education_json=r["education_json"],
                inferred_skills=r["inferred_skills"],
            )
            record = {"cand_id": cand_id, "text": text, "meta": meta}
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            if len(examples) < 3:
                examples.append(text)

            n += 1

    # Write preview
    preview_lines = []
    preview_lines.append(f"Generated documents: {n}")
    preview_lines.append(f"JSONL: {JSONL_PATH.resolve()}")
    preview_lines.append("")
    preview_lines.append("=== Examples (first 3) ===")
    preview_lines.extend(examples)
    preview_lines.append("")
    preview_lines.append("=== Tip ===")
    preview_lines.append("Open local/out/candidates.jsonl and inspect ~10 random entries to verify:")
    preview_lines.append("- compactness")
    preview_lines.append("- no PII leakage (names/emails/phones/addresses/LinkedIn URLs)")
    preview_lines.append("- skills cleaned (no ': null')")
    preview_lines.append("")

    PREVIEW_PATH.write_text("\n".join(preview_lines), encoding="utf-8")

    print(f"✅ Wrote {n} documents")
    print(f" - {JSONL_PATH}")
    print(f" - {PREVIEW_PATH}")
    print("\nExamples:")
    for ex in examples:
        print("  " + ex)


if __name__ == "__main__":
    main()