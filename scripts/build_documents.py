"""
Build anonymized, compact matching documents from the local SQLite DB.

Output:
- local/out/candidates.jsonl
- local/out/candidates_preview.txt
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from dbx.paths import DB, OUT


DB_PATH = DB
OUT_DIR = OUT
JSONL_PATH = OUT_DIR / "candidates.jsonl"
PREVIEW_PATH = OUT_DIR / "candidates_preview.txt"

MAX_SUMMARY_CHARS = 800
MAX_WORK_ITEMS = 6
MAX_EDU_ITEMS = 3
MAX_SKILLS = 30
MAX_INFERRED_SKILLS = 15
INCLUDE_WORK_DESCRIPTIONS = True
MAX_WORK_DESC_CHARS = 220
MAX_EDU_DESC_CHARS = 160


def safe_load_json(text: Optional[str], default):
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def norm_ws(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clip(text: str, max_chars: int) -> str:
    text = norm_ws(text)
    return text if len(text) <= max_chars else (text[: max_chars - 3].rstrip() + "...")


def clean_skill_token(token: str) -> Optional[str]:
    token = token.strip()
    if not token:
        return None
    token = re.sub(r"\s*:\s*(null|none)\s*$", "", token, flags=re.IGNORECASE)
    token = token.strip(" -\t")
    token = norm_ws(token)
    return token if token else None


def parse_skills(skills_raw: Optional[str], limit: int = MAX_SKILLS) -> List[str]:
    if not skills_raw:
        return []

    cleaned: List[str] = []
    seen = set()
    for part in skills_raw.split(","):
        token = clean_skill_token(part)
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def parse_inferred_skills(inferred: Optional[str], limit: int = MAX_INFERRED_SKILLS) -> List[str]:
    if not inferred:
        return []

    cleaned: List[str] = []
    seen = set()
    for part in re.split(r"[,\n;]+", inferred):
        token = clean_skill_token(part)
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def fmt_languages(languages_json: Optional[str]) -> str:
    languages = safe_load_json(languages_json, [])
    if not languages:
        return ""

    output = []
    for entry in languages:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        proficiency = entry.get("proficiency")
        if not name:
            continue
        if proficiency:
            output.append(f"{norm_ws(str(name))} ({norm_ws(str(proficiency))})")
        else:
            output.append(norm_ws(str(name)))
    return ", ".join(output)


def fmt_work(work_json: Optional[str], limit: int = MAX_WORK_ITEMS) -> str:
    work = safe_load_json(work_json, [])
    if not work:
        return ""

    items = []
    for entry in work[:limit]:
        if not isinstance(entry, dict):
            continue

        organization = entry.get("organization")
        title = entry.get("title")
        description = entry.get("description") if INCLUDE_WORK_DESCRIPTIONS else None

        parts = []
        if organization:
            parts.append(norm_ws(str(organization)))
        if title:
            parts.append(norm_ws(str(title)))

        base = " - ".join(parts) if parts else ""
        if description:
            description_text = clip(str(description), MAX_WORK_DESC_CHARS)
            if description_text:
                base = f"{base} | {description_text}" if base else description_text

        if base:
            items.append(base)

    return " ; ".join(items)


def fmt_education(education_json: Optional[str], limit: int = MAX_EDU_ITEMS) -> str:
    education = safe_load_json(education_json, [])
    if not education:
        return ""

    items = []
    for entry in education[:limit]:
        if not isinstance(entry, dict):
            continue

        degree = entry.get("degree")
        field_of_study = entry.get("fos")
        description = entry.get("description")

        parts = []
        if degree:
            parts.append(norm_ws(str(degree)))
        if field_of_study:
            parts.append(norm_ws(str(field_of_study)))
        if description:
            parts.append(clip(str(description), MAX_EDU_DESC_CHARS))

        item = " - ".join(parts)
        if item:
            items.append(item)

    return " ; ".join(items)


def build_document(
    cand_id: str,
    headline: Optional[str],
    summary: Optional[str],
    skills_raw: Optional[str],
    location_name: Optional[str],
    industry: Optional[str],
    current_company: Optional[str],
    current_position: Optional[str],
    languages_json: Optional[str],
    work_history_json: Optional[str],
    education_json: Optional[str],
    badges_job_seeker: Optional[int],
    badges_open_link: Optional[int],
    profile_snapshot_at: Optional[str],
    inferred_skills: Optional[str],
    total_role_months: Optional[int],
    current_role_tenure_months: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    headline_text = norm_ws(headline) if headline else ""
    summary_text = clip(summary, MAX_SUMMARY_CHARS) if summary else ""
    location_text = norm_ws(location_name) if location_name else ""
    industry_text = norm_ws(industry) if industry else ""
    current_company_text = norm_ws(current_company) if current_company else ""
    current_position_text = norm_ws(current_position) if current_position else ""

    skills_list = parse_skills(skills_raw)
    inferred_list = parse_inferred_skills(inferred_skills)
    languages_text = fmt_languages(languages_json)
    work_text = fmt_work(work_history_json)
    education_text = fmt_education(education_json)

    parts = [f"ID: {cand_id}"]
    if headline_text:
        parts.append(f"Headline: {headline_text}")
    if current_position_text or current_company_text:
        role_bits = [bit for bit in [current_position_text, current_company_text] if bit]
        parts.append(f"CurrentRole: {' at '.join(role_bits)}")
    if summary_text:
        parts.append(f"Summary: {summary_text}")
    if skills_list:
        parts.append(f"Skills: {', '.join(skills_list)}")
    if inferred_list:
        parts.append(f"InferredSkills: {', '.join(inferred_list)}")
    if work_text:
        parts.append(f"Experience: {work_text}")
    if education_text:
        parts.append(f"Education: {education_text}")
    if languages_text:
        parts.append(f"Languages: {languages_text}")
    if location_text:
        parts.append(f"Location: {location_text}")
    if industry_text:
        parts.append(f"Industry: {industry_text}")

    text = " | ".join(parts)
    meta = {
        "has_headline": bool(headline_text),
        "has_current_role": bool(current_position_text or current_company_text),
        "has_summary": bool(summary_text),
        "n_skills": len(skills_list),
        "n_inferred_skills": len(inferred_list),
        "has_work": bool(work_text),
        "has_education": bool(education_text),
        "has_languages": bool(languages_text),
        "badges_job_seeker": bool(badges_job_seeker),
        "badges_open_link": bool(badges_open_link),
        "profile_snapshot_at": profile_snapshot_at or "",
        "total_role_months": total_role_months,
        "current_role_tenure_months": current_role_tenure_months,
    }
    return text, meta


def fetch_candidates(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    rows = cursor.execute(
        """
        SELECT
          pt.cand_id,
          pt.headline,
          pt.summary,
          pt.skills_raw,
          pt.location_name,
          pt.industry,
          pt.current_company,
          pt.current_position,
          pt.languages_json,
          pt.work_history_json,
          pt.education_json,
          pt.badges_job_seeker,
          pt.badges_open_link,
          pt.profile_snapshot_at,
          pt.inferred_skills,
          rf.total_role_months,
          rf.current_role_tenure_months
        FROM candidate_profile_text pt
        LEFT JOIN candidate_rank_features rf ON rf.cand_id = pt.cand_id
        ORDER BY pt.cand_id
        """
    ).fetchall()
    return rows


def main(db_path: str = DB_PATH):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        rows = fetch_candidates(conn)

    examples: List[str] = []
    with open(JSONL_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            text, meta = build_document(
                cand_id=row["cand_id"],
                headline=row["headline"],
                summary=row["summary"],
                skills_raw=row["skills_raw"],
                location_name=row["location_name"],
                industry=row["industry"],
                current_company=row["current_company"],
                current_position=row["current_position"],
                languages_json=row["languages_json"],
                work_history_json=row["work_history_json"],
                education_json=row["education_json"],
                badges_job_seeker=row["badges_job_seeker"],
                badges_open_link=row["badges_open_link"],
                profile_snapshot_at=row["profile_snapshot_at"],
                inferred_skills=row["inferred_skills"],
                total_role_months=row["total_role_months"],
                current_role_tenure_months=row["current_role_tenure_months"],
            )
            handle.write(json.dumps({"cand_id": row["cand_id"], "text": text, "meta": meta}, ensure_ascii=False) + "\n")
            if len(examples) < 3:
                examples.append(text)

    preview_lines = [
        f"Generated documents: {len(rows)}",
        f"JSONL: {JSONL_PATH.resolve()}",
        "",
        "=== Examples (first 3) ===",
        *examples,
        "",
        "=== Tip ===",
        "Inspect local/out/candidates.jsonl and verify:",
        "- compactness",
        "- no PII leakage (names/emails/phones/addresses/LinkedIn URLs)",
        "- skills cleaned (no ': null')",
    ]
    PREVIEW_PATH.write_text("\n".join(preview_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} documents")
    print(f" - {JSONL_PATH}")
    print(f" - {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
