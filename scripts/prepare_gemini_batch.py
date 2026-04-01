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


def fmt_work_list(work_json: Optional[str], limit: int) -> List[str]:
    work = safe_load_json(work_json, [])
    if not work:
        return []
    items: List[str] = []
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
    return items


def fmt_edu_list(edu_json: Optional[str], limit: int) -> List[str]:
    edu = safe_load_json(edu_json, [])
    if not edu:
        return []
    items: List[str] = []
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
    return items


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
        SELECT
            pt.cand_id,
            pt.headline,
            pt.summary,
            pt.skills_raw,
            pt.location_name,
            pt.industry,
            pt.current_company,
            pt.current_position,
            pt.badges_job_seeker,
            pt.badges_open_link,
            pt.profile_snapshot_at,
            pt.work_history_json,
            pt.education_json,
            pt.inferred_skills,
            rf.listed_role_months_sum,
            rf.current_listed_role_months,
            rf.iam_role_months,
            rf.current_role_is_iam
        FROM candidate_profile_text pt
        LEFT JOIN candidate_rank_features rf ON rf.cand_id = pt.cand_id
        WHERE pt.cand_id IN ({placeholders})
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
            "location_name": r[4],
            "industry": r[5],
            "current_company": r[6],
            "current_position": r[7],
            "badges_job_seeker": r[8],
            "badges_open_link": r[9],
            "profile_snapshot_at": r[10],
            "work_history_json": r[11],
            "education_json": r[12],
            "inferred_skills": r[13],
            "listed_role_months_sum": r[14],
            "current_listed_role_months": r[15],
            "iam_role_months": r[16],
            "current_role_is_iam": r[17],
        }
    return out


def scrub_value(v: Any) -> Any:
    """Apply scrub_pii to strings recursively (dict/list)."""
    if v is None:
        return None
    if isinstance(v, str):
        return scrub_pii(v)
    if isinstance(v, list):
        return [scrub_value(x) for x in v]
    if isinstance(v, dict):
        return {k: scrub_value(val) for k, val in v.items()}
    return v


def build_candidate_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    cid = row["cand_id"]

    headline = norm_ws(row["headline"]) if row.get("headline") else ""
    summary = clip(row["summary"], MAX_SUMMARY_CHARS) if row.get("summary") else ""
    location_name = norm_ws(row["location_name"]) if row.get("location_name") else ""
    industry = norm_ws(row["industry"]) if row.get("industry") else ""
    current_company = norm_ws(row["current_company"]) if row.get("current_company") else ""
    current_position = norm_ws(row["current_position"]) if row.get("current_position") else ""

    skills = parse_skills(row.get("skills_raw"), MAX_SKILLS)
    inferred = parse_inferred_skills(row.get("inferred_skills"), MAX_INFERRED_SKILLS)
    work = fmt_work_list(row.get("work_history_json"), MAX_WORK_ITEMS)
    edu = fmt_edu_list(row.get("education_json"), MAX_EDU_ITEMS)

    obj: Dict[str, Any] = {"cand_id": cid}
    if headline:
        obj["headline"] = headline
    if current_position or current_company:
        obj["current_role"] = " at ".join([bit for bit in [current_position, current_company] if bit])
    if skills:
        obj["skills"] = skills
    if inferred:
        obj["inferred_skills"] = inferred
    if work:
        obj["work"] = work
    if edu:
        obj["education"] = edu
    if summary:
        obj["summary"] = summary
    if location_name:
        obj["location"] = location_name
    if industry:
        obj["industry"] = industry
    if row.get("badges_job_seeker") is not None:
        obj["open_to_work"] = bool(row["badges_job_seeker"])
    if row.get("badges_open_link") is not None:
        obj["open_to_contact"] = bool(row["badges_open_link"])
    if row.get("listed_role_months_sum") is not None:
        obj["listed_role_months_sum"] = row["listed_role_months_sum"]
    if row.get("current_listed_role_months") is not None:
        obj["current_listed_role_months"] = row["current_listed_role_months"]
    if row.get("iam_role_months") is not None:
        obj["iam_role_months"] = row["iam_role_months"]
    if row.get("current_role_is_iam") is not None:
        obj["current_role_is_iam"] = bool(row["current_role_is_iam"])
    if row.get("profile_snapshot_at"):
        obj["profile_snapshot_at"] = row["profile_snapshot_at"]

    # Scrub PII safely without touching JSON syntax
    return scrub_value(obj)


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


def build_prompt_json(job_description: str, candidates: List[Dict[str, Any]], rank_n: int = 50, explain_n: int = 20) -> str:
    payload = {
        "task": "recruiting_matching_engine",
        "job_description": job_description.strip(),
        "rank_n": rank_n,
        "explain_n": explain_n,
        "constraints": {
            "do_not_invent_facts": True,
            "use_only_candidate_fields": True,
            "cand_id_unique_in_output": True,
            "output_exactly_n": rank_n,
        },
        "candidates": candidates,
        "output_schema": {
            f"top{rank_n}": [
                {
                    "cand_id": "CAND_...",
                    "score": "integer 0-100",
                    "confidence": "low|medium|high",
                    "reasons": ["short bullet", "short bullet"],
                    "missing_requirements": ["short bullet"],
                }
            ]
        },
    }

    instructions = [
        "TASK: You are a recruiting matching engine.",
        "INPUT: You will receive a JSON payload with a job description and a list of anonymized candidates.",
        "OUTPUT: Return STRICT JSON only. No markdown. No prose outside JSON.",
        f"Return exactly top{rank_n} candidates, unique cand_id values.",
        "",
        "SCORING RULES (IMPORTANT):",
        "1) Infer the evaluation criteria ONLY from the job_description.",
        "2) Identify must-have requirements vs nice-to-have signals from the job_description.",
        "3) Score candidates 0-100 based on evidence present in candidate fields.",
        "4) Penalize missing must-haves strongly; do not penalize missing info unless it is required by the job.",
        "5) Do NOT invent facts. If something is not present, treat it as unknown.",
        "6) Prefer candidates with the clearest direct evidence for the job requirements.",
        "",
        f"EXPLANATIONS:",
        f"- For the top {explain_n}, provide 1-2 short bullets in reasons AND 0-2 bullets in missing_requirements.",
        f"- For ranks below top {explain_n}, keep reasons/missing_requirements empty arrays.",
        "",
        "INPUT_JSON:",
        json.dumps(payload, ensure_ascii=False),
    ]
    return "\n".join(instructions)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare anonymized Gemini batch prompt from Chroma IDs.")
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

    # NEW: control how many candidates we include from the ID file
    ap.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="How many candidate IDs (from the ids file) to include in the prompt (default: 500).",
    )

    # NEW: control what we ask Gemini to return
    ap.add_argument(
        "--rank-n",
        type=int,
        default=30,
        help="How many candidates Gemini should rank in the JSON output (default: 50).",
    )
    ap.add_argument(
        "--explain-n",
        type=int,
        default=20,
        help="For how many top candidates to provide short reasoning (default: 20).",
    )

    # NEW: allow using different id files e.g. top1200_ids.txt
    ap.add_argument(
        "--ids-path",
        type=str,
        default="",
        help="Optional path to ids file. Default: local/out/top500_ids.txt",
    )

    args = ap.parse_args()

    job_description = (args.job or " ".join(args.job_positional)).strip()
    if not job_description:
        job_description = read_job_description_from_prompt()
    if not job_description:
        raise SystemExit("Job description is empty. Provide --job \"...\" or paste it when prompted.")

    if args.top_n <= 0:
        raise SystemExit("--top-n must be > 0")
    if args.rank_n <= 0:
        raise SystemExit("--rank-n must be > 0")
    if args.explain_n < 0:
        raise SystemExit("--explain-n must be >= 0")
    if args.explain_n > args.rank_n:
        print("⚠️ --explain-n is greater than --rank-n; capping explain-n to rank-n.")
        args.explain_n = args.rank_n

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids_path = Path(args.ids_path).expanduser().resolve() if args.ids_path else TOP_IDS_PATH
    cand_ids = load_top_ids(ids_path)[: args.top_n]

    with sqlite3.connect(DB_PATH) as conn:
        profiles = fetch_profiles(conn, cand_ids)

    missing = 0
    candidate_objs: List[Dict[str, Any]] = []

    for cid in cand_ids:
        row = profiles.get(cid)
        if not row:
            missing += 1
            continue
        obj = build_candidate_obj(row)
        candidate_objs.append(obj)

    # JSONL for debugging (one candidate per line)
    with open(OUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for obj in candidate_objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    prompt_text = build_prompt_json(job_description, candidate_objs, rank_n=args.rank_n, explain_n=args.explain_n)
    OUT_PROMPT_PATH.write_text(prompt_text, encoding="utf-8")

    print(f"✅ Candidates found in DB: {len(candidate_objs)} (missing {missing})")
    print(f"\nFirst 3 candidate objs:")
    for o in candidate_objs[:3]:
        print("  " + json.dumps(o, ensure_ascii=False)[:400] + ("..." if len(json.dumps(o, ensure_ascii=False)) > 400 else ""))

    # Summary
    print(f"✅ Loaded IDs from: {ids_path}")
    print(f"✅ Using top-n IDs: {len(cand_ids)} (requested {args.top_n})")


if __name__ == "__main__":
    main()
