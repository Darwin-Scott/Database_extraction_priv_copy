# render_results.py
"""
Stage 3: Re-identification + output rendering.

Input:
- local/out/gemini_ranked.json (Gemini output with cand_id, score, reason)

Reads local SQLite:
- candidate (PII local)
- candidate_profile_text

Outputs (default):
- local/out/top{N}_results.csv
- local/out/top{N}_results.md
- prints top 10 to terminal

Usage:
  python scripts/render_results.py
  python scripts/render_results.py --in local/out/gemini_ranked.json --db local/candidates.db
  python scripts/render_results.py --rank-key top100
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dbx.paths import DB, OUT


DEFAULT_DB = str(DB)
DEFAULT_IN = OUT / "gemini_ranked.json"
OUT_DIR = OUT


def safe_json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_rank_key(obj: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Find key like top50/top100 with list content.
    Prefer the largest topN list if multiple exist.
    """
    best_key = None
    best_list = None
    best_n = -1
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        m = re.fullmatch(r"top(\d+)", k.strip())
        if not m:
            continue
        if not isinstance(v, list):
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best_key = k
            best_list = v

    if best_key and isinstance(best_list, list):
        return best_key, best_list

    raise ValueError("Input JSON must contain a key like 'top50'/'top100' with a list value.")


def _join_bullets(v: Any) -> str:
    """Convert list[str] bullets or str into a single compact string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        # keep only string-ish entries
        items = [str(x).strip() for x in v if x is not None and str(x).strip()]
        return "; ".join(items)
    return str(v).strip()


def load_ranked_list(obj: Dict[str, Any], rank_key: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    if rank_key.strip():
        rk = rank_key.strip()
        v = obj.get(rk)
        if not isinstance(v, list):
            raise ValueError(f"Input JSON must contain key '{rk}' as a list.")
        raw_list = v
        used_key = rk
    else:
        used_key, raw_list = infer_rank_key(obj)

    out: List[Dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        cid = item.get("cand_id")
        if not cid:
            continue

        # Backward compatible:
        # - new format: reasons (list), missing_requirements (list), confidence
        # - old format: reason (string)
        reasons = _join_bullets(item.get("reasons", item.get("reason", "")))
        missing = _join_bullets(item.get("missing_requirements", ""))
        confidence = str(item.get("confidence", "")).strip() if item.get("confidence") is not None else ""

        out.append(
            {
                "cand_id": str(cid),
                "score": item.get("score"),
                "confidence": confidence,
                "reasons": reasons,
                "missing_requirements": missing,
            }
        )
    return used_key, out


def fetch_candidate_data(conn: sqlite3.Connection, cand_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not cand_ids:
        return {}

    placeholders = ",".join(["?"] * len(cand_ids))
    query = f"""
    SELECT
      c.cand_id,
      c.full_name,
      c.profile_url,
      c.emails_json,
      c.phones_json,
      c.location_name,
      c.industry,
      pt.headline,
      pt.skills_raw,
      pt.languages_json
    FROM candidate c
    LEFT JOIN candidate_profile_text pt ON pt.cand_id = c.cand_id
    WHERE c.cand_id IN ({placeholders})
    """
    rows = conn.execute(query, cand_ids).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[r[0]] = {
            "cand_id": r[0],
            "full_name": r[1],
            "profile_url": r[2],
            "emails_json": r[3],
            "phones_json": r[4],
            "location_name": r[5],
            "industry": r[6],
            "headline": r[7],
            "skills_raw": r[8],
            "languages_json": r[9],
        }
    return out


def parse_json_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        return []
    return []


def render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "rank",
        "cand_id",
        "score",
        "confidence",
        "full_name",
        "headline",
        "emails",
        "phones",
        "profile_url",
        "reasons",
        "missing_requirements",
    ]
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")

    def esc(x):
        x = "" if x is None else str(x)
        x = x.replace("\n", " ").replace("|", "\\|")
        return x

    for r in rows:
        md.append("| " + " | ".join(esc(r.get(h, "")) for h in headers) + " |")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    parser.add_argument("--rank-key", type=str, default="", help="Optional: choose list key like top100 explicitly.")
    parser.add_argument("--out_csv", default="", help="Optional explicit output CSV path.")
    parser.add_argument("--out_md", default="", help="Optional explicit output Markdown path.")
    args = parser.parse_args()

    db_path = args.db
    in_path = Path(args.in_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Create mock output or run scripts/gemini_rank.py.")

    OUT_DIR.mkdir(exist_ok=True)

    ranked_obj = safe_json_load(in_path)
    used_key, ranked_list = load_ranked_list(ranked_obj, rank_key=args.rank_key)

    # derive N
    m = re.fullmatch(r"top(\d+)", used_key)
    n = int(m.group(1)) if m else len(ranked_list)

    out_csv = Path(args.out_csv) if args.out_csv else (OUT_DIR / f"top{n}_results.csv")
    out_md = Path(args.out_md) if args.out_md else (OUT_DIR / f"top{n}_results.md")

    cand_ids = [x["cand_id"] for x in ranked_list]

    with sqlite3.connect(db_path) as conn:
        db_data = fetch_candidate_data(conn, cand_ids)

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(ranked_list, start=1):
        cid = item["cand_id"]
        d = db_data.get(cid, {})
        emails = ", ".join(parse_json_list(d.get("emails_json")))
        phones = ", ".join(parse_json_list(d.get("phones_json")))

        results.append(
            {
                "rank": idx,
                "cand_id": cid,
                "score": item.get("score"),
                "confidence": item.get("confidence", ""),
                "reasons": item.get("reasons", ""),
                "missing_requirements": item.get("missing_requirements", ""),
                "full_name": d.get("full_name"),
                "headline": d.get("headline"),
                "emails": emails,
                "phones": phones,
                "profile_url": d.get("profile_url"),
            }
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "cand_id",
                "score",
                "confidence",
                "full_name",
                "headline",
                "emails",
                "phones",
                "profile_url",
                "reasons",
                "missing_requirements",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    out_md.write_text(render_markdown_table(results), encoding="utf-8")

    print("✅ Top 10 results (preview):\n")
    for r in results[:10]:
        print(f"{r['rank']:02d}. {r['cand_id']} | score={r['score']} | {r.get('full_name')} | {r.get('headline')}")
        reasons = r.get("reasons", "")
        missing = r.get("missing_requirements", "")
        if reasons:
            print("   - reasons: " + (reasons[:140] + "…" if len(reasons) > 140 else reasons))
        if missing:
            print("   - missing: " + (missing[:140] + "…" if len(missing) > 140 else missing))

    print(f"\n✅ Detected rank key: {used_key}")
    print(f"✅ Wrote CSV: {out_csv}")
    print(f"✅ Wrote Markdown: {out_md}")


if __name__ == "__main__":
    main()