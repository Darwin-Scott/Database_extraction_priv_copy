# render_results.py
"""
Stage 3: Re-identification + output rendering.

Input:
- out/gemini_ranked.json (Gemini output with cand_id, score, reason)

Reads local SQLite:
- candidate (PII local)
- candidate_profile_text (Category C local)

Outputs:
- out/top50_results.csv
- out/top50_results.md
- prints top 10 to terminal

Usage:
  python render_results.py
  python render_results.py --in out/gemini_ranked.json --db candidates.db
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB = "candidates.db"
DEFAULT_IN = Path("out") / "gemini_ranked.json"
OUT_DIR = Path("out")
OUT_CSV = OUT_DIR / "top50_results.csv"
OUT_MD = OUT_DIR / "top50_results.md"


def safe_json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ranked_list(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    top50 = obj.get("top50")
    if not isinstance(top50, list):
        raise ValueError("Input JSON must contain key 'top50' as a list.")
    # normalize items
    out = []
    for item in top50:
        if not isinstance(item, dict):
            continue
        cid = item.get("cand_id")
        if not cid:
            continue
        out.append(
            {
                "cand_id": str(cid),
                "score": item.get("score"),
                "reason": item.get("reason", ""),
            }
        )
    return out


def fetch_candidate_data(conn: sqlite3.Connection, cand_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Join candidate + candidate_profile_text for given IDs.
    Returns dict cand_id -> data dict
    """
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
    cur = conn.cursor()
    rows = cur.execute(query, cand_ids).fetchall()

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
    headers = ["rank", "cand_id", "score", "full_name", "headline", "emails", "phones", "profile_url", "reason"]
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        def esc(x):
            x = "" if x is None else str(x)
            x = x.replace("\n", " ").replace("|", "\\|")
            return x

        md.append("| " + " | ".join(esc(r.get(h, "")) for h in headers) + " |")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    parser.add_argument("--out_csv", default=str(OUT_CSV))
    parser.add_argument("--out_md", default=str(OUT_MD))
    args = parser.parse_args()

    db_path = args.db
    in_path = Path(args.in_path)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Create mock output or run gemini_rank.py.")

    OUT_DIR.mkdir(exist_ok=True)

    ranked_obj = safe_json_load(in_path)
    ranked_list = load_ranked_list(ranked_obj)

    cand_ids = [x["cand_id"] for x in ranked_list]

    with sqlite3.connect(db_path) as conn:
        db_data = fetch_candidate_data(conn, cand_ids)

    # Merge in rank order
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
                "reason": item.get("reason", ""),
                "full_name": d.get("full_name"),
                "headline": d.get("headline"),
                "emails": emails,
                "phones": phones,
                "profile_url": d.get("profile_url"),
            }
        )

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "cand_id", "score", "full_name", "headline", "emails", "phones", "profile_url", "reason"],
        )
        writer.writeheader()
        writer.writerows(results)

    # Write Markdown table
    out_md.write_text(render_markdown_table(results), encoding="utf-8")

    # Print top 10
    print("✅ Top 10 results (preview):\n")
    for r in results[:10]:
        print(f"{r['rank']:02d}. {r['cand_id']} | score={r['score']} | {r.get('full_name')} | {r.get('headline')}")
        reason = r.get("reason", "")
        if isinstance(reason, str) and reason:
            print("   - " + (reason[:140] + "…" if len(reason) > 140 else reason))

    print(f"\n✅ Wrote CSV: {out_csv}")
    print(f"✅ Wrote Markdown: {out_md}")
    print("\nNext step: (optional) build a Streamlit UI to display this table interactively.")


if __name__ == "__main__":
    main()