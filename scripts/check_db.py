import argparse
import json
import sqlite3
from collections import OrderedDict

from dbx.paths import DB

DB_PATH = DB


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def list_tables(conn):
    cur = conn.cursor()
    return [r[0] for r in cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
    """).fetchall()]


def table_columns(conn, table):
    cur = conn.cursor()
    return cur.execute(f"PRAGMA table_info({table});").fetchall()


def count_rows(conn, table):
    cur = conn.cursor()
    return cur.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]


def column_exists(conn, table, col):
    cols = [r[1] for r in table_columns(conn, table)]
    return col in cols


def safe_scalar(conn, sql, params=()):
    cur = conn.cursor()
    row = cur.execute(sql, params).fetchone()
    return row[0] if row else None


def print_kv(d: dict, indent=0):
    pad = " " * indent
    for k, v in d.items():
        print(f"{pad}{k}: {v}")


def sample_rows(conn, table, limit=3):
    cur = conn.cursor()
    return cur.execute(f"SELECT * FROM {table} LIMIT {limit};").fetchall()


def print_full_row(row: sqlite3.Row, row_num: int):
    print(f"  - row {row_num}")
    for key in row.keys():
        value = row[key]
        if isinstance(value, str):
            print(f"    {key}: {value}")
        else:
            print(f"    {key}: {json.dumps(value, ensure_ascii=False)}")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Inspect the SQLite DB content and schema.")
    parser.add_argument("--sample-limit", type=int, default=3, help="Example rows per table.")
    args = parser.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        tables = list_tables(conn)
        print_header(f"DB SNAPSHOT: {DB_PATH}")
        print("Tables:", ", ".join(tables) if tables else "(none)")

        # --- Quick totals ---
        print_header("Row counts per table")
        counts = OrderedDict()
        for t in tables:
            counts[t] = count_rows(conn, t)
        print_kv(counts)

        # --- Provenance / multi-CSV proof ---
        if "candidate" in tables and column_exists(conn, "candidate", "source_file"):
            print_header("Provenance: source_file distribution (proof of multi-CSV ingest)")
            cur = conn.cursor()
            rows = cur.execute("""
                SELECT source_file, COUNT(*) AS n
                FROM candidate
                GROUP BY source_file
                ORDER BY n DESC, source_file ASC;
            """).fetchall()

            if not rows:
                print("(no candidates)")
            else:
                total = sum(r["n"] for r in rows)
                print(f"Distinct source_file: {len(rows)} | Total candidates: {total}")
                for r in rows:
                    print(f"- {r['source_file']}: {r['n']}")

        if "candidate" in tables and column_exists(conn, "candidate", "source_imported_at"):
            print_header("Provenance: import time range")
            min_ts = safe_scalar(conn, "SELECT MIN(source_imported_at) FROM candidate;")
            max_ts = safe_scalar(conn, "SELECT MAX(source_imported_at) FROM candidate;")
            print(f"First import timestamp: {min_ts}")
            print(f"Last  import timestamp: {max_ts}")

        # --- Join/coverage checks ---
        print_header("Coverage checks")
        cov = OrderedDict()
        if "candidate" in tables:
            cov["candidate_total"] = safe_scalar(conn, "SELECT COUNT(*) FROM candidate;")

        if "candidate_profile_text" in tables:
            cov["profile_text_total"] = safe_scalar(conn, "SELECT COUNT(*) FROM candidate_profile_text;")
            cov["candidates_with_profile_text"] = safe_scalar(conn, """
                SELECT COUNT(*)
                FROM candidate c
                JOIN candidate_profile_text pt ON pt.cand_id = c.cand_id;
            """)
            cov["candidates_missing_profile_text"] = safe_scalar(conn, """
                SELECT COUNT(*)
                FROM candidate c
                LEFT JOIN candidate_profile_text pt ON pt.cand_id = c.cand_id
                WHERE pt.cand_id IS NULL;
            """)

        if "candidate" in tables and column_exists(conn, "candidate", "languages_json"):
            cov["candidate_with_languages_json"] = safe_scalar(conn, """
                SELECT COUNT(*)
                FROM candidate
                WHERE languages_json IS NOT NULL AND languages_json != '[]';
            """)

        if "candidate_messages" in tables:
            cov["messages_total"] = safe_scalar(conn, "SELECT COUNT(*) FROM candidate_messages;")
            cov["candidates_with_messages_row"] = safe_scalar(conn, """
                SELECT COUNT(*)
                FROM candidate c
                JOIN candidate_messages m ON m.cand_id = c.cand_id;
            """)

        if "candidate_rank_features" in tables:
            cov["rank_features_total"] = safe_scalar(conn, "SELECT COUNT(*) FROM candidate_rank_features;")
            cov["rank_features_with_listed_role_months_sum"] = safe_scalar(conn, """
                SELECT COUNT(*)
                FROM candidate_rank_features
                WHERE listed_role_months_sum IS NOT NULL;
            """)

        print_kv(cov)

        # --- “Are we deduping?” quick signals ---
        print_header("Dedupe / uniqueness sanity")
        if "candidate" in tables:
            # profile_url should be unique by schema; check anyway
            dup = safe_scalar(conn, """
                SELECT COUNT(*)
                FROM (
                    SELECT profile_url, COUNT(*) n
                    FROM candidate
                    GROUP BY profile_url
                    HAVING n > 1
                );
            """)
            print(f"profile_url duplicates found: {dup}")

            # cand_id duplicates
            dup_id = safe_scalar(conn, """
                SELECT COUNT(*)
                FROM (
                    SELECT cand_id, COUNT(*) n
                    FROM candidate
                    GROUP BY cand_id
                    HAVING n > 1
                );
            """)
            print(f"cand_id duplicates found: {dup_id}")

        # --- Per-table schema + samples (optional but useful) ---
        print_header(f"Samples (first {args.sample_limit} rows per table, full values)")
        for t in tables:
            print(f"\n[{t}] rows={counts[t]}")
            rows = sample_rows(conn, t, limit=args.sample_limit)
            if not rows:
                print("  (no rows)")
            else:
                for i, r in enumerate(rows, start=1):
                    print_full_row(r, i)

        # --- Extra: show a few missing-profile examples (actionable debug) ---
        if "candidate" in tables and "candidate_profile_text" in tables:
            print_header("Examples: candidates missing profile_text (up to 10)")
            cur = conn.cursor()
            rows = cur.execute("""
                SELECT c.cand_id, c.full_name, c.profile_url, c.source_file
                FROM candidate c
                LEFT JOIN candidate_profile_text pt ON pt.cand_id = c.cand_id
                WHERE pt.cand_id IS NULL
                LIMIT 10;
            """).fetchall()
            if not rows:
                print("(none)")
            else:
                for r in rows:
                    print(f"- {r['cand_id']} | {r['full_name']} | {r['source_file']} | {r['profile_url']}")

        # --- Languages examples ---
        if "candidate" in tables and column_exists(conn, "candidate", "languages_json"):
            print_header("Examples: languages_json filled (up to 5)")
            cur = conn.cursor()
            rows = cur.execute("""
                SELECT cand_id, languages_json
                FROM candidate
                WHERE languages_json IS NOT NULL AND languages_json != '[]'
                LIMIT 5;
            """).fetchall()
            if not rows:
                print("(none)")
            else:
                for r in rows:
                    print(f"- {r['cand_id']}: {r['languages_json']}")

        print_header("Done")


if __name__ == "__main__":
    main()
