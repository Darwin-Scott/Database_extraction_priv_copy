import sqlite3
import json
from textwrap import shorten
from dbx.paths import DB

DB_PATH = DB

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def list_tables(conn):
    cur = conn.cursor()
    return [r[0] for r in cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
    """).fetchall()]

def table_columns(conn, table):
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA table_info({table});").fetchall()
    # (cid, name, type, notnull, dflt_value, pk)
    return rows

def table_indexes(conn, table):
    cur = conn.cursor()
    idxs = cur.execute(f"PRAGMA index_list({table});").fetchall()
    # (seq, name, unique, origin, partial)
    details = []
    for idx in idxs:
        idx_name = idx[1]
        cols = cur.execute(f"PRAGMA index_info({idx_name});").fetchall()
        # (seqno, cid, name)
        details.append((idx, cols))
    return details

def count_rows(conn, table):
    cur = conn.cursor()
    return cur.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]

def sample_rows(conn, table, limit=2):
    cur = conn.cursor()
    return cur.execute(f"SELECT * FROM {table} LIMIT {limit};").fetchall()

def main():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row  # nicer access by column name

        # 1) tables overview
        print_header(f"SQLite DB: {DB_PATH}")
        tables = list_tables(conn)
        print("Tables found:", ", ".join(tables) if tables else "(none)")

        # 2) schema per table
        for t in tables:
            print_header(f"Schema for table: {t}")
            cols = table_columns(conn, t)
            print(f"Columns ({len(cols)}):")
            for cid, name, coltype, notnull, dflt, pk in cols:
                nn = "NOT NULL" if notnull else ""
                pk_s = "PK" if pk else ""
                dflt_s = f"DEFAULT {dflt}" if dflt is not None else ""
                meta = " ".join(x for x in [coltype, nn, dflt_s, pk_s] if x)
                print(f" - {name} [{meta}]")

            # indexes
            idx_details = table_indexes(conn, t)
            if idx_details:
                print("\nIndexes:")
                for (seq, idx_name, unique, origin, partial), cols in idx_details:
                    colnames = [c[2] for c in cols]
                    uniq = "UNIQUE" if unique else ""
                    print(f" - {idx_name} ({', '.join(colnames)}) {uniq}".rstrip())
            else:
                print("\nIndexes: (none)")

            # row count
            n = count_rows(conn, t)
            print(f"\nRow count: {n}")

            # sample rows (compact)
            rows = sample_rows(conn, t, limit=2)
            print("\nSample rows (first 2):")
            if not rows:
                print(" (no rows)")
            else:
                for r in rows:
                    # compact print: show key=value pairs, truncate long text
                    items = []
                    for k in r.keys():
                        v = r[k]
                        if isinstance(v, str):
                            v2 = shorten(v, width=120, placeholder="...")
                        else:
                            v2 = v
                        items.append(f"{k}={v2}")
                    print("  - " + " | ".join(items))

        # 3) Extra: show one joined sample + JSON lengths (your earlier checks)
        print_header("Join sanity check (candidate + candidate_profile_text)")
        cur = conn.cursor()
        row = cur.execute("""
            SELECT c.cand_id, c.full_name, c.profile_url, pt.headline,
                   pt.languages_json, pt.work_history_json, pt.education_json
            FROM candidate c
            LEFT JOIN candidate_profile_text pt ON pt.cand_id = c.cand_id
            LIMIT 1
        """).fetchone()

        if row:
            print("cand_id:", row[0])
            print("full_name:", row[1])
            print("profile_url:", row[2])
            print("headline:", row[3])

            langs = json.loads(row[4]) if row[4] else []
            work = json.loads(row[5]) if row[5] else []
            edu = json.loads(row[6]) if row[6] else []
            print("\nParsed JSON lengths:")
            print("languages:", len(langs), "work_history:", len(work), "education:", len(edu))
        print_header("Languages coverage check")

        cur = conn.cursor()

        # How many have languages_json not empty?
        n_with_lang = cur.execute("""
            SELECT COUNT(*)
            FROM candidate_profile_text
            WHERE languages_json IS NOT NULL AND languages_json != '[]'
        """).fetchone()[0]

        n_total = cur.execute("SELECT COUNT(*) FROM candidate_profile_text").fetchone()[0]

        print(f"Profiles with languages_json filled: {n_with_lang}/{n_total}")

        # Show a few examples where languages exist
        rows = cur.execute("""
            SELECT cand_id, languages_json
            FROM candidate_profile_text
            WHERE languages_json IS NOT NULL AND languages_json != '[]'
            LIMIT 5
        """).fetchall()

        print("\nExamples (first 5 with languages):")
        for r in rows:
            print(f"- {r[0]}: {r[1]}")

if __name__ == "__main__":
    main()