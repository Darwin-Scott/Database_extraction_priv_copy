import sqlite3

db_path = "candidates.db"

with sqlite3.connect(db_path) as conn:
    c = conn.cursor()
    print("candidate:", c.execute("SELECT COUNT(*) FROM candidate").fetchone()[0])
    print("profile_text:", c.execute("SELECT COUNT(*) FROM candidate_profile_text").fetchone()[0])
    print("messages:", c.execute("SELECT COUNT(*) FROM candidate_messages").fetchone()[0])

    # show one candidate
    row = c.execute("""
      SELECT cand_id, full_name, profile_url FROM candidate LIMIT 1
    """).fetchone()
    print("sample candidate:", row)
    