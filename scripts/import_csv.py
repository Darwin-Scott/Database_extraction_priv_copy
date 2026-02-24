import json
import sqlite3
import pandas as pd
import yaml
import csv
from pathlib import Path
from datetime import datetime

def detect_delimiter(csv_path, sample_bytes=20000):
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(sample_bytes)
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    return dialect.delimiter

def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v != "" else None
    return v

def collect_list(row, cols):
    vals = []
    for c in cols:
        if c in row:
            v = clean_value(row[c])
            if v is not None:
                vals.append(v)
    return vals

def build_pattern_items(row, spec, max_items):
    items = []
    for i in range(1, max_items + 1):
        item = {}
        for key, pattern in spec.items():
            col = pattern.format(i=i)
            if col in row:
                v = clean_value(row[col])
                if v is not None:
                    item[key] = v
        if item:
            items.append(item)
    return items

def build_languages(row, max_items=10):
    """
    Prefer structured columns: language_1..n + language_proficiency_1..n
    Fallback to 'languages' if list is empty.
    """
    items = []
    for i in range(1, max_items + 1):
        lang = clean_value(row.get(f"language_{i}"))
        prof = clean_value(row.get(f"language_proficiency_{i}"))
        if lang:
            entry = {"name": lang}
            if prof:
                entry["proficiency"] = prof
            items.append(entry)

    if items:
        return items

    # fallback
    fallback = clean_value(row.get("languages"))
    return [{"name": fallback}] if fallback else []

def import_csv_to_sqlite(csv_path, db_path="candidates.db", config_path="config.yml", reset_db=False):
    csv_path = Path(csv_path)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    delim = detect_delimiter(csv_path) if cfg["source"]["delimiter"] == "auto" else cfg["source"]["delimiter"]

    df = pd.read_csv(
        csv_path,
        sep=delim,
        engine="python",
        encoding=cfg["source"]["encoding"],
        encoding_errors="replace",
        on_bad_lines="skip"
    )

    unique_key = cfg["dedupe"]["unique_key"]
    prefix = cfg["id"]["prefix"]
    width = int(cfg["id"]["width"])

    imported_at = datetime.now().isoformat(timespec="seconds")
    source_file = str(csv_path.name)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")

        if reset_db:
            # Clears all tables (dev mode). Keeps schema intact.
            conn.execute("DELETE FROM candidate_messages;")
            conn.execute("DELETE FROM candidate_profile_text;")
            conn.execute("DELETE FROM candidate;")
            conn.commit()
            print("⚠️ Reset enabled: cleared existing tables before import.")

        # Preload existing profile_url → cand_id for dedupe
        existing = dict(conn.execute("SELECT profile_url, cand_id FROM candidate").fetchall())

        inserted = 0
        updated = 0

        for _, row in df.iterrows():
            row = row.to_dict()

            profile_url = clean_value(row.get(unique_key))
            if not profile_url:
                continue

            # dedupe / assign cand_id
            if profile_url in existing:
                cand_id = existing[profile_url]
                is_update = True
            else:
                next_num = conn.execute("SELECT COUNT(*) FROM candidate").fetchone()[0] + 1
                cand_id = f"{prefix}{next_num:0{width}d}"
                existing[profile_url] = cand_id
                is_update = False

            # --- candidate (PII/local) ---
            cand_cfg = cfg["tables"]["candidate"]
            pii_data = {k: clean_value(row.get(k)) for k in cand_cfg["pii_fields"]}

            emails = collect_list(row, cand_cfg["emails"])
            phones = collect_list(row, cand_cfg["phones"])
            websites = collect_list(row, cand_cfg["websites"])

            conn.execute(
                """
                INSERT INTO candidate (
                  cand_id, profile_url, full_name, first_name, last_name,
                  location_name, industry, address, avatar,
                  emails_json, phones_json, websites_json,
                  source_file, source_imported_at,
                  updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(profile_url) DO UPDATE SET
                  full_name=excluded.full_name,
                  first_name=excluded.first_name,
                  last_name=excluded.last_name,
                  location_name=excluded.location_name,
                  industry=excluded.industry,
                  address=excluded.address,
                  avatar=excluded.avatar,
                  emails_json=excluded.emails_json,
                  phones_json=excluded.phones_json,
                  websites_json=excluded.websites_json,
                  source_file=excluded.source_file,
                  source_imported_at=excluded.source_imported_at,
                  updated_at=datetime('now')
                """,
                (
                    cand_id,
                    profile_url,
                    pii_data.get("full_name"),
                    pii_data.get("first_name"),
                    pii_data.get("last_name"),
                    pii_data.get("location_name"),
                    pii_data.get("industry"),
                    pii_data.get("address"),
                    pii_data.get("avatar"),
                    json.dumps(emails, ensure_ascii=False),
                    json.dumps(phones, ensure_ascii=False),
                    json.dumps(websites, ensure_ascii=False),
                    source_file,
                    imported_at,
                ),
            )

            # --- candidate_profile_text (matching content local) ---
            headline = clean_value(row.get("headline"))
            summary = clean_value(row.get("summary"))
            skills = clean_value(row.get("skills"))

            # robust languages
            languages_json = build_languages(row, max_items=10)

            prof_cfg = cfg["tables"]["candidate_profile_text"]

            work_items = build_pattern_items(
                row,
                {
                    "organization": prof_cfg["work_history"]["organization"],
                    "title": prof_cfg["work_history"]["title"],
                    "description": prof_cfg["work_history"]["description"],
                    "start": prof_cfg["work_history"]["start"],
                    "end": prof_cfg["work_history"]["end"],
                },
                prof_cfg["work_history"]["max_items"],
            )

            edu_items = build_pattern_items(
                row,
                {
                    "school": prof_cfg["education"]["school"],
                    "degree": prof_cfg["education"]["degree"],
                    "fos": prof_cfg["education"]["fos"],
                    "description": prof_cfg["education"]["description"],
                    "start": prof_cfg["education"]["start"],
                    "end": prof_cfg["education"]["end"],
                },
                prof_cfg["education"]["max_items"],
            )

            tags = clean_value(row.get("tags"))
            note_public = clean_value(row.get("note"))

            conn.execute(
                """
                INSERT INTO candidate_profile_text (
                  cand_id, headline, summary, skills_raw,
                  languages_json, work_history_json, education_json,
                  tags, note_public, inferred_skills, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, datetime('now'))
                ON CONFLICT(cand_id) DO UPDATE SET
                  headline=excluded.headline,
                  summary=excluded.summary,
                  skills_raw=excluded.skills_raw,
                  languages_json=excluded.languages_json,
                  work_history_json=excluded.work_history_json,
                  education_json=excluded.education_json,
                  tags=excluded.tags,
                  note_public=excluded.note_public,
                  updated_at=datetime('now')
                """,
                (
                    cand_id,
                    headline,
                    summary,
                    skills,
                    json.dumps(languages_json, ensure_ascii=False),
                    json.dumps(work_items, ensure_ascii=False),
                    json.dumps(edu_items, ensure_ascii=False),
                    tags,
                    note_public,
                ),
            )

            # --- candidate_messages (sensitive local only) ---
            msg_cfg = cfg["tables"]["candidate_messages"]["fields"]
            msg_data = {k: clean_value(row.get(k)) for k in msg_cfg}

            if any(v is not None for v in msg_data.values()):
                conn.execute(
                    """
                    INSERT INTO candidate_messages (
                      cand_id, full_messaging_history,
                      last_sent_message_from, last_sent_message_text,
                      last_received_message_from, last_received_message_text,
                      last_sent_message_send_at, last_received_message_send_at,
                      updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(cand_id) DO UPDATE SET
                      full_messaging_history=excluded.full_messaging_history,
                      last_sent_message_from=excluded.last_sent_message_from,
                      last_sent_message_text=excluded.last_sent_message_text,
                      last_received_message_from=excluded.last_received_message_from,
                      last_received_message_text=excluded.last_received_message_text,
                      last_sent_message_send_at=excluded.last_sent_message_send_at,
                      last_received_message_send_at=excluded.last_received_message_send_at,
                      updated_at=datetime('now')
                    """,
                    (
                        cand_id,
                        msg_data.get("full_messaging_history"),
                        msg_data.get("last_sent_message_from"),
                        msg_data.get("last_sent_message_text"),
                        msg_data.get("last_received_message_from"),
                        msg_data.get("last_received_message_text"),
                        msg_data.get("last_sent_message_send_at"),
                        msg_data.get("last_received_message_send_at"),
                    ),
                )

            if is_update:
                updated += 1
            else:
                inserted += 1

        conn.commit()

    print(f"✅ Import done. Inserted: {inserted}, Updated (deduped): {updated}, Rows read: {len(df)}")
    print(f"ℹ️ Provenance: source_file={source_file}, source_imported_at={imported_at}")

if __name__ == "__main__":
    csv_file_path = r"DevOneIdent_170.csv"
    # Set reset_db=True ONLY if you want to wipe the DB tables before importing (dev mode)
    import_csv_to_sqlite(csv_file_path, reset_db=False)