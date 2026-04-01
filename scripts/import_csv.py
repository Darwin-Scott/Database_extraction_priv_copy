import csv
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from dbx.paths import DB


def detect_delimiter(csv_path, sample_bytes=20000):
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(sample_bytes)
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    return dialect.delimiter


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value != "" else None
    return value


def first_present(*values):
    for value in values:
        cleaned = clean_value(value)
        if cleaned is not None:
            return cleaned
    return None


def collect_list(row, cols):
    values = []
    for col in cols:
        if col in row:
            value = clean_value(row[col])
            if value is not None:
                values.append(value)
    return values


def build_languages(row, max_items=3):
    items = []
    for index in range(1, max_items + 1):
        language = clean_value(row.get(f"language_{index}"))
        proficiency = clean_value(row.get(f"language_proficiency_{index}"))
        if language:
            entry = {"name": language}
            if proficiency:
                entry["proficiency"] = proficiency
            items.append(entry)

    if items:
        return items

    fallback = clean_value(row.get("languages"))
    return [{"name": fallback}] if fallback else []


def normalize_row_dict(raw_row: dict) -> dict:
    return {key: clean_value(value) for key, value in raw_row.items()}


def import_csv_to_sqlite(csv_path, db_path=DB, config_path="config.yml", reset_db=False):
    csv_path = Path(csv_path)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    delimiter = detect_delimiter(csv_path) if cfg["source"]["delimiter"] == "auto" else cfg["source"]["delimiter"]

    dataframe = pd.read_csv(
        csv_path,
        sep=delimiter,
        engine="python",
        encoding=cfg["source"]["encoding"],
        encoding_errors="replace",
        on_bad_lines="skip",
    )

    unique_key = cfg["dedupe"]["unique_key"]
    prefix = cfg["id"]["prefix"]
    width = int(cfg["id"]["width"])
    imported_at = datetime.now().isoformat(timespec="seconds")
    source_file = str(csv_path.name)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")

        if reset_db:
            conn.execute("DELETE FROM candidate_messages;")
            conn.execute("DELETE FROM candidate_rank_features;")
            conn.execute("DELETE FROM candidate_profile_text;")
            conn.execute("DELETE FROM candidate_raw_import;")
            conn.execute("DELETE FROM candidate;")
            conn.commit()
            print("Reset enabled: cleared existing tables before import.")

        existing = dict(conn.execute("SELECT profile_url, cand_id FROM candidate").fetchall())

        inserted = 0
        updated = 0
        rows_with_unique_key = 0

        for _, raw_row in dataframe.iterrows():
            raw_row = raw_row.to_dict()
            row = normalize_row_dict(raw_row)

            profile_url = row.get(unique_key)
            if not profile_url:
                continue
            rows_with_unique_key += 1

            if profile_url in existing:
                cand_id = existing[profile_url]
                is_update = True
            else:
                next_num = conn.execute("SELECT COUNT(*) FROM candidate").fetchone()[0] + 1
                cand_id = f"{prefix}{next_num:0{width}d}"
                existing[profile_url] = cand_id
                is_update = False

            cand_cfg = cfg["tables"]["candidate"]
            pii_data = {key: row.get(key) for key in cand_cfg["pii_fields"]}
            emails = collect_list(row, cand_cfg["emails"])
            phones = collect_list(row, cand_cfg["phones"])

            full_name = first_present(row.get("original_full_name"), row.get("full_name"))
            first_name = first_present(row.get("original_first_name"), row.get("first_name"))
            last_name = first_present(row.get("original_last_name"), row.get("last_name"))
            primary_email = row.get("email")
            primary_phone = row.get("phone_1")
            languages_raw = row.get("languages")
            languages_json = build_languages(row, max_items=cfg["tables"]["candidate_profile_text"]["language_list"]["max_items"])

            conn.execute(
                """
                INSERT INTO candidate (
                  cand_id, profile_url, full_name, first_name, last_name,
                  primary_email, primary_phone, languages_raw, languages_json,
                  location_name, emails_json, phones_json,
                  source_file, source_imported_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(profile_url) DO UPDATE SET
                  full_name=excluded.full_name,
                  first_name=excluded.first_name,
                  last_name=excluded.last_name,
                  primary_email=excluded.primary_email,
                  primary_phone=excluded.primary_phone,
                  languages_raw=excluded.languages_raw,
                  languages_json=excluded.languages_json,
                  location_name=excluded.location_name,
                  emails_json=excluded.emails_json,
                  phones_json=excluded.phones_json,
                  source_file=excluded.source_file,
                  source_imported_at=excluded.source_imported_at,
                  updated_at=datetime('now')
                """,
                (
                    cand_id,
                    profile_url,
                    full_name,
                    first_name,
                    last_name,
                    primary_email,
                    primary_phone,
                    languages_raw,
                    json.dumps(languages_json, ensure_ascii=False),
                    pii_data.get("location_name"),
                    json.dumps(emails, ensure_ascii=False),
                    json.dumps(phones, ensure_ascii=False),
                    source_file,
                    imported_at,
                ),
            )

            conn.execute(
                """
                INSERT INTO candidate_raw_import (
                  cand_id, raw_profile_json, source_file, source_imported_at, updated_at
                )
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(cand_id) DO UPDATE SET
                  raw_profile_json=excluded.raw_profile_json,
                  source_file=excluded.source_file,
                  source_imported_at=excluded.source_imported_at,
                  updated_at=datetime('now')
                """,
                (
                    cand_id,
                    json.dumps(row, ensure_ascii=False),
                    source_file,
                    imported_at,
                ),
            )

            msg_cfg = cfg["tables"]["candidate_messages"]["fields"]
            msg_data = {key: row.get(key) for key in msg_cfg}
            if any(value is not None for value in msg_data.values()):
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

        conn.execute(
            """
            INSERT INTO import_journal (
              source_file, imported_at, rows_read, rows_with_unique_key, inserted, updated
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_file,
                imported_at,
                int(len(dataframe)),
                int(rows_with_unique_key),
                int(inserted),
                int(updated),
            ),
        )
        conn.commit()

    print(f"Import done. Inserted: {inserted}, Updated (deduped): {updated}, Rows read: {len(dataframe)}")
    print(f"Provenance: source_file={source_file}, source_imported_at={imported_at}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    else:
        csv_file_path = r"data\raw\DevOneIdent_170.csv"
    import_csv_to_sqlite(csv_file_path)
