import csv
import json
import re
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


def collect_list(row, cols):
    values = []
    for col in cols:
        if col in row:
            value = clean_value(row[col])
            if value is not None:
                values.append(value)
    return values


def first_present(*values):
    for value in values:
        cleaned = clean_value(value)
        if cleaned is not None:
            return cleaned
    return None


def build_pattern_items(row, spec, max_items):
    items = []
    for index in range(1, max_items + 1):
        item = {}
        for key, pattern in spec.items():
            col = pattern.format(i=index)
            if col in row:
                value = clean_value(row[col])
                if value is not None:
                    item[key] = value
        if item:
            items.append(item)
    return items


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


def clean_skill_token(token):
    token = clean_value(token)
    if token is None:
        return None
    token = re.sub(r"\s*:\s*(null|none|\d+)\s*$", "", token, flags=re.IGNORECASE)
    token = token.strip()
    return token if token else None


def count_skills(skills_raw):
    skills_raw = clean_value(skills_raw)
    if skills_raw is None:
        return 0

    seen = set()
    for part in str(skills_raw).split(","):
        token = clean_skill_token(part)
        if token is not None:
            seen.add(token.lower())
    return len(seen)


def parse_year_month(value):
    value = clean_value(value)
    if value is None:
        return None

    match = re.match(r"^\s*(\d{4})(?:\D+(\d{1,2}))?", str(value))
    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2) or 1)
    month = min(max(month, 1), 12)
    return year, month


def months_inclusive(start_ym, end_ym):
    if start_ym is None or end_ym is None:
        return None

    diff = (end_ym[0] - start_ym[0]) * 12 + (end_ym[1] - start_ym[1])
    if diff < 0:
        return None
    return diff + 1


def parse_iso_datetime(value):
    value = clean_value(value)
    if value is None:
        return None

    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def build_rank_features(work_items, languages_json, education_items, skills_raw, profile_snapshot_at, imported_at):
    reference_dt = parse_iso_datetime(imported_at) or datetime.utcnow()
    reference_ym = (reference_dt.year, reference_dt.month)

    durations = []
    for item in work_items:
        start_ym = parse_year_month(item.get("start"))
        end_ym = parse_year_month(item.get("end")) or reference_ym
        duration = months_inclusive(start_ym, end_ym)
        if duration is not None:
            durations.append(duration)

    current_role_tenure_months = None
    if work_items:
        current_start = parse_year_month(work_items[0].get("start"))
        current_end = parse_year_month(work_items[0].get("end")) or reference_ym
        current_role_tenure_months = months_inclusive(current_start, current_end)

    language_names = [str(entry.get("name", "")).strip().lower() for entry in languages_json if isinstance(entry, dict)]
    profile_age_days = None
    profile_snapshot_dt = parse_iso_datetime(profile_snapshot_at)
    if profile_snapshot_dt is not None:
        profile_age_days = max((reference_dt - profile_snapshot_dt).days, 0)

    return {
        "work_items_count": len(work_items),
        "total_role_months": sum(durations) if durations else None,
        "current_role_tenure_months": current_role_tenure_months,
        "longest_role_months": max(durations) if durations else None,
        "skills_count": count_skills(skills_raw),
        "languages_count": len(languages_json),
        "education_count": len(education_items),
        "has_german": int(any("deutsch" in name or "german" in name for name in language_names)),
        "has_english": int(any("engl" in name or "english" in name for name in language_names)),
        "profile_age_days": profile_age_days,
    }


def bool_to_int(value):
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    return None


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
            conn.execute("DELETE FROM candidate;")
            conn.commit()
            print("Reset enabled: cleared existing tables before import.")

        existing = dict(conn.execute("SELECT profile_url, cand_id FROM candidate").fetchall())

        inserted = 0
        updated = 0

        for _, row in dataframe.iterrows():
            row = row.to_dict()

            profile_url = clean_value(row.get(unique_key))
            if not profile_url:
                continue

            if profile_url in existing:
                cand_id = existing[profile_url]
                is_update = True
            else:
                next_num = conn.execute("SELECT COUNT(*) FROM candidate").fetchone()[0] + 1
                cand_id = f"{prefix}{next_num:0{width}d}"
                existing[profile_url] = cand_id
                is_update = False

            cand_cfg = cfg["tables"]["candidate"]
            pii_data = {key: clean_value(row.get(key)) for key in cand_cfg["pii_fields"]}

            emails = collect_list(row, cand_cfg["emails"])
            phones = collect_list(row, cand_cfg["phones"])

            full_name = first_present(row.get("original_full_name"), row.get("full_name"))
            first_name = first_present(row.get("original_first_name"), row.get("first_name"))
            last_name = first_present(row.get("original_last_name"), row.get("last_name"))
            primary_email = clean_value(row.get("email"))
            primary_phone = clean_value(row.get("phone_1"))

            conn.execute(
                """
                INSERT INTO candidate (
                  cand_id, profile_url, full_name, first_name, last_name,
                  primary_email, primary_phone, location_name,
                  emails_json, phones_json,
                  source_file, source_imported_at,
                  updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(profile_url) DO UPDATE SET
                  full_name=excluded.full_name,
                  first_name=excluded.first_name,
                  last_name=excluded.last_name,
                  primary_email=excluded.primary_email,
                  primary_phone=excluded.primary_phone,
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
                    pii_data.get("location_name"),
                    json.dumps(emails, ensure_ascii=False),
                    json.dumps(phones, ensure_ascii=False),
                    source_file,
                    imported_at,
                ),
            )

            prof_cfg = cfg["tables"]["candidate_profile_text"]

            headline = clean_value(row.get("headline"))
            summary = clean_value(row.get("summary"))
            skills = clean_value(row.get("skills"))
            location_name = clean_value(row.get("location_name"))
            industry = clean_value(row.get("industry"))
            current_company = first_present(row.get("original_current_company"), row.get("current_company"))
            current_position = first_present(row.get("original_current_company_position"), row.get("current_company_position"))
            badges_job_seeker = bool_to_int(row.get("badges_job_seeker"))
            badges_open_link = bool_to_int(row.get("badges_open_link"))
            profile_snapshot_at = clean_value(row.get("mini_profile_actual_at"))
            languages_raw = clean_value(row.get("languages"))

            languages_json = build_languages(row, max_items=prof_cfg["language_list"]["max_items"])

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
                    "degree": prof_cfg["education"]["degree"],
                    "fos": prof_cfg["education"]["fos"],
                    "description": prof_cfg["education"]["description"],
                },
                prof_cfg["education"]["max_items"],
            )

            conn.execute(
                """
                INSERT INTO candidate_profile_text (
                  cand_id, headline, summary, skills_raw,
                  location_name, industry, current_company, current_position,
                  badges_job_seeker, badges_open_link, profile_snapshot_at,
                  languages_raw, languages_json, work_history_json, education_json,
                  inferred_skills, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, datetime('now'))
                ON CONFLICT(cand_id) DO UPDATE SET
                  headline=excluded.headline,
                  summary=excluded.summary,
                  skills_raw=excluded.skills_raw,
                  location_name=excluded.location_name,
                  industry=excluded.industry,
                  current_company=excluded.current_company,
                  current_position=excluded.current_position,
                  badges_job_seeker=excluded.badges_job_seeker,
                  badges_open_link=excluded.badges_open_link,
                  profile_snapshot_at=excluded.profile_snapshot_at,
                  languages_raw=excluded.languages_raw,
                  languages_json=excluded.languages_json,
                  work_history_json=excluded.work_history_json,
                  education_json=excluded.education_json,
                  updated_at=datetime('now')
                """,
                (
                    cand_id,
                    headline,
                    summary,
                    skills,
                    location_name,
                    industry,
                    current_company,
                    current_position,
                    badges_job_seeker,
                    badges_open_link,
                    profile_snapshot_at,
                    languages_raw,
                    json.dumps(languages_json, ensure_ascii=False),
                    json.dumps(work_items, ensure_ascii=False),
                    json.dumps(edu_items, ensure_ascii=False),
                ),
            )

            rank_features = build_rank_features(
                work_items=work_items,
                languages_json=languages_json,
                education_items=edu_items,
                skills_raw=skills,
                profile_snapshot_at=profile_snapshot_at,
                imported_at=imported_at,
            )

            conn.execute(
                """
                INSERT INTO candidate_rank_features (
                  cand_id, work_items_count, total_role_months,
                  current_role_tenure_months, longest_role_months,
                  skills_count, languages_count, education_count,
                  has_german, has_english, profile_age_days, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(cand_id) DO UPDATE SET
                  work_items_count=excluded.work_items_count,
                  total_role_months=excluded.total_role_months,
                  current_role_tenure_months=excluded.current_role_tenure_months,
                  longest_role_months=excluded.longest_role_months,
                  skills_count=excluded.skills_count,
                  languages_count=excluded.languages_count,
                  education_count=excluded.education_count,
                  has_german=excluded.has_german,
                  has_english=excluded.has_english,
                  profile_age_days=excluded.profile_age_days,
                  updated_at=datetime('now')
                """,
                (
                    cand_id,
                    rank_features["work_items_count"],
                    rank_features["total_role_months"],
                    rank_features["current_role_tenure_months"],
                    rank_features["longest_role_months"],
                    rank_features["skills_count"],
                    rank_features["languages_count"],
                    rank_features["education_count"],
                    rank_features["has_german"],
                    rank_features["has_english"],
                    rank_features["profile_age_days"],
                ),
            )

            msg_cfg = cfg["tables"]["candidate_messages"]["fields"]
            msg_data = {key: clean_value(row.get(key)) for key in msg_cfg}

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

        conn.commit()

    print(f"Import done. Inserted: {inserted}, Updated (deduped): {updated}, Rows read: {len(dataframe)}")
    print(f"Provenance: source_file={source_file}, source_imported_at={imported_at}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    else:
        csv_file_path = r"data\raw\DevOneIdent_170.csv"
    import_csv_to_sqlite(csv_file_path)
