import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import yaml

from dbx.paths import DB


def clean_value(value):
    if value is None:
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


IAM_KEYWORDS = [
    "identity and access management",
    "identity management",
    "access management",
    "one identity",
    "oneidentity",
    "active directory",
    "berechtigungsmanagement",
    "sailpoint",
    "saviynt",
    "dirx",
    "bi-cube",
    "bicube",
    "ldap",
    "provisioning",
    "rezertifizierung",
    "recertification",
    "joiner",
    "mover",
    "leaver",
]


def looks_like_iam_role(item):
    haystack = " ".join(
        str(item.get(key, "")).strip().lower()
        for key in ("organization", "title", "description")
        if item.get(key)
    )
    return any(keyword in haystack for keyword in IAM_KEYWORDS)


def build_rank_features(work_items, education_items, skills_raw, profile_snapshot_at, imported_at):
    duration_reference_dt = parse_iso_datetime(profile_snapshot_at) or parse_iso_datetime(imported_at) or datetime.utcnow()
    freshness_reference_dt = parse_iso_datetime(imported_at) or datetime.utcnow()
    reference_ym = (duration_reference_dt.year, duration_reference_dt.month)

    durations = []
    iam_role_months = 0
    iam_role_count = 0
    for item in work_items:
        start_ym = parse_year_month(item.get("start"))
        end_ym = parse_year_month(item.get("end")) or reference_ym
        duration = months_inclusive(start_ym, end_ym)
        if duration is not None:
            durations.append(duration)
            if looks_like_iam_role(item):
                iam_role_months += duration
                iam_role_count += 1

    current_listed_role_months = None
    current_role_is_iam = None
    if work_items:
        current_start = parse_year_month(work_items[0].get("start"))
        current_end = parse_year_month(work_items[0].get("end")) or reference_ym
        current_listed_role_months = months_inclusive(current_start, current_end)
        current_role_is_iam = int(looks_like_iam_role(work_items[0]))

    profile_age_days = None
    profile_snapshot_dt = parse_iso_datetime(profile_snapshot_at)
    if profile_snapshot_dt is not None:
        profile_age_days = max((freshness_reference_dt - profile_snapshot_dt).days, 0)

    return {
        "work_items_count": len(work_items),
        "listed_role_months_sum": sum(durations) if durations else None,
        "current_listed_role_months": current_listed_role_months,
        "longest_listed_role_months": max(durations) if durations else None,
        "iam_role_months": iam_role_months if iam_role_count else None,
        "iam_role_count": iam_role_count,
        "current_role_is_iam": current_role_is_iam,
        "skills_count": count_skills(skills_raw),
        "education_count": len(education_items),
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


def feature_engineering(db_path=DB, config_path="config.yml", reset_derived=False):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")

        if reset_derived:
            conn.execute("DELETE FROM candidate_rank_features;")
            conn.execute("DELETE FROM candidate_profile_text;")

        rows = conn.execute(
            """
            SELECT cand_id, raw_profile_json, source_imported_at
            FROM candidate_raw_import
            ORDER BY cand_id
            """
        ).fetchall()

        prof_cfg = cfg["tables"]["candidate_profile_text"]
        processed = 0

        for cand_id, raw_profile_json, source_imported_at in rows:
            try:
                row = json.loads(raw_profile_json or "{}")
            except json.JSONDecodeError:
                row = {}

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
                  work_history_json, education_json,
                  inferred_skills, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, datetime('now'))
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
                    json.dumps(work_items, ensure_ascii=False),
                    json.dumps(edu_items, ensure_ascii=False),
                ),
            )

            rank_features = build_rank_features(
                work_items=work_items,
                education_items=edu_items,
                skills_raw=skills,
                profile_snapshot_at=profile_snapshot_at,
                imported_at=source_imported_at,
            )

            conn.execute(
                """
                INSERT INTO candidate_rank_features (
                  cand_id, work_items_count, listed_role_months_sum,
                  current_listed_role_months, longest_listed_role_months,
                  iam_role_months, iam_role_count, current_role_is_iam,
                  skills_count, education_count, profile_age_days, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(cand_id) DO UPDATE SET
                  work_items_count=excluded.work_items_count,
                  listed_role_months_sum=excluded.listed_role_months_sum,
                  current_listed_role_months=excluded.current_listed_role_months,
                  longest_listed_role_months=excluded.longest_listed_role_months,
                  iam_role_months=excluded.iam_role_months,
                  iam_role_count=excluded.iam_role_count,
                  current_role_is_iam=excluded.current_role_is_iam,
                  skills_count=excluded.skills_count,
                  education_count=excluded.education_count,
                  profile_age_days=excluded.profile_age_days,
                  updated_at=datetime('now')
                """,
                (
                    cand_id,
                    rank_features["work_items_count"],
                    rank_features["listed_role_months_sum"],
                    rank_features["current_listed_role_months"],
                    rank_features["longest_listed_role_months"],
                    rank_features["iam_role_months"],
                    rank_features["iam_role_count"],
                    rank_features["current_role_is_iam"],
                    rank_features["skills_count"],
                    rank_features["education_count"],
                    rank_features["profile_age_days"],
                ),
            )
            processed += 1

        conn.commit()

    print(f"Feature engineering done. Processed: {processed}")


if __name__ == "__main__":
    feature_engineering()
