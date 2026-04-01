-- schema.sql

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candidate (
  cand_id TEXT PRIMARY KEY,
  profile_url TEXT UNIQUE NOT NULL,

  full_name TEXT,
  first_name TEXT,
  last_name TEXT,
  primary_email TEXT,
  primary_phone TEXT,
  languages_raw TEXT,
  languages_json TEXT,

  location_name TEXT,
  emails_json TEXT,
  phones_json TEXT,

  source_file TEXT,
  source_imported_at TEXT,

  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))

);

CREATE TABLE IF NOT EXISTS candidate_profile_text (
  cand_id TEXT PRIMARY KEY,
  headline TEXT,
  summary TEXT,
  skills_raw TEXT,
  location_name TEXT,
  industry TEXT,
  current_company TEXT,
  current_position TEXT,
  badges_job_seeker INTEGER,
  badges_open_link INTEGER,
  profile_snapshot_at TEXT,
  work_history_json TEXT,
  education_json TEXT,

  inferred_skills TEXT,

  updated_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (cand_id) REFERENCES candidate(cand_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidate_rank_features (
  cand_id TEXT PRIMARY KEY,

  work_items_count INTEGER,
  listed_role_months_sum INTEGER,
  current_listed_role_months INTEGER,
  longest_listed_role_months INTEGER,
  iam_role_months INTEGER,
  iam_role_count INTEGER,
  current_role_is_iam INTEGER,
  skills_count INTEGER,
  education_count INTEGER,
  profile_age_days INTEGER,

  updated_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (cand_id) REFERENCES candidate(cand_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidate_messages (
  cand_id TEXT PRIMARY KEY,

  full_messaging_history TEXT,
  last_sent_message_from TEXT,
  last_sent_message_text TEXT,
  last_received_message_from TEXT,
  last_received_message_text TEXT,
  last_sent_message_send_at TEXT,
  last_received_message_send_at TEXT,

  updated_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (cand_id) REFERENCES candidate(cand_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS import_journal (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_file TEXT NOT NULL,
  imported_at TEXT NOT NULL,
  rows_read INTEGER NOT NULL,
  rows_with_unique_key INTEGER NOT NULL,
  inserted INTEGER NOT NULL,
  updated INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidate_profile_url ON candidate(profile_url);
