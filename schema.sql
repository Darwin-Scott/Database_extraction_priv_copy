-- schema.sql

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candidate (
  cand_id TEXT PRIMARY KEY,
  profile_url TEXT UNIQUE NOT NULL,

  full_name TEXT,
  first_name TEXT,
  last_name TEXT,

  location_name TEXT,
  industry TEXT,

  address TEXT,
  avatar TEXT,

  emails_json TEXT,
  phones_json TEXT,
  websites_json TEXT,

  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS candidate_profile_text (
  cand_id TEXT PRIMARY KEY,
  headline TEXT,
  summary TEXT,
  skills_raw TEXT,

  languages_json TEXT,
  work_history_json TEXT,
  education_json TEXT,

  tags TEXT,
  note_public TEXT,

  inferred_skills TEXT,

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

CREATE INDEX IF NOT EXISTS idx_candidate_profile_url ON candidate(profile_url);