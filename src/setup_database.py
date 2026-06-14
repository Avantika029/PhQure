import sqlite3
import os

# Create database inside data folder
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'phqure.db')

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.executescript("""

-- TABLE 1: Data sources
CREATE TABLE IF NOT EXISTS sources (
    source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    type        TEXT,
    description TEXT
);

-- TABLE 2: All URLs
CREATE TABLE IF NOT EXISTS urls (
    url_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_url        TEXT NOT NULL UNIQUE,
    label          INTEGER NOT NULL,
    source_id      INTEGER,
    date_collected TEXT,
    url_length     INTEGER,
    has_ip         INTEGER DEFAULT 0,
    has_https      INTEGER DEFAULT 0,
    has_at_symbol  INTEGER DEFAULT 0,
    has_hyphen     INTEGER DEFAULT 0,
    num_dots       INTEGER DEFAULT 0,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

-- TABLE 3: Daily collection stats
CREATE TABLE IF NOT EXISTS daily_stats (
    stat_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date           TEXT,
    source_id      INTEGER,
    count_phishing INTEGER DEFAULT 0,
    count_legit    INTEGER DEFAULT 0,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

-- INDEXES (for fast queries)
CREATE INDEX IF NOT EXISTS idx_label    ON urls(label);
CREATE INDEX IF NOT EXISTS idx_source   ON urls(source_id);
CREATE INDEX IF NOT EXISTS idx_date     ON urls(date_collected);
CREATE INDEX IF NOT EXISTS idx_length   ON urls(url_length);
CREATE INDEX IF NOT EXISTS idx_https    ON urls(has_https);

-- VIEWS
CREATE VIEW IF NOT EXISTS phishing_urls AS
    SELECT * FROM urls WHERE label = 1;

CREATE VIEW IF NOT EXISTS legit_urls AS
    SELECT * FROM urls WHERE label = 0;

CREATE VIEW IF NOT EXISTS source_summary AS
    SELECT
        s.name,
        COUNT(u.url_id)                                    AS total_urls,
        SUM(CASE WHEN u.label = 1 THEN 1 ELSE 0 END)      AS phishing,
        SUM(CASE WHEN u.label = 0 THEN 1 ELSE 0 END)      AS legitimate,
        MIN(u.date_collected)                              AS first_collected,
        MAX(u.date_collected)                              AS last_collected
    FROM sources s
    LEFT JOIN urls u ON s.source_id = u.source_id
    GROUP BY s.source_id;

""")

# Insert known sources
cur.executemany(
    "INSERT OR IGNORE INTO sources(name, type, description) VALUES (?,?,?)",
    [
        ("Tranco",    "legitimate", "Tranco Top-1M legitimate domains"),
        ("PhishTank", "phishing",   "Community-verified phishing URLs"),
        ("OpenPhish", "phishing",   "Live phishing feed, updated daily"),
        ("URLhaus",   "malicious",  "abuse.ch malicious URL feed via API"),
        ("Kaggle",    "mixed",      "Kaggle phishing datasets"),
    ]
)
conn.commit()
conn.close()

print("=" * 50)
print("  PhQure Database Created Successfully!")
print("=" * 50)
print(f"  Location : {os.path.abspath(DB_PATH)}")
print("  Tables   : urls, sources, daily_stats")
print("  Indexes  : 5 created")
print("  Views    : 3 created (phishing, legit, source_summary)")
print("=" * 50)