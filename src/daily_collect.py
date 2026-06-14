import sqlite3
import requests
import re
import os
from datetime import date

BASE = os.path.join(os.path.expanduser("~"), "Desktop",
                    "Dissertation", "PhQure")
DB   = os.path.join(BASE, "data", "phqure.db")

conn = sqlite3.connect(DB)
cur  = conn.cursor()

cur.execute("INSERT OR IGNORE INTO sources(name,type,description) VALUES(?,?,?)",
            ("OpenPhish-Daily", "phishing", "Daily OpenPhish live feed"))
conn.commit()
cur.execute("SELECT source_id FROM sources WHERE name='OpenPhish-Daily'")
source_id = cur.fetchone()[0]
today = str(date.today())

def extract_features(url):
    url = str(url).strip()
    return (
        len(url),
        1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        1 if url.lower().startswith("https") else 0,
        1 if "@" in url else 0,
        1 if (len(url.split("/")) > 2 and "-" in url.split("/")[2]) else 0,
        url.count(".")
    )

print("=" * 55)
print(f"  Daily Collection — {today}")
print("=" * 55)

total_inserted = 0

# ── OpenPhish ──
print("\n  Fetching OpenPhish feed...")
try:
    r = requests.get("https://openphish.com/feed.txt", timeout=15)
    urls = [u.strip() for u in r.text.splitlines()
            if u.strip() and u.strip().startswith("http")]

    batch = []
    for url in urls:
        f = extract_features(url)
        batch.append((url, 1, source_id, today,
                      f[0], f[1], f[2], f[3], f[4], f[5]))

    cur.executemany("""
        INSERT OR IGNORE INTO urls
        (raw_url, label, source_id, date_collected,
         url_length, has_ip, has_https,
         has_at_symbol, has_hyphen, num_dots)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, batch)
    conn.commit()
    inserted = cur.rowcount
    total_inserted += inserted
    print(f"  OpenPhish  : {len(urls):,} fetched  |  {inserted:,} new")

except Exception as e:
    print(f"  OpenPhish failed: {e}")

# ── URLhaus latest ──
print("\n  Fetching URLhaus recent...")
try:
    r = requests.post(
        "https://urlhaus-api.abuse.ch/v1/tag/",
        data={"tag": "phishing"},
        timeout=15
    )
    data = r.json()
    if data.get("query_status") == "ok":
        urls = data.get("urls", [])
        batch = []
        for entry in urls:
            url = str(entry.get("url", "")).strip()
            if not url:
                continue
            f = extract_features(url)
            batch.append((url, 1, source_id, today,
                          f[0], f[1], f[2], f[3], f[4], f[5]))
        cur.executemany("""
            INSERT OR IGNORE INTO urls
            (raw_url, label, source_id, date_collected,
             url_length, has_ip, has_https,
             has_at_symbol, has_hyphen, num_dots)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, batch)
        conn.commit()
        inserted = cur.rowcount
        total_inserted += inserted
        print(f"  URLhaus    : {len(urls):,} fetched  |  {inserted:,} new")
except Exception as e:
    print(f"  URLhaus failed: {e}")

# ── Log to daily_stats ──
cur.execute("""
    INSERT INTO daily_stats(date, source_id, count_phishing)
    VALUES (?, ?, ?)
""", (today, source_id, total_inserted))
conn.commit()

# ── Summary ──
cur.execute("SELECT COUNT(*) FROM urls")
db_total = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM urls WHERE label=1")
phishing = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM urls WHERE label=0")
legit = cur.fetchone()[0]

print(f"\n{'='*55}")
print(f"  Daily collection complete — {today}")
print(f"  New URLs added today : {total_inserted:,}")
print(f"  Total in database    : {db_total:,}")
print(f"  Phishing             : {phishing:,}")
print(f"  Legitimate           : {legit:,}")
print(f"{'='*55}")

conn.close()