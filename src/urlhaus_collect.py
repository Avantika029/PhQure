import sqlite3
import requests
import time
import os
import re
from datetime import date

BASE = os.path.join(os.path.expanduser("~"), "Desktop",
                    "Dissertation", "PhQure")
DB   = os.path.join(BASE, "data", "phqure.db")

conn = sqlite3.connect(DB)
cur  = conn.cursor()

# Make sure URLhaus is in sources
cur.execute("INSERT OR IGNORE INTO sources(name,type,description) VALUES(?,?,?)",
            ("URLhaus-API", "malicious", "abuse.ch URLhaus API tag queries"))
conn.commit()
cur.execute("SELECT source_id FROM sources WHERE name='URLhaus-API'")
source_id = cur.fetchone()[0]

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

def fetch_tag(tag):
    try:
        r = requests.post(
            "https://urlhaus-api.abuse.ch/v1/tag/",
            data={"tag": tag},
            timeout=15
        )
        return r.json()
    except Exception as e:
        print(f"  Request failed for tag '{tag}': {e}")
        return None

TAGS = [
    "phishing",
    "malware",
    "ransomware",
    "botnet_cc",
    "emotet",
    "AgentTesla",
    "FormBook",
    "Heodo",
    "Lokibot",
    "TrickBot",
]

print("=" * 55)
print("  URLhaus API Collection")
print("=" * 55)

total_inserted = 0
today = str(date.today())

for tag in TAGS:
    print(f"\n  Fetching tag: {tag} ...")
    data = fetch_tag(tag)

    if not data:
        continue

    if data.get("query_status") != "ok":
        print(f"  No results for tag: {tag}")
        continue

    urls = data.get("urls", [])
    print(f"  Found {len(urls):,} URLs")

    batch = []
    for entry in urls:
        url = str(entry.get("url", "")).strip()
        if not url or len(url) < 4:
            continue
        f = extract_features(url)
        batch.append((url, 1, source_id, today,
                      f[0], f[1], f[2], f[3], f[4], f[5]))

    if batch:
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
        print(f"  Inserted: {inserted:,} new URLs")

    time.sleep(1)

# ── Final count ──
cur.execute("SELECT COUNT(*) FROM urls")
total = cur.fetchone()[0]

print("\n" + "=" * 55)
print(f"  URLhaus collection complete")
print(f"  New URLs inserted : {total_inserted:,}")
print(f"  Total in database : {total:,}")
print("=" * 55)

conn.close()