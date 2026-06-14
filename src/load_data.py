import sqlite3
import pandas as pd
import os
import re
from datetime import date
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────
BASE    = os.path.join(os.path.expanduser("~"), "Desktop", "Dissertation", "PhQure")
DB      = os.path.join(BASE, "data", "phqure.db")
DATA    = os.path.join(BASE, "data")

# ── Connect ────────────────────────────────────────
conn = sqlite3.connect(DB)
cur  = conn.cursor()

# ── Feature extraction ─────────────────────────────
def features(url):
    url = str(url).strip()
    return {
        "url_length"   : len(url),
        "has_ip"       : 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        "has_https"    : 1 if url.lower().startswith("https") else 0,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_hyphen"   : 1 if (len(url.split("/")) > 2 and "-" in url.split("/")[2]) else 0,
        "num_dots"     : url.count("."),
    }

# ── Bulk insert ────────────────────────────────────
def insert(rows, source_name):
    cur.execute("SELECT source_id FROM sources WHERE name=?", (source_name,))
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO sources(name,type,description) VALUES(?,?,?)",
                    (source_name, "mixed", source_name))
        conn.commit()
        source_id = cur.lastrowid
    else:
        source_id = row[0]

    today = str(date.today())
    batch = []
    for url, label in rows:
        url = str(url).strip()
        if not url or len(url) < 4:
            continue
        f = features(url)
        batch.append((
            url, label, source_id, today,
            f["url_length"], f["has_ip"], f["has_https"],
            f["has_at_symbol"], f["has_hyphen"], f["num_dots"]
        ))

    # Insert in batches of 10,000
    inserted = 0
    for i in range(0, len(batch), 10000):
        cur.executemany("""
            INSERT OR IGNORE INTO urls
            (raw_url, label, source_id, date_collected,
             url_length, has_ip, has_https,
             has_at_symbol, has_hyphen, num_dots)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, batch[i:i+10000])
        conn.commit()
        inserted += cur.rowcount
        print(f"  {source_name}: {min(i+10000, len(batch)):,} / {len(batch):,} processed")

    return inserted


total = 0
print("\n" + "="*55)
print("  PhQure — Data Loading Pipeline")
print("="*55)


# ══════════════════════════════════════════════════
# SOURCE 1 — malicious_phish.csv
# columns: url, type
# types: phishing, benign, malware, defacement
# ══════════════════════════════════════════════════
print("\n[1/4] Loading malicious_phish.csv ...")
try:
    df = pd.read_csv(os.path.join(DATA, "malicious_phish.csv"))
    df.columns = df.columns.str.strip().str.lower()

    # Label mapping
    label_map = {
        "phishing"   : 1,
        "malware"    : 1,
        "defacement" : 1,
        "benign"     : 0,
    }
    df["label"] = df["type"].str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)

    rows = list(zip(df["url"], df["label"]))
    n = insert(rows, "Kaggle-MaliciousPhish")
    total += n
    print(f"  Done — {n:,} new URLs inserted")
    print(f"  Phishing/Malware : {(df['label']==1).sum():,}")
    print(f"  Benign           : {(df['label']==0).sum():,}")
except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════
# SOURCE 2 — tranco_KW6WW.csv
# No header. Format: rank, domain
# All legitimate (label = 0)
# ══════════════════════════════════════════════════
print("\n[2/4] Loading tranco (Top 1M legitimate domains) ...")
try:
    # Find the tranco file (name changes each download)
    tranco_file = None
    for f in os.listdir(DATA):
        if f.lower().startswith("tranco") and f.endswith(".csv"):
            tranco_file = f
            break

    if not tranco_file:
        print("  ERROR: tranco file not found in data folder")
    else:
        df = pd.read_csv(
            os.path.join(DATA, tranco_file),
            header=None,
            names=["rank", "domain"]
        )
        df = df.dropna(subset=["domain"])

        # Convert domain → full URL
        df["url"] = "https://" + df["domain"].str.strip()
        rows = list(zip(df["url"], [0] * len(df)))

        n = insert(rows, "Tranco-Top1M")
        total += n
        print(f"  Done — {n:,} new URLs inserted")
except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════
# SOURCE 3 — openphish.txt
# One URL per line, all phishing (label = 1)
# ══════════════════════════════════════════════════
print("\n[3/4] Loading openphish.txt ...")
try:
    with open(os.path.join(DATA, "openphish.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f.readlines()
                 if l.strip() and l.strip().startswith("http")]

    rows = [(url, 1) for url in lines]
    n = insert(rows, "OpenPhish")
    total += n
    print(f"  Done — {n:,} new URLs inserted")
except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════
# SOURCE 4 — phishing-links-ACTIVE.txt (mitchellkrogza)
# One URL per line, all phishing (label = 1)
# ══════════════════════════════════════════════════
print("\n[4/4] Loading phishing-links-ACTIVE.txt ...")
try:
    filepath = os.path.join(DATA, "phishing-links-ACTIVE.txt")
    if not os.path.exists(filepath):
        print("  SKIPPED — file not found (still downloading?)")
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines()
                     if l.strip() and not l.startswith("#")]
        rows = [(url, 1) for url in lines]
        n = insert(rows, "Mitchellkrogza-PhishDB")
        total += n
        print(f"  Done — {n:,} new URLs inserted")
except Exception as e:
    print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("  LOADING COMPLETE")
print("="*55)

cur.execute("SELECT COUNT(*) FROM urls")
db_total = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM urls WHERE label=1")
phishing_count = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM urls WHERE label=0")
legit_count = cur.fetchone()[0]

cur.execute("""
    SELECT s.name, COUNT(u.url_id)
    FROM urls u
    JOIN sources s ON u.source_id = s.source_id
    GROUP BY s.name
    ORDER BY COUNT(u.url_id) DESC
""")
by_source = cur.fetchall()

print(f"\n  Total URLs in database : {db_total:,}")
print(f"  Phishing / Malicious   : {phishing_count:,}")
print(f"  Legitimate             : {legit_count:,}")
print(f"\n  By Source:")
for name, count in by_source:
    print(f"    {name:<30} {count:,}")

print("="*55)
conn.close()