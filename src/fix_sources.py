import sqlite3
import os

DB = os.path.join(os.path.expanduser("~"), "Desktop",
                  "Dissertation", "PhQure", "data", "phqure.db")

conn = sqlite3.connect(DB)
cur  = conn.cursor()

cur.execute("""
    DELETE FROM sources
    WHERE source_id NOT IN (
        SELECT DISTINCT source_id FROM urls
    )
""")
deleted = cur.rowcount
conn.commit()

cur.execute("SELECT source_id, name, type FROM sources ORDER BY source_id")
rows = cur.fetchall()

print(f"Deleted {deleted} empty source rows")
print("Remaining sources:")
for r in rows:
    print(f"  {r}")

conn.close()