import sqlite3
import os
from collections import defaultdict

BASE = os.path.join(os.path.expanduser("~"), "Desktop",
                    "Dissertation", "PhQure")
DB   = os.path.join(BASE, "data", "phqure.db")

conn = sqlite3.connect(DB)
cur  = conn.cursor()

print("=" * 60)
print("  PhQure -- SQL Analysis Queries (Chapter 3)")
print("=" * 60)

# Q1
print("\n[Q1] Total URLs by Class")
print("-" * 40)
cur.execute("""
    SELECT
        CASE WHEN label = 1 THEN 'Phishing/Malicious'
             ELSE 'Legitimate' END AS class,
        COUNT(*) AS total,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM urls), 2) AS percentage
    FROM urls
    GROUP BY label
    ORDER BY label DESC
""")
for row in cur.fetchall():
    print("  {:<22} {:>10,}  ({}%)".format(row[0], row[1], row[2]))

# Q2
print("\n[Q2] URLs Collected Per Source")
print("-" * 40)
cur.execute("""
    SELECT
        s.name,
        s.type,
        COUNT(u.url_id) AS total_urls,
        SUM(CASE WHEN u.label=1 THEN 1 ELSE 0 END) AS phishing,
        SUM(CASE WHEN u.label=0 THEN 1 ELSE 0 END) AS legitimate
    FROM sources s
    LEFT JOIN urls u ON s.source_id = u.source_id
    GROUP BY s.source_id
    ORDER BY total_urls DESC
""")
print("  {:<30} {:<12} {:>10} {:>10} {:>10}".format(
    "Source","Type","Total","Phishing","Legit"))
for row in cur.fetchall():
    print("  {:<30} {:<12} {:>10,} {:>10,} {:>10,}".format(
        str(row[0]), str(row[1]), row[2] or 0, row[3] or 0, row[4] or 0))

# Q3
print("\n[Q3] URL Length Statistics by Class")
print("-" * 40)
cur.execute("""
    SELECT
        CASE WHEN label=1 THEN 'Phishing' ELSE 'Legitimate' END AS class,
        ROUND(AVG(url_length), 2) AS avg_length,
        MIN(url_length)           AS min_length,
        MAX(url_length)           AS max_length
    FROM urls
    GROUP BY label
""")
print("  {:<12} {:>8} {:>8} {:>8}".format("Class","Avg","Min","Max"))
for row in cur.fetchall():
    print("  {:<12} {:>8} {:>8} {:>8}".format(row[0], row[1], row[2], row[3]))

# Q4
print("\n[Q4] HTTPS Usage -- Phishing vs Legitimate")
print("-" * 40)
cur.execute("""
    SELECT
        CASE WHEN label=1 THEN 'Phishing' ELSE 'Legitimate' END AS class,
        SUM(has_https) AS uses_https,
        COUNT(*)       AS total,
        ROUND(SUM(has_https)*100.0/COUNT(*), 2) AS pct_https
    FROM urls
    GROUP BY label
""")
print("  {:<12} {:>12} {:>10} {:>10}".format("Class","Uses HTTPS","Total","% HTTPS"))
for row in cur.fetchall():
    print("  {:<12} {:>12,} {:>10,} {:>10}%".format(row[0], row[1], row[2], row[3]))

# Q5
print("\n[Q5] IP Address in URL -- Phishing vs Legitimate")
print("-" * 40)
cur.execute("""
    SELECT
        CASE WHEN label=1 THEN 'Phishing' ELSE 'Legitimate' END AS class,
        SUM(has_ip) AS has_ip_count,
        COUNT(*)    AS total,
        ROUND(SUM(has_ip)*100.0/COUNT(*), 2) AS pct_ip
    FROM urls
    GROUP BY label
""")
print("  {:<12} {:>10} {:>10} {:>10}".format("Class","Has IP","Total","% Has IP"))
for row in cur.fetchall():
    print("  {:<12} {:>10,} {:>10,} {:>10}%".format(row[0], row[1], row[2], row[3]))

# Q6
print("\n[Q6] @ Symbol Presence -- Phishing vs Legitimate")
print("-" * 40)
cur.execute("""
    SELECT
        CASE WHEN label=1 THEN 'Phishing' ELSE 'Legitimate' END AS class,
        SUM(has_at_symbol) AS has_at,
        COUNT(*)           AS total,
        ROUND(SUM(has_at_symbol)*100.0/COUNT(*), 2) AS pct_at
    FROM urls
    GROUP BY label
""")
print("  {:<12} {:>10} {:>10} {:>10}".format("Class","Has @","Total","% Has @"))
for row in cur.fetchall():
    print("  {:<12} {:>10,} {:>10,} {:>10}%".format(row[0], row[1], row[2], row[3]))

# Q7
print("\n[Q7] URL Length Percentiles (Window Function)")
print("-" * 40)
cur.execute("""
    SELECT
        CASE WHEN label=1 THEN 'Phishing' ELSE 'Legitimate' END AS class,
        url_length,
        NTILE(4) OVER (PARTITION BY label ORDER BY url_length) AS quartile
    FROM urls
""")
rows = cur.fetchall()
quartiles = defaultdict(list)
for cls, length, q in rows:
    quartiles[(cls, q)].append(length)

print("  {:<12} {:>10} {:>10} {:>10} {:>10}".format(
    "Class","Q1 (25%)","Q2 (50%)","Q3 (75%)","Q4 (100%)"))
for cls in ["Phishing", "Legitimate"]:
    vals = []
    for q in [1, 2, 3, 4]:
        lengths = quartiles[(cls, q)]
        vals.append(max(lengths) if lengths else 0)
    print("  {:<12} {:>10} {:>10} {:>10} {:>10}".format(cls, *vals))

# Q8
print("\n[Q8] Average Number of Dots in URL by Class")
print("-" * 40)
cur.execute("""
    SELECT
        CASE WHEN label=1 THEN 'Phishing' ELSE 'Legitimate' END AS class,
        ROUND(AVG(num_dots), 3) AS avg_dots,
        MAX(num_dots)           AS max_dots
    FROM urls
    GROUP BY label
""")
print("  {:<12} {:>10} {:>10}".format("Class","Avg Dots","Max Dots"))
for row in cur.fetchall():
    print("  {:<12} {:>10} {:>10}".format(row[0], row[1], row[2]))

# Q9
print("\n[Q9] CTE -- Multi-factor Risk Score (Top 10 Riskiest Phishing URLs)")
print("-" * 40)
cur.execute("""
    WITH risk_scores AS (
        SELECT
            url_id,
            raw_url,
            label,
            (has_ip * 3 +
             has_at_symbol * 3 +
             has_hyphen * 1 +
             CASE WHEN has_https = 0 THEN 2 ELSE 0 END +
             CASE WHEN url_length > 75 THEN 2 ELSE 0 END +
             CASE WHEN num_dots > 4 THEN 1 ELSE 0 END
            ) AS risk_score
        FROM urls
        WHERE label = 1
    )
    SELECT url_id, SUBSTR(raw_url, 1, 50) AS url_preview, risk_score
    FROM risk_scores
    ORDER BY risk_score DESC
    LIMIT 10
""")
print("  {:>8} {:<52} {:>5}".format("ID","URL (preview)","Risk"))
for row in cur.fetchall():
    print("  {:>8} {:<52} {:>5}".format(row[0], row[1], row[2]))

# Q10
print("\n[Q10] Running Total of URLs Collected (Window Function)")
print("-" * 40)
cur.execute("""
    SELECT
        date_collected,
        COUNT(*) AS added_today,
        SUM(COUNT(*)) OVER (ORDER BY date_collected) AS running_total
    FROM urls
    GROUP BY date_collected
    ORDER BY date_collected
""")
print("  {:<14} {:>10} {:>15}".format("Date","Added","Running Total"))
for row in cur.fetchall():
    print("  {:<14} {:>10,} {:>15,}".format(str(row[0]), row[1], row[2]))

print("\n" + "=" * 60)
print("  All 10 queries complete -- save this output for Chapter 3")
print("=" * 60)
conn.close()