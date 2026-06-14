import sqlite3
import pandas as pd
import os

BASE = os.path.join(os.path.expanduser("~"), "Desktop",
                    "Dissertation", "PhQure")
DB   = os.path.join(BASE, "data", "phqure.db")

print("=" * 55)
print("  Branch C — Dataset Preparation")
print("=" * 55)

conn = sqlite3.connect(DB)

# 50,000 URLs — balanced (25K phishing + 25K legit)
SAMPLE = 25000

print(f"\n  Loading {SAMPLE*2:,} URLs...")

phishing = pd.read_sql(f"""
    SELECT raw_url as url, label FROM urls
    WHERE label = 1
    LIMIT {SAMPLE}
""", conn)

legit = pd.read_sql(f"""
    SELECT raw_url as url, label FROM urls
    WHERE label = 0
    LIMIT {SAMPLE}
""", conn)

conn.close()

df = pd.concat([phishing, legit], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean URLs
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"].str.len() > 4]
df = df[df["url"].str.len() <= 512]  # DistilBERT max tokens

# Split 80/20
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"])

# Save
out = os.path.join(BASE, "data")
train_df.to_csv(os.path.join(out, "branch_c_train.csv"), index=False)
test_df.to_csv(os.path.join(out,  "branch_c_test.csv"),  index=False)

print(f"  Total URLs     : {len(df):,}")
print(f"  Training set   : {len(train_df):,}")
print(f"  Test set       : {len(test_df):,}")
print(f"  Phishing       : {(df['label']==1).sum():,}")
print(f"  Legitimate     : {(df['label']==0).sum():,}")
print(f"  Max URL length : {df['url'].str.len().max()}")
print(f"  Avg URL length : {df['url'].str.len().mean():.1f}")
print(f"\n  Saved:")
print(f"  - branch_c_train.csv")
print(f"  - branch_c_test.csv")
print("=" * 55)