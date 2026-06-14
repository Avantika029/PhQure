import sqlite3
import pandas as pd
import numpy as np
import re
import os
from urllib.parse import urlparse

BASE = os.path.join(os.path.expanduser("~"), "Desktop",
                    "Dissertation", "PhQure")
DB   = os.path.join(BASE, "data", "phqure.db")

conn = sqlite3.connect(DB)

# ── Feature extraction functions ──────────────────

def having_ip(url):
    return 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

def url_length(url):
    if len(url) < 54:   return 1
    elif len(url) <= 75: return 0
    else:               return -1

def shortening_service(url):
    pattern = r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net'
    return 1 if re.search(pattern, url) else 0

def having_at(url):
    return 1 if '@' in url else 0

def double_slash(url):
    return 1 if '//' in url[7:] else 0

def prefix_suffix(url):
    try:
        domain = urlparse(url).netloc
        return 1 if '-' in domain else 0
    except:
        return 0

def num_subdomains(url):
    try:
        domain = urlparse(url).netloc
        parts = domain.split('.')
        if len(parts) == 2:   return 1
        elif len(parts) == 3: return 0
        else:                 return -1
    except:
        return 0

def https_token(url):
    return 1 if url.startswith('https') else 0

def num_dots(url):
    return url.count('.')

def num_hyphens(url):
    return url.count('-')

def num_digits(url):
    return sum(c.isdigit() for c in url)

def num_special_chars(url):
    return sum(not c.isalnum() for c in url)

def url_depth(url):
    try:
        path = urlparse(url).path
        return len([p for p in path.split('/') if p])
    except:
        return 0

def has_port(url):
    try:
        port = urlparse(url).port
        return 1 if port and port not in [80, 443] else 0
    except:
        return 0

def path_length(url):
    try:
        return len(urlparse(url).path)
    except:
        return 0

def query_length(url):
    try:
        return len(urlparse(url).query)
    except:
        return 0

def has_query(url):
    try:
        return 1 if urlparse(url).query else 0
    except:
        return 0

def domain_length(url):
    try:
        return len(urlparse(url).netloc)
    except:
        return 0

def digit_ratio_url(url):
    try:
        digits = sum(c.isdigit() for c in url)
        return round(digits / len(url), 4) if url else 0
    except:
        return 0

def letter_ratio_url(url):
    try:
        letters = sum(c.isalpha() for c in url)
        return round(letters / len(url), 4) if url else 0
    except:
        return 0

def has_suspicious_words(url):
    words = ['secure', 'account', 'update', 'login', 'signin',
             'banking', 'confirm', 'verify', 'password', 'support',
             'paypal', 'ebay', 'amazon', 'apple', 'microsoft']
    url_lower = url.lower()
    return 1 if any(w in url_lower for w in words) else 0

def has_hex_encoding(url):
    return 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0

def tld_length(url):
    try:
        domain = urlparse(url).netloc
        parts = domain.split('.')
        return len(parts[-1]) if parts else 0
    except:
        return 0

def count_www(url):
    return url.lower().count('www')

def count_com(url):
    return url.lower().count('.com')

def has_iframe(url):
    return 0  # Cannot detect from URL alone — default 0

def mouse_over(url):
    return 0  # Cannot detect from URL alone — default 0

def right_click(url):
    return 0  # Cannot detect from URL alone — default 0

def forwarding(url):
    return 0  # Cannot detect from URL alone — default 0

def extract_all_features(url):
    url = str(url).strip()
    return {
        'having_ip'           : having_ip(url),
        'url_length'          : url_length(url),
        'shortening_service'  : shortening_service(url),
        'having_at'           : having_at(url),
        'double_slash'        : double_slash(url),
        'prefix_suffix'       : prefix_suffix(url),
        'num_subdomains'      : num_subdomains(url),
        'https_token'         : https_token(url),
        'num_dots'            : num_dots(url),
        'num_hyphens'         : num_hyphens(url),
        'num_digits'          : num_digits(url),
        'num_special_chars'   : num_special_chars(url),
        'url_depth'           : url_depth(url),
        'has_port'            : has_port(url),
        'path_length'         : path_length(url),
        'query_length'        : query_length(url),
        'has_query'           : has_query(url),
        'domain_length'       : domain_length(url),
        'digit_ratio'         : digit_ratio_url(url),
        'letter_ratio'        : letter_ratio_url(url),
        'suspicious_words'    : has_suspicious_words(url),
        'hex_encoding'        : has_hex_encoding(url),
        'tld_length'          : tld_length(url),
        'count_www'           : count_www(url),
        'count_com'           : count_com(url),
        'has_iframe'          : has_iframe(url),
        'mouse_over'          : mouse_over(url),
        'right_click'         : right_click(url),
        'forwarding'          : forwarding(url),
        'label'               : None,
    }

# ── Load URLs from database ────────────────────────
print("=" * 55)
print("  Branch A — Feature Extraction")
print("=" * 55)

# Use 100,000 URLs for training (50K phishing + 50K legit)
# Balanced dataset for fair evaluation
SAMPLE = 50000

print(f"\n  Loading {SAMPLE*2:,} URLs from database...")
print(f"  ({SAMPLE:,} phishing + {SAMPLE:,} legitimate)")

phishing = pd.read_sql(f"""
    SELECT raw_url, label FROM urls
    WHERE label = 1
    LIMIT {SAMPLE}
""", conn)

legit = pd.read_sql(f"""
    SELECT raw_url, label FROM urls
    WHERE label = 0
    LIMIT {SAMPLE}
""", conn)

df = pd.concat([phishing, legit], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
conn.close()

print(f"  Loaded {len(df):,} URLs")
print(f"\n  Extracting 29 features per URL...")
print(f"  This takes 5-10 minutes — please wait...")

# ── Extract features ───────────────────────────────
features = []
for i, row in enumerate(df.itertuples(), 1):
    f = extract_all_features(row.raw_url)
    f['label'] = row.label
    features.append(f)
    if i % 10000 == 0:
        print(f"  Progress: {i:,} / {len(df):,}")

feature_df = pd.DataFrame(features)

# ── Save to CSV ────────────────────────────────────
out_path = os.path.join(BASE, "data", "branch_a_features.csv")
feature_df.to_csv(out_path, index=False)

print(f"\n{'='*55}")
print(f"  Feature extraction complete")
print(f"  Total rows      : {len(feature_df):,}")
print(f"  Total features  : {len(feature_df.columns)-1}")
print(f"  Phishing        : {(feature_df['label']==1).sum():,}")
print(f"  Legitimate      : {(feature_df['label']==0).sum():,}")
print(f"  Saved to        : {out_path}")
print(f"{'='*55}")