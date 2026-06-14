import sqlite3
import qrcode
import os
from PIL import Image
from datetime import date

BASE     = os.path.join(os.path.expanduser("~"), "Desktop",
                        "Dissertation", "PhQure")
DB       = os.path.join(BASE, "data", "phqure.db")
QR_DIR   = os.path.join(BASE, "data", "qr_images")
PHISH_DIR= os.path.join(QR_DIR, "phishing")
LEGIT_DIR= os.path.join(QR_DIR, "legitimate")

# Create subfolders
os.makedirs(PHISH_DIR, exist_ok=True)
os.makedirs(LEGIT_DIR, exist_ok=True)

conn = sqlite3.connect(DB)
cur  = conn.cursor()

def make_qr(url, save_path):
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((224, 224), Image.LANCZOS)
        img.save(save_path)
        return True
    except Exception:
        return False

# ── How many to generate ──
PHISH_LIMIT = 15000   # 15,000 phishing QR codes
LEGIT_LIMIT = 15000   # 15,000 legitimate QR codes

print("=" * 55)
print("  PhQure — QR Code Generation Pipeline")
print("=" * 55)
print(f"  Target : {PHISH_LIMIT:,} phishing + {LEGIT_LIMIT:,} legitimate")
print(f"  Output : {QR_DIR}")
print("=" * 55)

# ── Already generated ──
existing_phish = set(os.listdir(PHISH_DIR))
existing_legit = set(os.listdir(LEGIT_DIR))
print(f"\n  Already generated:")
print(f"  Phishing   : {len(existing_phish):,}")
print(f"  Legitimate : {len(existing_legit):,}")

# ── Generate phishing QR codes ──
needed_phish = PHISH_LIMIT - len(existing_phish)
if needed_phish > 0:
    print(f"\n  Generating {needed_phish:,} phishing QR codes...")
    cur.execute("""
        SELECT url_id, raw_url FROM urls
        WHERE label = 1
        LIMIT ?
    """, (PHISH_LIMIT,))
    rows = cur.fetchall()

    done = 0
    skipped = 0
    for url_id, url in rows:
        filename = f"phish_{url_id}.png"
        if filename in existing_phish:
            continue
        path = os.path.join(PHISH_DIR, filename)
        if make_qr(url, path):
            done += 1
        else:
            skipped += 1
        if done % 500 == 0 and done > 0:
            print(f"  Phishing: {done:,} done  |  {skipped} failed")

    print(f"  Phishing QR done: {done:,} generated, {skipped} failed")

# ── Generate legitimate QR codes ──
needed_legit = LEGIT_LIMIT - len(existing_legit)
if needed_legit > 0:
    print(f"\n  Generating {needed_legit:,} legitimate QR codes...")
    cur.execute("""
        SELECT url_id, raw_url FROM urls
        WHERE label = 0
        LIMIT ?
    """, (LEGIT_LIMIT,))
    rows = cur.fetchall()

    done = 0
    skipped = 0
    for url_id, url in rows:
        filename = f"legit_{url_id}.png"
        if filename in existing_legit:
            continue
        path = os.path.join(LEGIT_DIR, filename)
        if make_qr(url, path):
            done += 1
        else:
            skipped += 1
        if done % 500 == 0 and done > 0:
            print(f"  Legitimate: {done:,} done  |  {skipped} failed")

    print(f"  Legitimate QR done: {done:,} generated, {skipped} failed")

# ── Final count ──
final_phish = len(os.listdir(PHISH_DIR))
final_legit = len(os.listdir(LEGIT_DIR))

print(f"\n{'='*55}")
print(f"  QR Generation Complete")
print(f"  Phishing QR images   : {final_phish:,}")
print(f"  Legitimate QR images : {final_legit:,}")
print(f"  Total QR images      : {final_phish + final_legit:,}")
print(f"  Saved to             : {QR_DIR}")
print(f"{'='*55}")

conn.close()