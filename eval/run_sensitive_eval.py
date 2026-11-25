"""Evaluate on-device sensitive-data detection.

Positives: screenshots of FAKE documents (random values that pass the real checksums), rendered as images and read
back through the same OCR used for indexing, so OCR mistakes count against us.
Negatives: every file of the synthetic corpus (receipts, boarding passes, chats, lecture slides...; ground-truth
text) and, when an index is given, the real files in it.

    python eval/run_sensitive_eval.py                       # positives + synthetic negatives (eval/data)
    RECALL_HOME=~/.recall_real python eval/run_sensitive_eval.py --index   # also count flags in that index
"""
import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall import extract, index, pii  # noqa: E402

R = random.Random(42)
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
NAMES = ["Priya Sharma", "Rahul Verma", "Ananya Gupta", "Kim Minji", "Arjun Patel", "Lee Jisoo", "Neha Singh"]


def digits(n):
    return "".join(R.choice("0123456789") for _ in range(n))


def aadhaar():
    while True:
        p = R.choice("23456789") + digits(10)
        for d in "0123456789":
            if pii.verhoeff_ok(p + d):
                return f"{p[:4]} {p[4:8]} {p[8:]}{d}"


def card():
    while True:
        p = "4" + digits(14)
        for d in "0123456789":
            if pii.luhn_ok(p + d):
                n = p + d
                return " ".join(n[i:i + 4] for i in range(0, 16, 4))


def pan():
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(R.choice(L) for _ in range(3)) + "P" + R.choice(L) + digits(4) + R.choice(L)


def rrn():
    return f"{R.randint(85, 99):02d}{R.randint(1, 12):02d}{R.randint(1, 28):02d}-{R.choice('5678')}{digits(6)}"


def ifsc():
    return R.choice(["SBIN", "HDFC", "ICIC", "UTIB", "PUNB"]) + "0" + digits(6)


TEMPLATES = {
    "aadhaar": lambda: ["Government of India", R.choice(NAMES), f"DOB: {R.randint(1,28):02d}/{R.randint(1,12):02d}/{R.randint(1990,2004)}",
                        R.choice(["FEMALE", "MALE"]), aadhaar(), "Aadhaar - Aam Aadmi ka Adhikar"],
    "pan": lambda: ["INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "Permanent Account Number Card", pan(), R.choice(NAMES)],
    "card": lambda: ["Credit Card Statement", R.choice(NAMES), f"Card {card()}", f"Due date: 2026-12-{R.randint(1,28):02d}",
                     f"Total due: Rs {R.randint(1000, 90000):,}"],
    "kr_rrn": lambda: ["ALIEN REGISTRATION CARD", R.choice(NAMES), f"Registration No. {rrn()}", "Status: D-2 Student"],
    "passport": lambda: ["REPUBLIC OF INDIA", f"Passport No: {R.choice('KLMNPRSTUVWZ')}{digits(7)}", R.choice(NAMES),
                         f"Date of Expiry: {R.randint(1,28):02d}/{R.randint(1,12):02d}/{R.randint(2028,2035)}"],
    "bank_account": lambda: ["Savings Account Passbook", R.choice(NAMES), f"Account No: {digits(R.choice([11, 12, 14]))}",
                             f"IFSC: {ifsc()}", "Branch: Main Road"],
    "password": lambda: [R.choice(["Home Wi-Fi", "Office network", "Netflix login"]), f"user: {R.choice(NAMES).split()[0].lower()}",
                         f"password: {R.choice(['maple', 'tiger', 'sunny', 'blue'])}{digits(4)}{R.choice('!@#$')}"],
    "api_key": lambda: ["# .env", "DEBUG=false", "OPENAI_API_KEY=sk-" + "".join(R.choice("abcdefghijkmnopqrstuvwxyzABCDEFGH0123456789") for _ in range(40))],
}


def render(lines, path):
    font = ImageFont.truetype(R.choice(FONTS), R.choice([28, 32, 36]))
    img = Image.new("RGB", (1000, 100 + 56 * len(lines)), R.choice(["white", "#f4f4f4", "#eef4ff"]))
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((40, 40 + 56 * i), l, font=font, fill="black")
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42, help="42 = dev set used while building; use another seed for a held-out test")
    ap.add_argument("--corpus", default=str(ROOT / "eval" / "data"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "sensitive_positives"))
    ap.add_argument("--index", action="store_true", help="also report files flagged in the current RECALL_HOME index")
    a = ap.parse_args()
    R.seed(a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_type, missed = {}, []
    for kind, make in TEMPLATES.items():
        for i in range(a.per_type):
            p = out / f"{kind}_{i}.png"
            render(make(), p)
            text = extract.extract(p)[0].text
            found = pii.scan(text)
            ok = kind in found
            per_type.setdefault(kind, []).append(ok)
            if not ok:
                missed.append((p.name, text.replace("\n", " | ")[:120]))
    print("Positives (fake documents -> image -> OCR -> detector)")
    for k, v in per_type.items():
        print(f"  {pii.LABELS[k]:<24} {sum(v)}/{len(v)}")
    tp = sum(sum(v) for v in per_type.values())
    n_pos = sum(len(v) for v in per_type.values())
    print(f"  detection rate: {tp}/{n_pos} = {tp / n_pos:.3f}")
    for name, text in missed:
        print(f"  missed {name}: {text}")

    man = Path(a.corpus) / "manifest.json"
    if man.exists():
        files = {n: v for n, v in json.loads(man.read_text())["files"].items() if not v.get("duplicate_of")}
        # The corpus has a few Wi-Fi notes with a real "Password: ..." line: those must be found, nothing else flagged.
        expected = {n for n, v in files.items() if "password:" in v["text"].lower()}
        flagged = {n: f for n, f in ((n, pii.scan(v["text"])) for n, v in files.items()) if f}
        found = [n for n in expected if "password" in flagged.get(n, {})]
        false_alarms = {n: f for n, f in flagged.items() if n not in expected}
        print(f"\nSynthetic corpus ({len(files)} files: receipts, boarding passes, chats, slides, ...)")
        print(f"  Wi-Fi password notes found: {len(found)}/{len(expected)}")
        print(f"  false alarms: {len(false_alarms)}/{len(files) - len(expected)}")
        for n, f in false_alarms.items():
            print(f"    {n}: {pii.describe(f)}")
    if a.index:
        con = index.connect()
        rows = con.execute("SELECT name, pii FROM files WHERE status='ok' AND duplicate_of IS NULL").fetchall()
        flagged = [(r["name"], json.loads(r["pii"])) for r in rows if r["pii"] and r["pii"] not in ("{}", "")]
        print(f"\nIndex at {index.DB_PATH}: {len(flagged)} of {len(rows)} files flagged")
        for n, f in flagged:
            print(f"  {n}: {pii.describe(f)}")


if __name__ == "__main__":
    main()
