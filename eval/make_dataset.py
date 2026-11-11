"""Generate a synthetic personal-files corpus with ground-truth queries.

Files get realistic, uninformative names (Screenshot_20260912_141233.png, IMG_4821.jpg, scan0003.pdf), so a
filename search cannot cheat. Each "target" file comes with natural-language queries that paraphrase its content.
Fillers (quotes, headlines, code, text-free photos, exact duplicates) pad the corpus to --n files.

    python eval/make_dataset.py --n 300 --out eval/data
"""
import argparse
import json
import os
import random
import shutil
import textwrap
from datetime import date, datetime, timedelta
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = "/usr/share/fonts/truetype"
FONTS = {
    "sans": f"{FONT_DIR}/dejavu/DejaVuSans.ttf",
    "bold": f"{FONT_DIR}/dejavu/DejaVuSans-Bold.ttf",
    "serif": f"{FONT_DIR}/dejavu/DejaVuSerif.ttf",
    "mono": f"{FONT_DIR}/dejavu/DejaVuSansMono.ttf",
    "lib": f"{FONT_DIR}/liberation/LiberationSans-Regular.ttf",
}


def font(name, size):
    try:
        return ImageFont.truetype(FONTS[name], size)
    except OSError:
        return ImageFont.load_default(size=size)


R = random.Random(7)
TODAY = date(2026, 10, 21)

def render(lines, style, title=None):
    """Render text lines into a screenshot-like image. Returns PIL image."""
    if style == "phone":
        W, fs, pad, dark = 720, 30, 40, R.random() < 0.35
        bg, fg, bar = ((18, 18, 20), (235, 235, 235), (40, 40, 48)) if dark else ((255, 255, 255), (20, 20, 20), (240, 240, 245))
        body_font, title_font = font("sans", fs), font("bold", 34)
    elif style == "web":
        W, fs, pad, bg, fg, bar = 1280, 26, 60, (255, 255, 255), (30, 30, 30), (225, 228, 235)
        body_font, title_font = font("lib", fs), font("bold", 40)
    elif style == "slide":
        W, fs, pad, bg, fg, bar = 1280, 34, 80, R.choice([(255, 255, 255), (245, 248, 255), (25, 35, 60)]), None, None
        fg = (255, 255, 255) if bg == (25, 35, 60) else (25, 30, 45)
        body_font, title_font = font("sans", fs), font("bold", 54)
    elif style == "receipt":
        W, fs, pad, bg, fg, bar = 560, 24, 36, (252, 252, 248), (25, 25, 25), None
        body_font, title_font = font("mono", fs), font("bold", 30)
    else:
        W, fs, pad, bg, fg, bar = 1240, 28, 90, (255, 255, 255), (20, 20, 20), None
        body_font, title_font = font("serif", fs), font("bold", 40)

    wrap = max(20, int((W - 2 * pad) / (fs * 0.56)))
    body = [w for l in lines for w in (textwrap.wrap(l, wrap) or [""])]
    line_h = int(fs * 1.55)
    top = 90 if style in ("phone", "web") else pad
    H = top + (int(title_font.size * 1.8) if title else 0) + line_h * len(body) + pad * 2
    if style == "slide":
        H = max(H, 720)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    if style == "phone":
        d.rectangle([0, 0, W, 80], fill=bar)
        d.text((24, 22), "9:41", font=font("bold", 26), fill=fg)
        d.text((W - 150, 22), "LTE 87%", font=font("sans", 24), fill=fg)
    elif style == "web":
        d.rectangle([0, 0, W, 70], fill=bar)
        d.rounded_rectangle([120, 16, W - 120, 54], 18, fill=(255, 255, 255))
        d.text((140, 22), "https://" + R.choice(["portal.knu.ac.kr", "mail.google.com", "eclass.univ.edu", "my.account.app"]),
               font=font("sans", 20), fill=(90, 90, 90))
    y = top
    if title:
        d.text((pad, y), title, font=title_font, fill=fg)
        y += int(title_font.size * 1.8)
    for l in body:
        d.text((pad, y), l, font=body_font, fill=fg)
        y += line_h
    if style == "paper":
        img = img.rotate(R.uniform(-1.2, 1.2), expand=False, fillcolor=(255, 255, 255)).filter(ImageFilter.GaussianBlur(0.6))
    return img


def save_image(img, out_dir, stamp):
    if R.random() < 0.7:
        name = f"Screenshot_{stamp:%Y%m%d_%H%M%S}.png"
        img.save(out_dir / name)
    else:
        name = f"IMG_{R.randint(1000, 9999)}.jpg"
        while (out_dir / name).exists():
            name = f"IMG_{R.randint(1000, 9999)}.jpg"
        img.save(out_dir / name, quality=R.choice([70, 80, 90]))
    return name


def save_pdf(lines, title, out_dir, scanned, counter):
    name = f"scan{counter:04d}.pdf" if scanned else R.choice(["document", "download", "notice", "file"]) + f"({counter}).pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    if scanned:
        img = render(lines, "paper", title)
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        page.insert_image(fitz.Rect(30, 30, 565, 30 + 535 * img.height / img.width), stream=buf.getvalue())
    else:
        page.insert_textbox(fitz.Rect(60, 60, 535, 120), title, fontsize=18, fontname="helv")
        page.insert_textbox(fitz.Rect(60, 120, 535, 800), "\n".join(lines), fontsize=11, fontname="helv")
    doc.save(out_dir / name)
    return name


COURSES = ["Machine Learning", "Operating Systems", "Computer Networks", "Linear Algebra", "Database Systems",
           "Computer Vision", "Natural Language Processing", "Signals and Systems", "Probability and Statistics",
           "Data Structures", "Software Engineering", "Deep Learning", "Korean Language II", "Microeconomics",
           "Digital Logic"]
STORES = ["Lotte Mart", "Olive Young", "Daiso", "GS25", "Starbucks", "Kyobo Bookstore", "Uniqlo", "Paris Baguette",
          "Emart", "ABC Mart"]
ITEMS = {"Lotte Mart": ["Milk 1L", "Eggs 10ea", "Rice 4kg", "Ramen 5pk", "Apples"], "Olive Young": ["Sunscreen SPF50", "Lip balm", "Face mask x5", "Shampoo"],
         "Daiso": ["Storage box", "Hangers x10", "Notebook A5", "USB cable"], "GS25": ["Triangle kimbap", "Banana milk", "Coffee can", "Sandwich"],
         "Starbucks": ["Iced Americano T", "Caffe Latte G", "Blueberry scone"], "Kyobo Bookstore": ["Pattern Recognition (Bishop)", "TOPIK II prep book", "Pen set"],
         "Uniqlo": ["Heattech crew neck", "Ultra light down jacket", "Socks 3P"], "Paris Baguette": ["Croissant", "Sweet red bean bread", "Strawberry cake"],
         "Emart": ["Laundry detergent", "Chicken breast 1kg", "Kimchi 2kg", "Orange juice"], "ABC Mart": ["Running shoes", "Shoe insoles"]}
SPECIALTIES = [("Dermatology", "skin doctor"), ("Dentistry", "dentist"), ("Ophthalmology", "eye doctor"),
               ("Orthopedics", "bone and joint specialist"), ("Otolaryngology (ENT)", "ear nose and throat doctor")]
DOCTORS = ["Kim Min-jun", "Park Ji-woo", "Lee Seo-yeon", "Choi Hyun-woo", "Jung Ha-eun"]
FRIENDS = ["Minji", "Arjun", "Sofia", "Daniel", "Yuna", "Priya"]
CAFES = [("Cafe Onion", "Anguk station"), ("Blue Bottle", "Seongsu station"), ("Starbucks Reserve", "Gangnam station"),
         ("Fritz Coffee", "Mapo station"), ("Ediya", "the main gate"), ("Twosome Place", "Dongseongno")]
CITIES = [("ICN", "Seoul Incheon", "NRT", "Tokyo"), ("ICN", "Seoul Incheon", "DEL", "New Delhi"),
          ("GMP", "Seoul Gimpo", "CJU", "Jeju"), ("ICN", "Seoul Incheon", "SIN", "Singapore"),
          ("PUS", "Busan Gimhae", "KIX", "Osaka"), ("ICN", "Seoul Incheon", "BKK", "Bangkok")]
AIRLINES = ["Korean Air", "Asiana Airlines", "Jeju Air", "Air India", "T'way Air"]
MEMOS = [("Rent October", "proof that I paid october rent"), ("Dinner split - BBQ", "money I sent for the barbecue dinner"),
         ("AI club membership fee", "payment for the AI club dues"), ("Phone bill Sep", "september phone bill transfer"),
         ("Textbook - used", "transfer for the second hand textbook"), ("Gym 3 months", "how much I paid for the gym membership")]
RECIPIENTS = ["Kim Landlord", "Arjun Sharma", "KNU AI Club", "SK Telecom", "Lee Seung-ho", "Fitness 24"]
WIFI = [("Grandma's house", "KT_GiGA_5G_A41F"), ("the library study room", "KNU-Library-Guest"),
        ("Cafe Mori", "CafeMori_Free"), ("the lab (Room 512)", "VisionLab-512"),
        ("the Airbnb in Busan", "Haeundae_Stay_2G"), ("the dorm lounge", "Dorm3-Lounge")]
SLIDES = [
    ("Backpropagation", ["Apply the chain rule layer by layer", "Compute dL/dW for every weight", "Gradients flow from output back to input", "Update weights with gradient descent"],
     "slide about how a neural network computes gradients for its weights"),
    ("Overfitting and Regularization", ["Model memorizes training data, fails on test data", "L2 penalty (weight decay)", "Dropout randomly zeroes activations", "Early stopping on validation loss"],
     "lecture on stopping a model from memorizing the training set"),
    ("TCP Three-Way Handshake", ["Client sends SYN", "Server replies SYN-ACK", "Client sends ACK", "Connection ESTABLISHED"],
     "how a client and server open a connection"),
    ("Deadlock: Coffman Conditions", ["Mutual exclusion", "Hold and wait", "No preemption", "Circular wait"],
     "the four requirements for processes getting stuck waiting on each other"),
    ("Eigenvalues and Eigenvectors", ["Av = lambda v", "Direction unchanged by the transformation", "det(A - lambda I) = 0", "Used in PCA"],
     "vectors that only get stretched, not rotated, by a matrix"),
    ("Database Normalization", ["1NF: atomic values", "2NF: no partial dependency", "3NF: no transitive dependency", "Goal: remove redundancy and update anomalies"],
     "slide about removing duplicated data across tables"),
    ("Convolutional Neural Networks", ["Learnable filters slide over the image", "Feature maps", "Max pooling reduces spatial size", "Weight sharing"],
     "slide on image filters and pooling layers"),
    ("Scaled Dot-Product Attention", ["softmax(QK^T / sqrt(d_k)) V", "Queries, keys and values", "Multi-head attention", "Used in Transformers"],
     "the formula transformers use to weight tokens against each other"),
    ("Bayes' Theorem", ["P(A|B) = P(B|A) P(A) / P(B)", "Prior, likelihood, posterior", "Update beliefs with evidence"],
     "rule for updating a probability after seeing new evidence"),
    ("Choosing the Learning Rate", ["Too large: loss diverges or oscillates", "Too small: very slow convergence", "Use schedules or warmup", "Adam adapts per parameter"],
     "what happens if the training step size is too big"),
    ("Processes vs Threads", ["Process: own address space", "Threads share memory within a process", "Context switch cost", "Race conditions need locks"],
     "difference between a program instance and lightweight execution units sharing memory"),
    ("Big-O Notation", ["O(1), O(log n), O(n), O(n log n), O(n^2)", "Upper bound on growth", "Ignore constants", "Worst-case analysis"],
     "how to describe how running time grows with input size"),
]
RECIPES = [("Tteokbokki", ["Rice cakes 300g", "Gochujang 2 tbsp", "Fish cakes", "Sugar 1 tbsp", "Green onion"], "that spicy recipe with rice cakes and red pepper paste"),
           ("Kimchi Fried Rice", ["Cooked rice 2 cups", "Aged kimchi 1 cup", "Spam", "Fried egg on top", "Sesame oil"], "recipe for frying leftover rice with fermented cabbage"),
           ("Dal Tadka", ["Yellow lentils (toor dal)", "Cumin seeds", "Garlic", "Ghee", "Turmeric"], "how to make the lentil curry with tempered spices"),
           ("Fluffy Pancakes", ["Flour 1.5 cups", "Baking powder", "Milk", "Egg", "Maple syrup"], "breakfast recipe with flour milk and maple syrup"),
           ("Pasta Carbonara", ["Spaghetti", "Guanciale", "Egg yolks", "Pecorino Romano", "Black pepper"], "italian pasta made with egg yolks and cured pork")]
ARTISTS = [("IU", "KSPO Dome"), ("Coldplay", "Goyang Stadium"), ("NewJeans", "Olympic Hall"), ("AP Dhillon", "Yes24 Live Hall"), ("Day6", "Jamsil Arena")]
BOOKS = [("Deep Learning", "Goodfellow"), ("Clean Code", "Robert Martin"), ("Sapiens", "Yuval Harari"), ("Operating System Concepts", "Silberschatz"), ("The Pragmatic Programmer", "Hunt")]
MALLS = ["COEX Mall", "Starfield Hanam", "Lotte World Mall", "Hyundai Department Store Daegu"]
COMPANIES = [("Naver", "ML Engineer Intern"), ("Samsung Research", "AI Researcher"), ("Kakao", "Backend Developer"), ("NVIDIA", "Solutions Architect Intern"), ("LG AI Research", "Data Scientist")]
NOTICES = [
    ("Alien Registration Card (ARC) Extension Checklist", ["Application form", "Passport and current ARC", "Certificate of enrollment", "Proof of residence (lease contract)", "Fee: 60,000 KRW revenue stamp", "Apply up to 4 months before expiry at the immigration office"],
     "documents needed to extend my visa / residence card"),
    ("Dormitory Regulations 2026", ["Curfew: gates close at 1:00 AM", "No cooking appliances in rooms", "Guests allowed in lobby only", "Quiet hours 11 PM - 7 AM", "Room inspection every month"],
     "what time does the dorm close at night"),
    ("Global Excellence Scholarship - Fall 2026", ["Eligibility: GPA 3.5 or higher", "Application deadline: November 14, 2026", "Submit transcript and recommendation letter", "Award: 50% tuition waiver"],
     "when is the scholarship application due"),
    ("Graduation Requirements - Computer Science", ["Total 130 credits", "Capstone design project required", "TOPIK level 4 or TOEIC 700", "Graduation thesis or portfolio"],
     "how many credits do I need to graduate"),
    ("Health Insurance for International Students", ["Mandatory National Health Insurance enrollment after 6 months", "Monthly premium approx. 76,000 KRW", "Pay at any bank or via app", "Covers hospital and pharmacy costs"],
     "monthly cost of the national health insurance for foreign students"),
    ("Lab Safety Training Notice", ["All graduate students must complete online safety training", "Deadline: October 31, 2026", "Certificate must be uploaded to the portal", "Non-completion blocks lab access"],
     "deadline for the mandatory safety course"),
]
QUOTES = ["The best way to predict the future is to invent it.", "Stay hungry, stay foolish.", "Done is better than perfect.",
          "Simplicity is the soul of efficiency.", "First, solve the problem. Then, write the code.", "Talk is cheap. Show me the code.",
          "It always seems impossible until it's done.", "Small steps every day."]
HEADLINES = ["Bank of Korea holds base rate steady", "New subway line opens in Gyeonggi province", "Typhoon expected to reach southern coast",
             "Semiconductor exports rise for third month", "City announces free bike rental weekend", "Researchers unveil low-power AI chip",
             "Han River fireworks festival this Saturday", "Housing prices cool in Seoul suburbs", "National team qualifies for World Cup",
             "Heat wave advisory issued for Daegu"]
CODE = ["def train(model, loader):\n    for x, y in loader:\n        loss = criterion(model(x), y)\n        loss.backward()",
        "SELECT name, COUNT(*) FROM orders\nGROUP BY name HAVING COUNT(*) > 5;", "git rebase -i HEAD~3\ngit push --force-with-lease",
        "docker run -p 8080:80 nginx:latest", "for i in range(10):\n    print(i ** 2)", "const [state, setState] = useState(null);"]


def rdate(lo=-120, hi=-1):
    d = TODAY + timedelta(days=R.randint(lo, hi))
    return datetime(d.year, d.month, d.day, R.randint(8, 23), R.randint(0, 59), R.randint(0, 59))


def fmt_day(d):
    return d.strftime("%a, %b %d, %Y").replace(" 0", " ")


def targets():
    """Yield (title, lines, style, queries, saved_at, as_pdf) for every target document."""
    for i, c in enumerate(COURSES):
        etype = R.choice(["Midterm", "Final", "Midterm"])
        d = TODAY + timedelta(days=R.randint(1, 50))
        room = f"Engineering Bldg. {R.randint(1, 4)}, Room {R.randint(101, 512)}"
        lines = [f"CSE{4000 + i * 7} {c}", f"{etype} Examination", f"Date: {fmt_day(d)}", f"Time: {R.choice(['9:00', '10:30', '13:00', '15:00'])} - {R.choice(['10:15', '12:00', '14:30', '16:30'])}",
                 f"Venue: {room}", "Closed book. Bring student ID.", "Calculators allowed" if R.random() < .5 else "No electronic devices"]
        q = R.choice([f"when is my {c} {etype.lower()} exam", f"which room is the {c} test in", f"{c} {etype.lower()} date and place",
                      f"screenshot with the {c} exam schedule"])
        yield ("Exam Schedule", lines, R.choice(["web", "phone"]), [q], rdate(), False)
    for i, c in enumerate(COURSES):
        n = R.randint(1, 5)
        d = TODAY + timedelta(days=R.randint(-10, 30))
        lines = [f"{c} - Homework {n}", f"Due: {fmt_day(d)} 23:59", "Submit through e-Class as a single PDF", "Late submissions: -10% per day",
                 R.choice(["Problems 1-6 from Chapter 4", "Implement the algorithm in Python and report results", "Group work allowed (max 2)"])]
        q = R.choice([f"deadline for {c} homework {n}", f"when do I have to submit assignment {n} for {c}", f"{c} hw{n} due date"])
        yield ("Assignment", lines, R.choice(["web", "phone"]), [q], rdate(), R.random() < 0.25)

    months = [TODAY.month - 3, TODAY.month - 2, TODAY.month - 1, TODAY.month]
    for s in STORES:
        mons = R.sample(months, 2) if R.random() < 0.6 else [R.choice(months)]
        for mon in mons:
            saved = datetime(TODAY.year, mon, R.randint(1, 28 if mon != TODAY.month else TODAY.day - 1), R.randint(9, 21), R.randint(0, 59))
            items = R.sample(ITEMS[s], min(len(ITEMS[s]), R.randint(2, 4)))
            prices = [R.randint(12, 400) * 100 for _ in items]
            lines = [s.upper(), f"{saved:%Y-%m-%d %H:%M}", "-" * 28] + [f"{it[:18]:<18}{p:>9,}" for it, p in zip(items, prices)] + \
                    ["-" * 28, f"TOTAL{sum(prices):>22,}", f"CARD {R.choice(['KB', 'Shinhan', 'Woori'])} ****{R.randint(1000, 9999)}", "Thank you!"]
            yield ("Receipt", lines, "receipt", None, saved, False, s, len(mons) > 1)

    for place, ssid in WIFI:
        pw = R.choice(["sunflower", "blueocean", "tiger", "maple"]) + str(R.randint(1000, 9999)) + R.choice(["!", "#", "@"])
        lines = ["Wi-Fi", f"Network: {ssid}", f"Password: {pw}", f"({place})"]
        yield ("Wi-Fi", lines, "phone", [R.choice([f"internet password at {place}", f"wifi details for {place}"])], rdate(), False)
    for (fc, fcity, tc, tcity), al in zip(CITIES, AIRLINES * 2):
        d = TODAY + timedelta(days=R.randint(5, 70))
        lines = ["BOARDING PASS", al, f"{fc} -> {tc}", f"{fcity} to {tcity}", f"Flight {al[:2].upper()}{R.randint(100, 999)}   Date {d:%d %b %Y}",
                 f"Boarding {R.randint(6, 21):02d}:{R.choice(['05', '20', '35', '50'])}   Gate {R.randint(10, 130)}   Seat {R.randint(10, 45)}{R.choice('ABCDEF')}",
                 f"Booking ref: {''.join(R.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=6))}"]
        yield ("Boarding Pass", lines, "phone", [R.choice([f"my plane ticket to {tcity}", f"what gate does my {tcity} flight leave from", f"booking reference for the {tcity} trip"])], rdate(), R.random() < .3)
    for (spec, lay), doc in zip(SPECIALTIES, DOCTORS):
        d = TODAY + timedelta(days=R.randint(2, 30))
        lines = ["Appointment confirmed", f"{R.choice(['Seoul', 'Kyungpook', 'Daegu'])} {spec.split()[0]} Clinic", f"Dr. {doc} - {spec}",
                 f"{fmt_day(d)} at {R.randint(9, 17)}:{R.choice(['00', '30'])}", "Please arrive 10 minutes early with your insurance card"]
        yield ("Appointment", lines, "phone", [f"when am I seeing the {lay}"], rdate(), False)
    for f, (cafe, st) in zip(FRIENDS, CAFES):
        day = R.choice(["Saturday", "Friday", "Sunday", "tomorrow"])
        t = f"{R.randint(1, 7)}:{R.choice(['00', '30'])} pm"
        lines = [f, f"{f}: are we still on for {day}?", "Me: yes!", f"{f}: let's meet at {cafe} near {st} at {t}", "Me: sounds good, see you there"]
        yield (None, lines, "phone", [f"where did {f} say we should meet"], rdate(), False)
    for (memo, para), rcp in zip(MEMOS, RECIPIENTS):
        lines = ["Transfer complete", f"To: {rcp}", f"Amount: {R.randint(10, 900) * 1000:,} KRW", f"From: {R.choice(['KB Kookmin', 'Toss Bank', 'Shinhan'])} ***-***-{R.randint(1000, 9999)}", f"Memo: {memo}"]
        yield ("Toss", lines, "phone", [para], rdate(), False)
    for title, bullets, q in SLIDES:
        yield (title, ["- " + b for b in bullets], "slide", [q], rdate(), R.random() < 0.3)
    for dish, ingr, q in RECIPES:
        yield (dish, ["Ingredients"] + ingr + ["Serves 2 - 20 minutes"], R.choice(["web", "phone"]), [q], rdate(), False)
    for artist, venue in ARTISTS:
        d = TODAY + timedelta(days=R.randint(10, 90))
        lines = [f"{artist} LIVE TOUR 2026", venue, f"{fmt_day(d)} 19:00", f"Section {R.choice('ABCDEFG')}{R.randint(1, 30)}  Row {R.randint(1, 40)}  Seat {R.randint(1, 60)}", "Mobile ticket - show QR at entrance"]
        yield ("Ticket", lines, "phone", [R.choice([f"{artist} concert seat number", f"ticket for the {artist} show"])], rdate(), False)
    for book, author in BOOKS:
        d = TODAY + timedelta(days=R.randint(1, 21))
        lines = ["Central Library - Loan Receipt", f"{book} / {author}", f"Due date: {d:%Y-%m-%d}", "Renew online up to 2 times"]
        yield (None, lines, R.choice(["receipt", "phone"]), [f"when is {book} due back at the library"], rdate(), False)
    for mall in MALLS:
        lines = ["Parking", mall, f"Level B{R.randint(1, 5)} - Zone {R.choice('ABCDEFGH')}{R.randint(1, 40)}", f"Entered {R.randint(10, 20)}:{R.randint(10, 59)}"]
        yield (None, lines, "phone", [f"where did I park at {mall}"], rdate(), False)
    for comp, role in COMPANIES:
        d = TODAY + timedelta(days=R.randint(3, 25))
        lines = [f"Interview Invitation - {comp}", f"Position: {role}", f"{fmt_day(d)} {R.randint(10, 17)}:00 KST", R.choice(["Online (Zoom)", "On-site"]), f"Interviewer: {R.choice(DOCTORS)}"]
        yield (None, lines, R.choice(["web", "phone"]), [R.choice([f"when is my interview with {comp}", f"{comp} interview time"])], rdate(), R.random() < .3)
    for title, lines, q in NOTICES:
        yield (title, lines, "paper", [q], rdate(), True)


def fillers(n):
    for _ in range(n):
        kind = R.random()
        if kind < 0.25:
            yield (None, [R.choice(QUOTES)], R.choice(["slide", "phone"]))
        elif kind < 0.5:
            yield ("News", R.sample(HEADLINES, 3), "web")
        elif kind < 0.65:
            yield (None, R.choice(CODE).split("\n"), "web")
        elif kind < 0.8:
            yield ("Lecture", ["- " + b for b in R.choice(SLIDES)[1][:2]] + ["- Review questions on the next page"], "slide")
        else:
            yield ("PHOTO", [], None)


def photo():
    img = Image.new("RGB", (1024, 768), tuple(R.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(25):
        x, y = R.randint(0, 1000), R.randint(0, 750)
        d.ellipse([x, y, x + R.randint(20, 300), y + R.randint(20, 300)], fill=tuple(R.randint(0, 255) for _ in range(3)))
    return img.filter(ImageFilter.GaussianBlur(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="total number of files")
    ap.add_argument("--out", default="eval/data")
    ap.add_argument("--seed", type=int, default=7, help="7 = test corpus; use another seed for a dev corpus")
    a = ap.parse_args()
    R.seed(a.seed)
    out = Path(a.out)
    files_dir = out / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True)
    manifest, queries, receipts, pdf_n = {}, [], {}, 0

    def emit(title, lines, style, saved, as_pdf):
        nonlocal pdf_n
        if as_pdf:
            pdf_n += 1
            name = save_pdf(lines, title or "", files_dir, scanned=(style == "paper" or R.random() < 0.4), counter=pdf_n)
        elif style is None:
            name = save_image(photo(), files_dir, saved)
        else:
            name = save_image(render(lines, style, title), files_dir, saved)
        ts = saved.timestamp()
        os.utime(files_dir / name, (ts, ts))
        manifest[name] = {"text": "\n".join(([title] if title else []) + lines), "style": style or "photo", "pdf": as_pdf, "saved": saved.isoformat()}
        return name

    for t in targets():
        if t[0] == "Receipt":
            title, lines, style, _, saved, as_pdf, store, multi = t
            name = emit(title, lines, style, saved, as_pdf)
            receipts.setdefault(store, []).append((name, saved))
            continue
        title, lines, style, qs, saved, as_pdf = t
        name = emit(title, lines, style, saved, as_pdf)
        queries += [{"query": q, "relevant": [name], "type": "content"} for q in qs]

    for store, recs in receipts.items():
        queries.append({"query": R.choice([f"receipt from {store}", f"how much did I spend at {store}"]),
                        "relevant": [n for n, _ in recs], "type": "content"})
        if len(recs) > 1:
            name, saved = recs[0]
            queries.append({"query": f"the {store} receipt from {saved:%B}", "relevant": [name], "type": "date"})

    n_fill = max(0, a.n - len(manifest))
    names = list(manifest)
    for title, lines, style in fillers(int(n_fill * 0.97)):
        emit(title, lines, style, rdate(-200, -1), style == "web" and R.random() < .15)
    for _ in range(a.n - len(manifest)):
        src = R.choice(names)
        dst = f"copy_{R.randint(100, 999)}_{src}"
        shutil.copy2(files_dir / src, files_dir / dst)
        manifest[dst] = {**manifest[src], "duplicate_of": src}

    (out / "manifest.json").write_text(json.dumps({"generated_on": TODAY.isoformat(), "files": manifest}, indent=1))
    with open(out / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"{len(manifest)} files ({pdf_n} PDFs), {len(queries)} queries -> {out}")


if __name__ == "__main__":
    main()
