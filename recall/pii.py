"""On-device sensitive-data detection and masking.

Runs on the text Recall already extracted (OCR / PDF / plain text), so it costs almost nothing at index time and
never needs the network. Each detector is a pattern plus, where one exists, a checksum (Verhoeff for Aadhaar,
Luhn for payment cards, a real calendar date for Korean resident numbers) or a context word ("passport",
"account no") to keep false alarms low.

Used for three things:
  * flag files that hold sensitive data (badge in the results, "Sensitive files" tab);
  * mask the values in snippets shown on screen unless the user asks to reveal them;
  * mask the values in excerpts BEFORE the optional answer step sends them to Nemotron.
"""
import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

LABELS = {
    "aadhaar": "Aadhaar number",
    "pan": "PAN",
    "card": "Payment card number",
    "kr_rrn": "Korean resident number",
    "passport": "Passport number",
    "bank_account": "Bank account number",
    "ifsc": "IFSC code",
    "password": "Password / PIN",
    "api_key": "API key / token",
    "us_ssn": "US SSN",
}

# ---- checksums -------------------------------------------------------------------------------------------------
_V_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_V_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_ok(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _V_D[c][_V_P[i % 8][int(ch)]]
    return c == 0


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _aadhaar(m):
    d = _digits(m.group(0))
    return len(d) == 12 and len(set(d)) > 2 and verhoeff_ok(d)


def _card(m):
    d = _digits(m.group(0))
    return 13 <= len(d) <= 19 and len(set(d)) > 1 and d[0] in "3456" and luhn_ok(d)


_PAN = re.compile(r"[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]")
_TO_LETTER = str.maketrans("0158", "OISB")   # common OCR confusions
_TO_DIGIT = str.maketrans("OoIlSB", "001158")


def _pan(m):
    v = m.group(0)
    if sum(c.isalpha() for c in v[:5]) < 3:
        return False
    fixed = v[:5].translate(_TO_LETTER) + v[5:9].translate(_TO_DIGIT) + v[9].translate(_TO_LETTER)
    return bool(_PAN.fullmatch(fixed))


def _kr_rrn(m):
    yy, mm, dd, g = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    year = (1800 if g in (9, 0) else 1900 if g in (1, 2, 5, 6) else 2000) + yy
    try:
        date(year, mm, dd)
        return True
    except ValueError:
        return False


@dataclass
class Detector:
    kind: str
    pattern: re.Pattern
    valid: Callable[[re.Match], bool] | None = None
    group: int = 0  # which group holds the value to mask


# Order matters: detectors with a context word come first, then longer numbers before shorter ones, so a 16-digit
# card number is never mistaken for a 12-digit Aadhaar number.
DETECTORS = [
    Detector("api_key", re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}|xox[bpas]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{35})\b")),
    Detector("password", re.compile(r"(?i)\b(?:password|passwd|pwd|passcode|wi-?fi\s+key|pin)\b\s*[:=]\s*(\S{3,})"), group=1),
    Detector("passport", re.compile(r"(?i)passport\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z]{1,2}\s?\d{6,8})\b"), group=1),
    Detector("bank_account", re.compile(r"(?i)\b(?:a/c|acct|account)\s*(?:no\.?|number|#)?\s*[:\-]?\s*(\d[\d ]{7,20}\d)\b"), group=1),
    Detector("us_ssn", re.compile(r"(?i)(?:ssn|social security(?: number| no\.?)?)\s*[:#]?\s*(\d{3}-\d{2}-\d{4})\b"), group=1),
    Detector("card", re.compile(r"(?<!\d)(?<!\d[ -])(?:\d[ -]?){12,18}\d(?![ -]?\d)"), _card),
    Detector("aadhaar", re.compile(r"(?<![\d-])(?<!\d )[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![ -]?\d)"), _aadhaar),
    Detector("kr_rrn", re.compile(r"(?<!\d)(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])-([1-8])\d{6}(?!\d)"), _kr_rrn),
    Detector("pan", re.compile(r"\b[A-Z0-9]{5}[0-9OISBl]{4}[A-Z0-9]\b"), _pan),
    Detector("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
]


def find(text: str) -> list[tuple[str, int, int]]:
    """All sensitive spans as (kind, start, end), non-overlapping, earlier detectors win."""
    spans: list[tuple[str, int, int]] = []
    for det in DETECTORS:
        for m in det.pattern.finditer(text or ""):
            if det.valid and not det.valid(m):
                continue
            s, e = m.span(det.group)
            if any(s < e2 and s2 < e for _, s2, e2 in spans):
                continue
            spans.append((det.kind, s, e))
    return sorted(spans, key=lambda x: x[1])


def scan(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for kind, _, _ in find(text):
        out[kind] = out.get(kind, 0) + 1
    return out


def _mask_value(kind: str, value: str) -> str:
    if kind in ("password", "api_key"):
        return "••••••"
    d = _digits(value)
    tail = d[-4:] if len(d) >= 8 else ""
    return f"[{LABELS[kind]} ••••{tail}]" if tail else f"[{LABELS[kind]}]"


def redact(text: str) -> tuple[str, int]:
    """Mask every sensitive value. Returns (masked_text, number_of_values_masked)."""
    spans = find(text)
    if not spans:
        return text, 0
    out, last = [], 0
    for kind, s, e in spans:
        out.append(text[last:s])
        out.append(_mask_value(kind, text[s:e]))
        last = e
    out.append(text[last:])
    return "".join(out), len(spans)


def describe(found: dict[str, int]) -> str:
    return ", ".join(LABELS.get(k, k) for k in sorted(found, key=lambda k: list(LABELS).index(k) if k in LABELS else 99))
