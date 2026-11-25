"""Functional stability + privacy tests.   pytest -q

Models must already be cached (run the app or eval once). Each test uses a fresh index in a temp dir.
"""
import os
import shutil
import socket
import sys
from datetime import date
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall import agent, index, pii  # noqa: E402
from recall.search import Filters, Searcher  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def text_image(path: Path, lines: list[str]):
    img = Image.new("RGB", (900, 120 + 50 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 32)
    for i, l in enumerate(lines):
        d.text((40, 40 + 50 * i), l, font=f, fill="black")
    img.save(path)


@pytest.fixture()
def corpus(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    text_image(files / "Screenshot_1.png", ["Operating Systems Midterm", "Date: Oct 30, 2026", "Room 301"])
    text_image(files / "Screenshot_2.png", ["Wi-Fi password: maple4821!", "Grandma's house"])
    (files / "notes.txt").write_text("Groceries: milk, eggs, kimchi. Call the landlord about the heater.")
    (files / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01")
    (files / "archive.xyz").write_bytes(b"unsupported")
    Image.new("RGB", (400, 300), (40, 120, 200)).save(files / "IMG_0001.jpg")
    shutil.copy(files / "Screenshot_1.png", files / "copy_of_exam.png")
    con = index.connect(tmp_path / "db.sqlite")
    return files, con


def test_index_handles_every_file_kind(corpus):
    files, con = corpus
    rep = index.index_folder(files, con)
    assert rep.scanned == 7
    assert rep.unsupported == 1
    assert rep.errors == 1
    assert rep.empty == 1
    assert rep.duplicates == 1
    assert rep.indexed == 3
    assert rep.visual == 3          # 2 screenshots + the text-free photo; the duplicate is skipped
    st = index.stats(con)
    assert st["searchable"] == 3 and st["photos_no_text"] == 1
    assert st["sensitive"] == 1     # the Wi-Fi password screenshot


def test_reindex_is_incremental_and_tracks_deletes(corpus):
    files, con = corpus
    index.index_folder(files, con)
    rep = index.index_folder(files, con)
    assert rep.indexed == 0 and rep.skipped_unchanged == 6
    (files / "notes.txt").unlink()
    index.index_folder(files, con)
    assert index.stats(con)["searchable"] == 2


def test_semantic_search_finds_ocr_text(corpus):
    files, con = corpus
    index.index_folder(files, con)
    s = Searcher(con)
    hits, _ = s.search("when is the OS midterm exam", "hybrid", k=3)
    assert hits[0].name == "Screenshot_1.png"
    hits, _ = s.search("internet code at grandma's", "semantic", k=3)
    assert hits[0].name == "Screenshot_2.png"


def test_empty_query_and_filters(corpus):
    files, con = corpus
    index.index_folder(files, con)
    s = Searcher(con)
    assert s.search("   ", "hybrid")[0] == []
    hits, _ = s.search("exam", "hybrid", filters=Filters(kinds=["pdf"]))
    assert hits == []
    hits, _ = s.search("groceries", "hybrid", filters=Filters(kinds=["text"]))
    assert [h.name for h in hits] == ["notes.txt"]


def test_missing_folder_raises(tmp_path):
    con = index.connect(tmp_path / "db.sqlite")
    with pytest.raises(FileNotFoundError):
        index.index_folder(tmp_path / "does-not-exist", con)


def test_rule_planner_dates_and_types():
    today = date(2026, 10, 21)
    p = agent.plan_rules("the starbucks receipt screenshot from September", today)
    assert p.file_types == ["image"] and (p.date_from, p.date_to) == ("2026-09-01", "2026-10-01")
    p = agent.plan_rules("pdf I saved last week", today)
    assert p.file_types == ["pdf"] and p.date_from == "2026-10-14"
    assert agent.plan_rules("when is my exam?", today).intent == "answer"


def test_planner_json_parsing_is_defensive():
    p = agent._parse_plan('<think>hmm</think> {"semantic_query": "exam date", "keywords": ["midterm"], '
                          '"file_types": ["image", "video"], "date_from": "last week", "intent": "answer"}', "q")
    assert p.file_types == ["image"] and p.date_from is None and p.keywords == ["midterm"]
    assert agent._parse_plan("no json here", "q") is None


def test_no_network_during_indexing_or_offline_search(corpus, monkeypatch):
    """Privacy: with all sockets blocked, indexing and offline agent search must still work."""
    files, con = corpus
    attempts = []

    def blocked(*a, **k):
        attempts.append(a)
        raise OSError("network blocked by test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    index.index_folder(files, con)
    res = agent.RecallAgent(Searcher(con), llm=None).run("operating systems midterm room")
    assert res.hits and res.hits[0].name == "Screenshot_1.png"
    assert attempts == [], f"unexpected network attempts: {attempts}"
    assert agent.LEDGER == []


class FakeNemotron:
    """Stands in for the Token Factory client: records what it was sent, returns canned plans."""
    model = "fake-nemotron"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, system, user, purpose, contains_file_content, max_tokens=1200, values_masked=0):
        self.calls.append((purpose, user, contains_file_content))
        return self.replies.pop(0)


def test_agent_replans_when_weak_and_only_answer_step_sees_file_text(corpus):
    files, con = corpus
    index.index_folder(files, con)
    llm = FakeNemotron([
        '{"semantic_query": "exam", "keywords": [], "file_types": ["pdf"], "intent": "answer"}',
        '<think>relax</think>{"semantic_query": "Operating Systems midterm exam date and room", '
        '"keywords": ["midterm", "room"], "file_types": [], "intent": "answer"}',
        "Oct 30, 2026 in Room 301 [1]",
    ])
    res = agent.RecallAgent(Searcher(con), llm).run("when is my OS midterm?", allow_answer=True)
    assert [s.name.split(" ")[0] for s in res.steps][:4] == ["plan", "retrieve", "re-plan", "retrieve"]
    assert res.hits[0].name == "Screenshot_1.png"
    assert res.answer.startswith("Oct 30")
    purposes = [(p, c) for p, _, c in llm.calls]
    assert purposes == [("plan", False), ("re-plan", False), ("answer", True)]
    for p, user, _ in llm.calls[:2]:
        assert "Screenshot_1" not in user and "Room 301" not in user


def _aadhaar_like(prefix11: str) -> str:
    """Append the Verhoeff check digit (fake but checksum-valid test number)."""
    for d in "0123456789":
        if pii.verhoeff_ok(prefix11 + d):
            return prefix11 + d


def test_sensitive_detectors_hit_real_formats_and_skip_lookalikes():
    a = _aadhaar_like("23456789012")
    text = (f"Aadhaar: {a[:4]} {a[4:8]} {a[8:]}\nPAN ABCPE1234F\nCard 4111 1111 1111 1111\n"
            "주민등록번호 990101-1234567\nPassport No: K1234567\nA/c No. 123456789012\nIFSC SBIN0001234\n"
            "Wi-Fi password: maple4821!\nkey sk-abcdefghijklmnopqrstuvwxyz123456")
    assert pii.scan(text) == {"aadhaar": 1, "pan": 1, "card": 1, "kr_rrn": 1, "passport": 1, "bank_account": 1,
                              "ifsc": 1, "password": 1, "api_key": 1}
    bad = str(int(a) + 1 if a[-1] != "9" else int(a) - 1)
    lookalikes = (f"Order {bad} shipped. Call 98765 43210. Room 301, Oct 30 2026. ISBN 978-3-16-148410-0. "
                  "Card 4111 1111 1111 1112. 991301-1234567. Total 1234 5678 9012 3456 won. The password is long.")
    assert pii.scan(lookalikes) == {}


def test_redact_masks_values_but_keeps_context():
    masked, n = pii.redact("Wi-Fi password: maple4821! and card 4111-1111-1111-1111")
    assert n == 2 and "maple4821" not in masked and "4111-1111" not in masked
    assert "Wi-Fi password:" in masked and "••••1111" in masked


def test_answer_step_never_sends_sensitive_values(corpus):
    files, con = corpus
    index.index_folder(files, con)
    llm = FakeNemotron(['{"semantic_query": "wifi password grandma house", "keywords": ["wi-fi", "password"], '
                        '"file_types": [], "intent": "answer"}', "It is masked [1]"])
    agent.LEDGER.clear()
    res = agent.RecallAgent(Searcher(con), llm).run("what is the wifi password at grandma's?", allow_answer=True)
    assert res.hits[0].name == "Screenshot_2.png" and res.hits[0].sensitive == {"password": 1}
    purpose, user, has_text = llm.calls[-1]
    assert purpose == "answer" and has_text
    assert "maple4821" not in user and "••••••" in user


def test_visual_search_finds_photo_with_no_text(corpus):
    files, con = corpus
    img = Image.new("RGB", (640, 480), (135, 206, 235))          # a drawn landscape: sky, field, sun
    d = ImageDraw.Draw(img)
    d.rectangle([0, 330, 640, 480], fill=(34, 139, 34))
    d.ellipse([480, 40, 580, 140], fill=(255, 215, 0))
    img.save(files / "IMG_0002.jpg")
    index.index_folder(files, con)
    s = Searcher(con)
    q = "a yellow sun in a blue sky above a green field"
    hits, _ = s.search(q, "visual", k=3)
    assert hits[0].name == "IMG_0002.jpg" and hits[0].visual_only
    hits, _ = s.search(q, "hybrid", k=3)
    assert hits[0].name == "IMG_0002.jpg"
    res = agent.RecallAgent(s, llm=None).run(q)
    assert res.hits[0].name == "IMG_0002.jpg"
    hits, _ = s.search("when is the OS midterm exam", "hybrid", k=3)
    assert hits[0].name == "Screenshot_1.png"   # text questions are not disturbed
