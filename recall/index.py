"""Local index: SQLite for metadata + FTS5 keyword index, float32 embedding blobs for semantic search."""
import hashlib
import json
import multiprocessing as mp
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import config, embed, extract, pii
from .config import DB_PATH, SUPPORTED_EXT, default_workers, kind_of

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    ext TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER,
    mtime REAL,
    sha1 TEXT,
    duplicate_of INTEGER,
    status TEXT NOT NULL,          -- ok | empty | error
    error TEXT,
    indexed_at REAL,
    pii TEXT                       -- JSON {kind: count} of sensitive values found; NULL = not scanned yet
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page INTEGER,
    method TEXT,
    text TEXT NOT NULL,
    emb BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE TABLE IF NOT EXISTS visuals (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    emb BLOB NOT NULL              -- CLIP image embedding, float32
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, name, chunk_id UNINDEXED, tokenize = 'porter unicode61'
);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    if "pii" not in {r[1] for r in con.execute("PRAGMA table_info(files)")}:
        con.execute("ALTER TABLE files ADD COLUMN pii TEXT")  # index built by an older version
        con.commit()
    return con


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class IndexReport:
    scanned: int = 0
    indexed: int = 0
    skipped_unchanged: int = 0
    duplicates: int = 0
    empty: int = 0
    errors: int = 0
    unsupported: int = 0
    visual: int = 0          # images given a CLIP embedding (incl. photos with no text)
    sensitive: int = 0       # files where sensitive data was found
    seconds: float = 0.0


def iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
            yield p


def _delete_file_rows(con: sqlite3.Connection, file_id: int) -> None:
    con.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)", (file_id,))
    con.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    con.execute("DELETE FROM visuals WHERE file_id = ?", (file_id,))
    con.execute("DELETE FROM files WHERE id = ?", (file_id,))


def index_folder(
    root: Path,
    con: sqlite3.Connection,
    progress: Callable[[int, int, str], None] | None = None,
    workers: int | None = None,
) -> IndexReport:
    """Incrementally index every supported file under `root`. Unchanged files are skipped.

    OCR/text extraction runs in `workers` processes (default config.default_workers()); embedding and DB writes stay in
    this process so SQLite has a single writer.
    """
    t0 = time.perf_counter()
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {root}")
    rep = IndexReport()
    workers = workers or default_workers()

    todo, dups, seen = [], [], {}
    for path in iter_files(root):
        rep.scanned += 1
        if path.suffix.lower() not in SUPPORTED_EXT:
            rep.unsupported += 1
            continue
        st = path.stat()
        row = con.execute("SELECT id, size, mtime FROM files WHERE path = ?", (str(path),)).fetchone()
        if row and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
            rep.skipped_unchanged += 1
            continue
        if row:
            _delete_file_rows(con, row["id"])
        sha = _sha1(path)
        base = dict(path=str(path), name=path.name, ext=path.suffix.lower(), kind=kind_of(path.suffix),
                    size=st.st_size, mtime=st.st_mtime, sha1=sha)
        if sha in seen or con.execute("SELECT 1 FROM files WHERE sha1 = ? AND duplicate_of IS NULL", (sha,)).fetchone():
            dups.append(base)
        else:
            seen[sha] = path
            todo.append(base)
    con.commit()

    total = len(todo)
    if workers > 1 and total > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, total), mp_context=ctx) as pool:
            futures = {pool.submit(_extract_job, b["path"]): b for b in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                b = futures[fut]
                if progress:
                    progress(i, total, b["name"])
                _store(con, b, *fut.result(), rep)
    else:
        for i, b in enumerate(todo, 1):
            if progress:
                progress(i, total, b["name"])
            _store(con, b, *_extract_job(b["path"]), rep)

    for b in dups:
        orig = con.execute("SELECT id FROM files WHERE sha1 = ? AND duplicate_of IS NULL", (b["sha1"],)).fetchone()
        rep.duplicates += 1
        _insert_file(con, {**b, "status": "ok", "duplicate_of": orig["id"] if orig else None, "indexed_at": time.time()})
    con.commit()

    for row in con.execute("SELECT id, path FROM files WHERE path LIKE ?", (str(root) + "%",)).fetchall():
        if not Path(row["path"]).exists():
            _delete_file_rows(con, row["id"])
    con.commit()
    backfill(con, rep)
    rep.seconds = time.perf_counter() - t0
    return rep


def _extract_job(path: str):
    """Runs in a worker process. Returns (segments, error)."""
    try:
        return extract.extract(Path(path)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:300]


def _store(con: sqlite3.Connection, base: dict, segments, error, rep: IndexReport) -> None:
    base = {**base, "indexed_at": time.time()}
    if error:
        rep.errors += 1
        _insert_file(con, {**base, "status": "error", "error": error})
        con.commit()
        return
    pieces = [(s.page, s.method, c) for s in segments for c in extract.chunk(s.text)]
    if not pieces:
        rep.empty += 1
        file_id = _insert_file(con, {**base, "status": "empty", "pii": "{}"})
        if base["kind"] == "image":
            _store_visuals(con, [(file_id, base["path"])], rep)
        con.commit()
        return
    found = pii.scan("\n".join(s.text for s in segments))
    rep.sensitive += bool(found)
    file_id = _insert_file(con, {**base, "status": "ok", "pii": json.dumps(found)})
    if base["kind"] == "image":
        _store_visuals(con, [(file_id, base["path"])], rep)
    stem = Path(base["path"]).stem
    vecs = embed.embed_passages([f"{stem}\n{c}" for _, _, c in pieces])
    for (page, method, text), v in zip(pieces, vecs):
        cid = con.execute(
            "INSERT INTO chunks(file_id, page, method, text, emb) VALUES (?,?,?,?,?)",
            (file_id, page, method, text, v.astype(np.float32).tobytes()),
        ).lastrowid
        con.execute("INSERT INTO chunks_fts(text, name, chunk_id) VALUES (?,?,?)", (text, stem, cid))
    rep.indexed += 1
    con.commit()


def _store_visuals(con: sqlite3.Connection, items: list[tuple[int, str]], rep: IndexReport | None = None) -> None:
    """CLIP-embed images. A failure only costs visual search for those files, never the text index."""
    if not config.VISUAL or not items:
        return
    from . import vision

    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        try:
            vecs = vision.embed_images([p for _, p in batch])
        except Exception:
            vecs = []
            for fid, p in batch:  # find the bad file, keep the rest
                try:
                    vecs.append(vision.embed_images([p])[0])
                except Exception:
                    vecs.append(None)
        for (fid, _), v in zip(batch, vecs):
            if v is not None:
                con.execute("INSERT OR REPLACE INTO visuals(file_id, emb) VALUES (?,?)", (fid, v.astype("float32").tobytes()))
                if rep:
                    rep.visual += 1


def backfill(con: sqlite3.Connection, rep: IndexReport | None = None) -> None:
    """Bring files indexed by an older version up to date: sensitive-data scan and CLIP embeddings."""
    for r in con.execute("SELECT id FROM files WHERE pii IS NULL AND status = 'ok' AND duplicate_of IS NULL").fetchall():
        text = "\n".join(c["text"] for c in con.execute("SELECT text FROM chunks WHERE file_id = ? ORDER BY id", (r["id"],)))
        found = pii.scan(text)
        if rep:
            rep.sensitive += bool(found)
        con.execute("UPDATE files SET pii = ? WHERE id = ?", (json.dumps(found), r["id"]))
    con.commit()
    todo = con.execute("SELECT f.id, f.path FROM files f LEFT JOIN visuals v ON v.file_id = f.id "
                       "WHERE f.kind = 'image' AND f.status IN ('ok','empty') AND f.duplicate_of IS NULL "
                       "AND v.file_id IS NULL").fetchall()
    _store_visuals(con, [(r["id"], r["path"]) for r in todo if Path(r["path"]).exists()], rep)
    con.commit()


def _insert_file(con: sqlite3.Connection, d: dict) -> int:
    cols = ",".join(d)
    return con.execute(f"INSERT INTO files({cols}) VALUES ({','.join('?' * len(d))})", list(d.values())).lastrowid


def stats(con: sqlite3.Connection) -> dict:
    q = lambda sql: con.execute(sql).fetchone()[0]
    return {
        "files": q("SELECT COUNT(*) FROM files"),
        "searchable": q("SELECT COUNT(*) FROM files WHERE status='ok' AND duplicate_of IS NULL"),
        "duplicates": q("SELECT COUNT(*) FROM files WHERE duplicate_of IS NOT NULL"),
        "empty": q("SELECT COUNT(*) FROM files WHERE status='empty'"),
        "errors": q("SELECT COUNT(*) FROM files WHERE status='error'"),
        "chunks": q("SELECT COUNT(*) FROM chunks"),
        "images": q("SELECT COUNT(*) FROM files WHERE kind='image'"),
        "pdfs": q("SELECT COUNT(*) FROM files WHERE kind='pdf'"),
        "visual": q("SELECT COUNT(*) FROM visuals"),
        "photos_no_text": q("SELECT COUNT(*) FROM files f JOIN visuals v ON v.file_id = f.id WHERE f.status='empty'"),
        "sensitive": q("SELECT COUNT(*) FROM files WHERE pii IS NOT NULL AND pii NOT IN ('{}', '') AND duplicate_of IS NULL"),
    }


def clear(con: sqlite3.Connection) -> None:
    con.executescript("DELETE FROM chunks_fts; DELETE FROM chunks; DELETE FROM visuals; DELETE FROM files;")
    con.commit()
