"""Local index: SQLite for metadata + FTS5 keyword index, float32 embedding blobs for semantic search."""
import hashlib
import multiprocessing as mp
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import embed, extract
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
    indexed_at REAL
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
    seconds: float = 0.0


def iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
            yield p


def _delete_file_rows(con: sqlite3.Connection, file_id: int) -> None:
    con.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)", (file_id,))
    con.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
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

    # Phase 1 (cheap, sequential): find new/changed files, hash them, detect exact duplicates.
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

    # Phase 2: extract text (parallel), then embed + store as results arrive.
    total = len(todo)
    if workers > 1 and total > 1:
        ctx = mp.get_context("spawn")  # never fork a process that may hold ONNX/CUDA sessions
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

    # Drop files that disappeared from disk.
    for row in con.execute("SELECT id, path FROM files WHERE path LIKE ?", (str(root) + "%",)).fetchall():
        if not Path(row["path"]).exists():
            _delete_file_rows(con, row["id"])
    con.commit()
    rep.seconds = time.perf_counter() - t0
    return rep


def _extract_job(path: str):
    """Runs in a worker process. Returns (segments, error)."""
    try:
        return extract.extract(Path(path)), None
    except Exception as e:  # corrupted file, unreadable PDF, etc.
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
        _insert_file(con, {**base, "status": "empty"})
        con.commit()
        return
    file_id = _insert_file(con, {**base, "status": "ok"})
    # Prefix the filename so a meaningful name still helps semantic retrieval.
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
    }


def clear(con: sqlite3.Connection) -> None:
    con.executescript("DELETE FROM chunks_fts; DELETE FROM chunks; DELETE FROM files;")
    con.commit()
