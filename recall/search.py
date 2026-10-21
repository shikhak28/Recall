"""Retrievers: filename baseline, keyword (FTS5/BM25), semantic (embeddings), and hybrid (reciprocal rank fusion)."""
import re
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np

from . import embed

STOPWORDS = set("""
a an the and or of to in on at for from by with about into over under is are was were be been being am do does did
have has had i me my mine we our you your he she it its they them their this that these those what which who whom
where when why how can could should would will shall may might must find show get give open locate search look
please file files image images picture photo screenshot screenshots pdf document doc docs one thing saved save
contained containing contains some any there here
""".split())

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def tokens(text: str) -> list[str]:
    return [t for t in (w.lower().strip("'-") for w in TOKEN_RE.findall(text)) if t and t not in STOPWORDS]


@dataclass
class Filters:
    kinds: list[str] = field(default_factory=list)  # subset of image/pdf/text; empty = all
    date_from: float | None = None  # unix timestamps on file mtime
    date_to: float | None = None

    def sql(self) -> tuple[str, list]:
        clauses, args = ["f.status = 'ok'", "f.duplicate_of IS NULL"], []
        if self.kinds:
            clauses.append(f"f.kind IN ({','.join('?' * len(self.kinds))})")
            args += self.kinds
        if self.date_from is not None:
            clauses.append("f.mtime >= ?")
            args.append(self.date_from)
        if self.date_to is not None:
            clauses.append("f.mtime < ?")
            args.append(self.date_to)
        return " AND ".join(clauses), args


@dataclass
class Hit:
    file_id: int
    path: str
    name: str
    kind: str
    mtime: float
    score: float
    page: int | None = None
    snippet: str = ""
    chunk_text: str = ""
    signals: dict = field(default_factory=dict)


class Searcher:
    # Weight of the keyword bonus in hybrid mode, chosen on a separate dev corpus (eval/tune_hybrid.py).
    kw_weight = 0.05

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self._cache_key = None
        self._ids: np.ndarray | None = None
        self._file_ids: np.ndarray | None = None
        self._mat: np.ndarray | None = None

    # ---------- embedding matrix, cached until the index changes ----------
    def _matrix(self):
        key = self.con.execute("SELECT COUNT(*), COALESCE(MAX(id),0) FROM chunks").fetchone()
        key = tuple(key)
        if key != self._cache_key:
            rows = self.con.execute("SELECT id, file_id, emb FROM chunks WHERE emb IS NOT NULL").fetchall()
            self._ids = np.array([r["id"] for r in rows], dtype=np.int64)
            self._file_ids = np.array([r["file_id"] for r in rows], dtype=np.int64)
            self._mat = (np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows])
                         if rows else np.zeros((0, 384), dtype=np.float32))
            self._cache_key = key
        return self._ids, self._file_ids, self._mat

    def _allowed_files(self, filters: Filters) -> dict[int, sqlite3.Row]:
        where, args = filters.sql()
        rows = self.con.execute(f"SELECT id, path, name, kind, mtime FROM files f WHERE {where}", args).fetchall()
        return {r["id"]: r for r in rows}

    # ---------- individual retrievers: each returns {file_id: (score, chunk_id)} ----------
    def _filename(self, query: str, allowed) -> dict[int, tuple[float, int | None]]:
        qt = set(tokens(query))
        out = {}
        for fid, r in allowed.items():
            name_toks = set(re.split(r"[^a-z0-9]+", r["name"].lower()))
            s = sum(1 for t in qt if any(n.startswith(t[:5]) for n in name_toks if len(n) > 1))
            if s:
                out[fid] = (float(s), None)
        return out

    def _keyword(self, query: str, allowed, extra_terms: list[str] | None = None) -> dict:
        terms = list(dict.fromkeys(tokens(query) + [t.lower() for t in (extra_terms or [])]))
        if not terms:
            return {}
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        rows = self.con.execute(
            "SELECT c.file_id, c.id AS cid, bm25(chunks_fts, 1.0, 0.6) AS s "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.chunk_id "
            "WHERE chunks_fts MATCH ? ORDER BY s LIMIT 400",
            (match,),
        ).fetchall()
        out = {}
        for r in rows:
            if r["file_id"] in allowed and r["file_id"] not in out:
                out[r["file_id"]] = (-r["s"], r["cid"])  # bm25: lower is better
        return out

    def _semantic(self, query: str, allowed) -> dict:
        ids, fids, mat = self._matrix()
        if len(ids) == 0:
            return {}
        sims = mat @ embed.embed_query(query)
        out = {}
        for i in np.argsort(-sims):
            fid = int(fids[i])
            if fid in allowed and fid not in out:
                out[fid] = (float(sims[i]), int(ids[i]))
            if len(out) >= 200:
                break
        return out

    # ---------- public API ----------
    def search(self, query: str, mode: str = "hybrid", filters: Filters | None = None, k: int = 10,
               semantic_query: str | None = None, keywords: list[str] | None = None) -> tuple[list[Hit], dict]:
        """Return top-k file hits and timing/diagnostic info."""
        t0 = time.perf_counter()
        filters = filters or Filters()
        query = (query or "").strip()
        if not query:
            return [], {"ms": 0.0, "top_semantic": 0.0}
        allowed = self._allowed_files(filters)
        sem_q = semantic_query or query

        kw = sem = fn = {}
        if mode == "filename":
            fn = self._filename(query, allowed)
            fused = {f: s for f, (s, _) in fn.items()}
        elif mode == "keyword":
            kw = self._keyword(query, allowed, keywords)
            fused = {f: s for f, (s, _) in kw.items()}
        elif mode == "semantic":
            sem = self._semantic(sem_q, allowed)
            fused = {f: s for f, (s, _) in sem.items()}
        else:  # hybrid: semantic cosine plus a max-normalised BM25 bonus for exact-term matches
            kw = self._keyword(query, allowed, keywords)
            sem = self._semantic(sem_q, allowed)
            floor = min((s for s, _ in sem.values()), default=0.0)
            kw_max = max((s for s, _ in kw.values()), default=0.0) or 1.0
            fused = {fid: sem.get(fid, (floor,))[0] + self.kw_weight * kw.get(fid, (0.0,))[0] / kw_max
                     for fid in set(sem) | set(kw)}

        top = sorted(fused, key=lambda f: -fused[f])[:k]
        qtoks = set(tokens(query) + [t.lower() for t in (keywords or [])])
        hits = []
        for fid in top:
            r = allowed[fid]
            cid = (sem.get(fid) or kw.get(fid) or (None, None))[1]
            chunk = self._chunk(cid) if cid else self._first_chunk(fid)
            hits.append(Hit(
                file_id=fid, path=r["path"], name=r["name"], kind=r["kind"], mtime=r["mtime"],
                score=fused[fid], page=chunk["page"] if chunk else None,
                chunk_text=chunk["text"] if chunk else "",
                snippet=best_lines(chunk["text"], qtoks) if chunk else "",
                signals={"semantic": sem.get(fid, (None,))[0], "keyword": kw.get(fid, (None,))[0],
                         "filename": fn.get(fid, (None,))[0]},
            ))
        top_sem = max((s for s, _ in sem.values()), default=0.0)
        return hits, {"ms": (time.perf_counter() - t0) * 1000, "top_semantic": top_sem,
                      "candidates": len(allowed), "keyword_hits": len(kw)}

    def _chunk(self, cid):
        return self.con.execute("SELECT page, text FROM chunks WHERE id = ?", (cid,)).fetchone()

    def _first_chunk(self, fid):
        return self.con.execute("SELECT page, text FROM chunks WHERE file_id = ? ORDER BY id LIMIT 1", (fid,)).fetchone()


def best_lines(text: str, qtoks: set[str], max_lines: int = 3, max_chars: int = 320) -> str:
    """Pick the lines of a chunk that best explain why it matched, kept in original order."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    def score(line):
        lt = set(tokens(line))
        return sum(1 for q in qtoks if any(w.startswith(q[:5]) for w in lt))
    ranked = sorted(range(len(lines)), key=lambda i: (-score(lines[i]), i))[:max_lines]
    if all(score(lines[i]) == 0 for i in ranked):
        ranked = list(range(min(max_lines, len(lines))))
    out = " · ".join(lines[i] for i in sorted(ranked))
    return out[:max_chars] + ("…" if len(out) > max_chars else "")
