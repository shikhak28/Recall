"""Retrievers: filename baseline, keyword (FTS5/BM25), semantic (embeddings), visual (CLIP), and hybrid fusion."""
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np

from . import config, embed

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
    kinds: list[str] = field(default_factory=list)
    date_from: float | None = None
    date_to: float | None = None

    def sql(self, textless: bool = False) -> tuple[str, list]:
        status = "f.status IN ('ok','empty')" if textless else "f.status = 'ok'"
        clauses, args = [status, "f.duplicate_of IS NULL"], []
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
    sensitive: dict = field(default_factory=dict)   # {kind: count} from recall.pii
    visual_only: bool = False                       # found by CLIP; the file has no text


NO_TEXT_SNIPPET = "No text in this image. Found by what the picture shows (on-device CLIP)."


class Searcher:
    kw_weight = 0.05
    # CLIP similarity -> the scale of text cosine scores. Calibrated on real photos: a matching description scores
    # about 0.27-0.31, unrelated ones 0.11-0.22. 0.26 maps to the agent's confidence line (0.62).
    # A photo must also stand out from the other images (vis_margin above their median): matching real photos stood
    # out by 0.13-0.19, while abstract text queries against random photos never exceeded 0.054.
    vis_min, vis_margin, vis_mid, vis_base, vis_scale = 0.24, 0.08, 0.26, 0.62, 5.0

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self._cache_key = None
        self._ids: np.ndarray | None = None
        self._file_ids: np.ndarray | None = None
        self._mat: np.ndarray | None = None
        self._vkey = None
        self._vids: np.ndarray | None = None
        self._vmat: np.ndarray | None = None

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

    def _vis_matrix(self):
        key = tuple(self.con.execute("SELECT COUNT(*), COALESCE(MAX(file_id),0) FROM visuals").fetchone())
        if key != self._vkey:
            rows = self.con.execute("SELECT file_id, emb FROM visuals").fetchall()
            self._vids = np.array([r["file_id"] for r in rows], dtype=np.int64)
            self._vmat = (np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows])
                          if rows else np.zeros((0, 512), dtype=np.float32))
            self._vkey = key
        return self._vids, self._vmat

    def _allowed_files(self, filters: Filters, textless: bool = False) -> dict[int, sqlite3.Row]:
        where, args = filters.sql(textless)
        rows = self.con.execute(f"SELECT id, path, name, kind, mtime, status, pii FROM files f WHERE {where}", args).fetchall()
        return {r["id"]: r for r in rows}

    def _visual(self, query: str, allowed) -> tuple[dict, float]:
        """Raw CLIP similarity for every allowed image with a visual embedding, and the median over ALL images."""
        if not config.VISUAL:
            return {}, 0.0
        vids, vmat = self._vis_matrix()
        if len(vids) == 0:
            return {}, 0.0
        from . import vision

        sims = vmat @ vision.embed_text(query)
        return {int(f): (float(v), None) for f, v in zip(vids, sims) if int(f) in allowed}, float(np.median(sims))

    def _vis_score(self, v: float) -> float:
        return self.vis_base + self.vis_scale * (v - self.vis_mid)

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
                out[r["file_id"]] = (-r["s"], r["cid"])
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

        kw = sem = fn = vis = {}
        top_vis = 0.0
        if mode == "visual":
            allowed = {f: r for f, r in self._allowed_files(filters, textless=True).items() if r["kind"] == "image"}
            vis, _ = self._visual(query, allowed)
            fused = {f: s for f, (s, _) in vis.items()}
        elif mode == "filename":
            fn = self._filename(query, allowed)
            fused = {f: s for f, (s, _) in fn.items()}
        elif mode == "keyword":
            kw = self._keyword(query, allowed, keywords)
            fused = {f: s for f, (s, _) in kw.items()}
        elif mode == "semantic":
            sem = self._semantic(sem_q, allowed)
            fused = {f: s for f, (s, _) in sem.items()}
        else:
            kw = self._keyword(query, allowed, keywords)
            sem = self._semantic(sem_q, allowed)
            floor = min((s for s, _ in sem.values()), default=0.0)
            kw_max = max((s for s, _ in kw.values()), default=0.0) or 1.0
            fused = {fid: sem.get(fid, (floor,))[0] + self.kw_weight * kw.get(fid, (0.0,))[0] / kw_max
                     for fid in set(sem) | set(kw)}
            # Photos with no text are invisible to the text retrievers; let CLIP bring them in when it is confident.
            if config.VISUAL and (not filters.kinds or "image" in filters.kinds):
                textless = {f: r for f, r in self._allowed_files(filters, textless=True).items()
                            if r["status"] == "empty" and r["kind"] == "image"}
                if textless:
                    vis, med = self._visual(query, textless)  # the user's own words describe the picture best
                    for fid, (v, _) in vis.items():
                        if v >= self.vis_min and v - med >= self.vis_margin:
                            fused[fid] = self._vis_score(v)
                            top_vis = max(top_vis, fused[fid])
                    allowed = {**allowed, **textless}

        top = sorted(fused, key=lambda f: -fused[f])[:k]
        qtoks = set(tokens(query) + [t.lower() for t in (keywords or [])])
        hits = []
        for fid in top:
            r = allowed[fid]
            cid = (sem.get(fid) or kw.get(fid) or (None, None))[1]
            chunk = self._chunk(cid) if cid else self._first_chunk(fid)
            no_text = r["status"] == "empty"
            hits.append(Hit(
                file_id=fid, path=r["path"], name=r["name"], kind=r["kind"], mtime=r["mtime"],
                score=fused[fid], page=chunk["page"] if chunk else None,
                chunk_text=chunk["text"] if chunk else "",
                snippet=best_lines(chunk["text"], qtoks) if chunk else (NO_TEXT_SNIPPET if no_text else ""),
                signals={"semantic": sem.get(fid, (None,))[0], "keyword": kw.get(fid, (None,))[0],
                         "filename": fn.get(fid, (None,))[0], "visual": vis.get(fid, (None,))[0]},
                sensitive=json.loads(r["pii"]) if r["pii"] else {},
                visual_only=no_text,
            ))
        top_sem = max(max((s for s, _ in sem.values()), default=0.0), top_vis)
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
