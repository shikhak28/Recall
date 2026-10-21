"""Evaluate Recall on the synthetic corpus: OCR quality, retrieval (Recall@5, MRR@10) per method, latency, dedup.

    python eval/run_eval.py                    # offline methods only
    NEBIUS_API_KEY=... python eval/run_eval.py # also the Nemotron agent
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("RECALL_HOME", str(ROOT / ".recall"))
sys.path.insert(0, str(ROOT.parent))

from recall import agent, index  # noqa: E402
from recall.search import Searcher  # noqa: E402

norm_toks = lambda s: re.findall(r"[a-z0-9]+", s.lower())


def ocr_eval(con, manifest):
    rows = con.execute("SELECT f.name, f.status, f.kind, f.duplicate_of, GROUP_CONCAT(c.text, '\n') AS text "
                       "FROM files f LEFT JOIN chunks c ON c.file_id = f.id GROUP BY f.id").fetchall()
    tok_recall, cer, ok, total, photo_ok, photos, per_style = [], [], 0, 0, 0, 0, {}
    for r in rows:
        m = manifest.get(r["name"])
        if not m or r["duplicate_of"]:
            continue
        got = r["text"] or ""
        if m["style"] == "photo":
            photos += 1
            photo_ok += not got.strip()
            continue
        truth = norm_toks(m["text"])
        have = set(norm_toks(got))
        tr = sum(t in have for t in truth) / max(1, len(truth))
        a, b = " ".join(truth), " ".join(norm_toks(got))
        c = 1 - SequenceMatcher(None, a, b, autojunk=False).ratio()
        tok_recall.append(tr)
        cer.append(c)
        total += 1
        ok += tr >= 0.9
        key = "pdf-scanned" if m["pdf"] and m["style"] == "paper" else ("pdf" if m["pdf"] else m["style"])
        per_style.setdefault(key, []).append(tr)
    return {
        "files_with_text": total,
        "extraction_success_rate": ok / max(1, total),
        "mean_token_recall": statistics.mean(tok_recall),
        "mean_char_error_approx": statistics.mean(cer),
        "text_free_photos_correctly_empty": f"{photo_ok}/{photos}",
        "token_recall_by_source": {k: round(statistics.mean(v), 3) for k, v in sorted(per_style.items())},
    }


def retrieval_eval(name, fn, queries):
    rec5, rr10, hit1, lat, per_type, fails = [], [], [], [], {}, []
    for q in queries:
        t = time.perf_counter()
        names = fn(q["query"])
        lat.append((time.perf_counter() - t) * 1000)
        rel = set(q["relevant"])
        r5 = len(rel & set(names[:5])) / len(rel)
        rr = next((1 / (i + 1) for i, n in enumerate(names[:10]) if n in rel), 0.0)
        rec5.append(r5)
        rr10.append(rr)
        hit1.append(float(bool(names) and names[0] in rel))
        per_type.setdefault(q["type"], []).append(r5)
        if rr == 0:
            fails.append({"query": q["query"], "expected": q["relevant"], "got": names[:3]})
    lat_sorted = sorted(lat)
    return {
        "method": name, "queries": len(queries),
        "recall@5": statistics.mean(rec5), "mrr@10": statistics.mean(rr10), "hit@1": statistics.mean(hit1),
        "recall@5_by_type": {k: round(statistics.mean(v), 3) for k, v in per_type.items()},
        "latency_ms_mean": statistics.mean(lat), "latency_ms_p95": lat_sorted[int(0.95 * (len(lat) - 1))],
        "failures": fails,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--no-agent", action="store_true", help="skip the Nemotron agent even if a key is set")
    a = ap.parse_args()
    data = Path(a.data)
    meta = json.loads((data / "manifest.json").read_text())
    manifest, today = meta["files"], date.fromisoformat(meta["generated_on"])
    queries = [json.loads(l) for l in open(data / "queries.jsonl")]

    con = index.connect()
    if a.reindex:
        index.clear(con)
    rep = index.index_folder(data / "files", con, progress=lambda i, n, f: print(f"\rindexing {i}/{n}", end="", flush=True))
    print()
    st = index.stats(con)
    n_files = st["files"]
    hashes = [index._sha1(p) for p in sorted((data / "files").iterdir())]
    expected_dups = len(hashes) - len(set(hashes))  # byte-identical files, incl. identical rendered fillers
    print("index:", rep, st)

    s = Searcher(con)
    s.search("warm up", "hybrid")  # load the embedding model before timing
    names = lambda hits: [h.name for h in hits]
    methods = {
        "Filename (baseline)": lambda q: names(s.search(q, "filename", k=10)[0]),
        "Keyword BM25": lambda q: names(s.search(q, "keyword", k=10)[0]),
        "Semantic": lambda q: names(s.search(q, "semantic", k=10)[0]),
        "Hybrid": lambda q: names(s.search(q, "hybrid", k=10)[0]),
        "Agent, offline planner": lambda q: names(agent.RecallAgent(s, None).run(q, k=10, today=today).hits),
    }
    llm = None if a.no_agent else agent.make_llm()
    if llm:
        methods[f"Agent, Nemotron planner ({llm.model})"] = lambda q: names(agent.RecallAgent(s, llm).run(q, k=10, today=today).hits)

    results = {"corpus": {"files": n_files, "queries": len(queries), "generated_on": meta["generated_on"]},
               "indexing": {"seconds": round(rep.seconds, 1), "files_per_sec": round(rep.indexed / rep.seconds, 2) if rep.indexed else None,
                            "note": "only newly indexed files are timed; use --reindex for a full timing"},
               "dedup": {"expected": expected_dups, "detected": st["duplicates"]},
               "ocr": ocr_eval(con, manifest), "retrieval": []}
    for name, fn in methods.items():
        print("evaluating", name)
        results["retrieval"].append(retrieval_eval(name, fn, queries))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    md = ["| Method | Recall@5 | MRR@10 | Hit@1 | Recall@5 (date queries) | Mean latency (ms) | p95 (ms) |", "|---|---|---|---|---|---|---|"]
    for r in results["retrieval"]:
        md.append(f"| {r['method']} | {r['recall@5']:.3f} | {r['mrr@10']:.3f} | {r['hit@1']:.3f} | "
                  f"{r['recall@5_by_type'].get('date', float('nan')):.3f} | {r['latency_ms_mean']:.0f} | {r['latency_ms_p95']:.0f} |")
    o = results["ocr"]
    md += ["", f"OCR: extraction success (token recall >= 0.9) {o['extraction_success_rate']:.1%} over {o['files_with_text']} files, "
               f"mean token recall {o['mean_token_recall']:.3f}, approx CER {o['mean_char_error_approx']:.3f}, "
               f"text-free photos left empty {o['text_free_photos_correctly_empty']}",
           f"By source: {o['token_recall_by_source']}",
           f"Dedup: {results['dedup']['detected']}/{results['dedup']['expected']} exact duplicates detected",
           f"Indexing: {results['indexing']}"]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    _chart(results, out / "retrieval.png")


def _chart(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rs = results["retrieval"]
    labels = [r["method"].split(" (")[0] if "Nemotron" not in r["method"] else "Agent, Nemotron" for r in rs]
    x = range(len(rs))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [r["recall@5"] for r in rs], w, label="Recall@5", color="#3B6EA8")
    ax.bar([i + w / 2 for i in x], [r["mrr@10"] for r in rs], w, label="MRR@10", color="#76B900")
    ax.axhline(0.8, ls="--", lw=1, color="#3B6EA8", alpha=.6)
    ax.axhline(0.7, ls=":", lw=1, color="#4d7a00", alpha=.8)
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Retrieval on {results['corpus']['files']} files / {results['corpus']['queries']} queries (dashed = proposal targets)")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
