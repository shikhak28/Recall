"""Score Recall on the real-file test set (real_test/): rank of the correct file for every question.

    RECALL_HOME=~/.recall_real python eval/run_real_test.py            # index real_test/files, then score
    RECALL_HOME=~/.recall_real python eval/run_real_test.py --modes agent hybrid filename

Prints one line per question and Hit@1 / Recall@5 / MRR@10 per question type, and writes
real_test/results.csv. Offline agent unless NEBIUS_API_KEY is set. Also lists the files flagged as sensitive.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall import agent, index, pii  # noqa: E402
from recall.search import Searcher  # noqa: E402

RT = ROOT / "real_test"


def group(t: str) -> str:
    t = t.lower()
    if "expected to fail" in t:
        return "korean"
    if "photo" in t:
        return "photo"
    if "image" in t:
        return "image text"
    return "documents"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["agent", "hybrid", "filename"],
                    choices=["agent", "hybrid", "semantic", "keyword", "filename", "visual"])
    ap.add_argument("--no-index", action="store_true", help="score the existing index without re-indexing")
    a = ap.parse_args()

    con = index.connect()
    if not a.no_index:
        print(index.index_folder(RT / "files", con))
    s = Searcher(con)
    llm = agent.make_llm()
    rows = list(csv.DictReader(open(RT / "test_questions.csv", encoding="utf-8-sig")))
    out, scores = [], {m: {} for m in a.modes}
    for r in rows:
        q, gold, g = r["question"], r["correct_file"], group(r["type"])
        line = {"#": r["#"], "question": q, "correct_file": gold, "type": r["type"]}
        for m in a.modes:
            if m == "agent":
                hits = agent.RecallAgent(s, llm).run(q, k=10).hits
            else:
                hits = s.search(q, m, k=10)[0]
            names = [h.name for h in hits]
            rank = names.index(gold) + 1 if gold in names else None
            line[f"rank_{m}"] = rank or "none"
            scores[m].setdefault(g, []).append(rank)
        out.append(line)
        print(f"{r['#']:>3} {g:<10} " + " ".join(f"{m}={line[f'rank_{m}']!s:<4}" for m in a.modes) + f"  {q[:60]}")

    def summary(ranks):
        n = len(ranks)
        return (sum(r == 1 for r in ranks) / n, sum(bool(r) and r <= 5 for r in ranks) / n,
                statistics.mean((1 / r if r else 0) for r in ranks), n)

    print("\nmode       group        Hit@1  Recall@5  MRR@10   n")
    for m in a.modes:
        groups = scores[m]
        main_ranks = [x for g, v in groups.items() if g != "korean" for x in v]
        for g, v in list(groups.items()) + [("ALL but Korean", main_ranks)]:
            h1, r5, mrr, n = summary(v)
            print(f"{m:<10} {g:<12} {h1:5.3f}  {r5:8.3f}  {mrr:6.3f}  {n:3d}")
    with open(RT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    flagged = con.execute("SELECT name, pii FROM files WHERE pii NOT IN ('{}','') AND pii IS NOT NULL "
                          "AND duplicate_of IS NULL ORDER BY name").fetchall()
    print(f"\nSensitive files flagged: {len(flagged)}")
    for r in flagged:
        print(f"  {r['name']}: {pii.describe(json.loads(r['pii']))}")
    print(f"\nwrote {RT / 'results.csv'}")


if __name__ == "__main__":
    main()
