"""Pick the hybrid keyword weight on a DEV corpus (different seed from the test corpus).

    python eval/make_dataset.py --seed 11 --n 150 --out eval/dev
    python eval/tune_hybrid.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["RECALL_HOME"] = str(ROOT / ".recall-dev")
sys.path.insert(0, str(ROOT.parent))

from recall import index  # noqa: E402
from recall.search import Searcher  # noqa: E402
from run_eval import retrieval_eval  # noqa: E402

data = ROOT / "dev"
queries = [json.loads(l) for l in open(data / "queries.jsonl")]
con = index.connect()
index.index_folder(data / "files", con)
s = Searcher(con)
print("| kw_weight | Recall@5 | MRR@10 |\n|---|---|---|")
for w in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    s.kw_weight = w
    r = retrieval_eval(str(w), lambda q: [h.name for h in s.search(q, "hybrid", k=10)[0]], queries)
    print(f"| {w} | {r['recall@5']:.3f} | {r['mrr@10']:.3f} |")
