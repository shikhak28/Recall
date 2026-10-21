"""Command line:  python -m recall index <folder>  |  python -m recall search "<query>" [--mode agent] [--answer]"""
import argparse
from datetime import datetime
from pathlib import Path

from . import agent, index
from .search import Searcher


def main():
    ap = argparse.ArgumentParser(prog="recall")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("folder")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--mode", default="agent", choices=["agent", "hybrid", "semantic", "keyword", "filename"])
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--answer", action="store_true", help="let Nemotron read the top-3 snippets to answer")
    sub.add_parser("stats")
    sub.add_parser("models", help="list Nemotron models visible on your Token Factory account")
    a = ap.parse_args()

    con = index.connect()
    if a.cmd == "index":
        rep = index.index_folder(Path(a.folder), con, progress=lambda i, n, f: print(f"\r{i}/{n} {f[:40]:<40}", end="", flush=True))
        print("\n", rep)
    elif a.cmd == "stats":
        print(index.stats(con))
    elif a.cmd == "models":
        llm = agent.make_llm()
        print("selected:", llm.model if llm else "(no NEBIUS_API_KEY)")
        if llm:
            print("\n".join(m.id for m in llm.client.models.list().data if "nemotron" in m.id.lower()))
    else:
        s = Searcher(con)
        if a.mode == "agent":
            res = agent.RecallAgent(s, agent.make_llm()).run(a.query, k=a.k, allow_answer=a.answer)
            for stp in res.steps:
                print(f"  [{stp.name}] {stp.detail}")
            hits = res.hits
            if res.answer:
                print("\nANSWER:", res.answer)
        else:
            hits, _ = s.search(a.query, a.mode, k=a.k)
        print()
        for i, h in enumerate(hits, 1):
            print(f"{i}. {h.name}  ({h.kind}, saved {datetime.fromtimestamp(h.mtime):%Y-%m-%d})\n   {h.snippet}\n   {h.path}")


if __name__ == "__main__":
    main()
