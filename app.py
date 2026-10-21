"""Recall — Streamlit UI.   streamlit run app.py"""
import html
import io
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st

from recall import agent, config, index
from recall.search import Searcher, tokens

st.set_page_config(page_title="Recall", page_icon="🔎", layout="wide")
DEMO_DIR = Path(__file__).parent / "eval" / "data" / "files"
UPLOAD_DIR = config.DATA_DIR / "uploads"
EXAMPLES = ["when is my machine learning exam?", "wifi password at grandma's house",
            "slide about stopping a model from memorizing training data", "the Starbucks receipt from September",
            "documents needed to extend my visa"]

st.markdown("""<style>
mark {background: #76b90033; padding: 0 2px; border-radius: 3px; color: inherit;}
.snip {font-size: 0.95rem; line-height: 1.5;}
.meta {color: #888; font-size: 0.82rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_con():
    return index.connect()


@st.cache_resource
def get_searcher():
    return Searcher(get_con())


@st.cache_resource
def get_llm():
    try:
        return agent.make_llm(), None
    except Exception as e:
        return None, str(e)


def open_file(path: str):
    if "microsoft" in platform.uname().release.lower():  # WSL -> open with the Windows default app
        win = subprocess.run(["wslpath", "-w", path], capture_output=True, text=True).stdout.strip()
        subprocess.Popen(["explorer.exe", win])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    elif platform.system() == "Windows":
        os.startfile(path)  # noqa
    else:
        subprocess.Popen(["xdg-open", path])


def highlight(text: str, query_terms: set[str]) -> str:
    out = html.escape(text)
    for t in sorted(query_terms, key=len, reverse=True):
        if len(t) >= 3:
            out = re.sub(rf"(?i)\b({re.escape(html.escape(t))}\w*)", r"<mark>\1</mark>", out)
    return out


@st.cache_data(max_entries=256)
def pdf_thumb(path: str, page: int) -> bytes:
    import pymupdf

    with pymupdf.open(path) as doc:
        pix = doc[max(0, (page or 1) - 1)].get_pixmap(dpi=60)
        return pix.tobytes("png")


def run_index(folder: Path):
    bar = st.sidebar.progress(0.0, text="Starting…")
    rep = index.index_folder(folder, get_con(), progress=lambda i, n, f: bar.progress(i / n, text=f"{i}/{n} · {f[:30]}"))
    bar.empty()
    st.sidebar.success(f"Indexed {rep.indexed} new · {rep.skipped_unchanged} unchanged · {rep.duplicates} duplicates · "
                       f"{rep.empty} without text · {rep.errors} errors · {rep.seconds:.0f}s")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Your files")
    folder = st.text_input("Folder to index", value=str(DEMO_DIR if DEMO_DIR.exists() else Path.home() / "Pictures"))
    c1, c2 = st.columns(2)
    if c1.button("Index folder", type="primary", width="stretch"):
        try:
            run_index(Path(folder).expanduser())
        except FileNotFoundError as e:
            st.error(str(e))
    if c2.button("Clear index", width="stretch"):
        index.clear(get_con())
        st.toast("Index cleared")
    ups = st.file_uploader("…or drop files", accept_multiple_files=True,
                           type=[e.strip(".") for e in sorted(config.SUPPORTED_EXT)])
    if ups and st.button("Index uploaded files", width="stretch"):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for u in ups:
            (UPLOAD_DIR / Path(u.name).name).write_bytes(u.getbuffer())
        run_index(UPLOAD_DIR)

    s = index.stats(get_con())
    m1, m2, m3 = st.columns(3)
    m1.metric("Searchable", s["searchable"])
    m2.metric("Images", s["images"])
    m3.metric("PDFs", s["pdfs"])
    st.caption(f"{s['duplicates']} duplicates collapsed · {s['empty']} without text · {s['errors']} unreadable")

    st.header("Search engine")
    llm, llm_err = get_llm()
    mode = st.radio("Mode", ["Agent", "Hybrid", "Semantic", "Keyword", "Filename only"], horizontal=False,
                    help="Agent = Nemotron plans the search (filters, synonyms) and retries when unsure. Others are the building blocks.")
    allow_answer = st.toggle("Let Nemotron read the top 3 snippets to answer questions", value=False,
                             help="Off: only your query text ever leaves the device. On: the top-3 extracted snippets are sent too.")
    if llm:
        st.success(f"Nemotron via Nebius Token Factory\n\n`{llm.model}`", icon="🟢")
    else:
        st.info("Offline planner (no NEBIUS_API_KEY). Everything runs locally." + (f"\n\n{llm_err}" if llm_err else ""), icon="🔒")

# ------------------------------------------------------------------ main
st.title("🔎 Recall")
st.caption("Find anything you saved — by what it *said*, not what it was called. OCR, embeddings and the index never leave this machine.")

tab_search, tab_privacy = st.tabs(["Search", "Privacy ledger"])
with tab_search:
    if "q" not in st.session_state:
        st.session_state.q = ""
    ex = st.columns(len(EXAMPLES))
    for col, e in zip(ex, EXAMPLES):
        if col.button(e, width="stretch"):
            st.session_state.q = e
    with st.form("search", border=False):
        q = st.text_input("What are you looking for?", key="q", placeholder="e.g. the screenshot with my exam date",
                          label_visibility="collapsed")
        st.form_submit_button("Search", type="primary")

    if q.strip():
        searcher = get_searcher()
        answer, steps = None, []
        if mode == "Agent":
            with st.spinner("Planning and searching…"):
                res = agent.RecallAgent(searcher, llm).run(q, k=10, allow_answer=allow_answer)
            hits, answer, steps, ms = res.hits, res.answer, res.steps, res.total_ms
            terms = set(tokens(q)) | {t.lower() for t in res.plan.keywords}
        else:
            m = {"Hybrid": "hybrid", "Semantic": "semantic", "Keyword": "keyword", "Filename only": "filename"}[mode]
            hits, info = searcher.search(q, m, k=10)
            ms, terms = info["ms"], set(tokens(q))

        if answer:
            st.info(answer, icon="💬")
        st.caption(f"{len(hits)} results · {ms:.0f} ms")
        if steps:
            with st.expander("Agent trace", expanded=False):
                for stp in steps:
                    st.markdown(f"**{stp.name}** · {stp.ms:.0f} ms")
                    st.json(stp.detail, expanded=False)
        if not hits:
            st.warning("Nothing matched. Try describing what was written in the file, or index another folder.")

        for rank, h in enumerate(hits, 1):
            with st.container(border=True):
                left, right = st.columns([1, 3])
                try:
                    if h.kind == "image":
                        left.image(h.path, width="stretch")
                    elif h.kind == "pdf":
                        left.image(pdf_thumb(h.path, h.page), width="stretch")
                    else:
                        left.code(h.chunk_text[:300])
                except Exception:
                    left.caption("(preview unavailable — file moved?)")
                sig = " · ".join(f"{k} {v:.2f}" for k, v in h.signals.items() if v is not None)
                saved = datetime.fromtimestamp(h.mtime).strftime("%Y-%m-%d")
                right.markdown(f"**{rank}. {html.escape(h.name)}**  \n<span class='meta'>{h.kind}"
                               f"{' · page ' + str(h.page) if h.kind == 'pdf' and h.page else ''} · saved {saved} · {sig}</span>",
                               unsafe_allow_html=True)
                right.markdown(f"<div class='snip'>{highlight(h.snippet, terms)}</div>", unsafe_allow_html=True)
                b1, b2, _ = right.columns([1, 1, 3])
                if b1.button("Open", key=f"open{h.file_id}"):
                    try:
                        open_file(h.path)
                    except Exception as e:
                        st.error(f"Could not open: {e}")
                if Path(h.path).exists():
                    b2.download_button("Download", Path(h.path).read_bytes(), file_name=h.name, key=f"dl{h.file_id}")
                with right.expander("Extracted text"):
                    st.text(h.chunk_text)
                right.caption(h.path)

with tab_privacy:
    st.subheader("Everything that left this device")
    st.write("OCR, embeddings, the index and ranking run locally. The only outbound calls are to Nemotron on "
             "Nebius Token Factory, and each one is logged here with its exact payload.")
    led = agent.LEDGER
    a, b, c = st.columns(3)
    a.metric("Requests", len(led))
    b.metric("Bytes sent", sum(e.bytes_sent for e in led))
    c.metric("Requests containing file text", sum(e.contains_file_content for e in led))
    for e in reversed(led[-50:]):
        with st.expander(f"{e.ts} · {e.purpose} · {e.host} · {e.bytes_sent} B · "
                         f"{'⚠ contains file snippets' if e.contains_file_content else 'query text only'}"):
            st.code(e.payload, language="json")
