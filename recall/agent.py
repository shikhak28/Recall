"""Agentic search loop: plan -> retrieve -> judge -> (re-plan) -> optional grounded answer.

The planner is NVIDIA Nemotron served by Nebius Token Factory. Privacy contract:
  * planning and re-planning send ONLY the user's query text (never file names or contents);
  * the optional answer step sends the top snippets, and only when the user turns it on.
Every outbound request is written to an in-memory ledger that the UI shows verbatim.
If no API key is configured, a deterministic rule-based planner keeps everything fully offline.
"""
import calendar
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

from . import config
from .search import Filters, Hit, Searcher, tokens

WEAK_SEMANTIC = 0.62


@dataclass
class LedgerEntry:
    ts: str
    purpose: str
    host: str
    model: str
    bytes_sent: int
    contains_file_content: bool
    payload: str


LEDGER: list[LedgerEntry] = []


@dataclass
class Plan:
    semantic_query: str
    keywords: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    intent: str = "find"
    source: str = "rules"

    def filters(self) -> Filters:
        ts = lambda s: datetime.strptime(s, "%Y-%m-%d").timestamp() if s else None
        kinds = [k for k in self.file_types if k in ("image", "pdf", "text")]
        return Filters(kinds=kinds, date_from=ts(self.date_from), date_to=ts(self.date_to))


MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
QUESTION_RE = re.compile(r"^(when|what|where|which|how much|how many|who|what's|whats)\b|\?$", re.I)


def plan_rules(query: str, today: date | None = None) -> Plan:
    """Offline planner: file-type words, relative/month dates, question detection."""
    today = today or date.today()
    q = query.lower()
    types = []
    if re.search(r"\b(screenshot|screen shot|image|photo|picture|pic|png|jpg)s?\b", q):
        types.append("image")
    if re.search(r"\b(pdf|scan|scanned)s?\b", q):
        types.append("pdf")
    d_from = d_to = None
    if "yesterday" in q:
        d_from, d_to = today - timedelta(days=1), today
    elif "today" in q:
        d_from, d_to = today, today + timedelta(days=1)
    elif "last week" in q or "past week" in q:
        d_from, d_to = today - timedelta(days=7), today + timedelta(days=1)
    elif "this month" in q:
        d_from, d_to = today.replace(day=1), today + timedelta(days=1)
    elif "last month" in q:
        first = today.replace(day=1)
        d_to = first
        d_from = (first - timedelta(days=1)).replace(day=1)
    else:
        m = re.search(r"\b(?:in|from|during|saved in|taken in)\s+(" + "|".join(MONTHS) + r")\b", q)
        if m:
            mon = MONTHS[m.group(1)]
            yr = today.year if mon <= today.month else today.year - 1
            d_from = date(yr, mon, 1)
            d_to = date(yr + (mon == 12), mon % 12 + 1, 1)
    fmt = lambda d: d.isoformat() if d else None
    return Plan(
        semantic_query=query,
        keywords=[],
        file_types=types,
        date_from=fmt(d_from),
        date_to=fmt(d_to),
        intent="answer" if QUESTION_RE.search(query.strip()) else "find",
        source="rules",
    )


PLANNER_PROMPT = """You are the query planner of Recall, a private search engine over a person's own screenshots, photos and PDFs.
Today is {today}. Turn the user's request into a JSON search plan. Respond with ONLY a JSON object, no prose:
{{
  "semantic_query": "a rewritten description of the text that would appear INSIDE the target file (expand synonyms, drop words like 'find the screenshot')",
  "keywords": ["3-8 exact words or numbers likely printed in the file, including synonyms, e.g. exam -> examination, midterm"],
  "file_types": ["image" and/or "pdf" and/or "text" -- only if the user explicitly names a type, else []],
  "date_from": "YYYY-MM-DD or null -- only if the user gives a time for when the file was SAVED",
  "date_to": "YYYY-MM-DD (exclusive) or null",
  "intent": "answer if the user asks a question whose answer is in the file, else find"
}}
Dates mentioned as the CONTENT of the file (e.g. 'the exam on Oct 23') are keywords, not date filters."""

REPLAN_PROMPT = """The previous search plan for this request returned weak results:
{previous}
Propose ONE different strategy as the same JSON schema: broaden or rephrase the semantic_query, use different keywords
(synonyms, abbreviations, related terms), and drop filters that may be too strict. Respond with ONLY the JSON object."""

ANSWER_PROMPT = """Answer the user's question using ONLY the numbered excerpts from their own files below.
Cite the excerpt number like [1]. If the excerpts do not contain the answer, say so in one sentence. Be brief.

Question: {question}

{excerpts}"""


class Nemotron:
    """Minimal OpenAI-compatible client for Nebius Token Factory with an outbound-data ledger."""

    def __init__(self, api_key: str = config.NEBIUS_API_KEY, base_url: str = config.NEBIUS_BASE_URL,
                 model: str = config.NEBIUS_MODEL):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=1)
        self.host = re.sub(r"^https?://([^/]+).*", r"\1", base_url)
        self.model = model or self._pick_model()

    def _pick_model(self) -> str:
        ids = [m.id for m in self.client.models.list().data]
        nemo = [i for i in ids if "nemotron" in i.lower()]
        if not nemo:
            raise RuntimeError("no Nemotron model visible on this Token Factory account; set NEBIUS_MODEL")
        for pref in ("nano", "super", ""):
            for i in nemo:
                if pref in i.lower() and not any(x in i.lower() for x in ("embed", "rerank", "guard", "reward")):
                    return i
        return nemo[0]

    def chat(self, system: str, user: str, purpose: str, contains_file_content: bool, max_tokens: int = 1200) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload = json.dumps(messages, ensure_ascii=False)
        LEDGER.append(LedgerEntry(datetime.now().isoformat(timespec="seconds"), purpose, self.host, self.model,
                                  len(payload.encode()), contains_file_content, payload))
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.0,
                                                   max_tokens=max_tokens)
        text = resp.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _parse_plan(text: str, query: str) -> Plan | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    valid_date = lambda s: s if isinstance(s, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else None
    return Plan(
        semantic_query=str(d.get("semantic_query") or query),
        keywords=[str(k) for k in (d.get("keywords") or []) if str(k).strip()][:10],
        file_types=[t for t in (d.get("file_types") or []) if t in ("image", "pdf", "text")],
        date_from=valid_date(d.get("date_from")),
        date_to=valid_date(d.get("date_to")),
        intent="answer" if d.get("intent") == "answer" else "find",
        source="nemotron",
    )


@dataclass
class Step:
    name: str
    detail: dict
    ms: float


@dataclass
class AgentResult:
    hits: list[Hit]
    plan: Plan
    steps: list[Step]
    answer: str | None = None
    total_ms: float = 0.0


class RecallAgent:
    def __init__(self, searcher: Searcher, llm: Nemotron | None = None, max_attempts: int = 3):
        self.searcher = searcher
        self.llm = llm
        self.max_attempts = max_attempts

    def run(self, query: str, k: int = 10, allow_answer: bool = False, today: date | None = None) -> AgentResult:
        t_all = time.perf_counter()
        today = today or date.today()
        steps: list[Step] = []

        plan = self._plan(query, today, steps)
        attempts: list[tuple[Plan, list[Hit], dict]] = []
        for attempt in range(self.max_attempts):
            hits, info = self.searcher.search(query, "hybrid", plan.filters(), k,
                                              semantic_query=plan.semantic_query, keywords=plan.keywords)
            attempts.append((plan, hits, info))
            weak = not hits or info["top_semantic"] < WEAK_SEMANTIC
            steps.append(Step(f"retrieve #{attempt + 1}", {
                "semantic_query": plan.semantic_query, "keywords": plan.keywords,
                "filters": {"types": plan.file_types, "from": plan.date_from, "to": plan.date_to},
                "results": len(hits), "top_similarity": round(info["top_semantic"], 3),
                "verdict": "weak -> re-plan" if weak and attempt + 1 < self.max_attempts else ("weak" if weak else "confident"),
            }, info["ms"]))
            if not weak:
                break
            if attempt + 1 < self.max_attempts:
                plan = self._replan(query, plan, today, steps)

        plan, hits, info = max(attempts, key=lambda a: (bool(a[1]), a[2]["top_semantic"]))

        answer = None
        if allow_answer and plan.intent == "answer" and hits and self.llm:
            answer = self._answer(query, hits[:3], steps)
        return AgentResult(hits, plan, steps, answer, (time.perf_counter() - t_all) * 1000)

    def _plan(self, query, today, steps) -> Plan:
        t = time.perf_counter()
        plan = None
        if self.llm:
            try:
                plan = _parse_plan(self.llm.chat(PLANNER_PROMPT.format(today=today.isoformat()), query,
                                                 "plan", contains_file_content=False), query)
            except Exception as e:
                steps.append(Step("plan (nemotron failed, using rules)", {"error": str(e)[:200]}, 0))
        plan = plan or plan_rules(query, today)
        steps.append(Step(f"plan ({plan.source})", asdict(plan), (time.perf_counter() - t) * 1000))
        return plan

    def _replan(self, query, prev: Plan, today, steps) -> Plan:
        t = time.perf_counter()
        new = None
        if self.llm:
            try:
                prev_json = json.dumps({k: v for k, v in asdict(prev).items() if k != "source"})
                text = self.llm.chat(PLANNER_PROMPT.format(today=today.isoformat()),
                                     query + "\n\n" + REPLAN_PROMPT.format(previous=prev_json),
                                     "re-plan", contains_file_content=False)
                new = _parse_plan(text, query)
            except Exception as e:
                steps.append(Step("re-plan (nemotron failed)", {"error": str(e)[:200]}, 0))
        if new is None:
            if prev.file_types or prev.date_from or prev.date_to:
                new = Plan(prev.semantic_query, prev.keywords, [], None, None, prev.intent, "rules")
            else:
                new = Plan(" ".join(tokens(query)) or query, tokens(query), [], None, None, prev.intent, "rules")
        steps.append(Step(f"re-plan ({new.source})", asdict(new), (time.perf_counter() - t) * 1000))
        return new

    def _answer(self, query, hits, steps) -> str | None:
        t = time.perf_counter()
        excerpts = "\n\n".join(f"[{i + 1}] ({h.name})\n{h.chunk_text[:600]}" for i, h in enumerate(hits))
        try:
            ans = self.llm.chat("You answer questions about the user's own files.",
                                ANSWER_PROMPT.format(question=query, excerpts=excerpts),
                                "answer", contains_file_content=True, max_tokens=800)
        except Exception as e:
            ans = None
            steps.append(Step("answer (failed)", {"error": str(e)[:200]}, 0))
        steps.append(Step("answer (nemotron)", {"excerpts_sent": len(hits)}, (time.perf_counter() - t) * 1000))
        return ans


def make_llm() -> Nemotron | None:
    if not config.NEBIUS_API_KEY:
        return None
    return Nemotron()
