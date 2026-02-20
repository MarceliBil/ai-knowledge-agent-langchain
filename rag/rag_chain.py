import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableMap, RunnableLambda, RunnablePassthrough
from langchain_anthropic import ChatAnthropic

from rag.retriever import get_retriever
from config.settings import get_settings


def get_llm():
    s = get_settings()
    return ChatAnthropic(
        model=s.llm_model,
        temperature=0
    )


def join_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def format_sources(docs):
    seen = set()
    lines = []
    for d in docs:
        md = d.metadata or {}
        file_name = md.get("file")
        if not file_name:
            path = md.get("source_path") or md.get("source")
            if path:
                file_name = str(path).strip().replace("\\", "/").split("/")[-1]
        file_name = str(file_name).strip() if file_name else "unknown"
        if file_name in seen:
            continue
        seen.add(file_name)
        lines.append(f"- {file_name}")
    return "\n".join(lines) if lines else ""


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Odpowiadaj wyłącznie na podstawie kontekstu, ale nie wspominaj, że to robisz, nie użwaj słowa \"kontekst\". tylko po prostu udziel konkretnej odpowiedzi naturalnym językiem. Jeśli kontekst nie pozwala odpowiedzieć na pytanie, odpowiedz dokładnie: Nie mam wiedzy na ten temat. Jeśli kontekst zawiera choćby część odpowiedzi, podaj wyłącznie to, co wynika z kontekstu, w maksymalnie 2 krótkich zdaniach. Nie dodawaj zastrzeżeń typu 'nie mam wystarczających informacji' ani przeprosin."),
        MessagesPlaceholder("chat_history"),
        ("human", """Kontekst:
{context}

Pytanie:
{input}""")
    ])


def get_contextualize_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Na podstawie HISTORII CZATU i OSTATNIEGO PYTANIA sformułuj jedno, samodzielne pytanie, "
            "które można wysłać do wyszukiwarki dokumentów. "
            "W samodzielnym pytaniu zawsze doprecyzuj temat (np. podróż służbowa / zwrot kosztów) i zachowaj sens pytania "
            "(np. podlegają vs nie podlegają zwrotowi). "
            "Jeśli ostatnie pytanie jest już samodzielne, zwróć je bez zmian. "
            "Zwróć wyłącznie tekst pytania, bez dodatkowych słów."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def get_judge_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Oceń, czy KONTEKST zawiera informacje istotne dla PYTANIA (nawet jeśli pozwalają odpowiedzieć tylko częściowo). Zwróć wyłącznie YES albo NO. Jeśli kontekst jest nie na temat, zwróć NO."),
        ("human", """PYTANIE:
{input}

KONTEKST:
{context}""")
    ])


def get_route_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Zwróć dokładnie jeden token: RECAP albo RAG. Zwróć RECAP, jeśli użytkownik pyta o historię tej rozmowy (np. o co pytał wcześniej, jakie było ostatnie pytanie, co mówił wcześniej, przypomnij). W przeciwnym razie zwróć RAG. Nie dodawaj żadnych innych słów."),
        ("human", "{input}")
    ])


def get_recap_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Użytkownik pyta, o co pytał wcześniej. "
            "Napisz krótkie, naturalne zdanie w 2. osobie (np. 'Pytałeś/aś o…'), które streszcza POPRZEDNIE pytanie użytkownika. "
            "Nie streszczaj całej rozmowy, tylko jedno poprzednie pytanie. "
            "Bez cudzysłowów, bez list, bez nagłówków. "
            "Nie używaj słów: kontekst, historia czatu. "
            "Jeśli brak wcześniejszego pytania, zwróć dokładnie: Nie mam wiedzy na ten temat."
        ),
        ("human", "Poprzednie pytanie użytkownika:\n{previous_question}"),
    ])


_STOPWORDS_PL = {
    "a",
    "aby",
    "albo",
    "ale",
    "bo",
    "co",
    "czy",
    "dla",
    "do",
    "gdzie",
    "i",
    "jak",
    "jaka",
    "jakie",
    "jaki",
    "jest",
    "kiedy",
    "która",
    "które",
    "który",
    "ma",
    "mam",
    "mi",
    "mnie",
    "na",
    "nad",
    "nie",
    "o",
    "od",
    "oraz",
    "po",
    "pod",
    "się",
    "są",
    "ta",
    "ten",
    "to",
    "tu",
    "w",
    "we",
    "z",
    "za",
    "ze",
}


_STOPWORDS_EN = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


_EN_GREETINGS = {
    "hello",
    "hi",
    "hey",
    "yo",
    "thanks",
    "thank",
    "please",
    "good",
    "morning",
    "afternoon",
    "evening",
}


def _is_probably_polish(text: str) -> bool:
    t = (text or "").casefold()
    if not t.strip():
        return True
    if re.search(r"[ąćęłńóśźż]", t):
        return True
    words = re.findall(r"[a-z]+", t)
    if not words:
        return True
    if len(words) <= 3 and all(w in _EN_GREETINGS for w in words):
        return False
    pl_hits = sum(1 for w in words if w in _STOPWORDS_PL)
    en_hits = sum(1 for w in words if w in _STOPWORDS_EN)
    return pl_hits >= en_hits


def _tokens_pl(text: str) -> set[str]:
    t = (text or "").casefold()
    raw = re.findall(r"[\wąćęłńóśźż]+", t)
    out: set[str] = set()
    for tok in raw:
        if len(tok) < 3:
            continue
        if tok in _STOPWORDS_PL:
            continue
        out.add(tok)
    return out


def _has_relevance_overlap(question: str, context: str, *, min_hits: int = 1) -> bool:
    q = _tokens_pl(question)
    if not q:
        return True
    c = _tokens_pl(context)
    return len(q & c) >= min_hits


_RECAP_PATTERNS = [
    re.compile(
        r"\b(przypomnij|przypomnij\s+mi|histori[ea]|co\s+(pyta(?:łem|łam)|mówi(?:łem|łam)|pisa(?:łem|łam))|"
        r"jakie\s+było\s+(moje\s+)?ostatnie\s+pytanie|o\s+co\s+pyta(?:łem|łam)(\s+(wcześniej|poprzednio))?|"
        r"podsumuj\s+(rozmowę|czat|naszą\s+rozmowę)|streść\s+(rozmowę|czat|naszą\s+rozmowę))\b",
        re.IGNORECASE,
    ),
]


def _detect_route(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "RAG"
    for p in _RECAP_PATTERNS:
        if p.search(t):
            return "RECAP"
    return "RAG"


def get_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_prompt()
    judge_prompt = get_judge_prompt()
    contextualize_prompt = get_contextualize_prompt()
    recap_prompt = get_recap_prompt()

    unknown = "Nie mam wiedzy na ten temat."
    polish_only = "Na ten moment jestem dostępny tylko w języku polskim."

    route_runnable = RunnableLambda(
        lambda x: _detect_route(x.get("input") or ""))

    start = RunnablePassthrough.assign(route=route_runnable)

    def _previous_question(x) -> str:
        history = x.get("chat_history") or []
        for m in reversed(history):
            if getattr(m, "type", "") != "human":
                continue
            c = str(getattr(m, "content", "") or "").strip()
            if c:
                return c
        return ""

    recap_chain = (
        RunnableLambda(lambda x: {"previous_question": _previous_question(x)})
        | RunnableBranch(
            (lambda x: not (x.get("previous_question") or "").strip(),
             RunnableLambda(lambda _: unknown)),
            recap_prompt | llm | StrOutputParser(),
        )
    )

    contextualize_runnable = RunnableBranch(
        (
            lambda x: not (x.get("chat_history") or []),
            RunnableLambda(lambda x: x.get("input") or ""),
        ),
        contextualize_prompt | llm | StrOutputParser(),
    )

    def _q(x):
        q = (x.get("standalone_question") or x.get("input") or "").strip()
        return q

    docs_runnable = RunnableLambda(lambda x: retriever.invoke(_q(x)))

    base = RunnablePassthrough.assign(
        standalone_question=contextualize_runnable)
    base = base.assign(docs=docs_runnable)
    base = base.assign(context=RunnableLambda(lambda x: join_docs(x["docs"])))
    base = base.assign(sources=RunnableLambda(
        lambda x: format_sources(x["docs"])))
    base = base.assign(overlap_ok=RunnableLambda(
        lambda x: _has_relevance_overlap(_q(x), x.get("context") or "")
    ))

    judge_runnable = (
        RunnableLambda(lambda x: {"input": _q(
            x), "context": x.get("context") or ""})
        | judge_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda s: "YES" if "YES" in str(s).strip().upper() else "NO")
    )
    base = base.assign(judge=judge_runnable)

    def _prep_for_answer(x):
        return {
            **x,
            "input": _q(x),
        }

    answer = RunnableBranch(
        (lambda x: not (x.get("context") or "").strip(),
         RunnableLambda(lambda _: unknown)),
        (lambda x: not bool(x.get("overlap_ok")),
         RunnableLambda(lambda _: unknown)),
        (lambda x: (x.get("judge") or "").strip().upper() != "YES",
         RunnableLambda(lambda _: unknown)),
        RunnableLambda(_prep_for_answer) | prompt | llm | StrOutputParser(),
    )

    out = RunnableMap(
        {
            "answer": answer,
            "sources": RunnableLambda(lambda x: x["sources"]),
        }
    )

    def _render(x):
        a = (x.get("answer") or "").strip()
        s = (x.get("sources") or "").strip()
        a = re.sub(r"^(zgodnie|na\s+podstawie)\s+z?\s*kontekst(em|u)\s*,\s*",
                   "", a, flags=re.IGNORECASE)
        low = a.casefold()
        if ("nie mam wiedzy" in low) or ("nie wiem" in low):
            return unknown
        if not s:
            return a or unknown
        return f"{a}\n\nSources:\n{s}"

    rag_chain = base | out | RunnableLambda(_render)

    return start | RunnableBranch(
        (lambda x: (x.get("route") or "").upper() == "RECAP", recap_chain),
        (lambda x: not _is_probably_polish(x.get("input") or ""),
         RunnableLambda(lambda _: polish_only)),
        rag_chain,
    )
