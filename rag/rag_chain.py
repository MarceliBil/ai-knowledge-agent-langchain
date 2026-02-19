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
        ("system", "Odpowiadaj wyłącznie na podstawie kontekstu. Jeśli kontekst nie pozwala odpowiedzieć na pytanie, odpowiedz dokładnie: Nie mam wiedzy na ten temat. Jeśli kontekst zawiera choćby część odpowiedzi, podaj wyłącznie to, co wynika z kontekstu, w maksymalnie 2 krótkich zdaniach. Nie dodawaj zastrzeżeń typu 'nie mam wystarczających informacji' ani przeprosin."),
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
        ("system", "Oceń, czy na podstawie KONTEKSTU da się jednoznacznie odpowiedzieć na PYTANIE. Zwróć wyłącznie YES albo NO. Jeśli kontekst jest nie na temat, zwróć NO."),
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


def get_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_prompt()
    judge_prompt = get_judge_prompt()
    route_prompt = get_route_prompt()
    contextualize_prompt = get_contextualize_prompt()

    unknown = "Nie mam wiedzy na ten temat."

    route_runnable = (
        route_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda s: "RECAP" if "RECAP" in str(s).strip().upper() else "RAG")
    )

    start = RunnablePassthrough.assign(route=route_runnable)

    def _recap(x):
        history = x.get("chat_history") or []
        last = ""
        for m in reversed(history):
            if getattr(m, "type", "") != "human":
                continue
            c = str(getattr(m, "content", "") or "").strip()
            if c:
                last = c
                break
        return last or unknown

    recap_chain = RunnableLambda(_recap)

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

    def _prep_for_answer(x):
        return {
            **x,
            "input": _q(x),
        }

    answer = RunnableBranch(
        (lambda x: not (x.get("context") or "").strip(),
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
        low = a.casefold()
        if ("nie mam wiedzy" in low) or ("nie wiem" in low):
            return unknown
        if not s:
            return a or unknown
        return f"{a}\n\nSources:\n{s}"

    rag_chain = base | out | RunnableLambda(_render)

    return start | RunnableBranch(
        (lambda x: (x.get("route") or "").upper() == "RECAP", recap_chain),
        rag_chain,
    )
