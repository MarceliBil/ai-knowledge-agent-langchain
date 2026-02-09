from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda, RunnablePassthrough
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
        file_name = (d.metadata or {}).get("file") or (
            d.metadata or {}).get("source") or "unknown"
        if file_name in seen:
            continue
        seen.add(file_name)
        lines.append(f"- {file_name}")
    return "\n".join(lines) if lines else "- none"


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Odpowiadaj wyłącznie na podstawie kontekstu. Jeśli nie ma odpowiedzi w dokumentach, powiedz że jej nie ma."),
        MessagesPlaceholder("chat_history"),
        ("human", """Kontekst:
{context}

Pytanie:
{input}""")
    ])


def get_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_prompt()

    docs_runnable = RunnableLambda(lambda x: retriever.invoke(x["input"]))

    base = RunnablePassthrough.assign(docs=docs_runnable)
    base = base.assign(context=RunnableLambda(lambda x: join_docs(x["docs"])))
    base = base.assign(sources=RunnableLambda(
        lambda x: format_sources(x["docs"])))

    out = RunnableMap(
        {
            "answer": (prompt | llm | StrOutputParser()),
            "sources": RunnableLambda(lambda x: x["sources"]),
        }
    )

    return base | out | RunnableLambda(lambda x: f"{x['answer']}\n\nSources:\n{x['sources']}")
