from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda, RunnablePassthrough
from langchain_anthropic import ChatAnthropic

from rag.retriever import get_retriever


def get_llm():
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0
    )


def join_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Odpowiadaj wyłącznie na podstawie kontekstu. Jeśli brak odpowiedzi w dokumentach — powiedz że jej nie ma."),
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

    retrieval_block = RunnableLambda(
        lambda x: retriever.invoke(x["input"])
    ) | RunnableLambda(join_docs)

    chain = (
        RunnableMap({
            "context": retrieval_block,
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"])
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
