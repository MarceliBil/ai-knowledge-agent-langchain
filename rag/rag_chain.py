from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough

from rag.retriever import get_retriever


def get_llm():
    return ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )


def get_prompt():
    return ChatPromptTemplate.from_template(
        """Odpowiadaj wyłącznie na podstawie dostarczonego kontekstu.
Jeśli odpowiedź nie znajduje się w kontekście - napisz że nie ma jej w dokumentach.

Kontekst:
{context}

Pytanie:
{question}

Odpowiedź:"""
    )


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_prompt()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
