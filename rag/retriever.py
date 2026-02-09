from rag.vector_store import get_vector_store


def get_retriever():
    return get_vector_store().as_retriever(search_type="hybrid")
