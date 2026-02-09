from rag.vector_store import get_vector_store
from config.settings import get_settings


def get_retriever():
    s = get_settings()
    search_type = getattr(s, "retrieval_search_type", "hybrid")
    search_kwargs = {}
    score_threshold = getattr(s, "retrieval_score_threshold", None)
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    return get_vector_store().as_retriever(
        search_type=search_type,
        k=s.retrieval_k,
        search_kwargs=search_kwargs,
    )
