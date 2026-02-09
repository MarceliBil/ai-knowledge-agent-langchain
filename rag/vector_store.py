from functools import lru_cache

from langchain_community.vectorstores.azuresearch import AzureSearch

from config.embeddings import get_embeddings
from config.settings import get_settings


@lru_cache(maxsize=1)
def get_vector_store() -> AzureSearch:
    s = get_settings()
    return AzureSearch(
        azure_search_endpoint=s.azure_search_endpoint,
        azure_search_key=s.azure_search_key,
        index_name=s.azure_search_index,
        embedding_function=get_embeddings().embed_query,
        content_key="content",
        vector_key="content_vector",
    )
