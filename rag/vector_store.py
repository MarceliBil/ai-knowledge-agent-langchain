import os
from langchain_community.vectorstores.azuresearch import AzureSearch
from config.embeddings import get_embeddings


def get_vector_store():
    return AzureSearch(
        azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_search_key=os.environ["AZURE_SEARCH_KEY"],
        index_name=os.environ["AZURE_SEARCH_INDEX"],
        embedding_function=get_embeddings().embed_query,
        content_key="content",
        vector_key="content_vector"
    )
