from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)
