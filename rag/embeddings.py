from langchain_openai import OpenAIEmbeddings

_embeddings = OpenAIEmbeddings()


def embed_chunks(chunks):
    return _embeddings.embed_documents([c.page_content for c in chunks])


def embed_query(text: str):
    return _embeddings.embed_query(text)
