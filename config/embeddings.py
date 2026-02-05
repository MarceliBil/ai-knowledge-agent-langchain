import os
from langchain_openai import AzureOpenAIEmbeddings


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version="2024-02-01"
    )
