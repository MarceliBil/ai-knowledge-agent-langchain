from dotenv import load_dotenv

from ingest.blob_loader import load_documents
from ingest.chunking import production_chunk_documents
from ingest.index_azure_search import index_documents


def run():
    load_dotenv()
    docs = load_documents()
    chunks = production_chunk_documents(docs)
    index_documents(chunks)


if __name__ == "__main__":
    run()
