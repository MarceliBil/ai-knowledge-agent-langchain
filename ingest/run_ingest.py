from ingest.drive_loader import load_docs
from ingest.chunking import production_chunk_documents
from ingest.index_to_db import save_chunks

docs = load_docs()
chunks = production_chunk_documents(docs)
save_chunks(chunks)
