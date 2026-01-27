import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


def enrich_metadata(chunks):
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["source"] = c.metadata.get("source", "google_drive")
        c.metadata["file"] = c.metadata.get("file_name", "unknown")
    return chunks


def add_chunk_hash(chunks):
    for c in chunks:
        h = hashlib.sha256(c.page_content.encode()).hexdigest()
        c.metadata["hash"] = h
    return chunks


def production_chunk_documents(docs):
    semantic_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=2000,
        chunk_overlap=0,
    )

    semantic_chunks = semantic_splitter.split_documents(docs)

    token_splitter = TokenTextSplitter(
        chunk_size=300,
        chunk_overlap=60,
    )

    final_chunks = []

    for doc in semantic_chunks:
        final_chunks.extend(token_splitter.split_documents([doc]))

    final_chunks = enrich_metadata(final_chunks)
    final_chunks = add_chunk_hash(final_chunks)

    return final_chunks
