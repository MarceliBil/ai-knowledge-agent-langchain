import base64

from rag.vector_store import get_vector_store


def chunk_id_from_doc_id(doc_id: str, chunk_position: int) -> str:
    raw = f"{doc_id}:{chunk_position}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def _stable_id_from_chunk(chunk) -> str:
    md = chunk.metadata or {}
    doc_id = md.get("source_path") or md.get(
        "blob_name") or md.get("file") or "unknown"
    pos = md.get("chunk_position")
    try:
        chunk_position = int(pos)
    except Exception:
        chunk_position = 0
    return chunk_id_from_doc_id(str(doc_id), chunk_position)


def index_documents(chunks) -> list[str]:
    store = get_vector_store()
    ids = [_stable_id_from_chunk(c) for c in chunks]
    return store.add_documents(chunks, ids=ids)
