import base64
import hashlib

from rag.vector_store import get_vector_store


def _stable_id_from_chunk(chunk) -> str:
    file_name = (chunk.metadata or {}).get("file") or "unknown"
    content = chunk.page_content or ""
    content_hash = (chunk.metadata or {}).get("chunk_hash")
    if not content_hash:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    position = (chunk.metadata or {}).get("chunk_position")
    position_part = str(position) if position is not None else ""

    raw = f"{file_name}:{content_hash}:{position_part}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def index_documents(chunks) -> list[str]:
    store = get_vector_store()
    ids = [_stable_id_from_chunk(c) for c in chunks]
    return store.add_documents(chunks, ids=ids)
