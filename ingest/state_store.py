from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256


@dataclass(frozen=True)
class DocState:
    doc_id: str
    etag: str
    chunk_count: int


def _prefix() -> str:
    v = os.environ.get("RAG_STATE_PREFIX", "_rag_state")
    v = v.strip().strip("/")
    return v or "_rag_state"


def _name(doc_id: str) -> str:
    h = sha256(doc_id.encode("utf-8")).hexdigest()
    return f"{_prefix()}/{h}.json"


def load_state(container_client, doc_id: str) -> DocState | None:
    blob = container_client.get_blob_client(_name(doc_id))
    try:
        data = json.loads(blob.download_blob().readall())
        return DocState(
            doc_id=str(data.get("doc_id") or doc_id),
            etag=str(data.get("etag") or ""),
            chunk_count=int(data.get("chunk_count") or 0),
        )
    except Exception:
        return None


def save_state(container_client, state: DocState) -> None:
    blob = container_client.get_blob_client(_name(state.doc_id))
    payload = json.dumps(
        {
            "doc_id": state.doc_id,
            "etag": state.etag,
            "chunk_count": int(state.chunk_count),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    blob.upload_blob(payload, overwrite=True, content_type="application/json")


def delete_state(container_client, doc_id: str) -> None:
    blob = container_client.get_blob_client(_name(doc_id))
    try:
        blob.delete_blob()
    except Exception:
        return
