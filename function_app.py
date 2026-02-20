from __future__ import annotations
import logging
import json
import os
import tempfile
import time
import traceback
from pathlib import PurePath
import azure.functions as func

from ingest.text_cleaning import normalize_extracted_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp()


@app.function_name(name="healthz")
@app.route(route="healthz", auth_level=func.AuthLevel.ANONYMOUS)
def healthz(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("healthz called")
    return func.HttpResponse("ok", status_code=200)


# ------------------------
# STORAGE
# ------------------------

def _container_client():
    from azure.storage.blob import ContainerClient

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = os.environ["AZURE_STORAGE_CONTAINER"]

    logger.info("Creating container client", extra={
                "container": container_name})

    return ContainerClient.from_connection_string(
        conn_str=conn_str,
        container_name=container_name
    )


def _blob_name_from_url(url: str) -> str:
    container = os.environ["AZURE_STORAGE_CONTAINER"].strip("/")
    marker = f"/{container}/"
    i = url.find(marker)

    if i == -1:
        name = PurePath(url).name
        logger.warning("Container marker not found in URL", extra={"url": url})
        return name

    name = url[i + len(marker):]
    logger.info("Parsed blob name", extra={"blob": name})
    return name


def _supported(blob_name: str) -> bool:
    ok = PurePath(blob_name).suffix.lower() in {".pdf", ".txt"}

    if not ok:
        logger.warning("Unsupported file type", extra={"blob": blob_name})

    return ok


def _download(container_client, blob_name: str, local_path: str) -> str:
    logger.info("Downloading blob", extra={"blob": blob_name})

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob = container_client.get_blob_client(blob=blob_name)

    with open(local_path, "wb") as f:
        stream = blob.download_blob()
        stream.readinto(f)

    etag = str(blob.get_blob_properties().etag or "")
    logger.info("Download complete", extra={"blob": blob_name, "etag": etag})
    return etag


def _load_docs(local_path: str, blob_name: str):
    suffix = PurePath(blob_name).suffix.lower()

    if suffix == ".pdf":
        logger.info("Loading PDF", extra={"blob": blob_name})
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(local_path).load()
    else:
        logger.info("Loading text file", extra={"blob": blob_name})
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(local_path, encoding=None,
                          autodetect_encoding=True).load()

    logger.info("Documents loaded", extra={
                "count": len(docs), "blob": blob_name})

    for d in docs:
        d.page_content = normalize_extracted_text(d.page_content)
        md = d.metadata or {}
        md["blob_name"] = blob_name
        md["source_path"] = blob_name
        md["source"] = "azure_blob"
        md["file"] = PurePath(blob_name).name
        md["doc_id"] = blob_name
        d.metadata = md

    return docs


def _delete_ids(doc_id: str, *_):
    from rag.vector_store import get_vector_store
    store = get_vector_store()

    safe = doc_id.replace("'", "''")

    results = store.client.search(
        search_text="*",
        filter=f"doc_id eq '{safe}'",
        select=["id"],
        top=1000
    )

    ids = [r["id"] for r in results]

    if not ids:
        logger.warning("No documents found to delete",
                       extra={"doc_id": doc_id})
        return

    store.delete(ids=ids)

    logger.info("Deleted chunks", extra={"count": len(ids), "doc_id": doc_id})


# ------------------------
# UPSERT
# ------------------------

def _handle_upsert(blob_name: str) -> None:
    logger.info("UPSERT start", extra={"blob": blob_name})

    if not _supported(blob_name):
        return

    from ingest.chunking import production_chunk_documents
    from ingest.index_azure_search import index_documents
    from ingest.state_store import DocState, load_state, save_state

    container = _container_client()
    doc_id = blob_name

    prev = load_state(container, doc_id)
    logger.info("Loaded previous state", extra={
                "doc_id": doc_id, "state": str(prev)})

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(
            temp_dir, container.container_name, blob_name)

        etag = _download(container, blob_name, local_path)

        if prev and prev.etag == etag:
            logger.info("ETag unchanged — skipping", extra={"doc_id": doc_id})
            return

        docs = _load_docs(local_path, blob_name)

    chunks = production_chunk_documents(docs)
    logger.info("Chunking complete", extra={
                "chunks": len(chunks), "doc_id": doc_id})

    _delete_ids(doc_id)

    logger.info("Indexing chunks", extra={"doc_id": doc_id})
    index_documents(chunks)

    save_state(container, DocState(
        doc_id=doc_id,
        etag=etag,
        chunk_count=len(chunks)
    ))

    logger.info("UPSERT completed", extra={"doc_id": doc_id})


# ------------------------
# DELETE
# ------------------------

def _handle_delete(blob_name: str) -> None:
    logger.info("DELETE start", extra={"blob": blob_name})

    from ingest.state_store import delete_state, load_state

    container = _container_client()
    doc_id = blob_name

    prev = load_state(container, doc_id)

    if not prev:
        logger.warning("Delete requested but no state found",
                       extra={"doc_id": doc_id})
        return

    _delete_ids(doc_id, 0, prev.chunk_count)
    delete_state(container, doc_id)

    logger.info("DELETE completed", extra={"doc_id": doc_id})


# ------------------------
# TRIGGER
# ------------------------

@app.function_name(name="blob_ingest")
@app.event_grid_trigger(arg_name="event", data_type="string")
def blob_ingest(event: func.EventGridEvent):

    logger.info("Event received")

    try:
        et = (event.event_type or "").lower()
        data = event.get_json() or {}

        logger.info("Event payload", extra={"type": et, "data": data})

        url = data.get("url")

        if not url:
            logger.error("Event missing URL field")
            return

        blob_name = _blob_name_from_url(url)

        if "blobdeleted" in et:
            _handle_delete(blob_name)
        else:
            _handle_upsert(blob_name)

        logger.info("Event processed successfully")

    except Exception as e:
        logger.error("Event processing failed", extra={"error": str(e)})
        logger.error(traceback.format_exc())
