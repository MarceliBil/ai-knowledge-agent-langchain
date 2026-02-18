from __future__ import annotations
import logging
import json
import os
import tempfile
import time
import traceback
from pathlib import PurePath
import azure.functions as func

logging.basicConfig(level=logging.WARNING)

app = func.FunctionApp()


@app.function_name(name="healthz")
@app.route(route="healthz", auth_level=func.AuthLevel.ANONYMOUS)
def healthz(req: func.HttpRequest) -> func.HttpResponse:
    logging.warning("healthz called")
    return func.HttpResponse("ok", status_code=200)


# ------------------------
# STORAGE
# ------------------------

def _container_client():
    logging.warning("Creating container client")

    from azure.storage.blob import ContainerClient

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = os.environ["AZURE_STORAGE_CONTAINER"]

    logging.warning(f"Container name: {container_name}")

    return ContainerClient.from_connection_string(
        conn_str=conn_str,
        container_name=container_name
    )


def _blob_name_from_url(url: str) -> str:
    logging.warning(f"Parsing blob name from URL: {url}")

    container = os.environ["AZURE_STORAGE_CONTAINER"].strip("/")
    marker = f"/{container}/"
    i = url.find(marker)

    if i == -1:
        name = PurePath(url).name
        logging.warning(f"Marker not found → fallback name: {name}")
        return name

    name = url[i + len(marker):]
    logging.warning(f"Parsed blob name: {name}")
    return name


def _supported(blob_name: str) -> bool:
    ok = PurePath(blob_name).suffix.lower() in {".pdf", ".txt"}
    logging.warning(f"Supported file? {blob_name} → {ok}")
    return ok


def _download(container_client, blob_name: str, local_path: str) -> str:
    logging.warning(f"Downloading blob: {blob_name}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    blob = container_client.get_blob_client(blob=blob_name)

    with open(local_path, "wb") as f:
        stream = blob.download_blob()
        stream.readinto(f)

    etag = str(blob.get_blob_properties().etag or "")
    logging.warning(f"Downloaded. ETag: {etag}")
    return etag


def _load_docs(local_path: str, blob_name: str):
    logging.warning(f"Loading document: {blob_name}")

    suffix = PurePath(blob_name).suffix.lower()

    if suffix == ".pdf":
        logging.warning("Using PDF loader")
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(local_path).load()
    else:
        logging.warning("Using Text loader")
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(local_path, encoding=None,
                          autodetect_encoding=True).load()

    logging.warning(f"Loaded docs count: {len(docs)}")

    for d in docs:
        md = d.metadata or {}
        md["blob_name"] = blob_name
        md["source_path"] = blob_name
        md["source"] = "azure_blob"
        md["file"] = PurePath(blob_name).name
        md["doc_id"] = blob_name
        d.metadata = md

    return docs


def _delete_ids(doc_id: str, *_):
    logging.warning(f"Deleting document from index: {doc_id}")

    from rag.vector_store import get_vector_store
    store = get_vector_store()

    logging.warning(f"DELETE FILTER VALUE: [{doc_id}]")

    try:
        store.delete(filter=f"doc_id eq '{doc_id}'")
        logging.warning("Delete completed")

    except Exception as e:
        logging.error("Delete failed")
        logging.error(str(e))


# ------------------------
# UPSERT
# ------------------------

def _handle_upsert(blob_name: str) -> None:
    logging.warning(f"UPSERT start → {blob_name}")

    if not _supported(blob_name):
        logging.warning("File type not supported — skipping")
        return

    from ingest.chunking import production_chunk_documents
    from ingest.index_azure_search import index_documents
    from ingest.state_store import DocState, load_state, save_state

    container = _container_client()
    doc_id = blob_name

    prev = load_state(container, doc_id)
    logging.warning(f"Previous state: {prev}")

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(
            temp_dir, container.container_name, blob_name)

        etag = _download(container, blob_name, local_path)

        if prev and prev.etag == etag:
            logging.warning("ETag unchanged → skipping")
            return

        docs = _load_docs(local_path, blob_name)

    logging.warning("Chunking docs")
    chunks = production_chunk_documents(docs)
    logging.warning(f"Chunks count: {len(chunks)}")

    logging.warning(f"Replacing document in index → {doc_id}")

    _delete_ids(doc_id)

    logging.warning("Indexing new chunks")
    index_documents(chunks)

    save_state(container, DocState(
        doc_id=doc_id,
        etag=etag,
        chunk_count=len(chunks)
    ))

    logging.warning("UPSERT completed")


# ------------------------
# DELETE
# ------------------------

def _handle_delete(blob_name: str) -> None:
    logging.warning(f"DELETE start → {blob_name}")

    from ingest.state_store import delete_state, load_state

    container = _container_client()
    doc_id = blob_name

    prev = load_state(container, doc_id)

    if not prev:
        logging.warning("No previous state — nothing to delete")
        return

    _delete_ids(doc_id, 0, prev.chunk_count)
    delete_state(container, doc_id)

    logging.warning("DELETE completed")


# ------------------------
# TRIGGER
# ------------------------

@app.function_name(name="blob_ingest")
@app.event_grid_trigger(arg_name="event", data_type="string")
def blob_ingest(event: func.EventGridEvent):

    logging.warning("=== EVENT RECEIVED ===")

    try:
        et = (event.event_type or "").lower()
        logging.warning(f"Event type: {et}")

        data = event.get_json() or {}
        logging.warning(f"Payload: {data}")

        url = data.get("url")

        if not url:
            logging.error("Event missing URL field")
            return

        blob_name = _blob_name_from_url(url)

        if "blobdeleted" in et:
            logging.warning("Trigger → DELETE")
            _handle_delete(blob_name)
        else:
            logging.warning("Trigger → UPSERT")
            _handle_upsert(blob_name)

        logging.warning("=== EVENT DONE ===")

    except Exception as e:
        logging.error("=== EVENT FAILED ===")
        logging.error(str(e))
        logging.error(traceback.format_exc())
