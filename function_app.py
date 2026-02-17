# from __future__ import annotations
# import json
# import os
# import tempfile
# from pathlib import PurePath
# import azure.functions as func
# from dotenv import load_dotenv
# print("FUNCTION APP FILE EXECUTED")


# load_dotenv()

# app = func.FunctionApp()


# @app.function_name(name="healthz")
# @app.route(route="healthz", auth_level=func.AuthLevel.ANONYMOUS)
# def healthz(req: func.HttpRequest) -> func.HttpResponse:
#     return func.HttpResponse("ok", status_code=200)


# def _container_client():
#     from azure.storage.blob import ContainerClient

#     conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
#     container_name = os.environ["AZURE_STORAGE_CONTAINER"]
#     return ContainerClient.from_connection_string(conn_str=conn_str, container_name=container_name)


# def _blob_name_from_url(url: str) -> str:
#     container = os.environ["AZURE_STORAGE_CONTAINER"].strip("/")
#     marker = f"/{container}/"
#     i = url.find(marker)
#     if i == -1:
#         return PurePath(url).name
#     return url[i + len(marker):]


# def _supported(blob_name: str) -> bool:
#     return PurePath(blob_name).suffix.lower() in {".pdf", ".txt"}


# def _download(container_client, blob_name: str, local_path: str) -> str:
#     os.makedirs(os.path.dirname(local_path), exist_ok=True)
#     blob = container_client.get_blob_client(blob=blob_name)
#     with open(local_path, "wb") as f:
#         stream = blob.download_blob()
#         stream.readinto(f)
#     return str(blob.get_blob_properties().etag or "")


# def _load_docs(local_path: str, blob_name: str):
#     suffix = PurePath(blob_name).suffix.lower()
#     if suffix == ".pdf":
#         from langchain_community.document_loaders import PyPDFLoader

#         docs = PyPDFLoader(local_path).load()
#     else:
#         from langchain_community.document_loaders import TextLoader

#         docs = TextLoader(local_path, encoding=None,
#                           autodetect_encoding=True).load()

#     for d in docs:
#         md = d.metadata or {}
#         md["blob_name"] = blob_name
#         md["source_path"] = blob_name
#         md["source"] = "azure_blob"
#         md["file"] = PurePath(blob_name).name
#         d.metadata = md

#     return docs


# def _delete_ids(doc_id: str, start: int, end: int) -> None:
#     if end <= start:
#         return
#     from rag.vector_store import get_vector_store
#     from ingest.index_azure_search import chunk_id_from_doc_id

#     store = get_vector_store()
#     ids = [chunk_id_from_doc_id(doc_id, i) for i in range(start, end)]
#     store.delete(ids=ids)


# def _handle_upsert(blob_name: str) -> None:
#     if not _supported(blob_name):
#         return

#     from ingest.chunking import production_chunk_documents
#     from ingest.index_azure_search import index_documents
#     from ingest.state_store import DocState, load_state, save_state

#     container = _container_client()
#     doc_id = blob_name
#     prev = load_state(container, doc_id)

#     with tempfile.TemporaryDirectory() as temp_dir:
#         local_path = os.path.join(
#             temp_dir, container.container_name, blob_name)
#         etag = _download(container, blob_name, local_path)
#         if prev and prev.etag == etag:
#             return
#         docs = _load_docs(local_path, blob_name)

#     chunks = production_chunk_documents(docs)
#     index_documents(chunks)

#     new_count = len(chunks)
#     old_count = prev.chunk_count if prev else 0
#     _delete_ids(doc_id, new_count, old_count)
#     save_state(container, DocState(doc_id=doc_id,
#                etag=etag, chunk_count=new_count))


# def _handle_delete(blob_name: str) -> None:
#     from ingest.state_store import delete_state, load_state

#     container = _container_client()
#     doc_id = blob_name
#     prev = load_state(container, doc_id)
#     if not prev:
#         return
#     _delete_ids(doc_id, 0, prev.chunk_count)
#     delete_state(container, doc_id)


# @app.function_name(name="blob_ingest")
# @app.event_grid_trigger(arg_name="event")
# def blob_ingest(event: func.EventGridEvent):
#     et = (event.event_type or "").lower()
#     data = event.get_json() or {}
#     url = str(data.get("url") or "")
#     blob_name = _blob_name_from_url(url) if url else ""
#     if not blob_name:
#         return
#     if "blobdeleted" in et:
#         _handle_delete(blob_name)
#     else:
#         _handle_upsert(blob_name)


# print("FUNCTION APP LOADED SUCCESSFULLY")

import azure.functions as func

app = func.FunctionApp()


@app.route(route="ping")
def ping(req: func.HttpRequest):
    return func.HttpResponse("pong")
