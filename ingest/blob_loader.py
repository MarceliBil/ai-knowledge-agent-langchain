import os
import tempfile
from pathlib import PurePath

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from ingest.text_cleaning import normalize_extracted_text


def load_documents():
    try:
        from azure.storage.blob import ContainerClient
    except ImportError as exc:
        raise ImportError(
            "Could not import azure storage blob python package. "
            "Please install it with `pip install azure-storage-blob`."
        ) from exc

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = os.environ["AZURE_STORAGE_CONTAINER"]

    container = ContainerClient.from_connection_string(
        conn_str=conn_str, container_name=container_name
    )

    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for blob in container.list_blobs():
            blob_name = blob.name
            suffix = PurePath(blob_name).suffix.lower()

            if suffix not in {".pdf", ".txt"}:
                continue

            local_path = os.path.join(temp_dir, container_name, blob_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            blob_client = container.get_blob_client(blob=blob_name)
            with open(local_path, "wb") as f:
                stream = blob_client.download_blob()
                stream.readinto(f)

            if suffix == ".pdf":
                file_docs = PyPDFLoader(local_path).load()
            else:
                file_docs = TextLoader(
                    local_path,
                    encoding=None,
                    autodetect_encoding=True,
                ).load()

            for d in file_docs:
                d.page_content = normalize_extracted_text(d.page_content)
                md = d.metadata or {}
                md["blob_name"] = blob_name
                md["source_path"] = blob_name
                md["source"] = "azure_blob"
                md["file"] = PurePath(blob_name).name
                md["doc_id"] = blob_name
                d.metadata = md
            docs.extend(file_docs)

    return docs
