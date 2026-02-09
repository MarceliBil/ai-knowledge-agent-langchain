import os
from pathlib import PurePath
from langchain_community.document_loaders import AzureBlobStorageContainerLoader


def load_documents():
    loader = AzureBlobStorageContainerLoader(
        conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
        container=os.environ["AZURE_STORAGE_CONTAINER"]
    )

    docs = loader.load()

    for d in docs:
        md = d.metadata or {}
        original_source = md.get("source")
        if original_source:
            md["source_path"] = str(original_source).strip()

        file_name = md.get("blob_name")
        if not file_name and original_source:
            file_name = PurePath(
                str(original_source).strip().replace("\\", "/")).name

        md["source"] = "azure_blob"
        md["file"] = (str(file_name).strip() if file_name else "unknown")
        d.metadata = md

    return docs
