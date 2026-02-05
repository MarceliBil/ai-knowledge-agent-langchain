import os
from langchain_community.document_loaders import AzureBlobStorageContainerLoader


def load_documents():
    loader = AzureBlobStorageContainerLoader(
        conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
        container=os.environ["AZURE_STORAGE_CONTAINER"]
    )

    docs = loader.load()

    for d in docs:
        d.metadata["source"] = "azure_blob"
        d.metadata["file"] = d.metadata.get("blob_name", "unknown")

    return docs
