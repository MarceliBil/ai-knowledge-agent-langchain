import os
import json
from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
import hashlib


load_dotenv()

creds_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

creds = service_account.Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

loader = GoogleDriveLoader(
    folder_id=folder_id,
    credentials=creds,
    recursive=True
)

docs = loader.load()


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

    return final_chunks


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


chunks = production_chunk_documents(docs)
enrich_metadata(chunks)
add_chunk_hash(chunks)
