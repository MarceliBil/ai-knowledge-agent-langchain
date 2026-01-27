import os
from dotenv import load_dotenv
from supabase import create_client
from rag.embeddings import embed_chunks

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_chunks(chunks):
    vectors = embed_chunks(chunks)

    rows = []
    for i, c in enumerate(chunks):
        rows.append({
            "content": c.page_content,
            "chunk_index": i,
            "hash": c.metadata["hash"],
            "embedding": vectors[i],
        })

    supabase.table("chunks").insert(rows).execute()
