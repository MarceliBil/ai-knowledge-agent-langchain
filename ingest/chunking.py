from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
import hashlib


def production_chunk_documents(docs):
    structural_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=[
            "\n## ",
            "\n# ",
            "\n### ",
            "\n- ",
            "\n• ",
            "\n1. ",
            "\nStep ",
            "\n\n",
            "\n",
            " "
        ]
    )

    stage1_chunks = structural_splitter.split_documents(docs)

    token_splitter = TokenTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    final_chunks = token_splitter.split_documents(stage1_chunks)

    total = len(final_chunks)

    for i, chunk in enumerate(final_chunks):
        content_hash = hashlib.sha256(
            chunk.page_content.encode()
        ).hexdigest()

        chunk.metadata.update({
            "chunk_id": i,
            "chunk_hash": content_hash,
            "chunk_position": i,
            "total_chunks": total,
            "source": chunk.metadata.get("source", "unknown"),
            "file": chunk.metadata.get("file", "unknown")
        })

    return final_chunks
