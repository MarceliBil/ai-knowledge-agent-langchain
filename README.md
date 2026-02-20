# 🤖 RAG Knowledge Agent (Azure + LangChain + Streamlit)

Live demo: https://knowledge-agent-mb.streamlit.app

A small RAG (Retrieval-Augmented Generation) app that answers questions **only from your organization’s documents** stored in **Azure Blob Storage** and indexed in **Azure AI Search**, using **LangChain**.

## ⚙️ What it does

- Streamlit chat UI (password-protected)
- Retrieves relevant chunks from Azure AI Search (hybrid search)
- Uses OpenAI embeddings for vectorization
- Uses Anthropic Claude for answering
- Shows source filenames used for the answer
- Event-driven ingestion via Azure Functions (Event Grid Blob events)
- Supported document types: `.pdf`, `.txt`

## 🧰 Tech stack

- Python
- Streamlit
- LangChain
- Azure Blob Storage
- Azure AI Search
- Anthropic (Claude) for LLM
- OpenAI for embeddings

## 🗂️ Project layout (high level)

- `streamlit_app.py` – Streamlit UI entrypoint
- `rag/` – RAG chain, retriever, Azure AI Search vector store
- `ingest/` – loaders (Blob), text cleaning, chunking, indexing
- `function_app.py` – Azure Function triggered by Event Grid to ingest/update/delete on blob changes
- `config/` – settings + embeddings

## 💬 Example questions

A sample document about business travel policies is available in the `sample_documents` folder.  
You can use it to test the agent with questions such as:

1. Jaki jest budżet na podróż służbową?
2. Jakie koszty podróży służbowej nie podlegają zwrotowi?
3. Jakie mamy przepisy dotyczące podróży zagranicznych?

The agent will answer only using information retrieved from indexed documents.

## 🌍 Language support

The current version supports Polish only (for both documents and queries).
