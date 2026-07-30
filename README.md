# Hybrid RAG Pipeline

A hybrid search RAG (Retrieval-Augmented Generation) pipeline built from scratch over internal documents with BM25, dense search, RRF fusion, and citations.

## Pipeline

1. **Ingest** - chunks documents, embeds them, stores in ChromaDB
2. **Search** - hybrid search combining dense (semantic) and BM25 (keyword) with RRF fusion
3. **Generate** - sends top chunks to Claude and returns a cited answer

## Stack

- `sentence-transformers` - embeddings (all-MiniLM-L6-v2)
- `chromadb` - vector store
- `rank-bm25` - keyword search
- `anthropic` - LLM generation

## Setup

```bash
pip install sentence-transformers chromadb rank-bm25 anthropic python-dotenv
```

Add your API key to a `.env` file:

## Usage

Add `.txt` files to the `docs/` folder, then:

```bash
python ingest.py
python generate.py
```

## How it works

Dense search finds semantically similar chunks. BM25 finds exact keyword matches. RRF fusion combines both ranked lists. Claude generates a cited answer from the top results.