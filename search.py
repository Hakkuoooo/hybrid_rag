import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import os

collection_name = 'internal_docs'

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(collection_name)

all_chunks=collection.get()['documents']
all_ids = collection.get()['ids']

def search(query):
    #load the model 
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    #encode the query 
    bm25 = BM25Okapi([chunk.split() for chunk in all_chunks])
    query_embedding = model.encode(query).tolist()
    #get the top 5 results using BM25
    dense_results = collection.query(query_embeddings=[query_embedding], n_results=5)
    #get the top 5 results using BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    #get the rank of the desne rsults 
    dense_ids = dense_results['ids'][0]
    bm25_top5 = np.argsort(bm25_scores)[::-1][:5]
    rrf_scores = {}
    for rank, chunk_id in enumerate(dense_ids):
        rrf_scores[chunk_id] = 1/(rank + 60)
    for rank, idx in enumerate(bm25_top5):
        chunk_id = all_ids[idx]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1/(rank + 60)

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks[:5]
if __name__ == "__main__":
    query = "how many days annual leave do I get"
    results = search(query)
    for chunk_id, score in results:
        print(f"{chunk_id} -> {score:.4f}")
