import os
from sentence_transformers import SentenceTransformer
import chromadb

DOCS_DIR='docs'
CHUNK_SIZE=500
CHUNK_OVERLAP=50
COLLECTION_NAME='internal_docs'

def chunk_text(text,chunk_size,chunk_overlap):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start=end-chunk_overlap
    return chunks   

def ingest_docs():
    #£load the model 
    model = SentenceTransformer("all-MiniLM-L6-v2")
    #createa a chroma client 
    client = chromadb.PersistentClient(path="./chroma_db")

    #create a collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection=client.create_collection(COLLECTION_NAME)

    #iterate over the documents in the docs directory
    all_chunks=[]
    all_ids=[]
    all_metadatas=[]
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepaths=os.path.join(DOCS_DIR,filename)
            with open(filepaths,'r') as f:
                text=f.read()
                chunks1=chunk_text(text,CHUNK_SIZE,CHUNK_OVERLAP)
                for i,chunk in enumerate(chunks1):
                    all_chunks.append(chunk)
                    all_ids.append(f"{filename}_chunk_{i}")
                    all_metadatas.append({"source":filename,"chunk_index":i})

    model_embeddings=model.encode(all_chunks)
    collection.add(
    documents=all_chunks,
    embeddings=model_embeddings.tolist(),
    metadatas=all_metadatas,
    ids=all_ids
    )   


if __name__ == "__main__":
    ingest_docs()


