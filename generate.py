from os import environ
import anthropic
from search import search, collection
api_key = environ.get("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=api_key)

def generate(query):
    #search for relevant chunks 
    results = search(query)
    #get actual text from the chunks
    retrieved_chunks = []
    for chunk_id, score in results:
        chunk = collection.get(ids=[chunk_id])
        retrieved_chunks.append({
            "text": chunk['documents'][0],
            "source": chunk['metadatas'][0]["source"],
        })

    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[{i+1}] source: {chunk['source']}\n{chunk['text']}\n\n"    
    
    prompt = f"Answer the question using only the context below. Cite sources using [1], [2] etc.\n\nContext:\n{context}\nQuestion: {query}"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

