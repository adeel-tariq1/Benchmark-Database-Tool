import chromadb
import numpy as np
from chromadb.config import Settings

def upload_to_chroma(embeddings, text):
    client_chroma = chromadb.Client()
    collection_name = 'my-first-tool'

    # Create or get existing collection
    if collection_name in [c.name for c in client_chroma.list_collections()]:
        collection = client_chroma.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    else:
        collection = client_chroma.create_collection(collection_name)
        print(f"ChromaDB collection '{collection_name}' created.")

    # Add data
    collection.add(
        ids=[str(i) for i in range(len(embeddings))],
        embeddings=embeddings,
        documents=text
    )

    print(f"✅ Added {len(embeddings)} vectors to ChromaDB.")
    return collection


def search_chroma(query_vec, top_k=3):
    client_chroma = chromadb.Client()
    collection_name = "my-first-tool"
    collection = client_chroma.get_collection(collection_name)

    # --- Ensure query vector shape ---
    if isinstance(query_vec, np.ndarray):
        query_vec = query_vec.flatten().astype("float32").tolist()
    elif isinstance(query_vec, list) and isinstance(query_vec[0], (list, np.ndarray)):
        # Handle nested list [[...]]
        query_vec = np.array(query_vec[0], dtype="float32").tolist()
    else:
        query_vec = [float(x) for x in query_vec]

    # --- Query collection ---
    results = collection.query(
    query_embeddings=[query_vec],           #type: ignore
    n_results=top_k,
    include=["documents", "distances"]
)

    texts = results['documents'][0]       #type: ignore
    distances = results['distances'][0]     #type: ignore

    similarities = [1 / (1 + dist) for dist in distances]  # smaller dist = higher sim

    formatted_results = list(zip(texts, similarities))

    return(formatted_results)
