from pinecone import Pinecone, ServerlessSpec
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv

from pinecone import Pinecone, ServerlessSpec
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv

# pinecone_handler.py
import numpy as np
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def upload_to_pinecone(embeddings, texts, index_name="my-first-tool"):
    api_key = os.getenv("PINECONE_APIKEY")
    if not api_key:
        raise RuntimeError("PINECONE_APIKEY not found in .env")

    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.shape[1],
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"ℹ️ Pinecone index '{index_name}' already exists.")

    index = pc.Index(index_name)

    # Add vectors in batches for reliability
    batch_size = 50
    for i in range(0, len(embeddings), batch_size):
        batch_vectors = [
            {
                "id": str(i + j),
                "values": embeddings[i + j].tolist(),
                "metadata": {"text": texts[i + j]}
            }
            for j in range(min(batch_size, len(embeddings) - i))
        ]
        index.upsert(vectors=batch_vectors)  #type: ignore

    print(f"✅ Uploaded {len(texts)} vectors to Pinecone.")
    return index_name 



def search_pinecone(query_vec, index_name="my-first-tool", top_k=3):
    load_dotenv(find_dotenv())
    api_key = os.getenv("PINECONE_APIKEY")
    if not api_key:
        raise RuntimeError("PINECONE_APIKEY not found in .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Convert to proper format
    query_vec = np.array(query_vec).astype("float32").tolist()

    # Include metadata so we can get the text
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True) #type: ignore

    hits = results.get("matches", [])   #type: ignore
    if not hits:
        return "No matches found."

    return [(hit["metadata"]["text"], hit["score"]) for hit in hits]