import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

def upload_to_qdrant(embeddings: np.ndarray, texts, batch_size=50):
    """Uploads embeddings and texts to Qdrant with automatic collection creation and batch logging."""

    # Load .env variables
    load_dotenv(find_dotenv())

    collection_name = "my-first-tool"
    api_key = os.getenv("QDRANT_API_KEY")

    if not api_key:
        raise EnvironmentError("❌ QDRANT_API_KEY not found in .env")

    # Initialize Qdrant client
    client = QdrantClient(
        url="https://f92aa350-d714-4b0c-9a4f-cb38ec3733e9.us-east-1-1.aws.cloud.qdrant.io",
        api_key=api_key,
        timeout=60,  # ⏱️ 60s timeout for safety
    )

    # Create collection if it doesn’t exist
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"🆕 Creating Qdrant collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embeddings.shape[1],
                distance=Distance.COSINE
            )
        )
    else:
        print(f"📁 Collection '{collection_name}' already exists.")

    embeddings = np.array(embeddings, dtype=np.float32)

    # Upload batches with clear ranges
    total_uploaded = 0
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        batch_embeddings = embeddings[start:end]
        batch_texts = texts[start:end]

        points = [
            PointStruct(
                id=start + idx,
                vector=vec.tolist(),
                payload={"text": txt}
            )
            for idx, (vec, txt) in enumerate(zip(batch_embeddings, batch_texts))
        ]

        client.upsert(collection_name=collection_name, points=points)
        total_uploaded += len(points)
        print(f"✅ Uploaded vectors {start}–{end - 1}")

    print(f"\n🎯 Successfully uploaded {total_uploaded} total vectors to Qdrant.\n")

    return client





def search_qdrant(query_vec, top_k=3, collection_name="my-first-tool"):
    """Search Qdrant using the modern query_points API."""

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url="https://f92aa350-d714-4b0c-9a4f-cb38ec3733e9.us-east-1-1.aws.cloud.qdrant.io",
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Ensure query_vec is a flat list
    if isinstance(query_vec, np.ndarray):
        query_vec = query_vec.flatten().tolist()
    else:
        query_vec = list(query_vec)

    # Perform search with new API
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vec,       # type: ignore
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    # Extract points
    points = response.points if hasattr(response, "points") else []

    if not points:
        return [("No matches found", 0.0)]

    # Extract text and score
    matches = [
        (point.payload.get("text", "No text found"), float(point.score))  #type: ignore
        for point in points
    ]

    return matches
    