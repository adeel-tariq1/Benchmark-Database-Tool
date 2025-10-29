import numpy as np
import faiss

_faiss_index = None
_stored_texts = None

def upload_to_faiss(embeddings, text_list):
    global _faiss_index, _stored_texts

    # Delete old index if exists
    if '_faiss_index' in globals() and _faiss_index is not None:
        del _faiss_index

    embeddings = np.array(embeddings).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)          # type: ignore

    _faiss_index = index
    _stored_texts = text_list.copy()

    print(f"✅ Added {len(embeddings)} vectors to FAISS index (cosine similarity mode)")


def search_faiss(query_vec, top_k=1):
    """Search FAISS index for most similar vectors."""
    if _faiss_index is None:
        raise ValueError("FAISS index not initialized. Run upload_to_faiss() first.")

    query_vec = np.array(query_vec).astype('float32')
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    # Normalize query vector (to match cosine similarity)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    scores, indices = _faiss_index.search(query_vec, top_k)   #type: ignore
    results = [(_stored_texts[i], float(scores[0][idx])) for idx, i in enumerate(indices[0])]  #type: ignore
    return results
 