import streamlit as st
import time
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from chroma_handler import upload_to_chroma, search_chroma
from faiss_handler import upload_to_faiss, search_faiss
from pinecone_handler import upload_to_pinecone, search_pinecone
from qdrant_handler import upload_to_qdrant, search_qdrant
from tabulate import tabulate

st.set_page_config(page_title="Vector DB Benchmark", layout="wide")

st.title(" Vector Databases Benchmark Tool")
st.write("Upload a PDF document, create embeddings, and compare search performance across multiple vector databases.")

import tempfile

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    file = PyPDFLoader(tmp_path)
    docs = file.load()

    st.success(f" PDF loaded and split into {len(docs)} pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    text = [doc.page_content for doc in chunks]

    st.success(f"PDF loaded and split into {len(text)} chunks.")

    """Generating embeddings"""
    st.info("Generating embeddings using SentenceTransformer...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)

    with open("text.pkl", "wb") as f:
        pickle.dump(text, f)
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    st.success(" Embeddings generated successfully!")

    """Uploading to vector Databases"""
    if st.button("Upload to all vector databases"):
        with st.spinner("Uploading embeddings to Qdrant, Pinecone, Chroma, and FAISS..."):
            upload_to_qdrant(embeddings, text)
            upload_to_pinecone(embeddings, text)
            upload_to_chroma(embeddings, text)
            upload_to_faiss(embeddings, text)
        st.success(" Upload completed!")

    query = st.text_input("Enter your search query", "Write the moral of the story in Chapter 5")

    if st.button("Run Benchmark"):
        query_vec = model.encode([query], convert_to_numpy=True)
        results = []

        def benchmark_search(fn, db_name, cost_label, scalability_label, *args):
            start = time.time()
            output = fn(query_vec, *args)
            duration = round(time.time() - start, 3)
            try:
                if isinstance(output, list) and len(output) > 0 and isinstance(output[0], tuple):
                    top_score = round(output[0][1], 3)
                else:
                    top_score = None
            except Exception:
                top_score = None
            results.append([db_name, duration, top_score, cost_label, scalability_label])

        cost_map = {
            "Qdrant": " Free (self-hosted)",
            "Pinecone": " Paid (cloud)",
            "Chroma": " Free (local)",
            "FAISS": " Free (local)",
        }
        scalability_map = {
            "Qdrant": " High (distributed)",
            "Pinecone": " High (cloud-native)",
            "Chroma": " Medium (local only)",
            "FAISS": " Low (in-memory)",
        }

        # Run benchmark
        benchmark_search(search_qdrant, "Qdrant", cost_map["Qdrant"], scalability_map["Qdrant"])
        benchmark_search(search_pinecone, "Pinecone", cost_map["Pinecone"], scalability_map["Pinecone"])
        benchmark_search(search_chroma, "Chroma", cost_map["Chroma"], scalability_map["Chroma"])
        benchmark_search(search_faiss, "FAISS", cost_map["FAISS"], scalability_map["FAISS"])

        st.subheader("📊 Vector Database Benchmark Results")
        st.table(results)

        # Best performers
        fastest = min(results, key=lambda x: x[1])
        most_accurate = max([r for r in results if r[2] is not None], key=lambda x: x[2])

        st.markdown(f" **Fastest DB:** {fastest[0]} ({fastest[1]}s)")
        st.markdown(f"**Most Accurate DB:** {most_accurate[0]} (Score {most_accurate[2]})")

        st.subheader("📘 Quick Summary of Each Vector Database")
        st.markdown("🔹 **Qdrant** → Open-source, production-grade vector DB with REST API and distributed scaling.")
        st.markdown("🔹 **Pinecone** → Fully managed cloud vector DB, fast, but requires paid plans for large data.")
        st.markdown("🔹 **Chroma** → Lightweight, local-only, easy to use for small/prototype projects.")
        st.markdown("🔹 **FAISS** → In-memory similarity search, very fast locally, not scalable by itself.")
