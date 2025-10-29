📊 Vector Database Benchmark Tool

A Streamlit-based application to benchmark and compare popular vector databases for document search and retrieval. Upload a PDF, generate embeddings with SentenceTransformers, and evaluate performance across Qdrant, Pinecone, Chroma, and FAISS in terms of speed, accuracy, cost, and scalability.

This tool is ideal for developers and data scientists who want to quickly assess which vector database fits their use case, whether for local prototyping or production-grade deployments.

Features

PDF Upload & Chunking

Upload any PDF document via a simple GUI.

Automatic splitting into smaller chunks for accurate embedding and search.

Adjustable chunk size and overlap for balancing speed vs accuracy.

Embeddings Generation

Uses sentence-transformers/all-MiniLM-L6-v2 for fast semantic embeddings.

Supports batch processing to improve speed for large documents.

Caches embeddings to avoid recomputation for repeated uploads.

Vector Database Upload

Supports multiple vector databases:

Qdrant – Open-source, distributed vector database.

Pinecone – Cloud-native, fully managed vector database.

Chroma – Local vector database, good for prototyping.

FAISS – In-memory similarity search library from Facebook.

Parallel upload to all databases to maximize speed.

Benchmarking & Search

Test a query against all vector databases.

Compare search time, top similarity score, cost, and scalability.

Highlights fastest and most accurate database automatically.

Interactive results displayed in tables with clear summaries.

Installation
1. Clone the repository
git clone https://github.com/yourusername/vector-db-benchmark.git
cd vector-db-benchmark

2. Set up Python environment

It’s recommended to use a virtual environment:

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


Note: Make sure to install the new Pinecone SDK (pip install pinecone) instead of pinecone-client.

4. Set environment variables

Create a .env file in the project root with your API keys:

QDRANT_API_KEY=your_qdrant_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

Usage

Run the Streamlit app:

streamlit run streamapp.py

Step-by-step workflow

Upload PDF – Select the PDF file to benchmark.

Generate embeddings – The app splits the PDF into chunks and computes embeddings.

Upload to databases – Click the button to upload embeddings to Qdrant, Pinecone, Chroma, and FAISS in parallel.

Run benchmark – Enter a search query and run the benchmark.

View results – See a table comparing speed, accuracy, cost, and scalability.

Quick summary – Check which database is the fastest and most accurate.

Configuration

Chunk Size & Overlap: Adjust in the Streamlit code for speed vs accuracy trade-offs:

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


Batch Size for Uploads: Configurable in upload_to_qdrant and other DB handlers for faster processing.

Performance Tips

Use larger chunk sizes (800–1200) for faster embedding and upload, at the cost of slightly less granular search.

For local testing, use FAISS or Chroma only; cloud DBs like Pinecone/Qdrant may add network latency.

Reusing embeddings from cached files speeds up repeated tests.
