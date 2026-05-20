import os

# Set dummy environment variables so imports of services don't fail during test collection
os.environ["LLAMA_CLOUD_API_KEY"] = "dummy_llama_cloud_key"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["GEMINI_FAST_MODEL"] = "gemini-1.5-flash"
os.environ["GEMINI_EMBEDDING_MODEL"] = "text-embedding-004"
