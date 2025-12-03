import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Paths
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
METADATA_STORE = os.getenv("METADATA_STORE", "./data/faiss_index/metadata.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "6"))

# Toggles
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_LLM_MODEL_PATH = os.getenv("LOCAL_LLM_MODEL_PATH", "./models/ggml-model.bin")
