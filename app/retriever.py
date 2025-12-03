# app/retriever.py
from pathlib import Path
import traceback

from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, TOP_K

# embeddings import fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFE
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFE
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings as HFE

from langchain_community.vectorstores import FAISS

# Prepare embedding object for loading indexes
embedding_model = HFE(model_name=EMBEDDING_MODEL)

def _faiss_base_folder(setting: str) -> Path:
    p = Path(setting)
    return p.parent if p.suffix else p

DEFAULT_FAISS_FOLDER = _faiss_base_folder(FAISS_INDEX_PATH)

def load_vectorstore(folder_path: str | Path):
    folder = Path(folder_path)
    if not folder.exists() or not any(folder.iterdir()):
        print(f"[retriever] No FAISS files in {folder}")
        return None
    try:
        vs = FAISS.load_local(str(folder), embedding_model, allow_dangerous_deserialization=True)
        try:
            print("[retriever] Loaded FAISS. ntotal:", getattr(vs.index, "ntotal", "unknown"))
        except Exception:
            pass
        return vs
    except Exception as e:
        print("[retriever] Failed to load FAISS from", folder, "error:", e)
        traceback.print_exc()
        return None

# Optionally pre-load default index if present
_default_vs = load_vectorstore(DEFAULT_FAISS_FOLDER) if DEFAULT_FAISS_FOLDER.exists() else None

def retrieve(query: str, top_k: int = TOP_K, index_folder: str | None = None):
    """
    Retrieve top_k results from the given index folder.
    If index_folder is None, use default loaded index (if any).
    Returns list of Document-like objects (or empty list).
    """
    vs = None
    if index_folder:
        vs = load_vectorstore(index_folder)
    else:
        vs = _default_vs

    if vs is None:
        print(f"[retrieve] No vectorstore available for index_folder={index_folder}")
        return []

    try:
        return vs.similarity_search(query, k=top_k)
    except Exception as e:
        print("[retrieve] similarity_search error:", e)
        traceback.print_exc()
        return []
