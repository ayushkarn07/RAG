# app/debug_query.py
from pathlib import Path
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, TOP_K
import sys

print("FAISS_INDEX_PATH:", FAISS_INDEX_PATH)
faiss_folder = Path(FAISS_INDEX_PATH)
if faiss_folder.suffix:
    faiss_folder = faiss_folder.parent
print("Using folder:", faiss_folder)
print("Files in folder:", [p.name for p in faiss_folder.glob("*")])

# load embeddings and FAISS
try:
    # import embeddings (robust)
    try:
        from langchain.embeddings import HuggingFaceEmbeddings as HFE
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFE
    emb = HFE(model_name=EMBEDDING_MODEL)
    print("Embedding model loaded:", EMBEDDING_MODEL)
except Exception as e:
    print("Failed to load embeddings:", e)
    sys.exit(1)

try:
    from langchain_community.vectorstores import FAISS
    if not any(faiss_folder.iterdir()):
        print("FAISS folder is empty!")
        sys.exit(0)

    vs = FAISS.load_local(str(faiss_folder), emb, allow_dangerous_deserialization=True)
    # try to print ntotal
    try:
        print("Index ntotal:", vs.index.ntotal)
    except Exception:
        print("Could not read vs.index.ntotal (depends on wrapper).")

    # prompt for test query
    q = input("Enter a test query (e.g. \"placement details\" or \"college name\"): ").strip()
    if not q:
        print("No query entered, exiting.")
        sys.exit(0)

    print(f"Running similarity_search for top {TOP_K} results...")
    try:
        results = vs.similarity_search(q, k=TOP_K)
    except Exception as e:
        print("similarity_search failed:", e)
        sys.exit(1)

    print("Found", len(results), "results. Showing snippets & metadata:")
    for i, r in enumerate(results, 1):
        txt = getattr(r, "page_content", None) or getattr(r, "content", None) or r.get("text", None) if isinstance(r, dict) else None
        print("----- RESULT", i, "-----")
        if txt:
            print(txt[:800].replace("\n", " "))
        else:
            # print known metadata fields
            try:
                print("raw object:", r)
            except:
                print("Could not print result content.")
        print("metadata:", getattr(r, "metadata", None))
        print()
except Exception as e:
    print("Error loading FAISS index:", e)
    sys.exit(1)
