# ingest.py
import os
from pathlib import Path
from typing import Tuple, List

from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import UPLOAD_DIR, FAISS_INDEX_PATH, EMBEDDING_MODEL

# Use the parent directory of the specific index path as the root for all indices
INDEX_ROOT = os.path.dirname(FAISS_INDEX_PATH)

EMBED_MODEL = EMBEDDING_MODEL

def _split_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)
    return [Document(page_content=t) for t in texts]

def ingest_pdf(pdf_path: str) -> Tuple[int, str]:
    """
    Index a PDF and save FAISS index.
    Returns (num_chunks, index_folder_or_error).
    """
    try:
        reader = PdfReader(pdf_path)
        full_text = []
        for page in reader.pages:
            try:
                full_text.append(page.extract_text() or "")
            except Exception:
                full_text.append("")
        text = "\n".join(full_text)
        docs = _split_text(text)
        if not docs:
            return 0, "No text extracted from PDF."

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        store = FAISS.from_documents(docs, embeddings)
        base = Path(pdf_path).stem
        index_folder = os.path.join(INDEX_ROOT, f"{base}_")
        Path(index_folder).mkdir(parents=True, exist_ok=True)
        store.save_local(index_folder)
        return len(docs), index_folder
    except Exception as e:
        return 0, str(e)

from urllib.parse import urljoin, urlparse

def ingest_url(url: str) -> Tuple[int, str]:
    """
    Fetch a URL and its sublinks (depth 1), extract text, index and save.
    Returns (num_chunks, index_folder_or_error).
    """
    try:
        # 1. Crawl main page and sublinks
        visited = set()
        to_visit = [url]
        base_domain = urlparse(url).netloc
        
        documents = []
        max_pages = 20
        
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue
            
            try:
                print(f"Crawling: {current_url}")
                resp = requests.get(current_url, timeout=10)
                if resp.status_code != 200:
                    continue
                
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extract text
                paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
                text = "\n\n".join(paragraphs)
                
                if text.strip():
                    # Create document with metadata
                    doc = Document(page_content=text, metadata={"source": current_url})
                    documents.append(doc)
                
                visited.add(current_url)
                
                # Find sublinks (only if we haven't reached max pages)
                if len(visited) < max_pages:
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        full_url = urljoin(current_url, href)
                        parsed = urlparse(full_url)
                        
                        # Only internal links
                        if parsed.netloc == base_domain and full_url not in visited and full_url not in to_visit:
                            to_visit.append(full_url)
                            
            except Exception as e:
                print(f"Failed to process {current_url}: {e}")
                continue

        if not documents:
            return 0, "No text found at URL or sublinks."

        # 2. Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(documents) # Preserves metadata

        # 3. Index
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        store = FAISS.from_documents(final_docs, embeddings)
        
        safe_name = url.replace("://", "_").replace("/", "_")[:50]
        index_folder = os.path.join(INDEX_ROOT, f"{safe_name}_")
        Path(index_folder).mkdir(parents=True, exist_ok=True)
        store.save_local(index_folder)
        
        return len(final_docs), index_folder
    except Exception as e:
        return 0, str(e)
