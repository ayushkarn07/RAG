# RAG Starter App

A simple RAG (Retrieval-Augmented Generation) application using Streamlit, LangChain, and FAISS.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Or if you are using `uv`:
    ```bash
    uv sync
    ```

2.  **Environment Variables**:
    Copy `.env.example` (if it exists) or create a `.env` file with your API keys:
    ```ini
    GROQ_API_KEY=your_groq_api_key
    USE_GROQ=true
    ```

## Running the App

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Features

-   **PDF Ingestion**: Upload and index PDF documents.
-   **Website Ingestion**: Index content from URLs.
-   **Q&A**: Ask questions about the indexed content.
-   **LLM Support**: Supports Groq and OpenAI (configurable).
