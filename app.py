import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import fitz  # PyMuPDF for PDFs
import easyocr
import docx
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import re, hashlib, time, os, io  # Import io
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Download NLTK data (if needed)
nltk.download('punkt', quiet=True)

# --- Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 10
RERANK_TOP_N = 5
EMBEDDING_MODEL = "embedding-001"
LLM_MODEL = "gemini-2.0-flash"
ENABLE_HYBRID_SEARCH = True
ENABLE_RERANKING = True
EMBEDDING_DIMENSION = 768  # Define embedding dimension as a constant

# --- Session State Initialization ---
if 'config' not in st.session_state:
    st.session_state.config = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k_retrieval": TOP_K_RETRIEVAL,
        "rerank_top_n": RERANK_TOP_N,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "enable_hybrid_search": ENABLE_HYBRID_SEARCH,
        "enable_reranking": ENABLE_RERANKING
    }
if 'system_logs' not in st.session_state:
    st.session_state.system_logs = []
if 'data' not in st.session_state:
    st.session_state.data = {}  # Maps chunk ID to chunk metadata and text
if 'corpus' not in st.session_state:
    st.session_state.corpus = []  # Tokenized text for BM25
if 'raw_docs' not in st.session_state:
    st.session_state.raw_docs = []  # Original chunk texts
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = {}  # Document-level info (filename, chunk IDs, etc.)
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'doc_id_counter' not in st.session_state:
    st.session_state.doc_id_counter = 0
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
if 'bm25_model' not in st.session_state:
    st.session_state.bm25_model = None

# --- Set up basic page config and custom CSS ---
st.set_page_config(page_title="Advanced Hybrid RAG System", layout="wide")
st.markdown(
    """
<style>
    .main-header { font-size: 2.5rem; margin-bottom: 1rem; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4CAF50, #8BC34A); }
</style>
""",
    unsafe_allow_html=True,
)

# --- Utility Logging ---
def log_event(event_type: str, details: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.system_logs.append(
        {"timestamp": timestamp, "event_type": event_type, "details": details}
    )

# --- Initialize Gemini & Cross-Encoder ---
def initialize_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY environment variable not found. Please set it before running the application."
        )
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(st.session_state.config["llm_model"])
        log_event(
            "Initialization", f"Gemini model initialized: {st.session_state.config['llm_model']}"
        )
        return model
    except Exception as e:
        log_event("Error", f"Gemini initialization failed: {str(e)}")
        st.error(
            f"Failed to initialize Gemini. Please check your API key and model name. Error: {e}"
        )
        return None

@st.cache_resource
def load_cross_encoder():
    try:
        ce = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
        log_event("Initialization", "Cross-encoder initialized: ms-marco-TinyBERT-L-2-v2")
        return ce
    except Exception as e:
        log_event("Error", f"Cross-encoder initialization failed: {str(e)}")
        st.error(
            f"Failed to initialize cross-encoder. Please check the model name or your internet connection. Error: {e}"
        )
        return None

model = initialize_gemini()
cross_encoder = load_cross_encoder()

# --- Text Preprocessing & Chunking ---
def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return word_tokenize(text)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    if len(text) <= chunk_size:
        return [text]
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                sentence_break = text.rfind(". ", start, end)
                if sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2
        chunks.append(text[start:end])
        start = end - chunk_overlap
        if (
            start + chunk_size > len(text)
            and len(chunks) > 1
            and len(chunks[-1]) < chunk_size // 2
        ):
            chunks[-2] += " " + chunks[-1]
            chunks.pop()
            break
    return chunks

# --- File Extraction Helpers ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        if not text.strip():
            try:
                reader = easyocr.Reader(["en"])
                text = ""
                for page in pdf_doc:
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes()
                    results = reader.readtext(img_bytes)
                    text += " ".join([result[1] for result in results]) + " "
                log_event("PDF Processing", f"Extracted {len(text)} characters using OCR")
            except Exception as e:
                log_event("Error", f"OCR failed for PDF: {e}")
                st.warning(
                    f"Text extraction from PDF might be incomplete due to OCR failure: {e}"
                )
        else:
            log_event("PDF Processing", f"Extracted {len(text)} characters")
        return text
    except Exception as e:
        log_event("Error", f"Error processing PDF: {e}")
        st.error(f"Could not process PDF file. Error: {e}")
        return ""

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("utf-8")
        log_event("TXT Processing", f"Extracted {len(text)} characters")
        return text
    except Exception as e:
        log_event("Error", f"Error processing TXT file: {e}")
        st.error(f"Could not process TXT file. Error: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        log_event("DOCX Processing", f"Extracted {len(text)} characters")
        return text
    except Exception as e:
        log_event("Error", f"Error processing DOCX file: {e}")
        st.error(f"Could not process DOCX file. Error: {e}")
        return ""

# --- Embedding Function ---
def get_gemini_embedding(text: str) -> np.ndarray | None:
    if model is None:
        st.error("Gemini model is not initialized. Please check the API key.")
        return None
    try:
        embedding = genai.embed_content(
            model=f"models/{st.session_state.config['embedding_model']}",
            content=text,
            task_type="retrieval_document",
        )
        return np.array(embedding["embedding"])
    except Exception as e:
        st.error(f"Error generating embedding for text: '{text[:50]}...'. Error: {e}")
        log_event("Error", f"Embedding generation failed: {str(e)}")
        return None

# --- Document Processing ---
def process_document(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type
    filename = uploaded_file.name

    with st.spinner(f"Processing {filename}..."):
        if file_type == "text/plain":
            content = extract_text_from_txt(file_bytes)
        elif file_type == "application/pdf":
            content = extract_text_from_pdf(file_bytes)
        elif (
            file_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            content = extract_text_from_docx(file_bytes)
        else:
            st.error(f"Unsupported file type: {file_type} for file: {filename}")
            return

        if not content.strip():
            st.warning(f"No text extracted from {filename}.")
            return

        chunks = chunk_text(
            content, st.session_state.config["chunk_size"], st.session_state.config["chunk_overlap"]
        )
        chunk_ids = []
        word_count = 0

        for i, chunk in enumerate(chunks):
            chunk_id = str(st.session_state.doc_id_counter)
            st.session_state.doc_id_counter += 1
            embedding = get_gemini_embedding(chunk)
            if embedding is None:
                continue
            word_count += len(chunk.split())
            st.session_state.data[chunk_id] = {
                "text": chunk,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "word_count": len(chunk.split()),
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": file_type,
                "doc_id": chunk_id,
            }
            chunk_ids.append(chunk_id)
            st.session_state.corpus.append(preprocess_text(chunk))
            st.session_state.raw_docs.append(chunk)
            st.session_state.faiss_index.add(embedding.reshape(1, -1))

        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]  # Hash the file content
        doc_key = f"{filename}_{file_hash}"
        st.session_state.document_chunks[doc_key] = {
            "filename": filename,
            "file_type": file_type,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_ids": chunk_ids,
            "word_count": word_count,
            "chunk_count": len(chunks),
        }
        log_event("Document Processing", f"Processed '{filename}' into {len(chunks)} chunks.")
        st.success(f"'{filename}' processed successfully into {len(chunks)} chunks.")
        st.session_state.bm25_model = rebuild_bm25()  # Rebuild BM25 here

# --- Document Deletion ---
def delete_document(doc_key: str):
    if doc_key not in st.session_state.document_chunks:
        return
    with st.spinner(f"Deleting '{st.session_state.document_chunks[doc_key]['filename']}'..."):
        chunk_ids_to_delete = st.session_state.document_chunks[doc_key]["chunk_ids"]
        indices_to_remove = []
        for i, doc in enumerate(st.session_state.raw_docs):
            for cid in chunk_ids_to_delete:
                if cid in st.session_state.data and st.session_state.data[cid]["text"] == doc:
                    indices_to_remove.append(i)
                    del st.session_state.data[cid]
                    break

        # Remove from raw_docs and corpus in reverse order to avoid index issues
        for index in sorted(list(set(indices_to_remove)), reverse=True):
            st.session_state.raw_docs.pop(index)
            st.session_state.corpus.pop(index)

        del st.session_state.document_chunks[doc_key]

        # Rebuild FAISS index from remaining data
        st.session_state.faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        embeddings = [
            get_gemini_embedding(doc["text"])
            for doc in st.session_state.data.values()
            if get_gemini_embedding(doc["text"]) is not None
        ]
        if embeddings:
            st.session_state.faiss_index.add(np.array(embeddings))

        log_event("Document Deletion", f"Deleted document with key {doc_key}")
        st.success(
            f"'{st.session_state.document_chunks.get(doc_key, {'filename': 'unknown'})['filename']}' deleted successfully."
        )
        st.session_state.bm25_model = rebuild_bm25()  # Rebuild BM25 here

# --- BM25 Rebuild ---
def rebuild_bm25():
    if st.session_state.corpus:
        return BM25Okapi(st.session_state.corpus)
    return None

# --- Hybrid Search ---
def hybrid_search_rag(query: str) -> str:
    start_time = time.time()
    results: Dict[str, Any] = {
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bm25_results": [],
        "vector_results": [],
        "reranked_results": [],
        "processing_time": 0,
        "answer": "",
    }

    if not st.session_state.data:
        results["answer"] = "No documents uploaded yet. Please upload some documents to begin."
        st.session_state.search_history.append(results)
        return results["answer"]

    bm25 = st.session_state.bm25_model  # Use session state instead
    bm25_docs: List[str] = []
    bm25_scores_dict: Dict[str, float] = {}
    if bm25 and st.session_state.config["enable_hybrid_search"]:
        tokenized_query = preprocess_text(query)
        scores = bm25.get_scores(tokenized_query)
        top_n = min(st.session_state.config["top_k_retrieval"], len(st.session_state.raw_docs))
        top_indices = np.argsort(scores)[-top_n:][::-1]
        for i in top_indices:
            if scores[i] > 0:
                doc_text = st.session_state.raw_docs[i]
                bm25_docs.append(doc_text)
                bm25_scores_dict[doc_text] = float(scores[i])
                doc_id = None
                for cid, info in st.session_state.data.items():
                    if info["text"] == doc_text:
                        doc_id = cid
                        break
                results["bm25_results"].append(
                    {
                        "doc_id": doc_id,
                        "score": float(scores[i]),
                        "text_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
                        "filename": st.session_state.data.get(doc_id, {}).get("filename", "Unknown"),
                    }
                )

    vector_docs: List[str] = []
    vector_scores_dict: Dict[str, float] = {}
    query_embedding = get_gemini_embedding(query)
    if query_embedding is not None:
        query_embedding = query_embedding.reshape(1, -1)
        k = min(st.session_state.config["top_k_retrieval"], st.session_state.faiss_index.ntotal)
        if k > 0:
            distances, indices = st.session_state.faiss_index.search(query_embedding, k)
            for i, idx in enumerate(indices[0]):
                cid = str(idx)
                if cid in st.session_state.data:
                    doc_text = st.session_state.data[cid]["text"]
                    vector_docs.append(doc_text)
                    similarity = 1.0 / (1.0 + distances[0][i])
                    vector_scores_dict[doc_text] = float(similarity)
                    results["vector_results"].append(
                        {
                            "doc_id": cid,
                            "score": float(similarity),
                            "text_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
                            "filename": st.session_state.data.get(cid, {}).get("filename", "Unknown"),
                        }
                    )

    combined_docs = list(set(bm25_docs + vector_docs))
    if not combined_docs:
        results["answer"] = f"No relevant documents found for query: {query}"
        st.session_state.search_history.append(results)
        return results["answer"]

    # Rerank using cross-encoder if enabled
    reranked: List[Dict[str, Any]] = []
    if st.session_state.config["enable_reranking"] and cross_encoder:
        doc_pairs = [[query, doc] for doc in combined_docs]
        try:
            cross_scores = cross_encoder.predict(doc_pairs)
            for score, doc in zip(cross_scores, combined_docs):
                doc_id = None
                for cid, info in st.session_state.data.items():
                    if info["text"] == doc:
                        doc_id = cid
                        break
                result = {
                    "doc_id": doc_id,
                    "cross_score": float(score),
                    "bm25_score": bm25_scores_dict.get(doc, 0.0),
                    "vector_score": vector_scores_dict.get(doc, 0.0),
                    "text_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "filename": st.session_state.data.get(doc_id, {}).get("filename", "Unknown"),
                    "source": [],
                }
                if doc in bm25_docs:
                    result["source"].append("BM25")
                if doc in vector_docs:
                    result["source"].append("Vector")
                reranked.append(result)
            reranked = sorted(reranked, key=lambda x: x["cross_score"], reverse=True)
            results["reranked_results"] = reranked
            top_docs_ids = [r["doc_id"] for r in reranked[: st.session_state.config["rerank_top_n"]]]
            top_docs = [
                st.session_state.data[doc_id]["text"]
                for doc_id in top_docs_ids
                if doc_id in st.session_state.data
            ]
        except Exception as e:
            st.error(f"Error during cross-encoding: {e}")
            log_event("Error", f"Cross-encoding failed: {str(e)}")
            # Fallback to simple weighted average if reranking fails
            reranked = []
            for doc in combined_docs:
                doc_id = None
                for cid, info in st.session_state.data.items():
                    if info["text"] == doc:
                        doc_id = cid
                        break
                bm25_score = bm25_scores_dict.get(doc, 0.0)
                vector_score = vector_scores_dict.get(doc, 0.0)
                combined_score = (bm25_score * 0.4) + (vector_score * 0.6)
                reranked.append(
                    {
                        "doc_id": doc_id,
                        "combined_score": combined_score,
                        "text_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                        "filename": st.session_state.data.get(doc_id, {}).get("filename", "Unknown"),
                    }
                )
            reranked = sorted(reranked, key=lambda x: x["combined_score"], reverse=True)
            results["reranked_results"] = reranked
            top_docs = [
                st.session_state.data[r["doc_id"]]["text"]
                for r in reranked[: st.session_state.config["rerank_top_n"]]
                if r["doc_id"] in st.session_state.data
            ]

    else:
        # Simple weighted average if not using reranking
        for doc in combined_docs:
            doc_id = None
            for cid, info in st.session_state.data.items():
                if info["text"] == doc:
                    doc_id = cid
                    break
            bm25_score = bm25_scores_dict.get(doc, 0.0)
            vector_score = vector_scores_dict.get(doc, 0.0)
            combined_score = (bm25_score * 0.4) + (vector_score * 0.6)
            reranked.append(
                {
                    "doc_id": doc_id,
                    "combined_score": combined_score,
                    "text_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "filename": st.session_state.data.get(doc_id, {}).get("filename", "Unknown"),
                }
            )
        reranked = sorted(reranked, key=lambda x: x["combined_score"], reverse=True)
        results["reranked_results"] = reranked
        top_docs = [
            st.session_state.data[r["doc_id"]]["text"]
            for r in reranked[: st.session_state.config["rerank_top_n"]]
            if r["doc_id"] in st.session_state.data
        ]

    # Build prompt from top documents
    if not top_docs:
        results["answer"] = "No relevant content found in the documents for your query."
    else:
        doc_texts = [f"Document {i+1}:\n{doc}\n{'-'*40}\n" for i, doc in enumerate(top_docs)]
        prompt = f"""You are a sophisticated Retrieval-Augmented Generation (RAG) assistant leveraging hybrid search techniques to answer questions based on the content of uploaded documents. You combine semantic understanding with keyword-based retrieval to provide the most relevant and accurate answers.

You are designed to provide extremely detailed and comprehensive responses, drawing upon all available information within the provided documents. Do not hesitate to include lengthy excerpts and elaborate explanations to fully address the user's query.

Based on the following relevant documents:
{'-'*40}
{' '.join(doc_texts)}
Answer this query clearly and comprehensively:
{query}
Cite specific information from the documents where appropriate. Provide detailed citations, including document names and relevant sections.
If the documents don't contain enough information, state what is missing and suggest potential sources for further research.
"""
        try:
            if model:
                response = model.generate_content(prompt)
                results["answer"] = response.text
            else:
                results["answer"] = "Gemini model not initialized. Please check the API key."
        except Exception as e:
            log_event("Error", f"Gemini response generation failed: {str(e)}")
            results["answer"] = f"Error generating response from Gemini: {str(e)}"

    results["processing_time"] = time.time() - start_time
    st.session_state.search_history.append(results)
    return results["answer"]

# --- Streamlit UI ---
st.title("Advanced Hybrid RAG System")
st.markdown("<div class='main-header'>Upload Documents and Ask Questions</div>", unsafe_allow_html=True)

# Sidebar for document management and app description
with st.sidebar:
    st.header("Configuration")
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=CHUNK_SIZE, step=100)
    rerank_top_n = st.slider("Rerank Top N", min_value=3, max_value=10, value=RERANK_TOP_N, step=1)

    st.session_state.config["chunk_size"] = chunk_size
    st.session_state.config["rerank_top_n"] = rerank_top_n

with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents", type=["txt", "pdf", "docx"], accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            process_document(file)

    # List uploaded documents with option to delete
    if st.session_state.document_chunks:
        st.markdown("### Uploaded Documents")
        for key, info in st.session_state.document_chunks.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{info['filename']}** – {info['chunk_count']} chunks, {info['word_count']} words"
                )
            with col2:
                if st.button("Delete", key=f"delete_{key}"):
                    delete_document(key)
                    st.rerun()  # Rerun to update the list of documents

    # Add a "Clear Database" button
    if st.button("Clear Knowledge Base"):
        with st.spinner("Clearing knowledge base..."):
            st.session_state.data = {}
            st.session_state.corpus = []
            st.session_state.raw_docs = []
            st.session_state.document_chunks = {}
            st.session_state.faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            st.session_state.bm25_model = None

            # Clear the uploaded_files state
            for key in st.session_state.keys():
                if key.startswith("file_uploader"):
                    del st.session_state[key]

            log_event("Knowledge Base", "Cleared all documents and reset knowledge base.")
            st.success("Knowledge base cleared successfully!")
            st.rerun()

    st.markdown("---")  # Add a visual separator
    st.markdown(
        """
    ## How This App Works

    This app is an Advanced Hybrid Retrieval-Augmented Generation (RAG) system. It allows you to upload documents (TXT, PDF, DOCX) and then ask questions about their content.

    Here's a breakdown of the process:

    1.  **Document Upload:** Upload your documents using the file uploader.
    2.  **Processing:** The app processes the documents by:
        *   Extracting text.
        *   Chunking the text into smaller segments.
        *   Generating embeddings (vector representations) for each chunk using Gemini.
        *   Building a search index (FAISS) for efficient retrieval.
        *   Building a BM25 index for keyword-based retrieval.
    3.  **Hybrid Search:** When you ask a question, the app performs a hybrid search:
        *   It uses both vector search (semantic similarity) and BM25 (keyword matching) to find relevant document chunks.
        *   The results are combined and reranked using a cross-encoder for improved accuracy.
    4.  **Answer Generation:** The app uses the reranked document chunks to generate an answer to your question using the Gemini language model.

    **Key Technologies:**

    *   **Streamlit:** For the user interface.
    *   **Google Gemini:** For embeddings and answer generation.
    *   **FAISS:** For efficient vector search.
    *   **BM25:** For keyword-based search.
    *   **Cross-Encoder:** For reranking search results.

    This app demonstrates advanced techniques in information retrieval and natural language processing.
    """
    )

    debug_mode = st.checkbox("Enable Debug Mode", key="debug_checkbox")
    # Debug mode moved to sidebar
    if debug_mode:
        st.markdown("### Debug Information")
        with st.expander("Session State"):
            st.write(st.session_state)
        if st.button("Show Data"):
            st.markdown("#### Document Content")
            for doc_id, doc_info in st.session_state.data.items():
                st.markdown(f"**Doc ID**: {doc_id}")
                st.markdown(f"**Filename**: {doc_info['filename']}")
                st.markdown(f"**Preview**: {doc_info['text'][:100]}...")
                st.markdown("---")

            st.markdown("#### Raw Documents (First 5)")
            for i, doc in enumerate(st.session_state.raw_docs[:5]):
                st.markdown(f"**Raw Doc {i}**: {doc[:100]}...")

            st.markdown("#### Corpus (First 5)")
            for i, tokens in enumerate(st.session_state.corpus[:5]):
                st.markdown(f"**Corpus {i}**: {tokens[:20]}...")

# Main area for search
st.header("Search")
query = st.text_input("Enter your query")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    elif not st.session_state.data:
        st.warning("Please upload documents before searching.")
    elif model is None:
        st.error("Gemini model is not initialized. Check your API key.")
    else:
        with st.spinner("Searching..."):
            answer = hybrid_search_rag(query)
            st.markdown("### Answer")
            st.write(answer)

# Display search history
if st.session_state.search_history:
    st.markdown("### Search History")
    for search in reversed(st.session_state.search_history):
        with st.expander(
            f"Query: {search['query']} ({search['timestamp']}) - {search['processing_time']:.2f} seconds"
        ):
            st.markdown(f"**Answer:** {search['answer']}")
            if search["bm25_results"]:
                st.markdown("**BM25 Results:**")
                for res in search["bm25_results"]:
                    st.markdown(
                        f"- {res['text_preview']} (Score: {res['score']:.4f}, File: {res['filename']})"
                    )
            if search["vector_results"]:
                st.markdown("**Vector Results:**")
                for res in search["vector_results"]:
                    st.markdown(
                        f"- {res['text_preview']} (Score: {res['score']:.4f}, File: {res['filename']})"
                    )
            if search["reranked_results"]:
                st.markdown("**Reranked Results:**")
                for res in search["reranked_results"]:
                    score_display = f"(Cross Score: {res.get('cross_score', 'N/A')}"
                    if "bm25_score" in res:
                        score_display += f", BM25: {res['bm25_score']:.4f}"
                    if "vector_score" in res:
                        score_display += f", Vector: {res['vector_score']:.4f}"
                    if "combined_score" in res:
                        score_display += f", Combined: {res['combined_score']:.4f}"
                    score_display += ")"
                    source_display = (
                        f" (Source: {', '.join(res.get('source', ['N/A']))})"
                        if res.get("source")
                        else ""
                    )
                    st.markdown(
                        f"- {res['text_preview']} {score_display} File: {res['filename']}{source_display}"
                    )