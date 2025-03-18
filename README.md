# Advanced Hybrid RAG System
![](https://media.licdn.com/dms/image/v2/D5612AQGeQcakFCOTDg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1723529970365?e=2147483647&v=beta&t=6RRU6WL33TcCbkQh9p8q-Cy6K3GoD4HeullY0ByMcWM)

## Overview

This project implements an **Advanced Hybrid Retrieval-Augmented Generation (RAG) System** using Streamlit. It allows users to upload documents (TXT, PDF, DOCX), processes the content into manageable chunks, and then performs a hybrid search by combining semantic vector search (using FAISS and Gemini embeddings) with keyword-based retrieval (using BM25). A cross-encoder reranks the results, and the Gemini language model generates detailed, citation-rich answers.

This system is designed to be used in a Canvas course environment where instructors or students can interact with the tool as part of a hands-on assignment or project demonstration.

---

## Features

- **Document Processing**  
  - Supports TXT, PDF, and DOCX file formats.
  - Extracts text (with OCR fallback for PDFs when necessary).
  - Splits text into chunks to maintain context.
  - Generates embeddings using the Google Gemini API.

- **Hybrid Search**  
  - **Vector Search:** Leverages FAISS to find semantically similar document chunks.
  - **Keyword Search:** Uses BM25 for precise keyword matching.
  - **Reranking:** Applies a cross-encoder to improve search result accuracy.

- **Answer Generation**  
  - Combines the best document excerpts into a detailed prompt.
  - Generates comprehensive answers using the Gemini language model with proper citations.

- **Interactive UI & Document Management**  
  - Built with Streamlit for an intuitive user interface.
  - Features include document upload, deletion, and a complete knowledge base reset.
  - Displays search history and debug information (if enabled).

- **Canvas Integration**  
  - This README and project are intended for integration into a Canvas course.
  - Instructors can provide students with a link to the deployed app.
  - The system can serve as a practical demonstration of hybrid search techniques and natural language processing in coursework.

---

## Prerequisites

- **Python:** 3.8 or later
- **Key Libraries/Frameworks:**
  - [Streamlit](https://streamlit.io)
  - [google-generativeai](https://pypi.org/project/google-generativeai/)
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [NumPy](https://numpy.org/)
  - [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  - [python-docx](https://python-docx.readthedocs.io/)
  - [Sentence Transformers](https://www.sbert.net/)
  - [rank_bm25](https://pypi.org/project/rank-bm25/)
  - [NLTK](https://www.nltk.org/)
  - [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
   
   *Alternatively, install libraries individually if necessary.*

4. **Download NLTK Data:**

   The application automatically downloads necessary NLTK data (e.g., `punkt`). To manually ensure it’s available, run:
   
   ```python
   import nltk
   nltk.download('punkt')
   ```

5. **Set Up Environment Variables:**

   Create a `.env` file in the project root with the following content:
   
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   Replace `your_gemini_api_key_here` with your actual Google Gemini API key.

---

## Configuration

Key configuration settings are defined at the top of the main script:

- **Chunking Settings:**  
  - `CHUNK_SIZE`: Maximum size for text chunks.
  - `CHUNK_OVERLAP`: Overlap between chunks to retain context.

- **Retrieval Settings:**  
  - `TOP_K_RETRIEVAL`: Number of top documents to retrieve.
  - `RERANK_TOP_N`: Number of documents for reranking.

- **Model Settings:**  
  - `EMBEDDING_MODEL`: Gemini model used for generating embeddings.
  - `LLM_MODEL`: Gemini model used for generating detailed responses.
  - Flags to enable hybrid search and reranking can be toggled.

These can be adjusted directly in the code or via the Streamlit sidebar during runtime.

---

## Usage

1. **Run the Application:**

   From your terminal, start the app by running:
   
   ```bash
   streamlit run app.py
   ```
   
   *(Replace `app.py` with your main Python file name if different.)*

2. **Upload Documents:**

   - Use the sidebar to upload TXT, PDF, or DOCX files.
   - The application processes each document by extracting text, chunking it, and generating embeddings.

3. **Ask Questions:**

   - Enter your query in the provided search box.
   - The system performs a hybrid search using both BM25 and vector search techniques.
   - A cross-encoder reranks the results, and the Gemini model generates a detailed answer with citations.

4. **Manage Documents:**

   - Delete individual documents or clear the entire knowledge base via sidebar options.
   - The application displays a search history and system logs for further insights.

---

## Troubleshooting

- **API Issues:**  
  Verify that your `.env` file contains a valid `GEMINI_API_KEY` and that you have an active internet connection.

- **Document Processing Errors:**  
  Ensure that your documents are in supported formats. For PDFs, if standard text extraction fails, the app will attempt OCR.

- **Performance:**  
  Consider adjusting `CHUNK_SIZE` and `CHUNK_OVERLAP` if processing very large documents.

---

## License & Acknowledgments

- *Include your license information here (e.g., MIT License).*
- **Acknowledgments:**  
  - [Streamlit](https://streamlit.io)  
  - [Google Gemini](https://developers.generativeai.google/)  
  - [FAISS](https://github.com/facebookresearch/faiss)  
  - [NLTK](https://www.nltk.org)  
  - Other libraries and resources as noted above.

---

This README provides a detailed overview and usage instructions for the Advanced Hybrid RAG System, tailored for integration and demonstration within a Canvas course. Feel free to modify and extend the content to suit your course requirements or project specifications.

