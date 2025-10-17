import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ...existing code...

# ================================
# 1️⃣ Extract Text from PDF
# ================================
def extract_text_from_pdf(pdf_file):
    # pdf_file is a Streamlit UploadedFile; read bytes and open with fitz
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ================================
# 2️⃣ Chunk Text into Pieces
# ================================
def chunk_text(text, chunk_size=500):
    if not text:
        return []
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ================================
# 3️⃣ Load Models (Embeddings + LLM)
# ================================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="ibm-granite/granite-3.2-2b-instruct")
    return embedder, generator

# ...existing code...

# ================================
# 4️⃣ Build and Search FAISS Index
# ================================
def build_faiss_index(chunks, embedder):
    if not chunks:
        return None
    # ensure numpy array with float32 dtype for faiss
    embeddings = np.array(embedder.encode(chunks))
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search(query, chunks, index, embedder, top_k=3):
    if index is None or not chunks:
        return []
    query_vec = np.array(embedder.encode([query])).astype("float32")
    # ensure k does not exceed number of vectors
    k = min(top_k, index.ntotal) if index.ntotal > 0 else 0
    if k == 0:
        return []
    D, I = index.search(query_vec, k)
    # filter out invalid indices (-1)
    results = []
    for idx in I[0]:
        if idx >= 0 and idx < len(chunks):
            results.append(chunks[idx])
    return results

# ================================
# 5️⃣ Streamlit Interface
# ================================
def main():
    st.set_page_config(page_title="StudyMate - AI Academic Assistant", layout="wide")
    st.title("📚 StudyMate – AI PDF Question-Answer Assistant")
    st.write("Upload your study materials and ask questions interactively!")

    # load models inside Streamlit run (cached by st.cache_resource)
    embedder, generator = load_models()

    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.info("Processing uploaded PDFs...")
        all_text = ""
        for pdf in uploaded_files:
            try:
                all_text += extract_text_from_pdf(pdf)
            except Exception as e:
                st.error(f"Failed to read {getattr(pdf, 'name', 'file')}: {e}")

        chunks = chunk_text(all_text)
        if not chunks:
            st.warning("No text could be extracted from the uploaded PDFs.")
            return

        index = build_faiss_index(chunks, embedder)
        if index is None or index.ntotal == 0:
            st.error("Failed to build vector index.")
            return

        st.success("✅ PDFs processed successfully! You can now ask questions.")

        query = st.text_input("Ask a question about your study materials:")
        if query:
            relevant_chunks = search(query, chunks, index, embedder)
            if not relevant_chunks:
                st.info("No relevant content found for that question.")
            else:
                context = " ".join(relevant_chunks)
                prompt = f"Answer the question based only on this content:\n\n{context}\n\nQuestion: {query}\nAnswer:"

                try:
                    gen = generator(prompt, max_new_tokens=300, do_sample=False)
                    # guard against differing return shapes/keys
                    if isinstance(gen, list) and gen:
                        answer = gen[0].get("generated_text", "") if isinstance(gen[0], dict) else str(gen[0])
                    elif isinstance(gen, dict):
                        answer = gen.get("generated_text", "") or gen.get("text", "")
                    else:
                        answer = str(gen)
                except Exception as e:
                    answer = f"Failed to generate answer: {e}"

                st.markdown("### 🧠 Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()
