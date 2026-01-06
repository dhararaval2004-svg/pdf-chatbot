import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
 # CHANGED: Using HuggingFace instead of Google
from langchain_community.vectorstores import FAISS

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- PDF TEXT EXTRACTION --------------------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# -------------------- TEXT CHUNKING --------------------
def get_text_chunks(raw_text):
    """Split text into smaller chunks for embedding"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(raw_text)


# -------------------- VECTOR STORE --------------------
def get_vectorstore(text_chunks):
    """
    Create vector store using FREE HuggingFace embeddings
    
    CHANGED: Now using HuggingFaceEmbeddings instead of GoogleGenerativeAIEmbeddings
    - Completely free (no API costs)
    - Runs locally (no rate limits)
    - No API key required
    - Model downloads once (~90MB) then cached
    """
    if not text_chunks:
        raise ValueError("No text chunks to embed")

    # Create embeddings using HuggingFace model (runs on your machine)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight, fast model
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU for faster processing
        encode_kwargs={'normalize_embeddings': True}  # Better similarity search results
    )

    # Create FAISS vector store from text chunks
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore

# -------------------- SIMILARITY SEARCH --------------------
def search_relevant_chunks(vectorstore, question, k=3):
    """
    Find the most relevant chunks for a question
    
    Parameters:
    - vectorstore: The FAISS database with all chunks
    - question: User's question as a string
    - k: How many chunks to retrieve (default: 3)
    
    Returns:
    - List of relevant text chunks
    """
    # Search for similar chunks
    # This converts the question to numbers and finds matching chunks
    relevant_docs = vectorstore.similarity_search(question, k=k)
    
    # Extract just the text from the results
    relevant_texts = [doc.page_content for doc in relevant_docs]
    
    return relevant_texts

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📚"
    )

    st.header("Chat with multiple PDFs 📚")
    
    # Info box about the free embeddings
    # st.info("💡 Using free local HuggingFace embeddings - no API costs or limits!")

    # REMOVED: Google API key check (not needed anymore)
    # No API key validation required for local embeddings

    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                st.stop()

            with st.spinner("Reading and indexing PDFs..."):
                # Step 1: Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the PDFs.")
                    st.stop()

                # Step 2: Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                if not text_chunks:
                    st.error("Failed to split text into chunks.")
                    st.stop()

                st.write(f"✅ Total chunks: {len(text_chunks)}")

                # Step 3: Create vector store with embeddings
                # Note: First run will download the model (~90MB), then it's cached
                with st.spinner("Creating embeddings (first run may take a moment)..."):
                    vectorstore = get_vectorstore(text_chunks)
                
                # Store in session state for later use
                st.session_state.vectorstore = vectorstore

                st.success("✅ PDFs processed successfully!")

    # Handle user questions
    if user_question:
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Please upload and process PDFs first.")

        else:    
            # Step 1: Search for relevant chunks
            with st.spinner("Searching for relevant information..."):
                relevant_chunks = search_relevant_chunks(
                    st.session_state.vectorstore, 
                    user_question,
                    k=3  # Get top 3 most relevant chunks
                )
        
            # Display what we found (for testing)
            st.write("### 🔍 Found these relevant sections:")
            for i, chunk in enumerate(relevant_chunks, 1):
                with st.expander(f"Chunk {i}"):
                    st.write(chunk)

# -------------------- RUN --------------------
if __name__ == "__main__":
    main()