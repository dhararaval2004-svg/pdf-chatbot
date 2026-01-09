import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS  
from langchain_groq import ChatGroq

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- THEME PERSISTENCE --------------------
THEME_FILE = "theme_preference.json"

def save_theme(theme):
    """Save theme preference to local file"""
    try:
        with open(THEME_FILE, 'w') as f:
            json.dump({'theme': theme}, f)
    except Exception as e:
        st.warning(f"Could not save theme preference: {e}")

def load_theme():
    """Load theme preference from local file"""
    try:
        if os.path.exists(THEME_FILE):
            with open(THEME_FILE, 'r') as f:
                data = json.load(f)
                return data.get('theme', 'light')
    except Exception as e:
        st.warning(f"Could not load theme preference: {e}")
    return 'light'

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
    if not text_chunks:
        raise ValueError("No text chunks to embed")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# -------------------- SIMILARITY SEARCH --------------------
def search_relevant_chunks(vectorstore, question, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in relevant_docs]


# -------------------- ANSWER GENERATION --------------------
def generate_answer(vectorstore, question):
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    relevant_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return {'result': response.content, 'source_documents': relevant_docs}


# -------------------- THEME HANDLER --------------------
def apply_theme(theme):
    transition_css = """
    .stApp,
    section[data-testid="stSidebar"],
    header[data-testid="stHeader"],
    input,
    textarea,
    .stButton > button {
        transition:
            background-color 0.25s ease,
            color 0.25s ease,
            border-color 0.25s ease,
            box-shadow 0.25s ease;
    }
    """

    if theme == "dark":
        st.markdown(f"""
        <style>
        {transition_css}
        .stApp {{ background-color: #0d1117 !important; color: #e6edf3 !important; }}
        header[data-testid="stHeader"] {{ background-color: #0d1117 !important; border-bottom: 1px solid #30363d !important; }}
        section[data-testid="stSidebar"] {{ background-color: #161b22 !important; }}
        h1,h2,h3,h4,h5,h6,p,span,label {{ color: #e6edf3 !important; }}
        input,textarea {{ background-color: #21262d !important; color: #e6edf3 !important; border-radius:8px !important; border:1px solid #30363d !important; }}
        div[data-testid="stFileUploader"] {{ background-color:#161b22 !important; border:1px dashed #30363d !important; border-radius:10px !important; padding:10px !important; }}
        .stButton > button {{ background-color:#238636 !important; color:white !important; border-radius:8px !important; }}
        .stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 6px 14px rgba(0,0,0,0.25); }}
        div[data-testid="stExpander"] {{ background-color:#161b22 !important; border:1px solid #30363d !important; border-radius:10px !important; }}
        </style>
        """, unsafe_allow_html=True)

    elif theme == "pink":
        st.markdown(f"""
        <style>
        {transition_css}
        .stApp {{ background-color:#fff1f5 !important; color:#3b0a1a !important; }}
        header[data-testid="stHeader"] {{ background-color:#ffe4ec !important; border-bottom:1px solid #f4b6c2 !important; }}
        section[data-testid="stSidebar"] {{ background-color:#ffe4ec !important; }}
        h1,h2,h3,h4,h5,h6,p,span,label {{ color:#3b0a1a !important; }}
        input,textarea {{ background-color:#fff7fa !important; color:#3b0a1a !important; border-radius:8px !important; border:1px solid #f4b6c2 !important; }}
        div[data-testid="stFileUploader"] {{ background-color:#fff7fa !important; border:1px dashed #f4b6c2 !important; border-radius:10px !important; padding:10px !important; }}
        .stButton > button {{ background-color:#ec407a !important; color:white !important; border-radius:8px !important; }}
        .stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 6px 14px rgba(236,64,122,0.35); }}
        div[data-testid="stExpander"] {{ background-color:#fff7fa !important; border-radius:10px !important; border:1px solid #f4b6c2 !important; }}
        </style>
        """, unsafe_allow_html=True)

    else:  # light
        st.markdown(f"""
        <style>
        {transition_css}
        .stApp {{ background-color:#ffffff !important; color:#000000 !important; }}
        header[data-testid="stHeader"] {{ background-color:#ffffff !important; border-bottom:1px solid #ddd !important; }}
        section[data-testid="stSidebar"] {{ background-color:#f5f7fb !important; }}
        h1,h2,h3,h4,h5,h6,p,span,label {{ color:#000000 !important; }}
        input,textarea {{ background-color:#f0f2f6 !important; color:#000000 !important; border-radius:8px !important; border:1px solid #ccc !important; }}
        div[data-testid="stFileUploader"] {{ background-color:#ffffff !important; border:1px dashed #ccc !important; border-radius:10px !important; padding:10px !important; }}
        .stButton > button {{ background-color:#1976d2 !important; color:white !important; border-radius:8px !important; }}
        .stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 6px 14px rgba(0,0,0,0.15); }}
        div[data-testid="stExpander"] {{ background-color:#ffffff !important; border-radius:10px !important; border:1px solid #ddd !important; }}
        </style>
        """, unsafe_allow_html=True)


# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")

    # Load saved theme on startup
    if "theme" not in st.session_state:
        st.session_state.theme = load_theme()

    # ----- Theme radio with callback -----
    def update_theme():
        new_theme = st.session_state.theme_radio
        st.session_state.theme = new_theme
        save_theme(new_theme)  # Save to file when changed

    with st.sidebar:
        st.radio(
            "🎨 Select Theme",
            ["light", "dark", "pink"],
            key="theme_radio",
            index=["light","dark","pink"].index(st.session_state.theme),
            on_change=update_theme
        )

    apply_theme(st.session_state.theme)

    st.header("Chat with multiple PDFs 📚")

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it to use answer generation.")
        st.stop()

    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                st.stop()

            # ---------- Pink Theme Loading Skeleton ----------
            if st.session_state.theme == "pink":
                # Create skeleton placeholder
                skeleton_placeholder = st.empty()
                skeleton_placeholder.markdown("""
                <div style="padding: 20px;">
                    <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                background-size: 200% 100%; 
                                animation: shimmer 1.5s infinite;
                                height: 30px; 
                                border-radius: 8px; 
                                margin-bottom: 15px;">
                    </div>
                    <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                background-size: 200% 100%; 
                                animation: shimmer 1.5s infinite;
                                height: 30px; 
                                border-radius: 8px; 
                                margin-bottom: 15px;">
                    </div>
                    <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                background-size: 200% 100%; 
                                animation: shimmer 1.5s infinite;
                                height: 30px; 
                                border-radius: 8px;">
                    </div>
                </div>
                <style>
                @keyframes shimmer {
                    0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; }
                }
                </style>
                """, unsafe_allow_html=True)
                
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    skeleton_placeholder.empty()
                    st.error("No readable text found in the PDFs.")
                    st.stop()

                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    skeleton_placeholder.empty()
                    st.error("Failed to split text into chunks.")
                    st.stop()

                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                
                # Clear skeleton and show success
                skeleton_placeholder.empty()
                st.success("✅ PDFs processed successfully!")
            
            else:
                # ---------- Default Loading for other themes ----------
                with st.spinner("Reading PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text found in the PDFs.")
                        st.stop()

                    with st.spinner("Splitting text..."):
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("Failed to split text into chunks.")
                            st.stop()
                        st.write(f"✅ Total chunks: {len(text_chunks)}")

                    with st.spinner("Creating embeddings..."):
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.vectorstore = vectorstore
                        st.success("✅ PDFs processed successfully!")

    # ---------- Handle questions ----------
    if user_question:
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Please upload and process PDFs first.")
        else:
            # ---------- Pink Theme Loading Skeleton ----------
            if st.session_state.theme == "pink":
                skeleton_placeholder = st.empty()
                skeleton_placeholder.markdown("""
                <div style="padding: 20px;">
                    <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                background-size: 200% 100%; 
                                animation: shimmer 1.5s infinite;
                                height: 40px; 
                                border-radius: 8px; 
                                margin-bottom: 10px;">
                    </div>
                    <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                background-size: 200% 100%; 
                                animation: shimmer 1.5s infinite;
                                height: 100px; 
                                border-radius: 8px;">
                    </div>
                </div>
                <style>
                @keyframes shimmer {
                    0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; }
                }
                </style>
                """, unsafe_allow_html=True)
                
                try:
                    result = generate_answer(st.session_state.vectorstore, user_question)
                    skeleton_placeholder.empty()
                    
                    st.write("### 💬 Answer:")
                    st.write(result['result'])

                    with st.expander("📄 Source Chunks Used"):
                        for i, doc in enumerate(result['source_documents'], 1):
                            st.write(f"**Chunk {i}:**")
                            st.write(doc.page_content)
                            st.write("---")
                except Exception as e:
                    skeleton_placeholder.empty()
                    st.error(f"Error generating answer: {str(e)}")
            
            else:
                # ---------- Default Loading for other themes ----------
                with st.spinner("Generating answer..."):
                    try:
                        result = generate_answer(st.session_state.vectorstore, user_question)
                        st.write("### 💬 Answer:")
                        st.write(result['result'])

                        with st.expander("📄 Source Chunks Used"):
                            for i, doc in enumerate(result['source_documents'], 1):
                                st.write(f"**Chunk {i}:**")
                                st.write(doc.page_content)
                                st.write("---")
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")


# -------------------- RUN --------------------
if __name__ == "__main__":
    main()