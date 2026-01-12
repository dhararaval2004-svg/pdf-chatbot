import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import json
from datetime import datetime
import sqlite3
import hashlib 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS  
from langchain_groq import ChatGroq

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- LOCAL STORAGE SETUP --------------------
DATA_FOLDER = "data"
PDF_FOLDER = os.path.join(DATA_FOLDER, "pdfs")
VECTORS_FOLDER = os.path.join(DATA_FOLDER, "vectors")
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.json")

# Create folders if they don't exist
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTORS_FOLDER, exist_ok=True)

# -------------------- AUTH DATABASE --------------------
AUTH_DB = "users.db"

def init_auth_db():
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        conn = sqlite3.connect(AUTH_DB)
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user is not None


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


# -------------------- LOCAL DATA MANAGEMENT --------------------
def save_metadata(metadata):
    """Save document metadata to JSON file"""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        st.error(f"Could not save metadata: {e}")


def load_metadata():
    """Load document metadata from JSON file"""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load metadata: {e}")
    return {}


def save_pdf_locally(pdf_file, pdf_id):
    """Save uploaded PDF to local data folder"""
    try:
        pdf_path = os.path.join(PDF_FOLDER, f"{pdf_id}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(pdf_file.getbuffer())
        return pdf_path
    except Exception as e:
        st.error(f"Could not save PDF: {e}")
        return None


def save_vectorstore_locally(vectorstore, vector_id):
    """Save FAISS vectorstore to local data folder"""
    try:
        vector_path = os.path.join(VECTORS_FOLDER, vector_id)
        vectorstore.save_local(vector_path)
        return vector_path
    except Exception as e:
        st.error(f"Could not save vectorstore: {e}")
        return None


def load_vectorstore_locally(vector_id):
    """Load FAISS vectorstore from local data folder"""
    try:
        vector_path = os.path.join(VECTORS_FOLDER, vector_id)
        if os.path.exists(vector_path):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vectorstore = FAISS.load_local(
                vector_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
    except Exception as e:
        st.error(f"Could not load vectorstore: {e}")
    return None


def list_saved_documents():
    """Get list of all saved documents"""
    metadata = load_metadata()
    return metadata.get('documents', [])


def delete_document_locally(doc_id):
    """Delete a document and its associated files"""
    try:
        # Delete PDF
        pdf_path = os.path.join(PDF_FOLDER, f"{doc_id}.pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Delete vectorstore folder
        vector_path = os.path.join(VECTORS_FOLDER, doc_id)
        if os.path.exists(vector_path):
            import shutil
            shutil.rmtree(vector_path)
        
        # Update metadata
        metadata = load_metadata()
        documents = metadata.get('documents', [])
        metadata['documents'] = [doc for doc in documents if doc['id'] != doc_id]
        save_metadata(metadata)
        
        return True
    except Exception as e:
        st.error(f"Could not delete document: {e}")
        return False


# -------------------- CHAT HISTORY FUNCTIONS --------------------
def init_chat_history():
    """Initialize chat history in session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_to_history(question, answer):
    """Add a Q&A pair to chat history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append({
        'timestamp': timestamp,
        'question': question,
        'answer': answer
    })

def clear_history():
    """Clear all chat history"""
    st.session_state.chat_history = []

def download_conversation():
    """Generate downloadable text file of conversation"""
    if not st.session_state.chat_history:
        return "No conversation history to download."
    
    content = "=== PDF Chat Conversation History ===\n\n"
    for i, chat in enumerate(st.session_state.chat_history, 1):
        content += f"{'='*50}\n"
        content += f"Conversation #{i}\n"
        content += f"Time: {chat['timestamp']}\n"
        content += f"\nQ: {chat['question']}\n"
        content += f"\nA: {chat['answer']}\n"
        content += f"{'='*50}\n\n"
    
    return content


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
        .chat-message {{ background-color: #161b22 !important; border: 1px solid #30363d !important; }}
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
        .chat-message {{ background-color: #fff7fa !important; border: 1px solid #f4b6c2 !important; }}
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
        .chat-message {{ background-color: #f0f2f6 !important; border: 1px solid #ddd !important; }}
        </style>

        """, unsafe_allow_html=True)

def auth_page():
    st.title("🔐 Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            if login_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Register", use_container_width=True):
            if register_user(new_user, new_pass):
                st.success("Account created! You can now login.")
            else:
                st.error("Username already exists")

# -------------------- MAIN APP --------------------

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")

    # Initialize chat history
    init_chat_history()
    
    # Initialize last processed question tracker
    if 'last_processed_question' not in st.session_state:
        st.session_state.last_processed_question = None

    # Load saved theme on startup
    if "theme" not in st.session_state:
        st.session_state.theme = load_theme()

    apply_theme(st.session_state.theme)

    # ========== HEADER WITH THEME BUTTONS ==========
    header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([4, 1, 1, 1, 1])
    
    with header_col1:
        st.header("Chat with multiple PDFs 📚")
    
    with header_col3:
        if st.button("☀️ Light", key="light_btn", use_container_width=True, 
                    type="primary" if st.session_state.theme == 'light' else "secondary"):
            st.session_state.theme = 'light'
            save_theme('light')
            st.rerun()
    
    with header_col4:
        if st.button("🌙 Dark", key="dark_btn", use_container_width=True,
                    type="primary" if st.session_state.theme == 'dark' else "secondary"):
            st.session_state.theme = 'dark'
            save_theme('dark')
            st.rerun()
    
    with header_col5:
        if st.button("💖 Pink", key="pink_btn", use_container_width=True,
                    type="primary" if st.session_state.theme == 'pink' else "secondary"):
            st.session_state.theme = 'pink'
            save_theme('pink')
            st.rerun()

    st.markdown("---")


    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it to use answer generation.")
        st.stop()

    # ========== MAIN CONTENT - TWO COLUMNS ==========
    col1, col2 = st.columns([2.5, 1])

    # ========== LEFT COLUMN - CHAT INTERFACE ==========
    with col1:
        st.subheader("💬 Ask Questions")
        
        # Question input
        user_question = st.text_input("Type your question here:", key="question_input")
        
        
        
    # ========== RIGHT COLUMN - HISTORY ONLY ==========
    with col2:
        st.subheader("📜 Chat History")
        
        # History controls
        col_clear, col_download = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                clear_history()
                st.rerun()
        with col_download:
            if st.session_state.chat_history:
                st.download_button(
                    label="💾 Download",
                    data=download_conversation(),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        # Display chat history
        if st.session_state.chat_history:
            st.write(f"**{len(st.session_state.chat_history)} conversation(s)**")
            
            # Show last 5 conversations
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
             with st.expander(f"💬 {chat['question'][:40]}...", expanded=False):
                st.caption(f"🕐 {chat['timestamp']}")
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer']}")


    # ========== SIDEBAR - PDF UPLOAD ==========
    with st.sidebar:
        st.subheader("📄 Your Documents")

        
        # Show currently loaded document
        if 'current_doc' in st.session_state:
            st.info(f"📖 Active: {st.session_state.current_doc['name']}")
            st.markdown("---")
        
        st.subheader("📤 Upload New PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process", use_container_width=True):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                st.stop()

            # Generate unique document ID
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            doc_name = ", ".join([pdf.name for pdf in pdf_docs])

            # Pink theme loading skeleton
            if st.session_state.theme == "pink":
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
                
                # Process PDFs
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
                
                # Save PDFs locally
                for pdf in pdf_docs:
                    save_pdf_locally(pdf, f"{doc_id}_{pdf.name}")
                
                # Save vectorstore locally
                save_vectorstore_locally(vectorstore, doc_id)
                
                # Save metadata
                metadata = load_metadata()
                if 'documents' not in metadata:
                    metadata['documents'] = []
                
                metadata['documents'].append({
                    'id': doc_id,
                    'name': doc_name,
                    'chunks': len(text_chunks),
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                })
                save_metadata(metadata)
                
                # Set as current document
                st.session_state.vectorstore = vectorstore
                st.session_state.current_doc = metadata['documents'][-1]
                
                skeleton_placeholder.empty()
                st.success(f"✅ Saved: {doc_name}")
                st.rerun()
            
            else:
                # Default loading for other themes
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
                        
                        # Save PDFs locally
                        for pdf in pdf_docs:
                            save_pdf_locally(pdf, f"{doc_id}_{pdf.name}")
                        
                        # Save vectorstore locally
                        save_vectorstore_locally(vectorstore, doc_id)
                        
                        # Save metadata
                        metadata = load_metadata()
                        if 'documents' not in metadata:
                            metadata['documents'] = []
                        
                        metadata['documents'].append({
                            'id': doc_id,
                            'name': doc_name,
                            'chunks': len(text_chunks),
                            'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })
                        save_metadata(metadata)
                        
                        # Set as current document
                        st.session_state.vectorstore = vectorstore
                        st.session_state.current_doc = metadata['documents'][-1]
                        
                        st.success(f"✅ Saved: {doc_name}")
                        st.rerun()

    # ========== HANDLE QUESTIONS - DISPLAY ANSWERS IN LEFT COLUMN ==========
    # CRITICAL FIX: Only process if question is new and different from last processed
    if user_question and user_question != st.session_state.last_processed_question:
        if "vectorstore" not in st.session_state:
            with col1:
                st.warning("⚠️ Please upload and process PDFs first.")
        else:
            with col1:
                # Pink theme loading skeleton
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
                        
                        answer_text = result['result']
                        
                        # Add to chat history
                        add_to_history(user_question, answer_text)
                        
                        # Mark this question as processed
                        st.session_state.last_processed_question = user_question
                        
                        # Display answer
                        st.write("### 💬 Answer:")
                        st.write(answer_text)
                        
                        # COPY BUTTON WITH ICON
                        if st.button("📋 Copy Answer", key="copy_btn", use_container_width=True):
                            st.code(answer_text, language=None)
                            st.success("✅ Use Ctrl+C to copy from the code block above!")

                        with st.expander("📄 Source Chunks Used"):
                            for i, doc in enumerate(result['source_documents'], 1):
                                st.write(f"**Chunk {i}:**")
                                st.write(doc.page_content)
                                st.write("---")
                        
                    except Exception as e:
                        skeleton_placeholder.empty()
                        st.error(f"Error generating answer: {str(e)}")
                
                else:
                    # Default loading for other themes
                    with st.spinner("Generating answer..."):
                        try:
                            result = generate_answer(st.session_state.vectorstore, user_question)
                            
                            answer_text = result['result']
                            
                            # Add to chat history
                            add_to_history(user_question, answer_text)
                            
                            # Mark this question as processed
                            st.session_state.last_processed_question = user_question
                            
                            # Display answer
                            st.write("### 💬 Answer:")
                            st.write(answer_text)
                            
                            # COPY BUTTON WITH ICON
                            if st.button("📋 Copy Answer", key="copy_btn", use_container_width=True):
                                st.code(answer_text, language=None)
                                st.success("✅ Use Ctrl+C to copy from the code block above!")

                            with st.expander("📄 Source Chunks Used"):
                                for i, doc in enumerate(result['source_documents'], 1):
                                    st.write(f"**Chunk {i}:**")
                                    st.write(doc.page_content)
                                    st.write("---")
                            
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")


# -------------------- RUN --------------------
if __name__ == "__main__":
    init_auth_db()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        auth_page()
    else:
        main()