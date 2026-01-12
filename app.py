import streamlit as st
import os
import pickle
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# -------------------- PAGE CONFIG (MUST BE FIRST) --------------------
st.set_page_config("Chat with PDFs", "📚", layout="wide")

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- DATA FOLDER SETUP --------------------
DATA_FOLDER = "data"
USERS_FILE = os.path.join(DATA_FOLDER, "users.json")
VECTORSTORE_PATH = os.path.join(DATA_FOLDER, "faiss_index")
CHUNKS_PATH = os.path.join(DATA_FOLDER, "chunks.pkl")
METADATA_PATH = os.path.join(DATA_FOLDER, "metadata.pkl")
SESSIONS_FOLDER = os.path.join(DATA_FOLDER, "sessions")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(SESSIONS_FOLDER, exist_ok=True)

# -------------------- USER AUTHENTICATION --------------------
def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(username, password, email):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists!"
    
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_at": datetime.now().isoformat()
    }
    save_users(users)
    return True, "Registration successful!"

def login_user(username, password):
    """Verify user login"""
    users = load_users()
    if username not in users:
        return False, "Username not found!"
    
    if users[username]["password"] == hash_password(password):
        return True, "Login successful!"
    return False, "Incorrect password!"

def show_login_page():
    """Display login/registration page"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 500px;
        margin: 100px auto;
        padding: 40px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>📚 PDF Chat App</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Login or Register to continue</p>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        # LOGIN TAB
        with tab1:
            st.subheader("Login to Your Account")
            
            login_username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("🚀 Login", use_container_width=True, type="primary"):
                    if login_username and login_password:
                        success, message = login_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please fill in all fields!")
        
        # REGISTER TAB
        with tab2:
            st.subheader("Create New Account")
            
            reg_username = st.text_input("Username", key="reg_username", placeholder="Choose a username")
            reg_email = st.text_input("Email", key="reg_email", placeholder="Enter your email")
            reg_password = st.text_input("Password", type="password", key="reg_password", placeholder="Choose a password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="Re-enter password")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("📝 Register", use_container_width=True, type="primary"):
                    if reg_username and reg_email and reg_password and reg_confirm:
                        if reg_password == reg_confirm:
                            if len(reg_password) >= 6:
                                success, message = register_user(reg_username, reg_password, reg_email)
                                if success:
                                    st.success(message)
                                    st.info("Please login with your credentials!")
                                else:
                                    st.error(message)
                            else:
                                st.error("Password must be at least 6 characters!")
                        else:
                            st.error("Passwords do not match!")
                    else:
                        st.warning("Please fill in all fields!")

# -------------------- SESSION MANAGEMENT --------------------
def get_user_sessions_file(username):
    """Get the sessions file path for a user"""
    return os.path.join(SESSIONS_FOLDER, f"{username}_sessions.json")

def load_user_sessions(username):
    """Load all sessions for a user"""
    sessions_file = get_user_sessions_file(username)
    if os.path.exists(sessions_file):
        with open(sessions_file, 'r') as f:
            return json.load(f)
    return {}

def save_user_sessions(username, sessions):
    """Save all sessions for a user"""
    sessions_file = get_user_sessions_file(username)
    with open(sessions_file, 'w') as f:
        json.dump(sessions, f, indent=2)

def generate_session_title(first_question):
    """Generate a meaningful title from first question"""
    words = first_question.split()[:5]
    title = " ".join(words)
    return title[:40] + "..." if len(title) > 40 else title

def create_new_session(username):
    """Create a new session for a user"""
    sessions = load_user_sessions(username)
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "title": f"New Chat {len(sessions) + 1}",
        "messages": []
    }
    save_user_sessions(username, sessions)
    return session_id

def add_message_to_session(username, session_id, question, answer):
    """Add a message to a session"""
    sessions = load_user_sessions(username)
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "title": generate_session_title(question),
            "messages": []
        }
    
    # Auto-generate title from first question
    if len(sessions[session_id]["messages"]) == 0:
        sessions[session_id]["title"] = generate_session_title(question)
    
    sessions[session_id]["messages"].append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })
    save_user_sessions(username, sessions)

def delete_session(username, session_id):
    """Delete a session"""
    sessions = load_user_sessions(username)
    if session_id in sessions:
        del sessions[session_id]
        save_user_sessions(username, sessions)

def export_session_to_text(session):
    """Export session to text format"""
    text = f"Chat Session: {session.get('title', 'Untitled')}\n"
    text += f"Created: {session.get('created_at', 'N/A')}\n"
    text += "=" * 60 + "\n\n"
    
    for msg in session.get("messages", []):
        text += f"Q: {msg.get('question', '')}\n"
        text += f"A: {msg.get('answer', '')}\n"
        text += f"Time: {msg.get('timestamp', 'N/A')}\n"
        text += "-" * 60 + "\n\n"
    
    return text

# -------------------- HELPER FUNCTIONS --------------------
def get_pdf_hash(pdf_docs):
    hasher = hashlib.md5()
    for pdf in pdf_docs:
        pdf.seek(0)
        hasher.update(pdf.read())
        pdf.seek(0)
    return hasher.hexdigest()

def save_vectorstore(vectorstore, chunks, pdf_hash, pdf_names):
    try:
        vectorstore.save_local(VECTORSTORE_PATH)
        
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
        
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump({
                'pdf_hash': pdf_hash,
                'timestamp': datetime.now().isoformat(),
                'pdf_names': pdf_names,
                'chunk_count': len(chunks)
            }, f)
        
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def load_vectorstore():
    try:
        if not os.path.exists(VECTORSTORE_PATH):
            return None, None, None
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        
        return vectorstore, chunks, metadata
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def get_storage_size():
    total_size = 0
    if os.path.exists(DATA_FOLDER):
        for dirpath, dirnames, filenames in os.walk(DATA_FOLDER):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def skeleton_loader():
    st.markdown("""
    <style>
    .skeleton {
        background: linear-gradient(90deg, #e5e7eb 25%, #f3f4f6 37%, #e5e7eb 63%);
        background-size: 400% 100%;
        animation: shimmer 1.4s ease infinite;
        height: 60px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    @keyframes shimmer {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
    }
    </style>
    <div class="skeleton"></div>
    """, unsafe_allow_html=True)

def apply_theme(theme):
    if theme == "Light Blue":
        css = """
        <style>
        .stApp {
            background-color: #F5F9FF !important;
            color: #0F172A !important;
        }
        .main .block-container {
            background-color: #F5F9FF !important;
        }
        [data-testid="stSidebar"] {
            background-color: #E8F0FE !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #E8F0FE !important;
        }
        .chat-message {
            background-color: #DBEAFE;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 3px solid #3B82F6;
        }
        .stTextInput > div > div > input {
            background-color: #FFFFFF !important;
        }
        </style>
        """
    elif theme == "Green":
        css = """
        <style>
        .stApp {
            background-color: #F0FDF4 !important;
            color: #052E16 !important;
        }
        .main .block-container {
            background-color: #F0FDF4 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #DCFCE7 !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #DCFCE7 !important;
        }
        .chat-message {
            background-color: #D1FAE5;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 3px solid #10B981;
        }
        .stTextInput > div > div > input {
            background-color: #FFFFFF !important;
        }
        </style>
        """
    elif theme == "Purple":
        css = """
        <style>
        .stApp {
            background-color: #FAF5FF !important;
            color: #3B0764 !important;
        }
        .main .block-container {
            background-color: #FAF5FF !important;
        }
        [data-testid="stSidebar"] {
            background-color: #F3E8FF !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #F3E8FF !important;
        }
        .chat-message {
            background-color: #E9D5FF;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 3px solid #A855F7;
        }
        .stTextInput > div > div > input {
            background-color: #FFFFFF !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        .stApp {
            background-color: #0E1117 !important;
            color: #FAFAFA !important;
        }
        .main .block-container {
            background-color: #0E1117 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #111827 !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #111827 !important;
        }
        .chat-message {
            background-color: #1F2937;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 3px solid #6B7280;
        }
        .stTextInput > div > div > input {
            background-color: #1F2937 !important;
            color: #FAFAFA !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

def generate_answer(vectorstore, question, chat_history=None):
    """Generate answer with conversation context"""
    try:
        # Get relevant documents
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([f"[Source {i+1}] {doc.page_content}" 
                               for i, doc in enumerate(docs)])

        # Build conversation history
        history_text = ""
        if chat_history and len(chat_history) > 0:
            # Include last 5 messages for context
            recent_history = chat_history[-5:]
            for msg in recent_history:
                history_text += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"

        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3
        )

        # Create enhanced prompt with history
        prompt = f"""You are a helpful assistant answering questions based on PDF documents.

Previous Conversation:
{history_text if history_text else "No previous conversation."}

Context from PDFs:
{context}

Instructions:
- Answer ONLY based on the provided PDF context
- If the answer is not in the PDFs, say "Answer not available in the uploaded PDFs"
- Be clear, concise, and accurate
- Consider the conversation history for context

Current Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content, docs
    
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        st.error(error_msg)
        return "I encountered an error processing your question. Please try again.", []

# -------------------- MAIN APP --------------------
def main():
    # ---------- SESSION STATE INIT ----------
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "loaded_from_storage" not in st.session_state:
        st.session_state.loaded_from_storage = False
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "theme" not in st.session_state:
        st.session_state.theme = "Light Blue"

    # ---------- CHECK LOGIN STATUS ----------
    if not st.session_state.logged_in:
        show_login_page()
        return

    # ---------- LOAD FROM STORAGE ON STARTUP ----------
    if not st.session_state.loaded_from_storage:
        with st.spinner("🔄 Loading data from storage..."):
            vectorstore, chunks, metadata = load_vectorstore()
            if vectorstore is not None:
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_processed = True
                st.session_state.metadata = metadata
                st.session_state.loaded_from_storage = True

    # Apply theme
    apply_theme(st.session_state.theme)

    # ---------- TOP BAR WITH THEME SELECTOR ----------
    col1, col2, col3 = st.columns([6, 2, 1])
    with col1:
        st.title("📚 Chat with Multiple PDFs")
    with col2:
        theme = st.selectbox(
            "🎨 Theme",
            ["Light Blue", "Green", "Purple", "Dark"],
            index=["Light Blue", "Green", "Purple", "Dark"].index(st.session_state.theme),
            key="theme_selector"
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
    with col3:
        st.write("")  # Spacing

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # User info
        st.info(f"👤 **{st.session_state.username}**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.current_session = None
            st.rerun()
        
        st.divider()
        
        # Show storage status
        if st.session_state.pdf_processed and st.session_state.metadata:
            st.success("📦 Data loaded")
            with st.expander("📊 Storage Details"):
                metadata = st.session_state.metadata
                st.write(f"**PDFs:** {', '.join(metadata.get('pdf_names', ['Unknown']))}")
                st.write(f"**Chunks:** {metadata.get('chunk_count', 'N/A')}")
                st.write(f"**Date:** {metadata.get('timestamp', 'N/A')[:10]}")
                st.write(f"**Size:** {get_storage_size():.2f} MB")
        
        st.subheader("📄 Upload PDFs")

        pdf_docs = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if st.button("🚀 Process PDFs", use_container_width=True, type="primary"):
            if not pdf_docs:
                st.warning("⚠️ Upload at least one PDF")
            else:
                placeholder = st.empty()
                with placeholder:
                    skeleton_loader()
                    skeleton_loader()

                current_hash = get_pdf_hash(pdf_docs)
                pdf_names = [pdf.name for pdf in pdf_docs]

                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(chunks)

                if save_vectorstore(vectorstore, chunks, current_hash, pdf_names):
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    
                    _, _, metadata = load_vectorstore()
                    st.session_state.metadata = metadata

                    placeholder.empty()
                    st.success("✅ PDFs processed!")
                    st.rerun()
                else:
                    placeholder.empty()
                    st.error("❌ Failed to save")

        st.divider()
        
        # Session management
        st.subheader("💬 Chat Sessions")
        
        if st.button("➕ New Session", use_container_width=True, type="primary"):
            session_id = create_new_session(st.session_state.username)
            st.session_state.current_session = session_id
            st.rerun()
        
        # Load and display sessions
        sessions = load_user_sessions(st.session_state.username)
        
        if sessions:
            st.caption(f"Total sessions: {len(sessions)}")
            
            # Scrollable session list
            for session_id in sorted(sessions.keys(), reverse=True):
                session = sessions[session_id]
                msg_count = len(session.get("messages", []))
                
                col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
                
                with col_s1:
                    is_current = st.session_state.current_session == session_id
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(
                        f"📝 {session.get('title', session_id[-8:])} ({msg_count})",
                        key=f"session_{session_id}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.current_session = session_id
                        st.rerun()
                
                with col_s2:
                    # Export button
                    if st.button("📥", key=f"export_{session_id}"):
                        export_text = export_session_to_text(session)
                        st.download_button(
                            label="Download",
                            data=export_text,
                            file_name=f"{session.get('title', 'chat')}.txt",
                            mime="text/plain",
                            key=f"download_{session_id}"
                        )
                
                with col_s3:
                    if st.button("🗑️", key=f"del_{session_id}"):
                        delete_session(st.session_state.username, session_id)
                        if st.session_state.current_session == session_id:
                            st.session_state.current_session = None
                        st.rerun()
        else:
            st.info("No sessions yet. Create one!")
        
        st.divider()
        
        if st.button("🗑️ Delete All Data", use_container_width=True):
            try:
                import shutil
                if os.path.exists(DATA_FOLDER):
                    for item in os.listdir(DATA_FOLDER):
                        if item != "users.json":
                            path = os.path.join(DATA_FOLDER, item)
                            if os.path.isfile(path):
                                os.remove(path)
                            elif os.path.isdir(path):
                                shutil.rmtree(path)
                
                st.session_state.vectorstore = None
                st.session_state.pdf_processed = False
                st.session_state.loaded_from_storage = False
                st.session_state.metadata = None
                st.session_state.current_session = None
                
                st.success("🗑️ Deleted!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ---------- MAIN CONTENT ----------
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY missing in .env file")
        st.stop()

    # Ensure a session exists
    if st.session_state.current_session is None:
        sessions = load_user_sessions(st.session_state.username)
        if not sessions:
            st.session_state.current_session = create_new_session(st.session_state.username)
        else:
            st.session_state.current_session = sorted(sessions.keys(), reverse=True)[0]

    # Get current session messages
    sessions = load_user_sessions(st.session_state.username)
    current_session_data = sessions.get(st.session_state.current_session, {})
    current_messages = current_session_data.get("messages", [])

    # Display chat history
    if current_messages:
        st.subheader(f"💬 {current_session_data.get('title', 'Current Session')}")
        
        for idx, msg in enumerate(current_messages):
            question_text = str(msg.get('question', '')).replace('<', '&lt;').replace('>', '&gt;')
            answer_text = str(msg.get('answer', '')).replace('<', '&lt;').replace('>', '&gt;')
            
            st.markdown(f"""
            <div class="chat-message">
                <p style="margin:0; font-size:0.9em; font-weight:600; color: #1E40AF;">🧑 Q:</p>
                <p style="margin:5px 0 10px 0; font-size:0.9em; line-height: 1.4;">{question_text}</p>
                <p style="margin:10px 0 5px 0; font-size:0.9em; font-weight:600; color: #059669;">🤖 A:</p>
                <p style="margin:5px 0; font-size:0.9em; line-height: 1.5; white-space: pre-wrap;">{answer_text}</p>
                <p style="margin:10px 0 0 0; font-size:0.75em; opacity:0.7;">🕒 {msg.get('timestamp', 'N/A')[11:19]}</p>
            </div>
            """, unsafe_allow_html=True)
        st.divider()

    # Question input
    user_question = st.text_input(
        "Ask a question from your PDFs",
        placeholder="e.g., What is the main topic of the document?",
        key="user_input"
    )

    if user_question:
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process PDFs first!")
        else:
            # Check if this is a new question
            if not current_messages or current_messages[-1]["question"] != user_question:
                answer_placeholder = st.empty()
                with answer_placeholder:
                    st.info("🤔 Thinking...")
                    skeleton_loader()

                # Generate answer with chat history context
                answer, docs = generate_answer(
                    st.session_state.vectorstore,
                    user_question,
                    chat_history=current_messages
                )

                answer_placeholder.empty()

                # Add to current session
                add_message_to_session(
                    st.session_state.username,
                    st.session_state.current_session,
                    user_question,
                    answer
                )
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(docs, 1):
                        st.caption(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "...")
                
                st.rerun()

if __name__ == "__main__":
    main()