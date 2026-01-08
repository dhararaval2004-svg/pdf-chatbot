import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- KEYWORD EXTRACTION --------------------
def extract_keywords(text, top_n=10):
    """
    Extract important keywords from text using simple frequency analysis
    
    Parameters:
    - text: The text to extract keywords from
    - top_n: Number of top keywords to return
    
    Returns:
    - List of important keywords
    """
    # Remove common stop words
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with',
        'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'was',
        'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'they', 'we', 'their',
        'there', 'here', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'about', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'cannot', 'find', 'information',
        'provided', 'documents', 'context', 'question', 'answer', 'based'
    }
    
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Get top N keywords
    top_keywords = [word for word, _ in word_freq.most_common(top_n)]
    
    return top_keywords


def extract_entities_simple(text):
    """
    Extract potential entities (capitalized words/phrases) from text
    
    Returns:
    - List of entities (proper nouns, names, etc.)
    """
    # Find capitalized words (potential entities)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Remove common sentence starters
    sentence_starters = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'In', 'On', 'At', 'By', 'For'}
    entities = [e for e in entities if e not in sentence_starters]
    
    # Return unique entities
    return list(set(entities))


def highlight_text(text, keywords=None, entities=None, enable_highlighting=True):
    """
    Highlight keywords and entities in text using HTML
    
    Parameters:
    - text: The text to highlight
    - keywords: List of keywords to highlight
    - entities: List of entities to highlight
    - enable_highlighting: Toggle highlighting on/off
    
    Returns:
    - HTML formatted text with highlights
    """
    if not enable_highlighting:
        return text
    
    highlighted_text = text
    
    # Highlight entities (proper nouns) in light blue
    if entities:
        for entity in sorted(entities, key=len, reverse=True):  # Sort by length to handle overlaps
            pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<span style="background-color: #E3F2FD; padding: 2px 4px; border-radius: 3px; font-weight: 500;">{entity}</span>',
                highlighted_text
            )
    
    # Highlight keywords in light yellow
    if keywords:
        for keyword in sorted(keywords, key=len, reverse=True):
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            # Only highlight if not already highlighted
            if f'>{keyword}<' not in highlighted_text:
                highlighted_text = pattern.sub(
                    f'<span style="background-color: #FFF9C4; padding: 2px 4px; border-radius: 3px; font-weight: 500;">{keyword}</span>',
                    highlighted_text
                )
    
    return highlighted_text


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
    """Create vector store using FREE HuggingFace embeddings"""
    if not text_chunks:
        raise ValueError("No text chunks to embed")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


# -------------------- SIMILARITY SEARCH --------------------
def search_relevant_chunks(vectorstore, question, k=3):
    """Find the most relevant chunks for a question"""
    relevant_docs = vectorstore.similarity_search(question, k=k)
    relevant_texts = [doc.page_content for doc in relevant_docs]
    return relevant_texts


# -------------------- ANSWER GENERATION --------------------
def generate_answer(vectorstore, question):
    """Generate an answer using retrieved chunks + LLM (RAG)"""
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

Use the following context to answer the question. 
If you cannot find the answer in the context, say "I cannot find that information in the provided documents."
Be concise and specific in your answer.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    
    return {
        'result': response.content,
        'source_documents': relevant_docs
    }


# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title="Chat with PDFs - Highlighted",
        page_icon="🔦"
    )

    st.header("🔦 Chat with PDFs (with Keyword Highlighting)")
    
    # Info box
    st.info("💡 Important keywords and entities are automatically highlighted in answers!")

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it to use answer generation.")
        st.stop()

    # Highlighting settings in sidebar
    with st.sidebar:
        st.subheader("🎨 Highlighting Settings")
        
        enable_highlighting = st.checkbox(
            "Enable keyword highlighting",
            value=True,
            help="Highlight important keywords and entities in answers"
        )
        
        if enable_highlighting:
            st.markdown("""
            **Color Legend:**
            - <span style="background-color: #FFF9C4; padding: 2px 6px; border-radius: 3px;">Keywords</span> - Important terms
            - <span style="background-color: #E3F2FD; padding: 2px 6px; border-radius: 3px;">Entities</span> - Names, places, etc.
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📄 Your Documents")

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
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the PDFs.")
                    st.stop()

                text_chunks = get_text_chunks(raw_text)

                if not text_chunks:
                    st.error("Failed to split text into chunks.")
                    st.stop()

                st.write(f"✅ Total chunks: {len(text_chunks)}")

                with st.spinner("Creating embeddings (first run may take a moment)..."):
                    vectorstore = get_vectorstore(text_chunks)
                
                st.session_state.vectorstore = vectorstore

                st.success("✅ PDFs processed successfully! You can now ask questions.")

    # User question input
    user_question = st.text_input("Ask a question about your documents:")

    # Handle user questions
    if user_question:
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Please upload and process PDFs first.")
        else:
            with st.spinner("🤔 Thinking and generating answer..."):
                try:
                    result = generate_answer(
                        st.session_state.vectorstore,
                        user_question
                    )
                    
                    answer_text = result['result']
                    
                    # Extract keywords and entities from the answer
                    keywords = extract_keywords(answer_text, top_n=8)
                    entities = extract_entities_simple(answer_text)
                    
                    # Display the answer with highlighting
                    st.write("### 💬 Answer:")
                    
                    if enable_highlighting:
                        # Highlight and display
                        highlighted_answer = highlight_text(
                            answer_text,
                            keywords=keywords,
                            entities=entities,
                            enable_highlighting=True
                        )
                        st.markdown(highlighted_answer, unsafe_allow_html=True)
                        
                        # Show extracted keywords below
                        with st.expander("🔑 Extracted Keywords & Entities"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Keywords:**")
                                if keywords:
                                    for kw in keywords:
                                        st.markdown(f'- <span style="background-color: #FFF9C4; padding: 2px 6px; border-radius: 3px;">{kw}</span>', unsafe_allow_html=True)
                                else:
                                    st.write("None detected")
                            
                            with col2:
                                st.write("**Entities:**")
                                if entities:
                                    for ent in entities:
                                        st.markdown(f'- <span style="background-color: #E3F2FD; padding: 2px 6px; border-radius: 3px;">{ent}</span>', unsafe_allow_html=True)
                                else:
                                    st.write("None detected")
                    else:
                        # Display without highlighting
                        st.write(answer_text)
                    
                    # Show source chunks
                    with st.expander("📄 View source chunks used"):
                        st.write("These are the document sections used to generate the answer:")
                        for i, doc in enumerate(result['source_documents'], 1):
                            st.write(f"**Chunk {i}:**")
                            
                            if enable_highlighting:
                                # Also highlight source chunks
                                chunk_keywords = extract_keywords(doc.page_content, top_n=5)
                                chunk_entities = extract_entities_simple(doc.page_content)
                                highlighted_chunk = highlight_text(
                                    doc.page_content,
                                    keywords=chunk_keywords,
                                    entities=chunk_entities,
                                    enable_highlighting=True
                                )
                                st.markdown(highlighted_chunk, unsafe_allow_html=True)
                            else:
                                st.write(doc.page_content)
                            
                            st.write("---")
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.write("Please check your GROQ_API_KEY and try again.")


if __name__ == "__main__":
    main()