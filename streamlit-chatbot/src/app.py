import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embedding model (CPU safe)
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embedding_model = get_embedding_model()

# Load FAISS Vector Store
VECTOR_STORE_DIR = "vector_store"

@st.cache_resource(show_spinner=False)
def get_vector_store(_embeddings):
    if not os.path.exists(VECTOR_STORE_DIR):
        raise FileNotFoundError(
            f"Vector store not found. Please run ingest.py first to create directory '{VECTOR_STORE_DIR}'"
        )
    return FAISS.load_local(
        VECTOR_STORE_DIR,
        _embeddings,
        allow_dangerous_deserialization=True
    )

vector_store = get_vector_store(embedding_model)

# Prompt template
template = """
You are an assistant specialized in Computer Organization and Architecture. The user will provide questions based on this subject.

If the question is related to COA, respond with:
---
Please answer the following question using the context provided:

Context: {context}

Question: {question}

Answer in a concise and informative manner.
Use bullet points if possible.
---

If the question is outside the topic of Computer Organization and Architecture, respond with:
---
I'm specialized in answering questions about Computer Organization and Architecture. 
Could you please ask a question related to this topic?
---
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Create conversational chain
def create_conversational_chain(vector_store, temperature=0.2, max_tokens=256, k_docs=1):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": k_docs}),
        memory=memory
    )
    return chain

conversation_chain = create_conversational_chain(vector_store)

# Handle user query
def handle_query(query, history, temperature=0.2, max_tokens=256, k_docs=1):
    chain = create_conversational_chain(vector_store, temperature, max_tokens, k_docs)
    docs = vector_store.similarity_search(query, k=k_docs)
    result = chain({"question": query, "chat_history": history})

    sources = []
    for i, doc in enumerate(docs):
        source_info = {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
            "score": f"#{i+1}"
        }
        sources.append(source_info)
    return result["answer"], sources

# Streamlit app configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .chat-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .source-card {
        background: #fff3e0;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
        font-size: 0.9rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG-Powered Document Chatbot</h1>
    <p>Ask questions about your documents with AI-powered retrieval and generation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings and info
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Model settings
    st.markdown("### Model Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max Tokens", 50, 500, 256, 50)
    k_docs = st.slider("Number of Documents", 1, 5, 1, 1)
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Statistics")
    if "history" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions Asked", len(st.session_state.history))
        with col2:
            st.metric("Session Duration", "Active")
    
    st.markdown("---")
    
    # Document info
    st.markdown("### 📚 Document Info")
    st.info("📄 PDF documents loaded and indexed")
    st.success("✅ Vector store ready")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.history = []
        st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 Chat with your documents")
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Use form for better input handling
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main concepts in computer architecture?",
            height=100
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            submit_btn = st.form_submit_button("🚀 Ask", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.form_submit_button("🗑️ Clear", use_container_width=True)
    
    # Handle clear button
    if clear_btn:
        st.session_state.history = []
        st.rerun()
    
    # Handle submit
    if submit_btn and user_query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                # Get answer and sources
                answer, sources = handle_query(user_query, st.session_state.history, temperature, max_tokens, k_docs)
                
                # Store in history with sources
                st.session_state.history.append((user_query, answer, sources))
                
                # Display the answer immediately
                st.markdown("### 💬 Response")
                
                # Question
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You:</strong> {user_query}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Source information
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=True):
                        for j, source in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {j+1}:</strong><br>
                                {source['content']}<br>
                                <small>Metadata: {source.get('metadata', {})}</small>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    with st.expander("📚 Sources"):
                        st.info("No source information available for this question.")
                
                st.success("✅ Answer generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try rephrasing your question or check your API key.")

with col2:
    st.markdown("### 📈 Quick Stats")
    
    # Metrics
    if st.session_state.history:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Questions", len(st.session_state.history))
        with col_b:
            # Handle both old (2-tuple) and new (3-tuple) formats
            total_length = 0
            for item in st.session_state.history:
                if len(item) >= 2:
                    total_length += len(item[0])  # item[0] is the question
            avg_length = total_length / len(st.session_state.history) if st.session_state.history else 0
            st.metric("Avg Q Length", f"{avg_length:.0f} chars")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    • Ask specific questions
    • Use keywords from your documents
    • Try different phrasings
    • Check the sources below answers
    """)

# Display conversation history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Previous Conversations")
    
    # Add a toggle to show/hide history
    show_history = st.checkbox("Show conversation history", value=True)
    
    if show_history:
        for i, history_item in enumerate(st.session_state.history):
            if len(history_item) == 3:
                question, answer, sources = history_item
            else:
                # Handle old format for backward compatibility
                question, answer = history_item
                sources = []
            
            # Create a container for each conversation
            with st.container():
                st.markdown(f"#### 💬 Conversation {i+1}")
                
                # Question
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Source information
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                        for j, source in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {j+1}:</strong><br>
                                {source['content']}<br>
                                <small>Metadata: {source.get('metadata', {})}</small>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    with st.expander("📚 Sources"):
                        st.info("No source information available for this question.")
                
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Powered by LangChain, FAISS, and Groq | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)