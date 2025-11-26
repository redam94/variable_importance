"""
Streaming Chat Page - Real-Time LLM Responses

This page provides instant, streaming responses for simple queries.
Use the main Chat page for data analysis with background processing.
"""

import streamlit as st
import sys
from pathlib import Path
import httpx
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import init_session_state
from session_persistence import save_streamlit_session
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Streaming Chat",
    page_icon="⚡",
    layout="wide"
)

init_session_state()

rag = st.session_state.get("rag")


# Custom CSS
st.markdown("""
<style>
    .streaming-note {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Streaming Chat")
st.caption("Real-time LLM responses - instant feedback with token-by-token streaming")

# Info banner
st.markdown("""
<div class="streaming-note">
    ⚡ <strong>Fast Mode:</strong> This page streams responses in real-time but doesn't execute code or analyze data.<br>
    📊 For data analysis with plots and calculations, use the main <strong>Chat</strong> page.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection
    def fetch_models():
        try:
            url = "http://100.91.155.118:11434/v1/models"
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return [model['id'] for model in response.json()['data']]
        except:
            pass
        return ["qwen3:30b"]
    
    models = fetch_models()
    current_model = st.session_state.get("selected_model", models[0])
    model = st.selectbox(
        "🤖 Model",
        options=models,
        index=models.index(current_model) if current_model in models else 0
    )
    
    if model != st.session_state.get("selected_model"):
        st.session_state.selected_model = model
        save_streamlit_session(st.session_state)
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    **Best for:**
    - General questions
    - Explanations
    - Conceptual discussions
    - Quick answers
    
    **Not for:**
    - Data analysis
    - Creating plots
    - Running code
    - Complex calculations
    
    *Use main Chat page for those!*
    """)
    
    st.divider()
    
    # Actions
    st.markdown("### 🎛️ Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            if "streaming_messages" in st.session_state:
                st.session_state.streaming_messages = []
            save_streamlit_session(st.session_state)
            st.rerun()
    with col2:
        if st.button("💾 Save", use_container_width=True):
            save_streamlit_session(st.session_state)
            st.success("Saved!")

# Initialize streaming messages (separate from main chat)
if "streaming_messages" not in st.session_state:
    st.session_state.streaming_messages = []

# Display chat history
for msg in st.session_state.streaming_messages:
    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input with streaming
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.streaming_messages.append(HumanMessage(content=prompt))
    save_streamlit_session(st.session_state)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Stream assistant response
    with st.chat_message("assistant"):
        try:
            # Initialize LLM with streaming
            llm = ChatOllama(
                model=model,
                base_url="http://100.91.155.118:11434",
                streaming=True  # Enable streaming!
            )
            
            if rag:
                # Create an agent with RAG context retrieval
                @tool
                def retrieve_context(query: str) -> str:
                    """Retrieve relevant context from RAG for the given query."""
                    return rag.get_context_summary(
                        query=query,
                        workflow_id=st.session_state.get("workflow_id"),
                        stage_name=st.session_state.get("stage_name"),
                        max_tokens=2000
                    )
                @tool
                def store_knowledge(document: str) -> str:
                    """Store useful information the user provide in the RAG system for future reference
                    along with the current time.
                    """
                    logger.info("Storing document in RAG...")
                    rag.add_summary(
                        summary=f"{document}",
                        stage_name=st.session_state.get("stage_name"),
                        workflow_id=st.session_state.get("workflow_id"),
                        metadata={"time": datetime.now().isoformat()}
                    )
                    return "Document stored successfully."
                
                agent = create_agent(
                    model=llm,
                    tools=[retrieve_context],
                    system_prompt="Before answering create a plan to query the RAG system. " \
                    "You may use the `retrieve_context` tool multiple times to get relevant information. " \
                    "If the context is sufficient, provide a concise answer. " \
                    "Make sure to provide accurate and concise responses based on the retrieved information."
                )
                
                def response_generator():
                    for token, metadata in agent.stream({"messages": [HumanMessage(content=prompt)]}, stream_mode='messages'):
                        if metadata['langgraph_node'] == 'model':
                            yield token.content
                # Stream the response token by token using the agent
                response = st.write_stream(
                   response_generator()
                )
            else:
                # Stream the response token by token
                response = st.write_stream(
                    llm.stream([HumanMessage(content=prompt)]), cursor="|"
                )
            
            # Save to chat history
            st.session_state.streaming_messages.append(AIMessage(content=response))
            save_streamlit_session(st.session_state)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("🔍 Debug Info"):
                st.code(traceback.format_exc())

# Instructions
if not st.session_state.streaming_messages:
    st.info("""
    👋 **Welcome to Streaming Chat!**
    
    This page provides **instant, real-time responses** perfect for:
    - 🤔 Asking questions
    - 📚 Learning concepts
    - 💬 General discussion
    - 🧠 Getting explanations
    
    **How it works:**
    - Type your question below
    - Watch the response appear **token by token** in real-time
    - Get instant feedback without waiting
    
    **Examples:**
    - "Explain how neural networks work"
    - "What is the difference between Python and JavaScript?"
    - "Tell me about quantum computing"
    - "How do I improve my programming skills?"
    
    ---
    
    💡 **Need data analysis?** Go to the main **Chat** page to:
    - Upload CSV files
    - Generate plots and visualizations
    - Run statistical analyses
    - Execute Python code
    """)

# Footer
st.divider()
st.caption("⚡ Streaming enabled • Instant responses • No background processing")