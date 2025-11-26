"""
Chat Page - Non-Blocking AI Interaction

Features:
- Background task execution
- Web search toggle for methodology research
- Real-time status updates
"""

import streamlit as st
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime
import httpx
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import init_session_state
from utils.background_tasks import get_task_manager
from session_persistence import save_streamlit_session
from langchain.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Chat - Data Science Agent",
    page_icon="💬",
    layout="wide"
)

init_session_state()

# CSS
st.markdown("""
<style>
    .status-running { background: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin-bottom: 1rem; }
    .status-done { background: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; margin-bottom: 1rem; }
    .status-error { background: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #dc3545; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("💬 Chat with AI Agent")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Model
    def fetch_models():
        try:
            r = httpx.get("http://100.91.155.118:11434/v1/models", timeout=5)
            return [m['id'] for m in r.json()['data'] if 'embedding' not in m['id']] if r.status_code == 200 else ["qwen3:30b"]
        except:
            return ["qwen3:30b"]
    
    models = fetch_models()
    model = st.selectbox("🤖 Model", models, index=0)
    st.session_state.selected_model = model
    
    st.divider()
    
    # Features
    st.markdown("### 🎚️ Features")
    
    web_search = st.toggle("🌐 Web Search", value=st.session_state.get("web_search_enabled", False),
                           help="Search documentation for methodology guidance")
    st.session_state.web_search_enabled = web_search
    
    rag = st.toggle("📚 Use History (RAG)", value=st.session_state.get("rag_enabled", True),
                    help="Use context from previous analyses")
    st.session_state.rag_enabled = rag
    
    st.divider()
    
    # File Upload
    st.markdown("### 📤 Data")
    
    current_file = st.session_state.get("uploaded_file_name")
    
    if current_file:
        st.success(f"✅ {current_file}")
        if st.button("❌ Remove File", use_container_width=True):
            if st.session_state.get("temp_file_path") and os.path.exists(st.session_state.temp_file_path):
                os.remove(st.session_state.temp_file_path)
            st.session_state.uploaded_file_name = None
            st.session_state.temp_file_path = None
            save_streamlit_session(st.session_state)
            st.rerun()
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                f.write(uploaded.getvalue())
                st.session_state.temp_file_path = f.name
                st.session_state.uploaded_file_name = uploaded.name
            save_streamlit_session(st.session_state)
            st.rerun()
    
    st.divider()
    
    # Task Status
    task_mgr = get_task_manager()
    task_id = st.session_state.get("current_task_id")
    
    if task_id:
        task = task_mgr.get_task(task_id)
        if task:
            if task.status.value == "running":
                st.info("⏳ Running...")
                if st.button("🔄 Refresh"):
                    st.rerun()
            elif task.status.value == "completed":
                st.success("✅ Done!")
            elif task.status.value == "failed":
                st.error("❌ Failed")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_task_id = None
            save_streamlit_session(st.session_state)
            st.rerun()
    with col2:
        if st.button("💾 Save", use_container_width=True):
            save_streamlit_session(st.session_state)
            st.toast("Saved!")

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Check running task
task_mgr = get_task_manager()
task_id = st.session_state.get("current_task_id")

if task_id:
    task = task_mgr.get_task(task_id)
    if task:
        if task.status.value == "running":
            st.markdown(f'<div class="status-running">⏳ <b>Processing:</b> {task.description}</div>', unsafe_allow_html=True)
            
        elif task.status.value == "completed" and task.result:
            st.markdown('<div class="status-done">✅ <b>Complete!</b></div>', unsafe_allow_html=True)
            
            result = task.result
            
            # Add results to messages (once)
            if not any(isinstance(m, AIMessage) and "Generated Code" in m.content for m in st.session_state.messages[-5:]):
                if result.get("code"):
                    st.session_state.messages.append(AIMessage(content=f"**Generated Code:**\n```python\n{result['code']}\n```"))
                
                output = result.get("output")
                if output and hasattr(output, 'stdout') and output.stdout:
                    st.session_state.messages.append(AIMessage(content=f"**Output:**\n```\n{output.stdout[:2000]}\n```"))
                
                if result.get("summary"):
                    st.session_state.messages.append(AIMessage(content=result["summary"]))
                
                save_streamlit_session(st.session_state)
            
            st.session_state.current_task_id = None
            
        elif task.status.value == "failed":
            st.markdown(f'<div class="status-error">❌ <b>Failed:</b> {task.error}</div>', unsafe_allow_html=True)
            st.session_state.current_task_id = None

# Display messages
for msg in st.session_state.messages:
    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
    with st.chat_message(role):
        content = msg.content
        
        if role == 'assistant' and "**Generated Code:**" in content:
            with st.expander("📝 Code", expanded=False):
                code = content.split("```python\n")[1].split("\n```")[0] if "```python" in content else content
                st.code(code, language="python")
        elif role == 'assistant' and "**Output:**" in content:
            with st.expander("💻 Output", expanded=False):
                output = content.split("```\n")[1].split("\n```")[0] if "```" in content else content
                st.text(output)
        else:
            st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask about your data..."):
    if not st.session_state.get("temp_file_path"):
        st.warning("⚠️ Upload a CSV file first")
        st.stop()
    
    if not Path(st.session_state.temp_file_path).exists():
        st.error("❌ File not found, please re-upload")
        st.session_state.uploaded_file_name = None
        st.session_state.temp_file_path = None
        save_streamlit_session(st.session_state)
        st.rerun()
    
    # Add message
    st.session_state.messages.append(HumanMessage(content=prompt))
    save_streamlit_session(st.session_state)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Start task
    with st.chat_message("assistant"):
        with st.spinner("🧠 Starting..."):
            from ai import workflow, Deps
            
            initial_state = {
                "messages": st.session_state.messages,
                "data_path": st.session_state.temp_file_path,
                "stage_name": st.session_state.get("stage_name", "analysis"),
                "workflow_id": st.session_state.get("workflow_id", "analysis_workflow"),
                "web_search_enabled": st.session_state.get("web_search_enabled", False),
            }
            
            deps: Deps = {
                "executor": st.session_state.executor,
                "output_manager": st.session_state.output_mgr,
                "rag": st.session_state.rag if st.session_state.get("rag_enabled", True) else None,
                "plot_cache": st.session_state.plot_cache,
                "llm": st.session_state.get("selected_model", "qwen3:30b"),
            }
            
            task_id = f"q_{datetime.now().strftime('%H%M%S')}"
            task_mgr.submit_task(
                task_id,
                f"Query: {prompt[:40]}...",
                workflow.ainvoke,
                initial_state,
                context=deps
            )
            
            st.session_state.current_task_id = task_id
            save_streamlit_session(st.session_state)
            st.info("✅ Started! You can navigate to other pages.")
    
    st.rerun()

# Empty state
if not st.session_state.messages:
    if st.session_state.get("uploaded_file_name"):
        st.success(f"✅ **Data loaded:** {st.session_state.uploaded_file_name}\n\nAsk me anything about your data!")
    else:
        st.info("👋 **Welcome!** Upload a CSV file in the sidebar to get started.")