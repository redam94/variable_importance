"""
Chat Page - Non-Blocking AI Interaction

Features:
- Background task execution
- Navigate away while processing
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

# Add parent directory to path
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

# Initialize
init_session_state()

# Custom CSS
st.markdown("""
<style>
    .task-running {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .task-completed {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .task-failed {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Chat with AI Agent")
st.caption("Ask questions about your data - processing runs in the background!")

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
    
    # File upload with improved UX
    st.markdown("### 📤 Data Upload")
    
    # Check if file is already uploaded
    current_file = st.session_state.get("uploaded_file_name")
    
    if current_file:
        # File is uploaded - show info with replace option
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; margin-bottom: 1rem;">
            <strong>✅ File Loaded</strong><br>
            <span style="font-size: 0.95rem;">📄 {current_file}</span><br>
            <span style="font-size: 0.85rem; color: #666;">
                Size: {st.session_state.get('uploaded_file_size', 0) / 1024:.2f} KB
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Replace file button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Replace File", use_container_width=True):
                # Clear current file
                st.session_state.uploaded_file_name = None
                st.session_state.temp_file_path = None
                st.session_state.uploaded_file_size = 0
                save_streamlit_session(st.session_state)
                st.rerun()
        with col2:
            if st.button("❌ Remove File", use_container_width=True, type="secondary"):
                # Remove file and clean up
                if st.session_state.get("temp_file_path"):
                    try:
                        import os
                        if os.path.exists(st.session_state.temp_file_path):
                            os.remove(st.session_state.temp_file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file: {e}")
                
                st.session_state.uploaded_file_name = None
                st.session_state.temp_file_path = None
                st.session_state.uploaded_file_size = 0
                save_streamlit_session(st.session_state)
                st.rerun()
    else:
        # No file uploaded - show uploader
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload a CSV file to analyze"
            # NOTE: No key here - file info stored separately in session state
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                st.session_state.temp_file_path = temp_file.name
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_size = uploaded_file.size
                save_streamlit_session(st.session_state)
                st.rerun()  # Rerun to show the file info card
    
    st.divider()
    
    # Task status
    st.markdown("### 🔄 Task Status")
    task_mgr = get_task_manager()
    current_task_id = st.session_state.get("current_task_id")
    
    if current_task_id:
        task = task_mgr.get_task(current_task_id)
        if task:
            if task.status.value == "running":
                st.info(f"⏳ Processing...")
                if st.button("🔄 Refresh Status"):
                    st.rerun()
            elif task.status.value == "completed":
                st.success("✅ Completed!")
            elif task.status.value == "failed":
                st.error("❌ Failed")
    
    st.divider()
    
    # Actions
    st.markdown("### 🎛️ Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_task_id = None
            save_streamlit_session(st.session_state)
            st.rerun()
    with col2:
        if st.button("💾 Save", use_container_width=True):
            save_streamlit_session(st.session_state)
            st.success("Saved!")

# Main content
# Check for running task
task_mgr = get_task_manager()
current_task_id = st.session_state.get("current_task_id")

if current_task_id:
    task = task_mgr.get_task(current_task_id)
    if task:
        if task.status.value == "running":
            # Check if task is actually running or was running before page refresh
            st.markdown(f"""
            <div class="task-running">
                ⏳ <strong>Task in progress...</strong><br>
                {task.description}<br>
                <small>Note: If you refreshed the page, the task may have been interrupted. 
                You can submit a new query if needed.</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-refresh while running
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Check Status", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("❌ Clear Task", use_container_width=True):
                    st.session_state.current_task_id = None
                    save_streamlit_session(st.session_state)
                    st.rerun()
            
        elif task.status.value == "completed":
            st.markdown(f"""
            <div class="task-completed">
                ✅ <strong>Analysis Complete!</strong><br>
                Results are ready below.
            </div>
            """, unsafe_allow_html=True)
            
            # Add results to messages if not already done
            if task.result:
                result = task.result
                
                # Check if we already added these results
                if not any(msg.content.startswith("Code:") for msg in st.session_state.messages[-3:]):
                    # Add code
                    if 'code' in result and result['code']:
                        st.session_state.messages.append(
                            AIMessage(content=f"Code:\n\n```python\n{result['code']}\n```")
                        )
                    
                    # Add output
                    if 'code_output' in result and result['code_output']:
                        co = result['code_output']
                        
                        st.session_state.messages.append(
                            AIMessage(content=f"**Output:**\n```\n{co.stdout}\n```")
                        )
                    
                    # Add summary
                    if 'summary' in result:
                        st.session_state.messages.append(
                            AIMessage(content=result['summary'])
                        )
                    
                    save_streamlit_session(st.session_state)
            
            # Clear task
            st.session_state.current_task_id = None
            save_streamlit_session(st.session_state)
            
        elif task.status.value == "failed":
            st.markdown(f"""
            <div class="task-failed">
                ❌ <strong>Task Failed</strong><br>
                Error: {task.error}
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.current_task_id = None
            save_streamlit_session(st.session_state)

# Display chat history
for message in st.session_state.messages:
    role = 'user' if isinstance(message, HumanMessage) else 'assistant'
    with st.chat_message(role):
        if role == 'assistant' and message.content.startswith("Code:"):
            # Render code blocks properly
            code_content = message.content[len("Code:"):].strip()
            with st.expander("📝 View Generated Code", expanded=False):
                st.code(code_content.replace("```python", "").replace("```", ""), language="python")
        elif role == 'assistant' and message.content.startswith("**Output:**"):
            output_content = message.content[len("**Output:**"):].strip()
            with st.expander("💻 View Code Output", expanded=False):
                st.text(output_content.replace("```", ""))
        else:
            st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask me anything about your data..."):
    # Check if data is uploaded
    if not st.session_state.get("temp_file_path"):
        st.warning("⚠️ Please upload a CSV file first (see sidebar)")
        st.stop()
    
    # Verify file still exists
    file_path = Path(st.session_state.temp_file_path)
    if not file_path.exists():
        st.error(f"❌ Data file not found. Please re-upload your file.")
        st.session_state.uploaded_file_name = None
        st.session_state.temp_file_path = None
        save_streamlit_session(st.session_state)
        st.rerun()
    
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    save_streamlit_session(st.session_state)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show processing indicator
    with st.chat_message("assistant"):
        with st.spinner("🧠 Starting analysis in background..."):
            # Import here to avoid circular imports
            from ai.model import code_workflow
            
            # Build initial state
            initial_state = {
                "messages": st.session_state.messages,
                "input_data_path": st.session_state.temp_file_path,
                "stage_name": st.session_state.get("stage_name", "analysis"),
                "workflow_id": st.session_state.get("workflow_id", "analysis_workflow"),
                "rag_context": {},
                "can_answer_from_rag": False,
                "stage_metadata": {}
            }
            
            # Build execution context
            execution_context = {
                'executor': st.session_state.executor,
                'output_manager': st.session_state.output_mgr,
                'plot_cache': st.session_state.plot_cache,
                'rag': st.session_state.rag
            }
            
            # Submit task to background
            task_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task = task_mgr.submit_task(
                task_id,
                f"Query: {prompt[:50]}...",
                code_workflow.ainvoke,
                initial_state,
                context=execution_context
            )
            
            st.session_state.current_task_id = task_id
            save_streamlit_session(st.session_state)
            
            st.info("✅ Task started in background! You can navigate to other pages.")
            st.info("💡 Click 'Check Status' or return to this page to see results.")
    
    st.rerun()

# Instructions
if not st.session_state.messages:
    # Check if file is uploaded
    if st.session_state.get("uploaded_file_name"):
        st.success(f"""
        ✅ **Data file loaded: {st.session_state.uploaded_file_name}**
        
        You're all set! Ask me questions about your data below.
        
        **Examples:**
        - "Show me the distribution of all numeric columns"
        - "Create a correlation matrix"
        - "What are the top 10 records by [column name]?"
        - "Analyze trends in [column name] over time"
        """)
    else:
        st.info("""
        👋 **Welcome to the Data Science Agent!**
        
        **To get started:**
        1. 📤 Upload a CSV file in the sidebar
        2. 💬 Ask questions about your data below
        
        **What I can do:**
        - Generate visualizations and plots
        - Perform statistical analysis
        - Create correlation matrices
        - Analyze distributions and trends
        - Answer questions about your data
        
        **Processing happens in the background** - feel free to explore other pages!
        """)