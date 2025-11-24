"""
Data Science Agent - Multi-Page App
Main Landing Page / Dashboard
"""

import streamlit as st
from pathlib import Path
# import sys

# # Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).parent))

from session_utils import init_session_state, get_all_workflows
from session_persistence import save_streamlit_session

# Page configuration
st.set_page_config(
    page_title="Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .dashboard-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Header
st.markdown('<p class="main-header">🤖 Data Science Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered analysis with intelligent context retrieval and non-blocking workflows</p>', unsafe_allow_html=True)

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

workflows = get_all_workflows()
total_workflows = len(workflows)
current_workflow_id = st.session_state.get("workflow_id", "analysis_workflow")

# Get current workflow stats
current_workflow_stages = 0
total_outputs = 0
if st.session_state.get("output_mgr"):
    workflow_dir = st.session_state.output_mgr.workflow_dir
    if workflow_dir.exists():
        current_workflow_stages = len([d for d in workflow_dir.iterdir() if d.is_dir()])

# RAG stats
rag_docs = 0
if st.session_state.get("rag") and st.session_state.rag.enabled:
    rag_stats = st.session_state.rag.get_stats()
    rag_docs = rag_stats.get("total_chunks", 0)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_workflows}</div>
        <div class="metric-label">Total Workflows</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{current_workflow_stages}</div>
        <div class="metric-label">Current Stages</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(st.session_state.get('messages', []))}</div>
        <div class="metric-label">Chat Messages</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{rag_docs}</div>
        <div class="metric-label">RAG Documents</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Quick actions
st.markdown("### 🚀 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💬 Start Chat", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Chat.py")

with col2:
    if st.button("🎨 View Artifacts", use_container_width=True):
        st.switch_page("pages/2_Artifacts.py")

with col3:
    if st.button("📚 Browse Workflows", use_container_width=True):
        st.switch_page("pages/3_Workflows.py")

st.divider()

# Recent workflows
st.markdown("### 📊 Recent Workflows")

if workflows:
    for workflow in workflows[:5]:  # Show 5 most recent
        with st.expander(f"📁 {workflow['workflow_id']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stages", workflow['total_stages'])
            with col2:
                st.metric("Outputs", workflow['total_outputs'])
            with col3:
                st.text(f"Created: {workflow['created_at'][:10]}")
            
            if st.button(f"Load Workflow", key=f"load_{workflow['workflow_id']}"):
                from session_utils import switch_workflow
                switch_workflow(workflow['workflow_id'])
                save_streamlit_session(st.session_state)
                st.success(f"✅ Switched to {workflow['workflow_id']}")
                st.rerun()
else:
    st.info("No workflows found. Start a new analysis in the Chat!")

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ Navigation")
    st.markdown("""
    Use the pages in the sidebar to:
    - **💬 Chat**: Interact with the AI agent
    - **🎨 Artifacts**: Browse generated plots and files
    - **📚 Workflows**: Explore past analyses
    - **🧠 RAG Explorer**: Search stored context
    """)
    
    st.divider()
    
    st.markdown("### ⚙️ Current Workflow")
    st.text(f"ID: {current_workflow_id}")
    st.text(f"Stages: {current_workflow_stages}")
    
    if st.button("🔄 New Workflow", use_container_width=True):
        from datetime import datetime
        new_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        from session_utils import switch_workflow
        switch_workflow(new_id)
        st.session_state.messages = []
        save_streamlit_session(st.session_state)
        st.success(f"✅ Created {new_id}")
        st.rerun()