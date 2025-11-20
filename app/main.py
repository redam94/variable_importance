from typing import Union
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from pathlib import Path
import httpx
import tempfile
import os
import asyncio
from model import code_workflow
from variable_importance.utils.code_executer import OutputCapturingExecutor 
from variable_importance.utils.output_manager import OutputManager
from plot_analysis_cache import PlotAnalysisCache
from context_rag import ContextRAG
from session_persistence import SessionPersistence, save_streamlit_session, load_streamlit_session

# Page configuration
st.set_page_config(
    page_title="Data Science Agent with RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .file-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stage-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .artifact-count {
        font-size: 0.9rem;
        color: #666;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .rag-status {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .rag-disabled {
        background-color: #ffebee;
    }
    .session-restored {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def fetch_models():
    """Fetch available models from Ollama"""
    try:
        url = "http://100.91.155.118:11434/v1/models"
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            return [model['id'] for model in response.json()['data']]
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
    return ["gpt-oss:20b"]  # Default fallback

def initialize_session_state():
    """
    Initialize all session state variables including RAG and caching.
    ENHANCED: Now loads from persistence first!
    """
    # Mark that we're initializing
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    # Try to restore from saved session first (only on first run)
    if not st.session_state.initialized:
        restored = load_streamlit_session(st.session_state)
        if restored:
            st.session_state.session_restored = True
            st.session_state.initialized = True
            
            # Recreate objects that weren't persisted
            if "executor" not in st.session_state:
                st.session_state.executor = OutputCapturingExecutor()
            if "plot_cache" not in st.session_state:
                st.session_state.plot_cache = PlotAnalysisCache()
            
            if "uploaded_file_size" not in st.session_state:
                st.session_state.uploaded_file_size = 0
            # Recreate output_mgr and rag based on workflow_id
            if "workflow_id" in st.session_state:
                if "output_mgr" not in st.session_state or st.session_state.output_mgr is None:
                    st.session_state.output_mgr = OutputManager(workflow_id=st.session_state.workflow_id)
                if "rag" not in st.session_state or st.session_state.rag is None:
                    st.session_state.rag = ContextRAG(
                        collection_name=st.session_state.workflow_id,
                        persist_directory="cache/rag_db"
                    )
            
            return
    
    # Initialize defaults if not restored
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "executor" not in st.session_state:
        st.session_state.executor = OutputCapturingExecutor()
    if "output_mgr" not in st.session_state:
        st.session_state.output_mgr = None
    if "temp_file_path" not in st.session_state:
        st.session_state.temp_file_path = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "workflow_id" not in st.session_state:
        st.session_state.workflow_id = "analysis_workflow"
    if "session_restored" not in st.session_state:
        st.session_state.session_restored = False
    
    # Initialize RAG and caching systems
    if "plot_cache" not in st.session_state:
        st.session_state.plot_cache = PlotAnalysisCache()
    if "rag" not in st.session_state:
        st.session_state.rag = ContextRAG(
            collection_name=st.session_state.workflow_id,
            persist_directory="cache/rag_db"
        )
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True
    
    st.session_state.initialized = True

def save_session_state():
    """
    Save current session state to disk.
    Call this after significant state changes.
    """
    save_streamlit_session(st.session_state)

def render_sidebar():
    """Render sidebar with configuration options including RAG controls"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        models = fetch_models()
        current_model = st.session_state.get("selected_model", models[0])
        model = st.selectbox(
            "🤖 Model",
            options=models,
            index=models.index(current_model) if current_model in models else 0,
            help="Select the LLM model to use",
            key="model_selector"
        )
        
        # Save model selection
        if model != st.session_state.get("selected_model"):
            st.session_state.selected_model = model
            save_session_state()
        
        st.divider()
        
        # Workflow configuration
        st.markdown("### 📁 Workflow Settings")
        work_id = st.text_input(
            "Workflow ID",
            value=st.session_state.workflow_id,
            help="Unique identifier for this workflow. RAG context is isolated per workflow.",
            key="workflow_id_input"
        )
        
        stage_name = st.text_input(
            "Stage Name",
            value=st.session_state.get("stage_name", "analysis"),
            help="Name of the current analysis stage",
            key="stage_name_input"
        )
        
        # Save stage name
        if stage_name != st.session_state.get("stage_name"):
            st.session_state.stage_name = stage_name
            save_session_state()
        
        # Initialize/update workflow components
        if work_id and work_id != st.session_state.workflow_id:
            st.session_state.workflow_id = work_id
            st.session_state.output_mgr = OutputManager(workflow_id=work_id)
            
            # Update RAG collection for new workflow
            st.session_state.rag = ContextRAG(
                collection_name=work_id,
                persist_directory="cache/rag_db"
            )
            save_session_state()
            st.success(f"✅ Workflow initialized: `{work_id}`")
            
        elif st.session_state.output_mgr is None:
            st.session_state.output_mgr = OutputManager(workflow_id=work_id)
            save_session_state()
        
        st.divider()
        
        # RAG System Controls
        st.markdown("### 🧠 RAG System")
        
        # RAG status
        if st.session_state.rag and st.session_state.rag.enabled:
            rag_stats = st.session_state.rag.get_stats()
            total_docs = rag_stats.get("total_documents", 0)
            
            st.markdown(f"""
            <div class="rag-status">
                ✅ RAG Active | {total_docs} documents
            </div>
            """, unsafe_allow_html=True)
            
            # Show document type breakdown
            if rag_stats.get("type_breakdown"):
                with st.expander("📊 Document Types", expanded=False):
                    for doc_type, count in rag_stats["type_breakdown"].items():
                        st.text(f"{doc_type}: {count}")
        else:
            st.markdown("""
            <div class="rag-status rag-disabled">
                ❌ RAG Disabled
            </div>
            """, unsafe_allow_html=True)
        
        # RAG controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear RAG", use_container_width=True, help="Clear all RAG documents for this workflow"):
                if st.session_state.rag:
                    st.session_state.rag.delete_by_workflow(st.session_state.workflow_id)
                    save_session_state()
                    st.success("RAG cleared!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True, help="Clear plot analysis cache"):
                st.session_state.plot_cache.clear()
                save_session_state()
                st.success("Cache cleared!")
                st.rerun()
        
        st.divider()
        
        # File upload
        st.markdown("### 📤 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload a CSV file for analysis",
            key="_file_uploader"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            if st.session_state.uploaded_file_name != uploaded_file.name:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    st.session_state.temp_file_path = temp_file.name
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                    save_session_state()
            
            # Display file info
            st.markdown(f"""
            <div class="file-info">
                <strong>📄 {st.session_state.uploaded_file_name}</strong><br>
                <span style="font-size: 0.9rem; color: #666;">
                    Size: {uploaded_file.size/ 1024:.2f} KB
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Actions
        st.markdown("### 🎛️ Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                save_session_state()
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Session management
        col3, col4 = st.columns(2)
        with col3:
            if st.button("💾 Save Session", use_container_width=True, help="Manually save current session"):
                save_session_state()
                st.success("Session saved!")
        with col4:
            if st.button("🔄 New Session", use_container_width=True, help="Start a new session"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key == '_file_uploader':
                        continue
                    del st.session_state[key]
                st.rerun()
        
        # Statistics
        if st.session_state.output_mgr:
            st.divider()
            st.markdown("### 📊 Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", len(st.session_state.messages))
            with col2:
                cache_stats = st.session_state.plot_cache.get_stats()
                st.metric("Cached Plots", cache_stats["total_entries"])
            
            workflow_dir = st.session_state.output_mgr.workflow_dir
            if workflow_dir.exists():
                stage_count = len([d for d in workflow_dir.iterdir() if d.is_dir()])
                st.metric("Stages", stage_count)
        
        return model, stage_name

def render_chat_message(message):
    """Render a single chat message with proper formatting"""
    role = 'user' if isinstance(message, HumanMessage) else 'assistant'
    content = message.content
    
    if role == 'user':
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Check for RAG context indicator
            if "=== Answer based on previous analysis ===" in content:
                st.info("📚 This answer is based on previous analysis results stored in RAG")
            
            # Check if this is a code message
            if "```python" in content and "Code:" in content:
                with st.expander("📝 Generated Code", expanded=False):
                    code_content = content.split("```python\n")[1].split("```")[0]
                    st.code(code_content, language="python")
            # Check if this is output
            elif "**Stdout:**" in content or "**Errors:**" in content:
                with st.expander("💻 Execution Output", expanded=False):
                    st.markdown(content)
            # Regular message
            else:
                st.markdown(content)

def render_artifacts_view(output_mgr, stage_name):
    """Render the artifacts viewer with organized display"""
    st.markdown("### 🎨 Generated Artifacts")
    
    if output_mgr is None:
        st.info("Initialize a workflow to view artifacts")
        return
    
    # Get workflow directory
    workflow_dir = output_mgr.workflow_dir
    
    if not workflow_dir.exists():
        st.warning("No artifacts found. Run an analysis first.")
        return
    
    # Find all stage directories
    stage_dirs = sorted([d for d in workflow_dir.iterdir() if d.is_dir()])
    
    if not stage_dirs:
        st.warning("No stage directories found.")
        return
    
    # Summary metrics
    total_plots = 0
    total_data_files = 0
    total_code_files = 0
    
    for stage_dir in stage_dirs:
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            total_plots += len(list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg")))
        
        data_dir = stage_dir / "data"
        if data_dir.exists():
            total_data_files += len(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json")))
        
        total_code_files += len(list(stage_dir.glob("*.py")))
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🖼️ Plots", total_plots)
    with col2:
        st.metric("📊 Data Files", total_data_files)
    with col3:
        st.metric("💾 Code Files", total_code_files)
    with col4:
        cache_stats = st.session_state.plot_cache.get_stats()
        st.metric("⚡ Cached", cache_stats["total_entries"])
    
    st.divider()
    
    # Display stages
    for stage_dir in stage_dirs:
        stage_display_name = stage_dir.name
        
        with st.expander(f"📂 {stage_display_name}", expanded=(stage_dir.name.endswith(stage_name))):
            # Plots
            plots_dir = stage_dir / "plots"
            if plots_dir.exists():
                plot_files = sorted(list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg")))
                
                if plot_files:
                    st.markdown(f"**🖼️ Plots ({len(plot_files)})**")
                    
                    # Display plots in columns
                    cols = st.columns(3)
                    for idx, plot_file in enumerate(plot_files):
                        with cols[idx % 3]:
                            # Check if cached
                            is_cached = st.session_state.plot_cache.get(str(plot_file)) is not None
                            if is_cached:
                                st.caption("⚡ Cached")
                            
                            st.image(str(plot_file), caption=plot_file.name, use_container_width=True)
                            
                            # Add download button
                            with open(plot_file, "rb") as f:
                                st.download_button(
                                    label="⬇️",
                                    data=f,
                                    file_name=plot_file.name,
                                    mime="image/png",
                                    key=f"_download_{stage_display_name}_{plot_file.stem}_{idx}"
                                )
                else:
                    st.info("No plots generated")
            
            # Data files
            data_dir = stage_dir / "data"
            if data_dir.exists():
                data_files = sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json")))
                
                if data_files:
                    st.markdown(f"**📊 Data Files ({len(data_files)})**")
                    for data_file in data_files:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"📄 {data_file.name}")
                        with col2:
                            with open(data_file, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download",
                                    data=f,
                                    file_name=data_file.name,
                                    key=f"_download_data_{stage_display_name}_{data_file.stem}"
                                )
            
            # Code
            code_files = sorted(list(stage_dir.glob("*.py")))
            if code_files:
                st.markdown(f"**💾 Code Files ({len(code_files)})**")
                for code_file in code_files:
                    with st.expander(f"📝 {code_file.name}"):
                        with open(code_file, 'r') as f:
                            code_content = f.read()
                            st.code(code_content, language="python")
                            st.download_button(
                                label="⬇️ Download Code",
                                data=code_content,
                                file_name=code_file.name,
                                key=f"_download_code_{stage_display_name}_{code_file.stem}"
                            )
            
            # Console output
            console_files = sorted(list(stage_dir.glob("console_output_*.txt")))
            if console_files:
                for file_idx, console_file in enumerate(console_files):
                    
                    with st.expander(f"💻 {console_file.stem.title()}"):
                        with open(console_file, 'r') as f:
                            st.text(f.read())
            
            # Execution info
            exec_info_file = stage_dir / "execution_info.json"
            if exec_info_file.exists():
                import json
                with open(exec_info_file, 'r') as f:
                    exec_info = json.load(f)
                
                with st.expander("ℹ️ Execution Info"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Success:** {'✅' if exec_info.get('success') else '❌'}")
                        st.markdown(f"**Execution Time:** {exec_info.get('execution_time_seconds', 0):.2f}s")
                    with col2:
                        st.markdown(f"**Generated Files:** {len(exec_info.get('generated_files', []))}")
                        if exec_info.get('error'):
                            st.error(f"Error: {exec_info['error']}")

async def process_user_message(prompt, temp_file_path, stage_name, output_mgr, executor, plot_cache, rag, workflow_id):
    """Process user message with enhanced RAG and caching context"""
    
    # Build initial state
    initial_state = {
        "messages": st.session_state.messages,
        "input_data_path": temp_file_path,
        "stage_name": stage_name,
        "workflow_id": workflow_id,
        "rag_context": {},
        "can_answer_from_rag": False,
        "stage_metadata": {}
    }
    
    # Build execution context with all components
    execution_context = {
        'executor': executor,
        'output_manager': output_mgr,
        'plot_cache': plot_cache,
        'rag': rag
    }
    
    response = await code_workflow.ainvoke(
        initial_state,
        context=execution_context
    )
    return response

def render_rag_insights():
    """Render insights about RAG usage"""
    if not st.session_state.rag or not st.session_state.rag.enabled:
        return
    
    with st.expander("🧠 RAG Insights", expanded=False):
        rag_stats = st.session_state.rag.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", rag_stats.get("total_documents", 0))
        with col2:
            plot_docs = rag_stats.get("type_breakdown", {}).get("plot_analysis", 0)
            st.metric("Plot Analyses", plot_docs)
        with col3:
            code_docs = rag_stats.get("type_breakdown", {}).get("code_execution", 0)
            st.metric("Code Executions", code_docs)
        
        # Show recent queries
        st.markdown("**Recent Context Usage:**")
        st.info("RAG system stores and retrieves context across queries within this workflow")

def render_chat_interface(model, stage_name):
    """Render the main chat interface with RAG awareness"""
    
    # Show session restored message
    if st.session_state.get("session_restored", False):
        st.markdown(f"""
        <div class="session-restored">
            ✅ Session restored! Your chat history and settings have been recovered.
        </div>
        """, unsafe_allow_html=True)
        # Clear the flag after showing once
        st.session_state.session_restored = False
    
    # Show RAG insights at the top
    render_rag_insights()
    
    # Display chat history
    for message in st.session_state.messages:
        render_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Display user message
        st.session_state.messages.append(HumanMessage(content=prompt))
        save_session_state()  # Save after adding message
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get agent response with RAG context
        with st.spinner("🧠 Checking existing context and analyzing..."):
            try:
                response = asyncio.run(process_user_message(
                    prompt,
                    st.session_state.temp_file_path,
                    stage_name,
                    st.session_state.output_mgr,
                    st.session_state.executor,
                    st.session_state.plot_cache,
                    st.session_state.rag,
                    st.session_state.workflow_id
                ))
                
                # Check if answer came from RAG
                if response.get('can_answer_from_rag'):
                    with st.chat_message("assistant", avatar="🤖"):
                        st.success("📚 Found relevant context from previous analysis!")
                
                # Display code if generated
                if 'code' in response and response['code']:
                    code = response['code']
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.expander("📝 Generated Code", expanded=True):
                            st.code(code, language="python")
                    st.session_state.messages.append(
                        AIMessage(content=f"Code:\n\n```python\n{code}\n```")
                    )
                
                # Display execution output
                if 'code_output' in response and response['code_output']:
                    code_output = response['code_output']
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.expander("💻 Execution Output", expanded=False):
                            if code_output.stdout:
                                st.markdown("**Output:**")
                                st.code(code_output.stdout)
                            if code_output.error:
                                st.markdown("**Errors:**")
                                st.code(code_output.error)
                    
                    st.session_state.messages.append(
                        AIMessage(content=f"**Stdout:**\n\n```\n{code_output.stdout}\n```\n\n**Errors:**\n```\n{code_output.error}\n```")
                    )
                
                # Display summary
                summary = response.get('summary', 'Analysis complete.')
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(summary)
                st.session_state.messages.append(AIMessage(content=summary))
                
                # Save session after all responses
                save_session_state()
                
                # Show success message with appropriate context
                if response.get('can_answer_from_rag'):
                    st.info("✅ Answered using existing context from RAG (no code execution needed)")
                elif 'code_output' in response and response['code_output'] and response['code_output'].success:
                    st.success("✅ Analysis completed successfully!")
                    st.info("💡 Results stored in RAG for future queries. View plots in **Artifacts** tab.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("🔍 Debug Info"):
                    st.code(traceback.format_exc())

def main():
    """Main application with enhanced RAG capabilities and session persistence"""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🤖 Data Science Agent with RAG</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis with intelligent context retrieval and persistent sessions</p>', unsafe_allow_html=True)
    
    # Sidebar
    model, stage_name = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎨 Artifacts", "🧠 RAG Explorer"])
    
    with tab1:
        render_chat_interface(model, stage_name)
    
    with tab2:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh", use_container_width=True, key="_refresh_artifacts"):
                st.rerun()
        render_artifacts_view(st.session_state.output_mgr, stage_name)
    
    with tab3:
        st.markdown("### 🧠 RAG Context Explorer")
        
        if st.session_state.rag and st.session_state.rag.enabled:
            # Query interface
            query = st.text_input("Search RAG context:", placeholder="Enter a query to search stored context...")
            
            if query:
                with st.spinner("Searching..."):
                    contexts = st.session_state.rag.query_relevant_context(
                        query=query,
                        workflow_id=st.session_state.workflow_id,
                        n_results=5
                    )
                    
                    if contexts:
                        st.success(f"Found {len(contexts)} relevant documents")
                        for ctx in contexts:
                            with st.expander(f"📄 {ctx['metadata'].get('type', 'document')}: {ctx['metadata'].get('stage_name', 'unknown')}"):
                                st.markdown("**Content:**")
                                st.text(ctx['document'][:500])
                                st.markdown("**Metadata:**")
                                st.json(ctx['metadata'])
                    else:
                        st.warning("No relevant context found")
            
            # RAG statistics
            st.divider()
            rag_stats = st.session_state.rag.get_stats()
            st.json(rag_stats)
        else:
            st.error("RAG system is not enabled")

if __name__ == "__main__":
    main()