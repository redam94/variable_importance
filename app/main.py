from typing import Union
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from pathlib import Path
import httpx
import tempfile
import os
import asyncio
from model import code_workflow, OutputCapturingExecutor, OutputManager

# Page configuration
st.set_page_config(
    page_title="Data Science Agent",
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
    """Initialize all session state variables"""
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

def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        models = fetch_models()
        model = st.selectbox(
            "🤖 Model",
            options=models,
            index=0,
            help="Select the LLM model to use"
        )
        
        st.divider()
        
        # Workflow configuration
        st.markdown("### 📁 Workflow Settings")
        work_id = st.text_input(
            "Workflow ID",
            value=st.session_state.workflow_id,
            help="Unique identifier for this workflow"
        )
        
        stage_name = st.text_input(
            "Stage Name",
            value="analysis",
            help="Name of the current analysis stage"
        )
        
        # Initialize output manager
        if work_id and work_id != st.session_state.workflow_id:
            st.session_state.workflow_id = work_id
            st.session_state.output_mgr = OutputManager(workflow_id=work_id)
            st.success(f"✅ Workflow initialized: `{work_id}`")
        elif st.session_state.output_mgr is None:
            st.session_state.output_mgr = OutputManager(workflow_id=work_id)
        
        st.divider()
        
        # File upload
        st.markdown("### 📤 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload a CSV file for analysis"
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
            
            # Display file info
            st.markdown(f"""
            <div class="file-info">
                <strong>📄 {uploaded_file.name}</strong><br>
                <span style="font-size: 0.9rem; color: #666;">
                    Size: {uploaded_file.size / 1024:.2f} KB
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
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Statistics
        if st.session_state.output_mgr:
            st.divider()
            st.markdown("### 📊 Statistics")
            st.metric("Messages", len(st.session_state.messages))
            
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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🖼️ Plots", total_plots)
    with col2:
        st.metric("📊 Data Files", total_data_files)
    with col3:
        st.metric("💾 Code Files", total_code_files)
    
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
                            st.image(str(plot_file), caption=plot_file.name, use_container_width=True)
                            # Add download button
                            with open(plot_file, "rb") as f:
                                st.download_button(
                                    label="⬇️",
                                    data=f,
                                    file_name=plot_file.name,
                                    mime="image/png",
                                    key=f"download_{plot_file.stem}_{idx}"
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
                                    key=f"download_data_{data_file.stem}"
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
                                key=f"download_code_{code_file.stem}"
                            )
            
            # Console output
            console_file = stage_dir / "console_output.txt"
            if console_file.exists():
                with st.expander("💻 Console Output"):
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

async def process_user_message(prompt, temp_file_path, stage_name, output_mgr, executor):
    """Process user message and get agent response"""
    response = await code_workflow.ainvoke(
        {
            "messages": st.session_state.messages,
            "input_data_path": temp_file_path,
            "stage_name": stage_name
        },
        context={
            'executor': executor,
            'output_manager': output_mgr
        }
    )
    return response

def render_chat_interface(model, stage_name):
    """Render the main chat interface"""
    
    # Display chat history
    for message in st.session_state.messages:
        render_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Display user message
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get agent response
        with st.spinner("🤔 Analyzing..."):
            try:
                response = asyncio.run(process_user_message(
                    prompt,
                    st.session_state.temp_file_path,
                    stage_name,
                    st.session_state.output_mgr,
                    st.session_state.executor
                ))
                
                # Display code if generated
                if 'code' in response:
                    code = response['code']
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.expander("📝 Generated Code", expanded=True):
                            st.code(code, language="python")
                    st.session_state.messages.append(
                        AIMessage(content=f"Code:\n\n```python\n{code}\n```")
                    )
                
                # Display execution output
                if 'code_output' in response:
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
                
                # Show success message
                if 'code_output' in response and response['code_output'].success:
                    st.success("✅ Analysis completed successfully!")
                    # Automatically switch to artifacts tab hint
                    st.info("💡 View generated plots in the **Artifacts** tab")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("🔍 Debug Info"):
                    st.code(traceback.format_exc())

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🤖 Data Science Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered data analysis and visualization</p>', unsafe_allow_html=True)
    
    # Sidebar
    model, stage_name = render_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🎨 Artifacts"])
    
    with tab1:
        render_chat_interface(model, stage_name)
    
    with tab2:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh", use_container_width=True, key="refresh_artifacts"):
                st.rerun()
        render_artifacts_view(st.session_state.output_mgr, stage_name)

if __name__ == "__main__":
    main()