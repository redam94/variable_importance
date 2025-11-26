"""
Code Editor & Executor Page

Interactive code editor with execution capabilities.
Features:
- Syntax-highlighted code editing
- Execute code in isolated environment
- Save/load code snippets
- View execution results and outputs
- Integration with session persistence
"""

import streamlit as st
import asyncio
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from variable_importance.utils.code_executer import OutputCapturingExecutor
from variable_importance.utils.output_manager import OutputManager
from session_persistence import save_streamlit_session
from session_utils import init_session_state

# Page configuration
st.set_page_config(
    page_title="Code Editor",
    page_icon="💻",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .code-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .execution-success {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
    }
    .execution-error {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin-bottom: 1rem;
    }
    .snippet-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .snippet-card:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


def initialize_code_editor_state():
    """Initialize session state for code editor."""
    if "code_editor_initialized" not in st.session_state:
        st.session_state.code_editor_initialized = True
    
    if "current_code" not in st.session_state:
        st.session_state.current_code = """import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Your code here
print("Hello from Code Editor!")
"""
    
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    
    if "saved_snippets" not in st.session_state:
        st.session_state.saved_snippets = {}
    
    if "code_executor" not in st.session_state:
        st.session_state.code_executor = OutputCapturingExecutor()


def save_code_snippet(name: str, code: str, description: str = ""):
    """Save a code snippet to session state."""
    st.session_state.saved_snippets[name] = {
        "code": code,
        "description": description,
        "saved_at": datetime.now().isoformat()
    }
    save_streamlit_session(st.session_state)
    logger.info(f"💾 Saved code snippet: {name}")


def load_code_snippet(name: str) -> str:
    """Load a code snippet from session state."""
    if name in st.session_state.saved_snippets:
        return st.session_state.saved_snippets[name]["code"]
    return ""


async def execute_code_async(code: str, working_dir: Path = None):
    """Execute code asynchronously and return results."""
    executor = st.session_state.code_executor
    
    if working_dir is None:
        working_dir = Path("cache/code_editor_outputs")
        working_dir.mkdir(parents=True, exist_ok=True)
    
    result = await executor.execute_code(
        code=code,
        working_dir=working_dir
    )
    
    # Add to history
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "code": code[:100] + "..." if len(code) > 100 else code,
        "success": result.success,
        "execution_time": result.execution_time_seconds,
        "has_output": bool(result.stdout),
        "has_error": bool(result.stderr)
    }
    st.session_state.execution_history.insert(0, history_entry)
    
    # Keep only last 10 executions
    if len(st.session_state.execution_history) > 10:
        st.session_state.execution_history = st.session_state.execution_history[:10]
    
    save_streamlit_session(st.session_state)
    
    return result


def render_sidebar():
    """Render sidebar with saved snippets and history."""
    with st.sidebar:
        st.markdown("### 💾 Saved Snippets")
        
        # Add new snippet
        with st.expander("➕ Save Current Code", expanded=False):
            snippet_name = st.text_input("Snippet Name", key="snippet_name_input")
            snippet_desc = st.text_area("Description (optional)", key="snippet_desc_input", height=80)
            
            if st.button("Save Snippet", use_container_width=True):
                if snippet_name:
                    save_code_snippet(
                        snippet_name,
                        st.session_state.current_code,
                        snippet_desc
                    )
                    st.success(f"✅ Saved: {snippet_name}")
                    st.rerun()
                else:
                    st.error("Please enter a snippet name")
        
        # Display saved snippets
        if st.session_state.saved_snippets:
            st.divider()
            for name, snippet in st.session_state.saved_snippets.items():
                with st.expander(f"📄 {name}"):
                    st.caption(snippet.get("description", "No description"))
                    st.caption(f"Saved: {snippet['saved_at'][:10]}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load", key=f"load_{name}", use_container_width=True):
                            st.session_state.current_code = snippet["code"]
                            st.success(f"✅ Loaded: {name}")
                            st.rerun()
                    with col2:
                        if st.button("Delete", key=f"delete_{name}", use_container_width=True):
                            del st.session_state.saved_snippets[name]
                            save_streamlit_session(st.session_state)
                            st.success(f"🗑️ Deleted: {name}")
                            st.rerun()
        else:
            st.info("No saved snippets yet")
        
        # Execution history
        st.divider()
        st.markdown("### 📊 Execution History")
        
        if st.session_state.execution_history:
            for i, entry in enumerate(st.session_state.execution_history[:5], 1):
                status = "✅" if entry["success"] else "❌"
                time_str = entry["timestamp"][11:19]  # HH:MM:SS
                st.markdown(f"""
                <div class="snippet-card">
                    {status} {time_str}<br>
                    <small>{entry['execution_time']:.2f}s</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No executions yet")
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.execution_history = []
            save_streamlit_session(st.session_state)
            st.rerun()


def render_code_templates():
    """Render quick code templates."""
    st.markdown("### 📋 Quick Templates")
    
    templates = {
        "Data Analysis": """import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('your_data.csv')

# Display basic info
print(df.head())
print(df.describe())
""",
        "Visualization": """import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sample Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig("plot.png")
plt.close()

print("Plot saved!")
""",
        "Statistics": """import pandas as pd
import numpy as np
from scipy import stats

# Sample data
data = np.random.randn(100)

# Statistical measures
print(f"Mean: {np.mean(data):.4f}")
print(f"Median: {np.median(data):.4f}")
print(f"Std Dev: {np.std(data):.4f}")

# Hypothesis testing
t_stat, p_value = stats.ttest_1samp(data, 0)
print(f"\\nT-test: t={t_stat:.4f}, p={p_value:.4f}")
"""
    }
    
    cols = st.columns(len(templates))
    for col, (name, code) in zip(cols, templates.items()):
        with col:
            if st.button(name, use_container_width=True):
                st.session_state.code_editor_area = code
                st.rerun()


def render_execution_result(result):
    """Render execution result with formatted output."""
    if result.success:
        st.markdown("""
        <div class="execution-success">
            ✅ Execution Successful
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="execution-error">
            ❌ Execution Failed
        </div>
        """, unsafe_allow_html=True)
    
    # Execution info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Execution Time", f"{result.execution_time_seconds:.2f}s")
    with col2:
        st.metric("Generated Files", len(result.generated_files))
    with col3:
        status = "Success" if result.success else "Failed"
        st.metric("Status", status)
    
    # Output tabs
    tab1, tab2, tab3 = st.tabs(["📄 Output", "❌ Errors", "📁 Generated Files"])
    
    with tab1:
        if result.stdout:
            st.code(result.stdout, language="text")
        else:
            st.info("No output")
    
    with tab2:
        if result.stderr:
            st.code(result.stderr, language="text")
        elif result.error:
            st.code(result.error, language="text")
        else:
            st.success("No errors")
    
    with tab3:
        if result.generated_files:
            st.markdown(f"**Generated {len(result.generated_files)} files:**")
            for file in result.generated_files:
                st.text(f"📄 {file}")
                
                # Try to display images
                file_path = Path(result.working_dir) / file
                if file_path.exists() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    st.image(str(file_path), caption=file, use_container_width=True)
        else:
            st.info("No files generated")


def main():
    """Main code editor page."""
    init_session_state()
    initialize_code_editor_state()
    
    # Header
    st.markdown('<p class="code-header">💻 Code Editor & Executor</p>', unsafe_allow_html=True)
    st.markdown("Write and execute Python code with instant feedback")
    
    # Sidebar
    render_sidebar()
    
    # Quick templates
    with st.expander("📋 Quick Templates", expanded=False):
        render_code_templates()
    
    # Code editor
    st.markdown("### ✏️ Code Editor")
    st.session_state.code_editor_area = st.session_state.current_code
    code = st.text_area(
        "Write your Python code:",
        height=400,
        key="code_editor_area",
        help="Write Python code to execute. Libraries like pandas, numpy, matplotlib are available."
    )
    
    # Update current code
    st.session_state.current_code = code
    
    # Execution controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        execute_button = st.button("▶️ Execute Code", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear Code", use_container_width=True):
            st.session_state.current_code = "# Your code here\n"
            st.rerun()
    
    with col3:
        if st.button("💾 Quick Save", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_code_snippet(
                f"auto_{timestamp}",
                st.session_state.current_code,
                "Auto-saved code"
            )
            st.success("✅ Quick saved!")
    
    with col4:
        if st.button("🏠 Back to Main", use_container_width=True):
            st.switch_page("Dashboard.py")
    
    # Execute code
    if execute_button:
        if not code.strip():
            st.warning("⚠️ Please enter some code to execute")
        else:
            with st.spinner("🔄 Executing code..."):
                try:

                    selected_stage = st.session_state.get("current_stage", "analysis")
                    working_dir = Path(st.session_state.output_mgr.workflow_dir) / selected_stage / 'execution'
                    result = asyncio.run(execute_code_async(code, working_dir=working_dir))
                    
                    st.divider()
                    st.markdown("### 📊 Execution Results")
                    render_execution_result(result)
                    
                except Exception as e:
                    st.error(f"❌ Execution error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Debug Info"):
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()