"""
Chat Page - Non-Blocking AI Interaction with Real-Time Progress

Features:
- Background task execution
- Web search toggle for methodology research
- Auto-refreshing progress updates (no manual refresh needed)
- Intermediate output display
"""

import streamlit as st
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime
import httpx
from loguru import logger
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import init_session_state
from utils.background_tasks import get_task_manager, TaskStatus
from utils.progress_events import get_reader, ProgressReader
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
    .status-running { 
        background: #fff3cd; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 4px solid #ffc107; 
        margin-bottom: 1rem; 
    }
    .status-done { 
        background: #d4edda; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 4px solid #28a745; 
        margin-bottom: 1rem; 
    }
    .status-error { 
        background: #f8d7da; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 4px solid #dc3545; 
        margin-bottom: 1rem; 
    }
    .progress-event {
        padding: 0.3rem 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        background: #f8f9fa;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .stage-running { background: #ffc107; color: #000; }
    .stage-completed { background: #28a745; color: #fff; }
    .stage-failed { background: #dc3545; color: #fff; }
</style>
""", unsafe_allow_html=True)

st.title("💬 Chat with AI Agent")


@st.fragment(run_every=2)  # Auto-refresh every 2 seconds
def auto_refresh_progress():
    """Auto-refreshing progress display fragment."""
    task_id = st.session_state.get("current_task_id")
    if not task_id:
        return
    
    task_mgr = get_task_manager()
    task = task_mgr.get_task(task_id)
    
    if not task:
        return
    
    # Check if task is still running
    if task.status == TaskStatus.RUNNING:
        display_progress_events(task_id, is_running=True)
    elif task.status == TaskStatus.COMPLETED:
        # Task completed - trigger full page rerun to show results
        st.session_state._task_just_completed = True
        st.rerun()
    elif task.status == TaskStatus.FAILED:
        # Task failed - trigger full page rerun
        st.session_state._task_just_failed = True
        st.rerun()


def display_progress_events(task_id: str, is_running: bool = False):
    """Display progress events for a task."""
    reader = get_reader(task_id)
    status = reader.get_latest_status()
    
    if not status:
        if is_running:
            st.info("⏳ Initializing workflow...")
        return
    
    # Stage status badges
    stage_status = status.get("stage_status", {})
    
    st.markdown("**Workflow Progress:**")
    cols = st.columns(4)
    stage_order = ["gather_context", "plan_and_decide", "execute", "summarize"]
    
    # Also check for "answer" stage (alternative to execute)
    if "answer" in stage_status and "execute" not in stage_status:
        stage_order[2] = "answer"
    
    for i, stage in enumerate(stage_order):
        status_val = stage_status.get(stage, "pending")
        
        if status_val == "running":
            badge = "🔄"
            color = "#ffc107"
        elif status_val == "completed":
            badge = "✅"
            color = "#28a745"
        elif status_val == "failed":
            badge = "❌"
            color = "#dc3545"
        else:
            badge = "⏳"
            color = "#6c757d"
        
        display_name = stage.replace('_', ' ').title()
        if stage == "gather_context":
            display_name = "Context"
        elif stage == "plan_and_decide":
            display_name = "Plan"
        
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: {color}20; border-radius: 0.5rem; border: 2px solid {color}40;">
                <div style="font-size: 1.5rem;">{badge}</div>
                <div style="font-size: 0.7rem; font-weight: 500;">{display_name}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Latest events
    events = reader.get_all_events()
    if events:
        # Current activity indicator
        latest = events[-1]
        current_msg = latest.get("message", "Processing...")
        
        if is_running:
            st.markdown(f"**Current:** {current_msg}")
        
        with st.expander(f"📋 Activity Log ({len(events)} events)", expanded=is_running):
            # Show last 15 events in reverse order
            for event in reversed(events[-15:]):
                event_type = event.get("event_type", "")
                message = event.get("message", "")
                stage = event.get("stage", "")
                timestamp = event.get("timestamp", "")[:19]
                
                # Icon based on event type
                icon_map = {
                    "stage_start": "🚀",
                    "stage_end": "🏁",
                    "progress": "📍",
                    "code_generated": "✍️",
                    "code_executing": "⚡",
                    "code_output": "💻",
                    "error": "❌",
                    "rag_query": "📚",
                    "web_search": "🌐",
                    "plot_analyzed": "📊",
                    "summary": "📝",
                }
                icon = icon_map.get(event_type, "•")
                
                st.markdown(f"""
                <div class="progress-event">
                    {icon} <strong>[{stage}]</strong> {message}
                    <span style="float: right; color: #6c757d; font-size: 0.75rem;">{timestamp[11:]}</span>
                </div>
                """, unsafe_allow_html=True)


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
    
    web_search = st.toggle(
        "🌐 Web Search (DuckDuckGo)", 
        value=st.session_state.get("web_search_enabled", False),
        help="Search the web for methodology guidance using DuckDuckGo"
    )
    st.session_state.web_search_enabled = web_search
    
    rag = st.toggle(
        "📚 Use History (RAG)", 
        value=st.session_state.get("rag_enabled", True),
        help="Use context from previous analyses"
    )
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
            if task.status == TaskStatus.RUNNING:
                st.info("⏳ Running... (auto-updating)")
            elif task.status == TaskStatus.COMPLETED:
                st.success("✅ Done!")
            elif task.status == TaskStatus.FAILED:
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
        if task.status == TaskStatus.RUNNING:
            st.markdown(f'<div class="status-running">⏳ <b>Processing:</b> {task.description}</div>', unsafe_allow_html=True)
            logger.info(f"⏳ Task running: {task.task_id} - {task.description}")
            # Auto-refreshing progress display
            auto_refresh_progress()
            
        elif task.status == TaskStatus.COMPLETED and task.result:
            st.markdown('<div class="status-done">✅ <b>Complete!</b></div>', unsafe_allow_html=True)
            logger.info(f"✅ Task completed: {task.task_id}")
            result = task.result
            
            # Show final progress summary (static, not auto-refresh)
            display_progress_events(task_id, is_running=False)
            
            # Debug: show result structure
            with st.expander("🔍 Debug: Result Structure", expanded=False):
                st.write(f"**Result keys:** {list(result.keys())}")
                if result.get("summary"):
                    st.write(f"**Summary length:** {len(result.get('summary', ''))} chars")
                    st.text(result.get("summary", "")[:500] + "..." if len(result.get("summary", "")) > 500 else result.get("summary", ""))
                if result.get("code"):
                    st.write(f"**Code length:** {len(result.get('code', ''))} chars")
                if result.get("error"):
                    st.error(f"**Error:** {result.get('error')}")
            
            # Check if we've already processed this task's results
            # Use a more robust check: look for task ID marker or the actual summary
            task_marker = f"<!-- task:{task_id} -->"
            already_processed = any(
                isinstance(m, AIMessage) and task_marker in m.content 
                for m in st.session_state.messages[-10:]
            )
            
            if not already_processed:
                logger.info(f"🆕 Processing results for task: {task.task_id}")
                # Add code if present
                if result.get("code"):
                    code_msg = f"**Generated Code:**\n```python\n{result['code']}\n```"
                    st.session_state.messages.append(AIMessage(content=code_msg))
                
                # Add output if present
                output = result.get("output")
                if output and hasattr(output, 'stdout') and output.stdout:
                    output_msg = f"**Output:**\n```\n{output.stdout[:2000]}\n```"
                    st.session_state.messages.append(AIMessage(content=output_msg))
                
                # Add summary (always expected)
                summary = result.get("summary")
                if summary:
                    # Include hidden task marker to prevent duplicates
                    summary_msg = f"{task_marker}\n{summary}"
                    st.session_state.messages.append(AIMessage(content=summary_msg))
                    logger.info(f"✅ Added summary to chat ({len(summary)} chars)")
                else:
                    # No summary - log for debugging
                    logger.warning(f"⚠️ Task completed but no summary in result. Keys: {list(result.keys())}")
                    st.session_state.messages.append(AIMessage(content=f"{task_marker}\n⚠️ Analysis completed but no summary was generated. Please try again."))
            
            # Clear task ID BEFORE saving so it persists
            st.session_state.current_task_id = None
            save_streamlit_session(st.session_state)
            
        elif task.status == TaskStatus.FAILED:
            st.markdown(f'<div class="status-error">❌ <b>Failed:</b> {task.error}</div>', unsafe_allow_html=True)
            
            # Show progress events even on failure
            display_progress_events(task_id, is_running=False)
            
            # Clear task ID and save
            st.session_state.current_task_id = None
            save_streamlit_session(st.session_state)

# Display messages
for msg in st.session_state.messages:
    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
    with st.chat_message(role):
        content = msg.content
        
        # Strip hidden task markers
        if "<!-- task:" in content:
            content = "\n".join(line for line in content.split("\n") if not line.startswith("<!-- task:"))
        
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
            from ai import Deps
            from ai.workflow_v2 import workflow
            from utils.progress_events import get_emitter
            
            # Create unique task ID
            task_id = f"q_{datetime.now().strftime('%H%M%S')}"
            
            # Create progress emitter for this task
            progress_emitter = get_emitter(task_id)
            
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
                "progress_emitter": progress_emitter,  # Add progress emitter
            }
            
            task_mgr.submit_task(
                task_id,
                f"Query: {prompt[:40]}...",
                workflow.ainvoke,
                initial_state,
                context=deps
            )
            
            st.session_state.current_task_id = task_id
            save_streamlit_session(st.session_state)
            st.info("✅ Started! You can navigate to other pages. Progress will appear above.")
    
    st.rerun()

# Empty state
if not st.session_state.messages:
    if st.session_state.get("uploaded_file_name"):
        st.success(f"✅ **Data loaded:** {st.session_state.uploaded_file_name}\n\nAsk me anything about your data!")
    else:
        st.info("""
        👋 **Welcome!** Upload a CSV file in the sidebar to get started.
        
        **Features:**
        - 🌐 **Web Search**: Toggle DuckDuckGo search for methodology guidance
        - 📊 **Auto-Updating Progress**: Watch stages complete in real-time (every 2s)
        - 📋 **Activity Log**: Detailed event tracking for debugging
        - 🔄 **Background Processing**: Navigate freely while analysis runs
        """)