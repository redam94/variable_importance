"""
Shared Session State Utilities

Provides consistent session state initialization and management across all pages.
"""

import streamlit as st
from pathlib import Path
from loguru import logger
from datetime import datetime

from variable_importance.utils.code_executer import OutputCapturingExecutor
from variable_importance.utils.output_manager import OutputManager
from ai.memory.plot_analysis_cache import PlotAnalysisCache
from ai.memory.context_rag import ContextRAG
from session_persistence import load_streamlit_session
from utils.background_tasks import get_task_manager


def init_session_state():
    """
    Initialize session state with all required variables.
    Call this at the start of every page.
    """
    # Mark initialization
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    # Try to restore from saved session (only once)
    if not st.session_state.initialized:
        # Clean widget keys before restoring
        _clean_widget_keys()
        
        restored = load_streamlit_session(st.session_state)
        if restored:
            st.session_state.session_restored = True
            st.session_state.initialized = True
            
            # Recreate non-persistent objects
            _recreate_runtime_objects()
            return
    
    # Initialize defaults if not restored
    _init_defaults()
    st.session_state.initialized = True


def _clean_widget_keys():
    """Remove widget keys that can't be set via session state."""
    widget_keys = [
        "file_uploader",
        "_file_uploader",
        "model_selector",
        "workflow_id_input",
        "stage_name_input",
    ]
    
    for key in widget_keys:
        if key in st.session_state:
            del st.session_state[key]


def _init_defaults():
    """Initialize default session state values."""
    defaults = {
        "messages": [],
        "temp_file_path": None,
        "uploaded_file_name": None,
        "uploaded_file_size": 0,
        "workflow_id": "analysis_workflow",
        "stage_name": "analysis",
        "selected_model": "qwen3:30b",
        "session_restored": False,
        "current_task_id": None,
        "rag_enabled": True,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize runtime objects
    _recreate_runtime_objects()


def _recreate_runtime_objects():
    """Recreate objects that can't be pickled or should always be fresh."""
    
    # Executor - contains code execution environment
    if "executor" not in st.session_state:
        st.session_state.executor = OutputCapturingExecutor()
    
    # Plot cache - file-based cache system
    if "plot_cache" not in st.session_state:
        st.session_state.plot_cache = PlotAnalysisCache()
    
    # Task manager - ALWAYS recreate! Contains threads and cannot be pickled
    # Each page load needs a fresh task manager instance
    st.session_state.task_manager = get_task_manager()
    
    # Create output_mgr and rag based on workflow_id
    workflow_id = st.session_state.get("workflow_id", "analysis_workflow")
    
    # Output manager - manages workflow artifacts
    if "output_mgr" not in st.session_state or st.session_state.output_mgr is None:
        st.session_state.output_mgr = OutputManager(workflow_id=workflow_id)
    
    # RAG - contains ChromaDB client which can't be pickled
    if "rag" not in st.session_state or st.session_state.rag is None:
        st.session_state.rag = ContextRAG(
            collection_name=workflow_id,
            persist_directory="cache/rag_db"
        )


def get_workflow_manager() -> OutputManager:
    """Get the current workflow's output manager."""
    return st.session_state.output_mgr


def get_all_workflows() -> list[dict]:
    """
    Get list of all available workflows.
    
    Returns:
        List of workflow info dictionaries
    """
    base_dir = Path("results")
    logger.debug(f"Looking for workflows in: {base_dir.resolve()}")
    if not base_dir.exists():
        logger.warning(f"Base directory does not exist: {base_dir.resolve()}")
        return []
    
    workflows = []
    for workflow_dir in base_dir.iterdir():
        logger.debug(f"Checking workflow directory: {workflow_dir.resolve()}")
        if not workflow_dir.is_dir():
            continue
        
        # Get workflow info
        manifest_file = workflow_dir
        
        import json
        try:
           
            
            workflows.append({
                "workflow_id": workflow_dir.name,
                "path": str(workflow_dir),
                "created_at": datetime.fromtimestamp(workflow_dir.stat().st_ctime).isoformat(),
                "total_stages": len([d for d in workflow_dir.iterdir() if d.is_dir()]),
                "total_outputs": sum(1 for _ in workflow_dir.rglob('*') if _.is_file()),
                "manifest": {}
            })
        except Exception as e:
            logger.warning(f"⚠️ Failed to read manifest for {workflow_dir}: {e}")

    # Sort by creation date (newest first)
    workflows.sort(key=lambda x: x["created_at"], reverse=True)
    logger.debug(f"Found {len(workflows)} workflows")
    return workflows


def get_workflow_stages(workflow_id: str) -> list[dict]:
    """
    Get all stages for a specific workflow.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        List of stage info dictionaries
    """
    workflow_dir = Path("results") / workflow_id
    if not workflow_dir.exists():
        return []
    
    stages = []
    for stage_dir in sorted(workflow_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        
        # Get stage info
        exec_info_file = stage_dir / "execution_info.json"
        stage_info = {
            "stage_name": stage_dir.name,
            "path": str(stage_dir),
            "has_plots": False,
            "plot_count": 0,
            "has_data": False,
            "data_file_count": 0,
            "has_code": False,
            "code_file_count": 0,
            "execution_info": None
        }
        
        # Count plots
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
            if plot_files:
                stage_info["has_plots"] = True
                stage_info["plot_count"] = len(plot_files)
        
        # Count data files
        data_dir = stage_dir / "data"
        if data_dir.exists():
            data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
            if data_files:
                stage_info["has_data"] = True
                stage_info["data_file_count"] = len(data_files)
        
        # Count code files
        code_files = list(stage_dir.glob("*.py"))
        if code_files:
            stage_info["has_code"] = True
            stage_info["code_file_count"] = len(code_files)
        
        # Load execution info
        if exec_info_file.exists():
            import json
            try:
                with open(exec_info_file, 'r') as f:
                    stage_info["execution_info"] = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to read execution info: {e}")
        
        stages.append(stage_info)
    
    return stages


def switch_workflow(workflow_id: str):
    """
    Switch to a different workflow.
    
    Args:
        workflow_id: Workflow to switch to
    """
    st.session_state.workflow_id = workflow_id
    st.session_state.output_mgr = OutputManager(workflow_id=workflow_id)
    st.session_state.rag = ContextRAG(
        collection_name=workflow_id,
        persist_directory="cache/rag_db"
    )
    logger.info(f"🔄 Switched to workflow: {workflow_id}")