"""
Workflows Page - Browse All Previous Workflows and Stages

Explore past analyses, switch between workflows, and view historical artifacts.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import (
    init_session_state, 
    get_all_workflows, 
    get_workflow_stages,
    switch_workflow
)
from session_persistence import save_streamlit_session

st.set_page_config(
    page_title="Workflows - Data Science Agent",
    page_icon="📚",
    layout="wide"
)

init_session_state()

st.title("📚 Workflow Browser")
st.caption("Explore all your past analyses and workflows")

workflows = get_all_workflows()
# Get all workflows
if st.button("🔄 Refresh Workflows", use_container_width=True):
    workflows = get_all_workflows()
    st.success("✅ Workflows refreshed")


if not workflows:
    st.info("No workflows found yet. Start a new analysis in the Chat!")
    st.stop()

# Summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Workflows", len(workflows))
with col2:
    total_stages = sum(w['total_stages'] for w in workflows)
    st.metric("Total Stages", total_stages)
with col3:
    total_outputs = sum(w['total_outputs'] for w in workflows)
    st.metric("Total Outputs", total_outputs)

st.divider()

# Search and filter
search = st.text_input("🔍 Search workflows", placeholder="Enter workflow ID or keywords...")

# Filter workflows
filtered_workflows = workflows
if search:
    filtered_workflows = [
        w for w in workflows 
        if search.lower() in w['workflow_id'].lower()
    ]

# Display workflows
st.markdown(f"### Workflows ({len(filtered_workflows)})")

for workflow in filtered_workflows:
    workflow_id = workflow['workflow_id']
    created_at = workflow['created_at']
    total_stages = workflow['total_stages']
    total_outputs = workflow['total_outputs']
    
    # Check if this is the current workflow
    is_current = workflow_id == st.session_state.get('workflow_id')
    
    with st.expander(
        f"{'🟢 ' if is_current else ''}📁 {workflow_id}", 
        expanded=is_current
    ):
        # Workflow metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stages", total_stages)
        with col2:
            st.metric("Outputs", total_outputs)
        with col3:
            st.text(f"Created:\n{created_at[:19]}")
        with col4:
            if is_current:
                st.success("✅ Current")
            else:
                if st.button("Load", key=f"load_{workflow_id}", use_container_width=True):
                    switch_workflow(workflow_id)
                    save_streamlit_session(st.session_state)
                    st.success(f"✅ Switched to {workflow_id}")
                    st.rerun()
        
        st.divider()
        
        # Get stages
        stages = get_workflow_stages(workflow_id)
        
        if not stages:
            st.info("No stages found in this workflow")
            continue
        
        # Display stages
        st.markdown("**Stages:**")
        
        for stage in stages:
            stage_name = stage['stage_name']
            
            # Stage info
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{stage_name}**")
            with col2:
                st.caption(f"📊 {stage['plot_count']} plots | 📈 {stage['data_file_count']} data | 💾 {stage['code_file_count']} code")
            
            # Execution info
            if stage['execution_info']:
                exec_info = stage['execution_info']
                success = exec_info.get('success', False)
                exec_time = exec_info.get('execution_time_seconds', 0)
                
                status_icon = "✅" if success else "❌"
                st.caption(f"{status_icon} Execution time: {exec_time:.2f}s")
                
                if not success and exec_info.get('error'):
                    st.error(f"Error: {exec_info['error']}")
            
            # Preview plots
            if stage['has_plots']:
                plots_dir = Path(stage['path']) / "plots"
                plot_files = sorted(list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg")))
                
                if plot_files:
                    with st.expander(f"🖼️ View Plots ({len(plot_files)})", expanded=False):
                        cols = st.columns(3)
                        for idx, plot_file in enumerate(plot_files[:6]):  # Show max 6
                            with cols[idx % 3]:
                                st.image(str(plot_file), caption=plot_file.name, use_container_width=True)
                        
                        if len(plot_files) > 6:
                            st.caption(f"... and {len(plot_files) - 6} more plots")
            
            # Preview code
            if stage['has_code']:
                code_files = list(Path(stage['path']).glob("*.py"))
                if code_files:
                    with st.expander(f"💾 View Code ({len(code_files)})", expanded=False):
                        for code_file in code_files[:2]:  # Show max 2
                            st.markdown(f"**{code_file.name}:**")
                            with open(code_file, 'r') as f:
                                code_content = f.read()
                                st.code(code_content[:500] + ("..." if len(code_content) > 500 else ""), 
                                       language="python")
                        
                        if len(code_files) > 2:
                            st.caption(f"... and {len(code_files) - 2} more code files")
            
            st.markdown("---")
        
        # Workflow actions
        st.markdown("**Actions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            manifest_file = Path(workflow['path']) / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest_data = f.read()
                st.download_button(
                    label="📄 Download Manifest",
                    data=manifest_data,
                    file_name=f"{workflow_id}_manifest.json",
                    mime="application/json",
                    key=f"dl_manifest_{workflow_id}"
                )
        
        with col2:
            summary_file = Path(workflow['path']) / "SUMMARY.md"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = f.read()
                st.download_button(
                    label="📋 Download Summary",
                    data=summary_data,
                    file_name=f"{workflow_id}_SUMMARY.md",
                    mime="text/markdown",
                    key=f"dl_summary_{workflow_id}"
                )
        
        with col3:
            if st.button("🗑️ Delete Workflow", key=f"del_{workflow_id}", type="secondary"):
                import shutil
                try:
                    shutil.rmtree(workflow['path'])
                    st.success(f"✅ Deleted {workflow_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Workflow Management")
    
    if st.button("➕ Create New Workflow", use_container_width=True, type="primary"):
        new_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        switch_workflow(new_id)
        st.session_state.messages = []
        save_streamlit_session(st.session_state)
        st.success(f"✅ Created {new_id}")
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📊 Current Workflow")
    st.text(f"{st.session_state.get('workflow_id', 'None')}")
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Click on a workflow** to see its stages
    - **Load a workflow** to switch to it
    - **Delete old workflows** to free up space
    - **Download manifests** for record keeping
    """)