"""
Artifacts Page - View Generated Plots and Files

Browse all artifacts from the current workflow with live updates.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import init_session_state

st.set_page_config(
    page_title="Artifacts - Data Science Agent",
    page_icon="🎨",
    layout="wide"
)

init_session_state()

st.title("🎨 Generated Artifacts")
st.caption("Browse plots, data files, and code from your current workflow")

# Refresh button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
with col2:
    view_mode = st.selectbox("View", ["Grid", "List"], label_visibility="collapsed")

st.divider()

output_mgr = st.session_state.get("output_mgr")
if not output_mgr:
    st.warning("No workflow initialized. Go to Chat to start!")
    st.stop()

workflow_dir = output_mgr.workflow_dir
if not workflow_dir.exists():
    st.info("No artifacts yet. Run an analysis in Chat to generate outputs!")
    st.stop()

# Find all stage directories
stage_dirs = sorted([d for d in workflow_dir.iterdir() if d.is_dir()])

if not stage_dirs:
    st.warning("No stage directories found.")
    st.stop()

# Summary metrics
total_plots = 0
total_data_files = 0
total_code_files = 0
total_model_files = 0

for stage_dir in stage_dirs:
    plots_dir = stage_dir / "plots"
    if plots_dir.exists():
        total_plots += len(list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg")))
    
    data_dir = stage_dir / "data"
    if data_dir.exists():
        total_data_files += len(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json")))
    
    total_code_files += len(list(stage_dir.glob("*.py")))
    model_dir = stage_dir / "models"
    if model_dir.exists():
        total_model_files += len(list(model_dir.glob("*")))

# Display summary
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📊 Stages", len(stage_dirs))
with col2:
    st.metric("🖼️ Plots", total_plots)
with col3:
    st.metric("📈 Data Files", total_data_files)
with col4:
    st.metric("💾 Code Files", total_code_files)
with col5:
    st.metric("🧠 Model Files", total_model_files)

st.divider()

# Display stages
for stage_dir in stage_dirs:
    stage_name = stage_dir.name
    
    with st.expander(f"📂 {stage_name}", expanded=True):
        # Plots
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plot_files = sorted(list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg")))
            
            if plot_files:
                st.markdown(f"**🖼️ Plots ({len(plot_files)})**")
                
                if view_mode == "Grid":
                    # Display plots in grid
                    cols = st.columns(3)
                    for idx, plot_file in enumerate(plot_files):
                        with cols[idx % 3]:
                            # Check if cached
                            is_cached = False
                            if st.session_state.get("plot_cache"):
                                is_cached = st.session_state.plot_cache.get(str(plot_file)) is not None
                            
                            if is_cached:
                                st.caption("⚡ Cached")
                            
                            st.image(str(plot_file), caption=plot_file.name, use_container_width=True)
                            
                            # Download button
                            with open(plot_file, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download",
                                    data=f,
                                    file_name=plot_file.name,
                                    mime="image/png",
                                    key=f"download_plot_{stage_name}_{idx}",
                                    use_container_width=True
                                )
                else:
                    # List view
                    for plot_file in plot_files:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(f"📊 {plot_file.name}")
                        with col2:
                            with open(plot_file, "rb") as f:
                                st.download_button(
                                    label="⬇️",
                                    data=f,
                                    file_name=plot_file.name,
                                    key=f"download_plot_list_{plot_file.stem}",
                                    use_container_width=True
                                )
        
        # Data files
        data_dir = stage_dir / "data"
        if data_dir.exists():
            data_files = sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json")))
            
            if data_files:
                st.markdown(f"**📈 Data Files ({len(data_files)})**")
                for data_file in data_files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"📄 {data_file.name}")
                    with col2:
                        with open(data_file, "rb") as f:
                            st.download_button(
                                label="⬇️",
                                data=f,
                                file_name=data_file.name,
                                key=f"download_data_{data_file.stem}",
                                use_container_width=True
                            )
        
        # Code files
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
                            key=f"download_code_{code_file.stem}",
                            use_container_width=False
                        )
        
        # Console output
        console_files = sorted(list(stage_dir.glob("console_output_*.txt")))
        if console_files:
            st.markdown(f"**💻 Console Output**")
            for console_file in console_files:
                with st.expander(f"📋 {console_file.name}"):
                    with open(console_file, 'r') as f:
                        st.text(f.read())
        model_files = sorted(list(stage_dir.glob("models/*")))
        if model_files:
            st.markdown(f"**🧠 Model Files ({len(model_files)})**")
            for model_file in model_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"🗄️ {model_file.name}")
                with col2:
                    with open(model_file, "rb") as f:
                        st.download_button(
                            label="⬇️",
                            data=f,
                            file_name=model_file.name,
                            key=f"download_model_{model_file.stem}",
                            use_container_width=True
                        )

# Sidebar info
with st.sidebar:
    st.markdown("### 📊 Workflow Info")
    st.text(f"ID: {st.session_state.get('workflow_id', 'N/A')}")
    st.text(f"Stages: {len(stage_dirs)}")
    
    st.divider()
    
    st.markdown("### ⚡ Cache Status")
    if st.session_state.get("plot_cache"):
        cache_stats = st.session_state.plot_cache.get_stats()
        st.text(f"Cached plots: {cache_stats['total_entries']}")
        
        if st.button("🗑️ Clear Cache"):
            st.session_state.plot_cache.clear()
            st.success("Cache cleared!")
            st.rerun()