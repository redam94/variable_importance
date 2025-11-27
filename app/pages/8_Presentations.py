"""
Presentations Page - Generate PowerPoint from Analysis Results

Create professional presentations with:
- LLM-planned structure
- Intelligent plot-to-slide matching
- Multiple images per slide
- python-pptx for reliable PPTX creation
"""

import streamlit as st
import asyncio
from pathlib import Path
import sys
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import init_session_state, get_workflow_stages
from session_persistence import save_streamlit_session

st.set_page_config(
    page_title="Presentations - Data Science Agent",
    page_icon="📽️",
    layout="wide"
)

init_session_state()

st.title("📽️ Presentation Generator")
st.caption("Create PowerPoint presentations from your analysis with intelligent plot matching")


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Workflow
    workflow_id = st.session_state.get("workflow_id", "analysis_workflow")
    st.text(f"Workflow: {workflow_id}")
    
    # Stage selection
    stages = get_workflow_stages(workflow_id)
    stage_names = [s["stage_name"] for s in stages] if stages else ["analysis"]
    
    selected_stage = st.selectbox(
        "📂 Source Stage",
        options=stage_names,
        help="Stage containing plots and analysis results"
    )
    
    # Show available plots
    if stages:
        stage_info = next((s for s in stages if s["stage_name"] == selected_stage), None)
        if stage_info:
            plot_count = stage_info.get("plot_count", 0)
            st.metric("Available Plots", plot_count)
            
            # Preview plots
            if plot_count > 0:
                stage_path = Path(stage_info.get("path", ""))
                plots_dir = stage_path / "plots"
                if plots_dir.exists():
                    with st.expander("🖼️ Preview Plots"):
                        plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
                        for img_file in plot_files[:4]:
                            st.image(str(img_file), caption=img_file.name, use_container_width=True)
                        if len(plot_files) > 4:
                            st.caption(f"... and {len(plot_files) - 4} more")
    
    st.divider()
    
    # Theme selection
    st.markdown("### 🎨 Theme")
    themes = {
        "Corporate Blue": {"primary": "1F4E79", "accent": "2E86AB"},
        "Forest Green": {"primary": "2D5016", "accent": "4A7C23"},
        "Sunset Orange": {"primary": "C84C09", "accent": "E07B39"},
        "Royal Purple": {"primary": "4A2C6A", "accent": "7B4FA2"},
        "Slate Gray": {"primary": "3D4852", "accent": "606F7B"},
    }
    
    theme_name = st.selectbox("Color Theme", list(themes.keys()))
    selected_theme = themes[theme_name]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div style="background:#{selected_theme["primary"]};'
            f'color:white;padding:0.5rem;border-radius:0.25rem;text-align:center">'
            f'Primary</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div style="background:#{selected_theme["accent"]};'
            f'color:white;padding:0.5rem;border-radius:0.25rem;text-align:center">'
            f'Accent</div>',
            unsafe_allow_html=True
        )


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("### 📝 Presentation Details")

topic = st.text_input(
    "Presentation Topic",
    placeholder="e.g., Q3 Sales Analysis, Customer Segmentation Results...",
    help="Main topic or title for your presentation"
)

col1, col2 = st.columns([1, 2])
with col1:
    num_slides = st.slider("Number of Slides", min_value=4, max_value=15, value=6)
with col2:
    custom_instructions = st.text_area(
        "Custom Instructions (optional)",
        placeholder="e.g., Focus on revenue trends, Use multi_image layout for comparisons, Emphasize key findings...",
        height=80
    )

st.divider()

# Generate button
if st.button("🚀 Generate Presentation", type="primary", use_container_width=True, disabled=not topic):
    if not topic:
        st.warning("Please enter a presentation topic")
    else:
        progress_container = st.container()
        progress_text = progress_container.empty()
        progress_bar = progress_container.progress(0)
        
        progress_messages = []
        
        def update_progress(message: str):
            progress_messages.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
            progress_text.markdown(f"**Status:** {message}")
            
            # Update progress bar based on keywords
            msg_lower = message.lower()
            if "starting" in msg_lower or "planning" in msg_lower:
                progress_bar.progress(0.1)
            elif "planned" in msg_lower:
                progress_bar.progress(0.25)
            elif "matching" in msg_lower:
                progress_bar.progress(0.35)
            elif "matched" in msg_lower:
                progress_bar.progress(0.45)
            elif "building" in msg_lower:
                progress_bar.progress(0.55)
            elif "slide" in msg_lower:
                # Increment for each slide
                current = progress_bar._value if hasattr(progress_bar, '_value') else 0.55
                if isinstance(current, (int, float)):
                    progress_bar.progress(min(0.9, current + 0.05))
            elif "saved" in msg_lower:
                progress_bar.progress(1.0)
        
        with st.spinner("Generating presentation..."):
            try:
                from ai.pptx import PresentationAgent
                
                agent = PresentationAgent(
                    llm_model=st.session_state.get("selected_model", "qwen3:30b"),
                    rag=st.session_state.get("rag"),
                    output_manager=st.session_state.get("output_mgr"),
                    progress_callback=update_progress,
                )
                
                result = run_async(agent.create_presentation(
                    topic=topic,
                    workflow_id=workflow_id,
                    stage_name=selected_stage,
                    num_slides=num_slides,
                    custom_instructions=custom_instructions or "",
                ))
                
                progress_bar.progress(1.0)
                
                if result.success:
                    st.success(f"✅ Created presentation with {result.slide_count} slides!")
                    
                    # Download button
                    output_path = Path(result.output_path)
                    if output_path.exists():
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Presentation",
                                data=f,
                                file_name=output_path.name,
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                type="primary",
                                use_container_width=True,
                            )
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Slides", result.slide_count)
                    with col2:
                        st.metric("Plots Used", len(result.plots_used))
                    with col3:
                        st.metric("File", output_path.name if output_path else "N/A")
                    
                    # Plots used
                    if result.plots_used:
                        with st.expander("📊 Plots Included", expanded=True):
                            cols = st.columns(min(3, len(result.plots_used)))
                            for i, plot_name in enumerate(result.plots_used):
                                with cols[i % 3]:
                                    st.text(f"• {plot_name}")
                                    # Try to show thumbnail
                                    if stages:
                                        stage_info = next((s for s in stages if s["stage_name"] == selected_stage), None)
                                        if stage_info:
                                            plot_path = Path(stage_info["path"]) / "plots" / plot_name
                                            if plot_path.exists():
                                                st.image(str(plot_path), use_container_width=True)
                    
                    # Generation log
                    with st.expander("📋 Generation Log"):
                        for msg in progress_messages:
                            st.text(msg)
                else:
                    st.error(f"❌ Generation failed: {result.error}")
                    with st.expander("📋 Log"):
                        for msg in progress_messages:
                            st.text(msg)
                    
            except ImportError as e:
                st.error(f"❌ Import error: {e}")
                st.info("Make sure the ai.pptx module exists in app/ai/pptx/")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("🔍 Debug"):
                    st.code(traceback.format_exc())

# Empty state / help
if not topic:
    st.info("""
    ### 🚀 Getting Started
    
    1. **Enter a topic** - The main title/theme for your presentation
    2. **Select source stage** - Where your analysis plots are stored
    3. **Choose theme & slides** - Customize colors and length
    4. **Generate** - AI creates your presentation
    
    ---
    
    **How it works:**
    
    The AI will:
    - 📋 **Plan structure** - Create appropriate slides based on your analysis context
    - 🎯 **Match plots** - Intelligently assign visualizations to relevant slides
    - 🖼️ **Multi-image support** - Include 1-4 images per slide for comparisons
    - 📐 **Fit images** - Automatically scale images while preserving aspect ratio
    - 📝 **Generate content** - Create bullet points and speaker notes
    - 📄 **Build PPTX** - Create downloadable PowerPoint using python-pptx
    
    ---
    
    **Layout Types:**
    
    | Layout | Description |
    |--------|-------------|
    | `title` | Opening slide with presentation title |
    | `content` | Text-focused with optional image |
    | `image` | Large single image with minimal text |
    | `multi_image` | 2-4 images for comparisons |
    | `two_column` | Side-by-side content |
    | `summary` | Closing slide with key takeaways |
    
    ---
    
    💡 **Tip:** Run an analysis in Chat first to generate plots and context for your presentation.
    """)

# Footer
st.divider()
st.caption("Built with python-pptx • Images automatically scaled to fit • Multiple images per slide supported")