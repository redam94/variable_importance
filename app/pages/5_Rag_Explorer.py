"""
RAG Explorer Page - Search and Browse Stored Context

Explore the knowledge base built from your analyses.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import init_session_state

st.set_page_config(
    page_title="RAG Explorer - Data Science Agent",
    page_icon="🧠",
    layout="wide"
)

init_session_state()

st.title("🧠 RAG Context Explorer")
st.caption("Search and explore the knowledge base built from your analyses")

rag = st.session_state.get("rag")

if not rag or not rag.enabled:
    st.error("❌ RAG system is not enabled or available")
    st.info("💡 RAG is automatically enabled when you run analyses. Try uploading data and asking questions in the Chat!")
    st.stop()

# Get RAG statistics
rag_stats = rag.get_stats()

# Display stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Chunks", rag_stats.get("total_chunks", 0))
with col2:
    plot_docs = rag_stats.get("type_breakdown", {}).get("plot_analysis", 0)
    st.metric("Plot Analyses", plot_docs)
with col3:
    code_docs = rag_stats.get("type_breakdown", {}).get("code_execution", 0)
    st.metric("Code Executions", code_docs)
with col4:
    summary_docs = rag_stats.get("type_breakdown", {}).get("summary", 0)
    st.metric("Summaries", summary_docs)

st.divider()

# Query interface
st.markdown("### 🔍 Search Knowledge Base")

query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., correlation analysis, sales distribution, error handling..."
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    n_results = st.number_input("Results", min_value=1, max_value=50, value=10)
with col2:
    doc_type_filter = st.selectbox(
        "Filter by type",
        ["All", "Plot Analysis", "Code Execution", "Summary"]
    )

if query:
    with st.spinner("Searching..."):
        # Map display names to actual types
        doc_type_map = {
            "All": None,
            "Plot Analysis": ["plot_analysis"],
            "Code Execution": ["code_execution"],
            "Summary": ["summary"]
        }
        
        doc_types = doc_type_map[doc_type_filter]
        
        contexts = rag.query_relevant_context(
            query=query,
            workflow_id=st.session_state.get("workflow_id"),
            doc_types=doc_types,
            n_results=n_results
        )
        
        if contexts:
            st.success(f"✅ Found {len(contexts)} relevant chunks")
            
            for i, ctx in enumerate(contexts, 1):
                with st.expander(
                    f"**#{i}** - {ctx['metadata'].get('type', 'unknown').title()} "
                    f"(Stage: {ctx['metadata'].get('stage_name', 'unknown')})",
                    expanded=(i <= 3)
                ):
                    # Metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Metadata:**")
                        metadata = ctx['metadata'].copy()
                        # Clean up metadata for display
                        display_metadata = {
                            "Type": metadata.get('type', 'unknown'),
                            "Stage": metadata.get('stage_name', 'unknown'),
                            "Workflow": metadata.get('workflow_id', 'unknown'),
                            "Timestamp": metadata.get('timestamp', 'unknown')[:19] if 'timestamp' in metadata else 'unknown',
                            "Chunk": f"{metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
                        }
                        
                        if 'plot_name' in metadata:
                            display_metadata["Plot"] = metadata['plot_name']
                        
                        for key, value in display_metadata.items():
                            st.text(f"{key}: {value}")
                    
                    with col2:
                        if ctx.get('distance') is not None:
                            relevance = (1 - ctx['distance']) * 100
                            st.metric("Relevance", f"{relevance:.1f}%")
                    
                    st.markdown("**Content:**")
                    st.text_area(
                        "Content",
                        value=ctx['document'],
                        height=200,
                        key=f"content_{i}",
                        label_visibility="collapsed"
                    )
                    
                    # Full metadata
                    with st.expander("View full metadata"):
                        st.json(ctx['metadata'])
        else:
            st.warning("No relevant context found. Try a different query or check your workflow ID.")

st.divider()

# RAG Management
st.markdown("### ⚙️ RAG Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Clear Current Workflow", use_container_width=True):
        workflow_id = st.session_state.get("workflow_id")
        rag.delete_by_workflow(workflow_id)
        st.success(f"✅ Cleared RAG for {workflow_id}")
        st.rerun()

with col2:
    if st.button("🗑️ Clear All RAG Data", use_container_width=True, type="secondary"):
        rag.clear_all()
        st.success("✅ Cleared all RAG data")
        st.rerun()

with col3:
    if st.button("📊 Refresh Stats", use_container_width=True):
        st.rerun()

# Detailed statistics
with st.expander("📈 Detailed Statistics"):
    st.json(rag_stats)

# Sidebar
with st.sidebar:
    st.markdown("### 💡 About RAG")
    st.markdown("""
    The RAG (Retrieval-Augmented Generation) system stores context from:
    
    - **Plot Analyses**: Vision LLM interpretations
    - **Code Executions**: Code and outputs
    - **Summaries**: Analysis summaries
    
    This allows the AI to:
    - Remember past analyses
    - Answer questions without re-running code
    - Build on previous work
    - Learn from errors
    """)
    
    st.divider()
    
    st.markdown("### 🔧 Configuration")
    st.text(f"Chunk Size: {rag.chunker.chunk_size}")
    st.text(f"Chunk Overlap: {rag.chunker.overlap}")
    st.text(f"Collection: {rag.collection.name}")
    
    st.divider()
    
    st.markdown("### 🎯 Tips")
    st.markdown("""
    - Use specific keywords
    - Search by error messages
    - Filter by document type
    - Explore nearby chunks
    """)