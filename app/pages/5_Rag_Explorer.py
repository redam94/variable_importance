"""
RAG Explorer Page - Search and Browse Stored Context

Explore the knowledge base built from your analyses.
Features:
- Search by document type (code, web results, summaries, plots)
- View all chunks from a single source
- Code-aware chunk display
- Web result browsing
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

# Custom CSS
st.markdown("""
<style>
    .chunk-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .chunk-card.code-chunk {
        background: #1e1e1e;
        border-color: #333;
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #6c757d;
    }
    .web-result-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #90caf9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .type-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .type-code { background: #e8f5e9; color: #2e7d32; }
    .type-web { background: #e3f2fd; color: #1565c0; }
    .type-plot { background: #fff3e0; color: #e65100; }
    .type-summary { background: #f3e5f5; color: #7b1fa2; }
</style>
""", unsafe_allow_html=True)

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
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.metric("Total Chunks", rag_stats.get("total_chunks", 0))
with col2:
    plot_docs = rag_stats.get("type_breakdown", {}).get("plot_analysis", 0)
    st.metric("Plots", plot_docs)
with col3:
    code_docs = rag_stats.get("type_breakdown", {}).get("code_execution", 0)
    st.metric("Code", code_docs)
with col4:
    web_docs = rag_stats.get("type_breakdown", {}).get("web_result", 0)
    st.metric("Web", web_docs)
with col5:
    enriched_docs = rag_stats.get("enriched_web_chunks", 0)
    st.metric("🔷 Enriched", enriched_docs)
with col6:
    doc_chunks = rag_stats.get("document_chunks", 0)
    st.metric("📚 Docs", doc_chunks)
with col7:
    code_chunks = rag_stats.get("code_chunks", 0)
    st.metric("💻 Code", code_chunks)

st.divider()

# Tabs for different views
tab_search, tab_web, tab_code, tab_docs, tab_browse = st.tabs([
    "🔍 Search", 
    "🌐 Web Results", 
    "💻 Code Chunks",
    "📚 Documents",
    "📋 Browse All"
])

# =============================================================================
# TAB 1: GENERAL SEARCH
# =============================================================================
with tab_search:
    st.markdown("### 🔍 Search Knowledge Base")
    
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., correlation analysis, sales distribution, error handling...",
        key="search_query"
    )
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        n_results = st.number_input("Results", min_value=1, max_value=50, value=10, key="search_n")
    with col2:
        doc_type_filter = st.selectbox(
            "Filter by type",
            ["All", "Plot Analysis", "Code Execution", "Web Result", "Summary"],
            key="search_type"
        )
    with col3:
        code_only = st.checkbox("Code only", value=False, help="Only return chunks containing code")
    
    if query:
        with st.spinner("Searching..."):
            doc_type_map = {
                "All": None,
                "Plot Analysis": ["plot_analysis"],
                "Code Execution": ["code_execution"],
                "Web Result": ["web_result"],
                "Summary": ["summary"]
            }
            
            doc_types = doc_type_map[doc_type_filter]
            
            contexts = rag.query_relevant_context(
                query=query,
                workflow_id=st.session_state.get("workflow_id"),
                doc_types=doc_types,
                code_only=code_only,
                n_results=n_results
            )
            
            if contexts:
                st.success(f"✅ Found {len(contexts)} relevant chunks")
                
                for i, ctx in enumerate(contexts, 1):
                    metadata = ctx['metadata']
                    doc_type = metadata.get('type', 'unknown')
                    has_code = metadata.get('has_code', False)
                    
                    # Type badge
                    type_class = {
                        'code_execution': 'type-code',
                        'web_result': 'type-web',
                        'plot_analysis': 'type-plot',
                        'summary': 'type-summary'
                    }.get(doc_type, '')
                    
                    # Header
                    header = f"**#{i}** "
                    if has_code:
                        header += "💻 "
                    header += f"{doc_type.replace('_', ' ').title()} "
                    header += f"(Stage: {metadata.get('stage_name', 'unknown')})"
                    
                    with st.expander(header, expanded=(i <= 3)):
                        # Metadata row
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if ctx.get('distance') is not None:
                                relevance = (1 - ctx['distance']) * 100
                                st.metric("Relevance", f"{relevance:.1f}%")
                        with col2:
                            chunk_info = f"{metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
                            st.metric("Chunk", chunk_info)
                        with col3:
                            st.metric("Size", f"{metadata.get('chunk_size', 0)} chars")
                        
                        # Content
                        if has_code and ctx['document'].strip().startswith('```'):
                            # Display as code
                            code_content = ctx['document']
                            # Extract language if present
                            if code_content.startswith('```'):
                                lines = code_content.split('\n')
                                lang = lines[0][3:] or 'python'
                                code_body = '\n'.join(lines[1:-1]) if lines[-1] == '```' else '\n'.join(lines[1:])
                                st.code(code_body, language=lang)
                            else:
                                st.code(ctx['document'], language='python')
                        else:
                            st.text_area(
                                "Content",
                                value=ctx['document'],
                                height=150,
                                key=f"content_{i}",
                                label_visibility="collapsed"
                            )
                        
                        # View related chunks button
                        if metadata.get('total_chunks', 1) > 1:
                            if st.button(f"📑 View all {metadata.get('total_chunks')} chunks from this source", key=f"view_all_{i}"):
                                st.session_state[f"expand_source_{i}"] = True
                        
                        # Full metadata
                        with st.expander("View full metadata", expanded=False):
                            st.json(metadata)
            else:
                st.warning("No relevant context found. Try a different query.")


# =============================================================================
# TAB 2: WEB RESULTS BROWSER
# =============================================================================
with tab_web:
    st.markdown("### 🌐 Web Search Results")
    st.caption("Browse and search through stored web research")
    
    # Web-specific search
    web_query = st.text_input(
        "Search web results:",
        placeholder="e.g., pandas best practices, sklearn tutorial...",
        key="web_query"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        web_n_results = st.number_input("Max results", min_value=1, max_value=30, value=10, key="web_n")
    with col2:
        enriched_only = st.checkbox("🔷 Enriched only", value=False, 
            help="Show only results enriched with full page content via crawl4ai")
    
    # Get web results
    if web_query:
        if enriched_only:
            web_contexts = rag.query_enriched_web_context(
                query=web_query,
                workflow_id=st.session_state.get("workflow_id"),
                n_results=web_n_results
            )
        else:
            web_contexts = rag.query_relevant_context(
                query=web_query,
                workflow_id=st.session_state.get("workflow_id"),
                doc_types=["web_result"],
                n_results=web_n_results
            )
    else:
        # Show all web results if no query
        try:
            if enriched_only:
                all_results = rag.collection.get(
                    where={"$and": [{"type": "web_result"}, {"enriched": True}]},
                    limit=50
                )
            else:
                all_results = rag.collection.get(
                    where={"type": "web_result"},
                    limit=50
                )
            web_contexts = []
            if all_results['documents']:
                for i, doc in enumerate(all_results['documents']):
                    web_contexts.append({
                        'document': doc,
                        'metadata': all_results['metadatas'][i],
                        'distance': None
                    })
        except Exception as e:
            st.error(f"Error fetching web results: {e}")
            web_contexts = []
    
    if web_contexts:
        # Count enriched
        enriched_count = sum(1 for ctx in web_contexts if ctx['metadata'].get('enriched'))
        st.success(f"📚 Found {len(web_contexts)} web result chunks ({enriched_count} enriched 🔷)")
        
        # Group by URL for cleaner display
        url_groups = {}
        for ctx in web_contexts:
            url = ctx['metadata'].get('url', 'unknown')
            if url not in url_groups:
                url_groups[url] = {
                    'title': ctx['metadata'].get('title', 'Unknown'),
                    'query': ctx['metadata'].get('search_query', ''),
                    'source': ctx['metadata'].get('source', 'web'),
                    'enriched': ctx['metadata'].get('enriched', False),
                    'query_used': ctx['metadata'].get('query_used', ''),
                    'content_length': ctx['metadata'].get('content_length', 0),
                    'chunks': []
                }
            url_groups[url]['chunks'].append(ctx)
        
        # Count enriched sources
        enriched_sources = sum(1 for data in url_groups.values() if data['enriched'])
        st.info(f"📄 {len(url_groups)} unique sources ({enriched_sources} enriched 🔷)")
        
        for url, data in url_groups.items():
            # Build header with enriched indicator
            title_display = data['title'][:60] + "..." if len(data['title']) > 60 else data['title']
            enriched_badge = " 🔷" if data['enriched'] else ""
            
            with st.expander(f"🔗 {title_display}{enriched_badge}", expanded=False):
                # Source info
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**URL:** [{url}]({url})")
                    st.markdown(f"**Search Query:** {data['query']}")
                    if data['query_used'] and data['query_used'] != data['query']:
                        st.markdown(f"**Query Used:** {data['query_used']}")
                with col2:
                    st.markdown(f"**Source:** {data['source']}")
                    if data['enriched']:
                        st.markdown("**Status:** 🔷 Enriched (full page)")
                        st.markdown(f"**Content:** ~{data['content_length']} chars")
                    else:
                        st.markdown("**Status:** Snippet only")
                    st.markdown(f"**Chunks:** {len(data['chunks'])}")
                
                st.divider()
                
                # Show all chunks from this source
                for i, chunk in enumerate(data['chunks']):
                    chunk_idx = chunk['metadata'].get('chunk_index', i)
                    total = chunk['metadata'].get('total_chunks', len(data['chunks']))
                    has_code = chunk['metadata'].get('has_code', False)
                    
                    st.markdown(f"**Chunk {chunk_idx + 1}/{total}** {'💻' if has_code else ''}")
                    
                    if has_code and chunk['document'].strip().startswith('```'):
                        lines = chunk['document'].split('\n')
                        lang = lines[0][3:] or 'text'
                        code_body = '\n'.join(lines[1:-1]) if lines[-1] == '```' else '\n'.join(lines[1:])
                        st.code(code_body, language=lang)
                    else:
                        st.text_area(
                            f"Content {i}",
                            value=chunk['document'],
                            height=100,
                            key=f"web_chunk_{url}_{i}",
                            label_visibility="collapsed"
                        )
                    
                    if i < len(data['chunks']) - 1:
                        st.markdown("---")
    else:
        if enriched_only:
            st.info("No enriched web results found. Enriched results contain full page content extracted via crawl4ai.")
        else:
            st.info("No web results stored yet. Enable web search in Chat to start collecting results.")


# =============================================================================
# TAB 3: CODE CHUNKS BROWSER
# =============================================================================
with tab_code:
    st.markdown("### 💻 Code Chunks")
    st.caption("Browse stored code snippets and execution results")
    
    code_query = st.text_input(
        "Search code:",
        placeholder="e.g., load csv, create plot, correlation...",
        key="code_query"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        code_n_results = st.number_input("Max results", min_value=1, max_value=30, value=10, key="code_n")
    
    # Get code chunks
    if code_query:
        code_contexts = rag.query_relevant_context(
            query=code_query,
            workflow_id=st.session_state.get("workflow_id"),
            doc_types=["code_execution"],
            code_only=True,
            n_results=code_n_results
        )
    else:
        # Show all code chunks
        try:
            all_results = rag.collection.get(
                where={"$and": [{"type": "code_execution"}, {"has_code": True}]},
                limit=50
            )
            code_contexts = []
            if all_results['documents']:
                for i, doc in enumerate(all_results['documents']):
                    code_contexts.append({
                        'document': doc,
                        'metadata': all_results['metadatas'][i],
                        'distance': None
                    })
        except Exception as e:
            # Fallback without compound filter
            try:
                all_results = rag.collection.get(
                    where={"type": "code_execution"},
                    limit=100
                )
                code_contexts = []
                if all_results['documents']:
                    for i, doc in enumerate(all_results['documents']):
                        if all_results['metadatas'][i].get('has_code'):
                            code_contexts.append({
                                'document': doc,
                                'metadata': all_results['metadatas'][i],
                                'distance': None
                            })
            except Exception as e2:
                st.error(f"Error fetching code: {e2}")
                code_contexts = []
    
    if code_contexts:
        st.success(f"💻 Found {len(code_contexts)} code chunks")
        
        # Group by stage
        stage_groups = {}
        for ctx in code_contexts:
            stage = ctx['metadata'].get('stage_name', 'unknown')
            if stage not in stage_groups:
                stage_groups[stage] = []
            stage_groups[stage].append(ctx)
        
        for stage, chunks in stage_groups.items():
            with st.expander(f"📂 {stage} ({len(chunks)} code chunks)", expanded=True):
                for i, chunk in enumerate(chunks):
                    metadata = chunk['metadata']
                    timestamp = metadata.get('timestamp', '')[:19]
                    
                    st.markdown(f"**Code #{i+1}** - {timestamp}")
                    
                    code_content = chunk['document']
                    # Clean up code display
                    if code_content.strip().startswith('```'):
                        lines = code_content.split('\n')
                        lang = lines[0][3:] or 'python'
                        code_body = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
                        st.code(code_body, language=lang)
                    else:
                        st.code(code_content, language='python')
                    
                    # Copy button
                    if st.button(f"📋 Copy", key=f"copy_code_{stage}_{i}"):
                        st.toast("Code copied to clipboard!")
                    
                    if i < len(chunks) - 1:
                        st.markdown("---")
    else:
        st.info("No code chunks stored yet. Run some analyses in Chat to populate the code library.")


# =============================================================================
# TAB 4: DOCUMENTS BROWSER
# =============================================================================
with tab_docs:
    st.markdown("### 📚 Uploaded Documents")
    st.caption("Browse documents added via the Resources page")
    
    # Search documents
    doc_query = st.text_input(
        "Search documents:",
        placeholder="e.g., methodology, tutorial, regression...",
        key="doc_query"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        doc_n_results = st.number_input("Max results", min_value=1, max_value=30, value=10, key="doc_n")
    
    # Get documents
    if doc_query:
        doc_contexts = rag.query_documents(
            query=doc_query,
            workflow_id=st.session_state.get("workflow_id"),
            n_results=doc_n_results
        )
    else:
        # Show all documents
        try:
            all_results = rag.collection.get(
                where={"type": "document"},
                limit=50
            )
            doc_contexts = []
            if all_results['documents']:
                for i, doc in enumerate(all_results['documents']):
                    doc_contexts.append({
                        'document': doc,
                        'metadata': all_results['metadatas'][i],
                        'distance': None
                    })
        except Exception as e:
            st.error(f"Error fetching documents: {e}")
            doc_contexts = []
    
    if doc_contexts:
        st.success(f"📚 Found {len(doc_contexts)} document chunks")
        
        # Group by title
        title_groups = {}
        for ctx in doc_contexts:
            title = ctx['metadata'].get('title', 'Unknown')
            if title not in title_groups:
                title_groups[title] = {
                    'source_type': ctx['metadata'].get('source_type', 'unknown'),
                    'url': ctx['metadata'].get('url'),
                    'source_path': ctx['metadata'].get('source_path'),
                    'content_length': ctx['metadata'].get('content_length', 0),
                    'tags': ctx['metadata'].get('tags', ''),
                    'chunks': []
                }
            title_groups[title]['chunks'].append(ctx)
        
        st.info(f"📄 {len(title_groups)} unique documents")
        
        for title, data in title_groups.items():
            source_type = data['source_type']
            type_icon = {
                'pdf': '📕',
                'txt': '📄',
                'md': '📝',
                'url': '🌐',
                'csv': '📊',
                'json': '📋'
            }.get(source_type, '📄')
            
            with st.expander(f"{type_icon} {title}", expanded=False):
                # Document info
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Type:** {source_type}")
                    if data.get('url'):
                        st.markdown(f"**URL:** [{data['url']}]({data['url']})")
                    if data.get('source_path'):
                        st.markdown(f"**File:** {data['source_path']}")
                    if data.get('tags'):
                        st.markdown(f"**Tags:** {data['tags']}")
                with col2:
                    st.markdown(f"**Content:** ~{data['content_length']:,} chars")
                    st.markdown(f"**Chunks:** {len(data['chunks'])}")
                
                st.divider()
                
                # Show chunks
                for i, chunk in enumerate(data['chunks']):
                    chunk_idx = chunk['metadata'].get('chunk_index', i)
                    total = chunk['metadata'].get('total_chunks', len(data['chunks']))
                    has_code = chunk['metadata'].get('has_code', False)
                    
                    st.markdown(f"**Chunk {chunk_idx + 1}/{total}** {'💻' if has_code else ''}")
                    
                    if has_code and chunk['document'].strip().startswith('```'):
                        lines = chunk['document'].split('\n')
                        lang = lines[0][3:] or 'text'
                        code_body = '\n'.join(lines[1:-1]) if lines[-1] == '```' else '\n'.join(lines[1:])
                        st.code(code_body, language=lang)
                    else:
                        st.text_area(
                            f"Content {i}",
                            value=chunk['document'],
                            height=100,
                            key=f"doc_chunk_{title}_{i}",
                            label_visibility="collapsed"
                        )
                    
                    if i < len(data['chunks']) - 1:
                        st.markdown("---")
    else:
        st.info("No documents uploaded yet. Go to the **Resources** page to add PDFs, text files, or scrape URLs.")
        if st.button("📚 Go to Resources", use_container_width=True):
            st.switch_page("pages/7_Resources.py")


# =============================================================================
# TAB 5: BROWSE ALL
# =============================================================================
with tab_browse:
    st.markdown("### 📋 Browse All Stored Context")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        browse_type = st.selectbox(
            "Document Type",
            ["All", "code_execution", "web_result", "plot_analysis", "summary", "document"],
            key="browse_type"
        )
    with col2:
        browse_limit = st.number_input("Limit", min_value=10, max_value=200, value=50, key="browse_limit")
    with col3:
        show_code_only = st.checkbox("Code chunks only", key="browse_code_only")
    with col4:
        show_enriched_only = st.checkbox("🔷 Enriched only", key="browse_enriched_only",
            help="Show only enriched web results")
    
    # Fetch documents
    try:
        conditions = []
        
        if browse_type != "All":
            conditions.append({"type": browse_type})
        if show_code_only:
            conditions.append({"has_code": True})
        if show_enriched_only:
            conditions.append({"enriched": True})
        
        if len(conditions) == 0:
            where_filter = None
        elif len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}
        
        all_docs = rag.collection.get(
            where=where_filter,
            limit=browse_limit
        )
        
        if all_docs['documents']:
            st.success(f"📄 Showing {len(all_docs['documents'])} documents")
            
            # Statistics
            type_counts = {}
            for meta in all_docs['metadatas']:
                t = meta.get('type', 'unknown')
                type_counts[t] = type_counts.get(t, 0) + 1
            
            cols = st.columns(len(type_counts))
            for i, (t, count) in enumerate(type_counts.items()):
                with cols[i]:
                    st.metric(t.replace('_', ' ').title(), count)
            
            st.divider()
            
            # Display documents
            for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                doc_type = metadata.get('type', 'unknown')
                has_code = metadata.get('has_code', False)
                is_enriched = metadata.get('enriched', False)
                stage = metadata.get('stage_name', 'unknown')
                
                # Icon based on type
                icon = {
                    'code_execution': '💻',
                    'web_result': '🌐',
                    'plot_analysis': '📊',
                    'summary': '📝'
                }.get(doc_type, '📄')
                
                header = f"{icon} {doc_type.replace('_', ' ').title()} - {stage}"
                if has_code:
                    header += " 💻"
                if is_enriched:
                    header += " 🔷"
                
                with st.expander(header, expanded=False):
                    # Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.text(f"Type: {doc_type}")
                        st.text(f"Stage: {stage}")
                    with col2:
                        chunk_info = f"{metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
                        st.text(f"Chunk: {chunk_info}")
                        st.text(f"Size: {metadata.get('chunk_size', len(doc))} chars")
                    with col3:
                        st.text(f"Has Code: {has_code}")
                        if doc_type == 'web_result':
                            st.text(f"Enriched: {is_enriched}")
                        if metadata.get('url'):
                            st.text(f"URL: {metadata['url'][:30]}...")
                    
                    st.divider()
                    
                    # Content
                    if has_code and doc.strip().startswith('```'):
                        lines = doc.split('\n')
                        lang = lines[0][3:] or 'python'
                        code_body = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
                        st.code(code_body, language=lang)
                    elif has_code:
                        st.code(doc, language='python')
                    else:
                        st.text_area(
                            "Content",
                            value=doc,
                            height=120,
                            key=f"browse_doc_{i}",
                            label_visibility="collapsed"
                        )
                    
                    # Full metadata expander
                    with st.expander("Full Metadata"):
                        st.json(metadata)
        else:
            st.info("No documents found with the selected filters.")
            
    except Exception as e:
        st.error(f"Error browsing documents: {e}")
        import traceback
        with st.expander("Debug"):
            st.code(traceback.format_exc())


st.divider()

# =============================================================================
# RAG MANAGEMENT
# =============================================================================
st.markdown("### ⚙️ RAG Management")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.rerun()

with col2:
    if st.button("🗑️ Clear Web Results", use_container_width=True):
        try:
            results = rag.collection.get(where={"type": "web_result"})
            if results['ids']:
                rag.collection.delete(ids=results['ids'])
                st.success(f"✅ Cleared {len(results['ids'])} web result chunks")
                st.rerun()
            else:
                st.info("No web results to clear")
        except Exception as e:
            st.error(f"Error: {e}")

with col3:
    if st.button("🗑️ Clear Current Workflow", use_container_width=True):
        workflow_id = st.session_state.get("workflow_id")
        rag.delete_by_workflow(workflow_id)
        st.success(f"✅ Cleared RAG for {workflow_id}")
        st.rerun()

with col4:
    if st.button("🗑️ Clear All RAG Data", use_container_width=True, type="secondary"):
        rag.clear_all()
        st.success("✅ Cleared all RAG data")
        st.rerun()

# Detailed statistics
with st.expander("📈 Detailed Statistics"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Document Types:**")
        for doc_type, count in rag_stats.get("type_breakdown", {}).items():
            st.text(f"  {doc_type}: {count}")
    
    with col2:
        st.markdown("**Chunk Types:**")
        for chunk_type, count in rag_stats.get("chunk_type_breakdown", {}).items():
            st.text(f"  {chunk_type}: {count}")
    
    st.divider()
    st.markdown("**Configuration:**")
    st.text(f"Contextual Chunking: {rag_stats.get('contextual_chunking', True)}")
    st.text(f"Chunks with Section Context: {rag_stats.get('chunks_with_section_context', 0)}")
    
    with st.expander("Raw Stats"):
        st.json(rag_stats)

# Sidebar
with st.sidebar:
    st.markdown("### 💡 About RAG")
    st.markdown("""
    The RAG system stores context from:
    
    - **🌐 Web Results**: Search findings from DuckDuckGo
      - 🔷 = Enriched (full page via crawl4ai)
    - **💻 Code Executions**: Code and outputs
    - **📊 Plot Analyses**: Vision LLM interpretations
    - **📝 Summaries**: Analysis summaries
    - **📚 Documents**: Uploaded PDFs, text files, URLs
    
    **Contextual Chunking:**
    - Sections preserved under headers
    - Lists kept together
    - Code blocks as atomic units
    - Section context included in chunks
    
    **Enriched Web Results:**
    When enabled, top search results are crawled for full page content.
    """)
    
    st.divider()
    
    st.markdown("### 🔧 Configuration")
    st.text(f"Chunk Size: {rag.chunker.chunk_size}")
    st.text(f"Chunk Overlap: {rag.chunker.overlap}")
   # st.text(f"Contextual: {rag.chunker.contextual}")
    st.text(f"Collection: {rag.collection.name}")
    
    st.divider()
    
    st.markdown("### 🎯 Search Tips")
    st.markdown("""
    - **Code search**: Use function names, library calls
    - **Web search**: Use topic keywords
    - **🔷 Enriched only**: Filter for detailed content
    - **Filter by type** for focused results
    """)