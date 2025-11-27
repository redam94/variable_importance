"""
Resources Page - Add External Documents to RAG

Upload PDFs, text files, or scrape URLs to enhance the knowledge base.
Features:
- PDF upload and text extraction
- Text/Markdown file upload
- URL scraping with crawl4ai
- Preview before adding
- Document management (view, delete)
"""

import streamlit as st
from pathlib import Path
import sys
import tempfile
import asyncio
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_utils import init_session_state

st.set_page_config(
    page_title="Resources - Data Science Agent",
    page_icon="📚",
    layout="wide"
)

init_session_state()

# Custom CSS
st.markdown("""
<style>
    .resource-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .resource-type-pdf { border-left: 4px solid #dc3545; }
    .resource-type-txt { border-left: 4px solid #28a745; }
    .resource-type-url { border-left: 4px solid #007bff; }
    .resource-type-md { border-left: 4px solid #6f42c1; }
    .preview-box {
        background: #f1f3f4;
        border-radius: 0.5rem;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Resource Library")
st.caption("Add external documents and web content to enhance your analysis context")

rag = st.session_state.get("rag")

if not rag or not rag.enabled:
    st.error("❌ RAG system is not enabled or available")
    st.stop()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{text}")
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        st.error("PyMuPDF not installed. Install with: pip install pymupdf")
        return ""
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""


def extract_text_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from text-based files."""
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        st.error(f"Text extraction error: {e}")
        return ""


async def scrape_url(url: str) -> tuple[str, str]:
    """Scrape URL content using crawl4ai."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        
        browser_cfg = BrowserConfig(headless=True, verbose=False)
        crawler_cfg = CrawlerRunConfig(
            word_count_threshold=50,
            excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style'],
            remove_overlay_elements=True
        )
        
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=crawler_cfg),
                timeout=30
            )
            
            if result.success:
                title = result.metadata.get('title', url)
                content = result.markdown or ""
                return title, content
            else:
                return "", ""
                
    except ImportError:
        st.error("crawl4ai not installed. Install with: pip install crawl4ai")
        return "", ""
    except asyncio.TimeoutError:
        st.error("URL scraping timed out (30s)")
        return "", ""
    except Exception as e:
        st.error(f"Scraping error: {e}")
        return "", ""


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# =============================================================================
# TABS
# =============================================================================

tab_upload, tab_url, tab_manage = st.tabs([
    "📤 Upload Files",
    "🌐 Scrape URL", 
    "📋 Manage Resources"
])


# =============================================================================
# TAB 1: FILE UPLOAD
# =============================================================================
with tab_upload:
    st.markdown("### 📤 Upload Documents")
    st.caption("Support for PDF, TXT, MD, and other text files")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md", "text", "rst", "csv", "json"],
        help="Upload PDF or text-based files to add to the knowledge base"
    )
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        file_ext = Path(filename).suffix.lower()
        
        st.info(f"📄 **{filename}** ({len(file_bytes) / 1024:.1f} KB)")
        
        # Extract content based on file type
        with st.spinner("Extracting content..."):
            if file_ext == '.pdf':
                content = extract_pdf_text(file_bytes)
                source_type = "pdf"
            else:
                content = extract_text_file(file_bytes, filename)
                source_type = file_ext.replace('.', '') or "txt"
        
        if content:
            # Preview
            st.markdown("**Preview:**")
            preview_text = content[:3000] + ("..." if len(content) > 3000 else "")
            st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{len(content):,}")
            with col2:
                word_count = len(content.split())
                st.metric("Words", f"{word_count:,}")
            with col3:
                # Estimate chunks
                chunk_estimate = max(1, len(content) // rag.chunker.chunk_size)
                st.metric("Est. Chunks", chunk_estimate)
            
            # Custom title
            custom_title = st.text_input(
                "Document Title",
                value=Path(filename).stem,
                help="Title for this document in the knowledge base"
            )
            
            # Tags (optional)
            tags = st.text_input(
                "Tags (optional)",
                placeholder="e.g., methodology, reference, tutorial",
                help="Comma-separated tags for organization"
            )
            
            # Add button
            if st.button("➕ Add to Knowledge Base", type="primary", use_container_width=True):
                with st.spinner("Adding document..."):
                    metadata = {}
                    if tags:
                        metadata["tags"] = tags
                    metadata["filename"] = filename
                    metadata["file_size"] = len(file_bytes)
                    
                    chunks_added = rag.add_document(
                        content=content,
                        title=custom_title,
                        source_type=source_type,
                        workflow_id=st.session_state.get("workflow_id", "default"),
                        source_path=filename,
                        metadata=metadata
                    )
                    
                    if chunks_added:
                        st.success(f"✅ Added **{custom_title}** ({chunks_added} chunks)")
                        st.balloons()
                    else:
                        st.error("Failed to add document")
        else:
            st.warning("Could not extract content from this file")


# =============================================================================
# TAB 2: URL SCRAPING
# =============================================================================
with tab_url:
    st.markdown("### 🌐 Scrape Web Content")
    st.caption("Add content from websites to your knowledge base")
    
    url_input = st.text_input(
        "Enter URL",
        placeholder="https://scikit-learn.org/stable/modules/ensemble.html",
        help="Enter a URL to scrape and add to the knowledge base"
    )
    
    # Quick URL suggestions
    st.markdown("**Quick Add:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Pandas Docs", use_container_width=True):
            url_input = "https://pandas.pydata.org/docs/user_guide/index.html"
            st.session_state.url_to_scrape = url_input
    with col2:
        if st.button("🤖 Sklearn Docs", use_container_width=True):
            url_input = "https://scikit-learn.org/stable/user_guide.html"
            st.session_state.url_to_scrape = url_input
    with col3:
        if st.button("📈 Seaborn Docs", use_container_width=True):
            url_input = "https://seaborn.pydata.org/tutorial.html"
            st.session_state.url_to_scrape = url_input
    with col4:
        if st.button("📉 Statsmodels", use_container_width=True):
            url_input = "https://www.statsmodels.org/stable/user-guide.html"
            st.session_state.url_to_scrape = url_input
    
    # Use stored URL if set
    if "url_to_scrape" in st.session_state and not url_input:
        url_input = st.session_state.url_to_scrape
    
    if url_input and url_input.startswith("http"):
        if st.button("🔍 Preview Content", use_container_width=True):
            with st.spinner(f"Scraping {url_input}..."):
                title, content = run_async(scrape_url(url_input))
                
                if content:
                    st.session_state.scraped_content = content
                    st.session_state.scraped_title = title
                    st.session_state.scraped_url = url_input
                else:
                    st.error("Failed to scrape content from URL")
        
        # Show preview if we have scraped content
        if "scraped_content" in st.session_state and st.session_state.get("scraped_url") == url_input:
            content = st.session_state.scraped_content
            title = st.session_state.scraped_title
            
            st.success(f"✅ Scraped: **{title}**")
            
            # Preview
            st.markdown("**Preview:**")
            preview_text = content[:3000] + ("..." if len(content) > 3000 else "")
            st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{len(content):,}")
            with col2:
                word_count = len(content.split())
                st.metric("Words", f"{word_count:,}")
            with col3:
                chunk_estimate = max(1, len(content) // rag.chunker.chunk_size)
                st.metric("Est. Chunks", chunk_estimate)
            
            # Custom title
            custom_title = st.text_input(
                "Document Title",
                value=title,
                key="url_title",
                help="Title for this document in the knowledge base"
            )
            
            # Tags
            tags = st.text_input(
                "Tags (optional)",
                placeholder="e.g., documentation, tutorial, reference",
                key="url_tags"
            )
            
            # Add button
            if st.button("➕ Add to Knowledge Base", type="primary", use_container_width=True, key="add_url"):
                with st.spinner("Adding content..."):
                    metadata = {}
                    if tags:
                        metadata["tags"] = tags
                    
                    chunks_added = rag.add_url_content(
                        url=url_input,
                        content=content,
                        title=custom_title,
                        workflow_id=st.session_state.get("workflow_id", "default"),
                        metadata=metadata
                    )
                    
                    if chunks_added:
                        st.success(f"✅ Added **{custom_title}** ({chunks_added} chunks)")
                        # Clear cached content
                        del st.session_state.scraped_content
                        del st.session_state.scraped_title
                        del st.session_state.scraped_url
                        st.balloons()
                    else:
                        st.error("Failed to add content")
    
    elif url_input:
        st.warning("Please enter a valid URL starting with http:// or https://")


# =============================================================================
# TAB 3: MANAGE RESOURCES
# =============================================================================
with tab_manage:
    st.markdown("### 📋 Manage Resources")
    st.caption("View and manage documents in your knowledge base")
    
    # Filters
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Get all documents
    workflow_id = st.session_state.get("workflow_id")
    documents = rag.get_all_documents(workflow_id=workflow_id)
    
    if documents:
        # Stats
        total_docs = len(documents)
        total_chunks = sum(d['chunk_count'] for d in documents)
        total_chars = sum(d['content_length'] for d in documents)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", total_docs)
        with col2:
            st.metric("Total Chunks", total_chunks)
        with col3:
            st.metric("Total Content", f"{total_chars / 1000:.1f}K chars")
        with col4:
            # Type breakdown
            types = {}
            for d in documents:
                t = d['source_type']
                types[t] = types.get(t, 0) + 1
            st.metric("Types", ", ".join(f"{k}:{v}" for k, v in types.items()))
        
        st.divider()
        
        # Search within documents
        search_query = st.text_input(
            "🔍 Search documents",
            placeholder="Search within your uploaded documents...",
            key="doc_search"
        )
        
        if search_query:
            search_results = rag.query_documents(
                query=search_query,
                workflow_id=workflow_id,
                n_results=10
            )
            
            if search_results:
                st.success(f"Found {len(search_results)} matching chunks")
                
                for i, result in enumerate(search_results, 1):
                    meta = result['metadata']
                    with st.expander(f"**{i}. {meta.get('title', 'Unknown')}** ({meta.get('source_type', '')})", expanded=(i <= 3)):
                        # Relevance
                        if result.get('distance') is not None:
                            relevance = (1 - result['distance']) * 100
                            st.metric("Relevance", f"{relevance:.1f}%")
                        
                        # Content
                        st.text_area(
                            "Content",
                            value=result['document'],
                            height=150,
                            key=f"search_result_{i}",
                            label_visibility="collapsed"
                        )
            else:
                st.warning("No matching content found")
        
        st.divider()
        
        # Document list
        st.markdown("**All Documents:**")
        
        for doc in sorted(documents, key=lambda x: x['timestamp'] or '', reverse=True):
            source_type = doc['source_type']
            type_icon = {
                'pdf': '📕',
                'txt': '📄',
                'md': '📝',
                'url': '🌐',
                'csv': '📊',
                'json': '📋'
            }.get(source_type, '📄')
            
            type_class = f"resource-type-{source_type}"
            
            with st.expander(f"{type_icon} **{doc['title']}**", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(f"Type: {source_type}")
                    st.text(f"Chunks: {doc['chunk_count']}")
                with col2:
                    st.text(f"Size: {doc['content_length']:,} chars")
                    if doc.get('timestamp'):
                        st.text(f"Added: {doc['timestamp'][:10]}")
                with col3:
                    if doc.get('url'):
                        st.markdown(f"[🔗 Source]({doc['url']})")
                    if doc.get('source_path'):
                        st.text(f"File: {doc['source_path']}")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{doc['title']}_{doc.get('timestamp', '')}", type="secondary"):
                    if rag.delete_document(doc['title'], workflow_id):
                        st.success(f"Deleted: {doc['title']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete document")
    else:
        st.info("No documents added yet. Upload files or scrape URLs to get started!")
        
        # Quick start guide
        st.markdown("""
        ### 🚀 Quick Start
        
        **1. Upload Files**
        - Go to the **Upload Files** tab
        - Drop a PDF, TXT, or MD file
        - Preview and add to knowledge base
        
        **2. Scrape URLs**
        - Go to the **Scrape URL** tab
        - Enter a documentation URL
        - Preview content and add
        
        **3. Use in Analysis**
        - Documents are automatically available in Chat
        - RAG will retrieve relevant context
        - Ask questions about your uploaded content
        """)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 📚 About Resources")
    st.markdown("""
    Add external documents to enhance your analysis:
    
    **Supported Files:**
    - 📕 PDF documents
    - 📄 Text files (.txt)
    - 📝 Markdown (.md)
    - 📊 CSV files
    - 📋 JSON files
    
    **Web Scraping:**
    - Documentation pages
    - Tutorials
    - Reference materials
    
    **Usage:**
    Documents are automatically searched when you ask questions in Chat.
    """)
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Add methodology docs for better analysis
    - Upload relevant papers or guides
    - Scrape library documentation
    - Use tags to organize content
    """)
    
    st.divider()
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    docs = rag.get_all_documents(workflow_id=st.session_state.get("workflow_id"))
    st.metric("Documents", len(docs))
    
    rag_stats = rag.get_stats()
    st.metric("Total RAG Chunks", rag_stats.get("total_chunks", 0))