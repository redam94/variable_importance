"""
Documents Router - Document upload and URL scraping.

Provides:
- POST /documents/upload - Upload document to RAG
- POST /documents/scrape-url - Scrape URL and add to RAG
- GET /documents/list/{workflow_id} - List documents in workflow
- DELETE /documents/{workflow_id}/{title} - Delete document
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List, Annotated
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from loguru import logger

from schemas import (
    DocumentUploadResponse,
    DocumentSourceType,
    URLScrapeRequest,
    URLScrapeResponse,
    ErrorResponse,
)
from dependencies import RAGManager
from routers.auth_route import get_current_active_user, User


router = APIRouter(prefix="/documents", tags=["Documents"])


# =============================================================================
# HELPERS
# =============================================================================

def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{text}")
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF not installed. Install with: pip install pymupdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


def extract_text_file(file_bytes: bytes) -> str:
    """Extract text from text-based files with encoding detection."""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


async def scrape_url_content(url: str, timeout: int = 30) -> tuple:
    """Scrape URL content using crawl4ai."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        
        browser_cfg = BrowserConfig(headless=True, verbose=False)
        crawler_cfg = CrawlerRunConfig(
            word_count_threshold=50,
            excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
            remove_overlay_elements=True,
        )
        
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=crawler_cfg),
                timeout=timeout,
            )
            
            if result.success:
                title = result.metadata.get("title", url)
                content = result.markdown or ""
                return title, content
            else:
                return "", ""
                
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="crawl4ai not installed. Install with: pip install crawl4ai"
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"URL scraping timed out ({timeout}s)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {e}")


# =============================================================================
# UPLOAD ENDPOINT
# =============================================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload document to RAG",
    description="Upload a PDF, TXT, MD, CSV, or JSON file to add to the RAG knowledge base."
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    workflow_id: str = Form(..., description="Workflow to add document to"),
    title: Optional[str] = Form(None, description="Custom title (uses filename if not provided)"),
    current_user: User = Depends(get_current_active_user),
) -> DocumentUploadResponse:
    """
    Upload a document and add it to the RAG knowledge base.
    
    Supported formats:
    - PDF (.pdf)
    - Text (.txt)
    - Markdown (.md)
    - CSV (.csv)
    - JSON (.json)
    
    Requires authentication.
    """
    rag = await RAGManager.get_rag(workflow_id)
    
    if not rag or not rag.enabled:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    # Determine file type
    filename = file.filename or "document"
    extension = Path(filename).suffix.lower()
    
    type_map = {
        ".pdf": DocumentSourceType.PDF,
        ".txt": DocumentSourceType.TXT,
        ".md": DocumentSourceType.MD,
        ".csv": DocumentSourceType.CSV,
        ".json": DocumentSourceType.JSON,
    }
    
    if extension not in type_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported: {list(type_map.keys())}"
        )
    
    source_type = type_map[extension]
    doc_title = title or Path(filename).stem
    
    # Read file content
    file_bytes = await file.read()
    
    # Extract text based on file type
    if source_type == DocumentSourceType.PDF:
        content = extract_pdf_text(file_bytes)
    else:
        content = extract_text_file(file_bytes)
    
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="No text content extracted from file")
    
    # Add to RAG
    try:
        chunk_count = rag.add_document(
            content=content,
            title=doc_title,
            source_type=source_type.value,
            workflow_id=workflow_id,
            source_path=filename,
            metadata={"original_filename": filename},
        )
        
        logger.info(f"📄 Uploaded {doc_title}: {chunk_count} chunks, {len(content)} chars")
        
        return DocumentUploadResponse(
            success=True,
            title=doc_title,
            source_type=source_type,
            chunk_count=chunk_count or 0,
            content_length=len(content),
            workflow_id=workflow_id,
            message=f"Successfully added '{doc_title}' to knowledge base ({chunk_count} chunks)",
        )
        
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# URL SCRAPING ENDPOINT
# =============================================================================

@router.post(
    "/scrape-url",
    response_model=URLScrapeResponse,
    summary="Scrape URL and add to RAG",
    description="Scrape content from a URL and add it to the RAG knowledge base."
)
async def scrape_url(
    request: URLScrapeRequest,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> URLScrapeResponse:
    """
    Scrape a URL and add its content to RAG.
    
    Uses crawl4ai for intelligent web scraping with:
    - Markdown extraction
    - Navigation/footer removal
    - Clean text output
    
    Requires authentication.
    """
    rag = await RAGManager.get_rag(request.workflow_id)
    
    if not rag or not rag.enabled:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    # Scrape URL
    logger.info(f"🌐 Scraping URL: {request.url}")
    title, content = await scrape_url_content(request.url)
    
    if not content or not content.strip():
        raise HTTPException(
            status_code=400,
            detail="No content extracted from URL. The page may be empty or require JavaScript."
        )
    
    # Use custom title or scraped title
    doc_title = request.title or title or request.url
    
    # Add to RAG
    try:
        chunk_count = rag.add_url_content(
            url=request.url,
            content=content,
            title=doc_title,
            workflow_id=request.workflow_id,
            metadata={"scraped_title": title},
        )
        
        logger.info(f"🌐 Scraped {doc_title}: {chunk_count} chunks, {len(content)} chars")
        
        return URLScrapeResponse(
            success=True,
            url=request.url,
            title=doc_title,
            chunk_count=chunk_count or 0,
            content_length=len(content),
            workflow_id=request.workflow_id,
            message=f"Successfully scraped and added '{doc_title}' ({chunk_count} chunks)",
        )
        
    except Exception as e:
        logger.error(f"Failed to add URL content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LIST & DELETE
# =============================================================================

@router.get(
    "/list/{workflow_id}",
    summary="List documents in workflow",
    description="Get all documents added to a workflow's RAG."
)
async def list_documents(workflow_id: str) -> list:
    """List all documents in the workflow's knowledge base."""
    rag = await RAGManager.get_rag(workflow_id)
    
    if not rag or not rag.enabled:
        return []
    
    try:
        documents = rag.get_all_documents(workflow_id=workflow_id)
        return documents
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []


@router.delete(
    "/{workflow_id}/{title}",
    summary="Delete document",
    description="Delete a document from the workflow's RAG."
)
async def delete_document(workflow_id: str, title: str) -> dict:
    """Delete a document from the knowledge base."""
    rag = await RAGManager.get_rag(workflow_id)
    
    if not rag or not rag.enabled:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        success = rag.delete_document(title=title, workflow_id=workflow_id)
        
        if success:
            return {"success": True, "message": f"Deleted '{title}'"}
        else:
            raise HTTPException(status_code=404, detail=f"Document '{title}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BULK UPLOAD
# =============================================================================

@router.post(
    "/upload-multiple",
    summary="Upload multiple documents",
    description="Upload multiple documents at once."
)
async def upload_multiple_documents(
    files: List[UploadFile] = File(..., description="Documents to upload"),
    workflow_id: str = Form(..., description="Workflow to add documents to"),
) -> dict:
    """Upload multiple documents to RAG."""
    results = []
    errors = []
    
    for file in files:
        try:
            # Reuse single upload logic
            file_bytes = await file.read()
            filename = file.filename or "document"
            extension = Path(filename).suffix.lower()
            
            type_map = {
                ".pdf": "pdf",
                ".txt": "txt",
                ".md": "md",
                ".csv": "csv",
                ".json": "json",
            }
            
            if extension not in type_map:
                errors.append({"file": filename, "error": f"Unsupported type: {extension}"})
                continue
            
            source_type = type_map[extension]
            doc_title = Path(filename).stem
            
            # Extract content
            if source_type == "pdf":
                content = extract_pdf_text(file_bytes)
            else:
                content = extract_text_file(file_bytes)
            
            if not content.strip():
                errors.append({"file": filename, "error": "No content extracted"})
                continue
            
            # Add to RAG
            rag = await RAGManager.get_rag(workflow_id)
            if rag and rag.enabled:
                chunk_count = rag.add_document(
                    content=content,
                    title=doc_title,
                    source_type=source_type,
                    workflow_id=workflow_id,
                    source_path=filename,
                )
                results.append({
                    "file": filename,
                    "title": doc_title,
                    "chunks": chunk_count,
                    "size": len(content),
                })
                
        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})
    
    return {
        "uploaded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }