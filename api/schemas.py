"""
API Schemas - Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowAction(str, Enum):
    """Possible workflow actions."""

    EXECUTE = "execute"
    ANSWER = "answer"


class DocumentSourceType(str, Enum):
    """Supported document source types."""

    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    JSON = "json"
    URL = "url"


# =============================================================================
# WORKFLOW SCHEMAS
# =============================================================================

class DataFileUploadResponse(BaseModel):
    """Response after uploading a data file for workflow analysis."""
    
    success: bool
    filename: str
    file_path: str = Field(..., description="Temporary path to use as data_path in workflow")
    file_size: int
    content_type: str
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "filename": "sales_data.csv",
                "file_path": "/tmp/workflow_data/abc123/sales_data.csv",
                "file_size": 1024,
                "content_type": "text/csv",
                "message": "File uploaded successfully"
            }
        }
    }

class WorkflowRequest(BaseModel):
    """Request to run the workflow."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    stage_name: str = Field(
        default="analysis", description="Stage name within workflow"
    )
    query: str = Field(..., description="User query to process")
    data_path: Optional[str] = Field("/Users/redam94/Coding/Ideas/variable_importance/results/workflow_20251129_122152/analysis/execution/tmpaty2dvcy.csv", description="Path to data file (CSV)")
    web_search_enabled: bool = Field(default=True, description="Enable web search")

    model_config = {
        "json_schema_extra": {
            "example": {
                "workflow_id": "analysis_001",
                "stage_name": "exploration",
                "query": "Analyze the correlation between price and sales",
                "data_path": "/data/sales.csv",
                "web_search_enabled": False,
            }
        }
    }


class WorkflowMessage(BaseModel):
    """A message from the workflow."""

    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Response from workflow execution."""

    workflow_id: str
    stage_name: str
    status: str = Field(..., description="Status: completed, error, in_progress")
    message: WorkflowMessage
    action_taken: Optional[str] = None
    code_executed: Optional[str] = None
    plots: Optional[List[str]] = None
    summary: Optional[str] = None
    error: Optional[str] = None


class WorkflowStatus(BaseModel):
    """Status of a running workflow."""

    workflow_id: str
    stage_name: str
    status: str
    current_node: Optional[str] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    elapsed_seconds: Optional[float] = None


# =============================================================================
# CHAT SCHEMAS
# =============================================================================


class ChatMessage(BaseModel):
    """Chat message for streaming endpoint."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request for streaming chat with RAG."""

    workflow_id: str = Field(..., description="Workflow context for RAG")
    message: str = Field(..., description="User message")
    history: Optional[List[ChatMessage]] = Field(
        default=None, description="Previous messages"
    )
    model: Optional[str] = Field(default="qwen3:30b", description="LLM model to use")

    model_config = {
        "json_schema_extra": {
            "example": {
                "workflow_id": "analysis_001",
                "message": "What patterns did we find in the sales data?",
                "history": [
                    {"role": "user", "content": "Analyze sales trends"},
                    {"role": "assistant", "content": "I found three key patterns..."},
                ],
            }
        }
    }


class ChatChunk(BaseModel):
    """Streaming chat chunk."""

    content: str
    done: bool = False
    tool_call: Optional[str] = None


# =============================================================================
# DOCUMENT SCHEMAS
# =============================================================================


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    success: bool
    title: str
    source_type: DocumentSourceType
    chunk_count: int
    content_length: int
    workflow_id: str
    message: str


class URLScrapeRequest(BaseModel):
    """Request to scrape a URL."""

    url: str = Field(..., description="URL to scrape")
    workflow_id: str = Field(..., description="Workflow to add content to")
    title: Optional[str] = Field(
        None, description="Custom title (auto-detected if not provided)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://docs.python.org/3/tutorial/datastructures.html",
                "workflow_id": "analysis_001",
                "title": "Python Data Structures",
            }
        }
    }


class URLScrapeResponse(BaseModel):
    """Response after URL scraping."""

    success: bool
    url: str
    title: str
    chunk_count: int
    content_length: int
    workflow_id: str
    message: str


# =============================================================================
# RAG SCHEMAS
# =============================================================================


class RAGQueryRequest(BaseModel):
    """Request to query RAG directly."""

    query: str = Field(..., description="Search query")
    workflow_id: Optional[str] = None
    n_results: int = Field(default=5, ge=1, le=50)
    doc_types: Optional[List[str]] = None


class RAGChunk(BaseModel):
    """A single RAG result chunk."""

    content: str
    metadata: Dict[str, Any]
    relevance_score: float


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""

    query: str
    results: List[RAGChunk]
    total_found: int


# =============================================================================
# HEALTH & STATUS
# =============================================================================


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    rag_enabled: bool = False
    rag_chunks: int = 0
    ollama_reachable: bool = False


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    status_code: int
