"""
Data Science Agent API

FastAPI backend for the data science workflow and RAG system.

Endpoints:
- /auth/* - Authentication (JWT, API keys)
- /workflow/* - Workflow execution
- /chat/* - Streaming chat with RAG
- /documents/* - Document upload and URL scraping
- /ws/* - WebSocket real-time updates
- /health - Health check
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys

from dependencies import lifespan, settings, check_ollama_health, RAGManager
from schemas import HealthResponse, ErrorResponse
from auth import init_auth

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)


# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Data Science Agent API",
    description="""
API for the AI-powered data science workflow system.

## Features

- **Authentication**: JWT tokens and API keys
- **Workflow Execution**: Run LangGraph workflows for data analysis
- **Streaming Chat**: Real-time chat with RAG context retrieval
- **Document Management**: Upload PDFs, text files, or scrape URLs
- **WebSocket**: Real-time progress updates

## Authentication

Most endpoints require authentication. Options:
1. **JWT Token**: POST to `/auth/token` with username/password
2. **API Key**: Create via `/auth/api-keys`, pass in `X-API-Key` header

Default credentials (change in production):
- Username: `admin`
- Password: `changeme123`
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {request.method} {request.url.path} [{response.status_code}]")
    return response


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    import traceback

    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500,
        ).model_dump(),
    )


# =============================================================================
# ROUTERS
# =============================================================================

from routers import workflow, chat, documents, websocket, auth_route, workflow_async, files, websocket_tasks

app.include_router(auth_route.router)
app.include_router(workflow.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(websocket.router)
app.include_router(workflow_async.router)
app.include_router(files.router)
app.include_router(websocket_tasks.router)

# =============================================================================
# ROOT ENDPOINTS
# =============================================================================


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "Data Science Agent API", "docs": "/docs"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """
    Check API health including RAG and Ollama status.
    """
    ollama_ok = await check_ollama_health()

    # Check default RAG
    rag_enabled = False
    rag_chunks = 0
    try:
        rag = await RAGManager.get_rag("default")
        if rag and rag.enabled:
            rag_enabled = True
            stats = rag.get_stats()
            rag_chunks = stats.get("total_chunks", 0)
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        version="1.0.0",
        rag_available=rag_enabled,
        rag_chunks=rag_chunks,
        ollama_connected=ollama_ok,
    )


@app.get(
    "/models",
    tags=["System"],
    summary="List available models",
)
async def list_models() -> dict:
    """Get list of available Ollama models."""
    from dependencies import get_available_models

    models = await get_available_models()
    return {
        "models": models,
        "default": settings.DEFAULT_MODEL,
    }


# =============================================================================
# DEVELOPMENT SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
