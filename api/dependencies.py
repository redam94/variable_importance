"""
API Dependencies - Shared state and dependency injection.

Provides singleton instances of:
- ContextRAG for knowledge retrieval
- Workflow for LangGraph execution
- OutputManager for file management
"""

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import httpx
from loguru import logger


# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings:
    """API configuration settings."""
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://100.91.155.118:11434"
    DEFAULT_MODEL: str = "qwen3:30b"
    CODE_MODEL: str = "qwen3-coder:30b"
    VISION_MODEL: str = "qwen3-vl:30b"
    
    # RAG
    RAG_PERSIST_DIR: str = "cache/rag_db"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    
    # Results
    RESULTS_DIR: str = "results"
    
    # API
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]


settings = Settings()


# =============================================================================
# SINGLETON MANAGERS
# =============================================================================

class RAGManager:
    """Manages ContextRAG instances per workflow."""
    
    _instances: Dict[str, Any] = {}
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_rag(cls, workflow_id: str):
        """Get or create RAG instance for workflow."""
        async with cls._lock:
            if workflow_id not in cls._instances:
                try:
                    # Import here to avoid circular imports
                    from variable_importance.memory.context_rag import ContextRAG
                    
                    cls._instances[workflow_id] = ContextRAG(
                        collection_name=workflow_id,
                        persist_directory=settings.RAG_PERSIST_DIR,
                        chunk_size=settings.RAG_CHUNK_SIZE,
                        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                    )
                    logger.info(f"📚 Created RAG instance for workflow: {workflow_id}")
                except ImportError as e:
                    logger.error(f"Failed to import ContextRAG: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Failed to create RAG: {e}")
                    return None
            
            return cls._instances[workflow_id]
    
    @classmethod
    def get_rag_sync(cls, workflow_id: str):
        """Synchronous version for non-async contexts."""
        if workflow_id not in cls._instances:
            try:
                from variable_importance.memory.context_rag import ContextRAG
                
                cls._instances[workflow_id] = ContextRAG(
                    collection_name=workflow_id,
                    persist_directory=settings.RAG_PERSIST_DIR,
                    chunk_size=settings.RAG_CHUNK_SIZE,
                    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                )
                logger.info(f"📚 Created RAG instance for workflow: {workflow_id}")
            except Exception as e:
                logger.error(f"Failed to create RAG: {e}")
                return None
        
        return cls._instances[workflow_id]


class WorkflowManager:
    """Manages workflow instances and execution."""
    
    _workflow = None
    _executor = None
    
    @classmethod
    def get_workflow(cls):
        """Get the compiled workflow graph."""
        if cls._workflow is None:
            try:
                from variable_importance.ai.workflow import workflow
                cls._workflow = workflow
                logger.info("✅ Workflow loaded")
            except ImportError as e:
                logger.error(f"Failed to import workflow: {e}")
                return None
        return cls._workflow
    
    @classmethod
    def get_executor(cls):
        """Get code executor instance."""
        if cls._executor is None:
            try:
                from variable_importance.utils.code_executer import OutputCapturingExecutor
                cls._executor = OutputCapturingExecutor(timeout=120)
                logger.info("✅ Executor initialized")
            except ImportError as e:
                logger.error(f"Failed to import executor: {e}")
                return None
        return cls._executor


class OutputManagerRegistry:
    """Manages OutputManager instances per workflow."""
    
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def get_output_manager(cls, workflow_id: str):
        """Get or create OutputManager for workflow."""
        if workflow_id not in cls._instances:
            try:
                from variable_importance.utils.output_manager import OutputManager
                cls._instances[workflow_id] = OutputManager(workflow_id=workflow_id)
                logger.info(f"📁 Created OutputManager for: {workflow_id}")
            except ImportError as e:
                logger.error(f"Failed to import OutputManager: {e}")
                return None
        return cls._instances[workflow_id]


# =============================================================================
# DEPENDENCY FUNCTIONS
# =============================================================================

async def get_rag(workflow_id: str):
    """Dependency to get RAG instance."""
    return await RAGManager.get_rag(workflow_id)


def get_workflow():
    """Dependency to get workflow."""
    return WorkflowManager.get_workflow()


def get_executor():
    """Dependency to get executor."""
    return WorkflowManager.get_executor()


def get_output_manager(workflow_id: str):
    """Dependency to get output manager."""
    return OutputManagerRegistry.get_output_manager(workflow_id)


async def check_ollama_health() -> bool:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


async def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [
                    m["id"] for m in data.get("data", [])
                    if "embedding" not in m["id"].lower()
                ]
    except Exception as e:
        logger.warning(f"Failed to fetch models: {e}")
    return [settings.DEFAULT_MODEL]


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app):
    """Manage startup and shutdown."""
    logger.info("🚀 Starting API...")
    
    # Initialize authentication
    from auth import init_auth
    init_auth()
    
    # Pre-warm commonly used components
    try:
        default_rag = await RAGManager.get_rag("default")
        if default_rag:
            logger.info(f"📚 Default RAG ready: {default_rag.get_stats().get('total_chunks', 0)} chunks")
    except Exception as e:
        logger.warning(f"Could not initialize default RAG: {e}")
    
    # Check Ollama
    if await check_ollama_health():
        logger.info(f"✅ Ollama connected at {settings.OLLAMA_BASE_URL}")
    else:
        logger.warning(f"⚠️ Ollama not reachable at {settings.OLLAMA_BASE_URL}")
    
    yield
    
    logger.info("👋 Shutting down API...")