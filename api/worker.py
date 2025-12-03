"""
arq Worker - Runs workflows in a separate process.

Start with:
    arq worker.WorkerSettings
    
Or:
    python -m worker
    
This runs OUTSIDE of FastAPI, so long-running workflows
don't block the API server.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from arq import cron
from arq.connections import RedisSettings
from loguru import logger
import redis.asyncio as redis

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_queue import TaskQueue, TaskStatus, TaskInfo


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def get_redis_settings() -> RedisSettings:
    """Parse REDIS_URL into RedisSettings."""
    url = REDIS_URL.replace("redis://", "")
    if "/" in url:
        host_port, db = url.rsplit("/", 1)
        db = int(db)
    else:
        host_port = url
        db = 0
    
    if ":" in host_port:
        host, port = host_port.split(":")
        port = int(port)
    else:
        host = host_port
        port = 6379
    
    return RedisSettings(host=host, port=port, database=db)


# -----------------------------------------------------------------------------
# Worker Context
# -----------------------------------------------------------------------------

async def startup(ctx: dict) -> None:
    """
    arq startup hook - runs once when worker starts.
    
    Creates Redis connection on the WORKER's event loop.
    """
    logger.info("🚀 Initializing worker...")
    
    # Create Redis connection on THIS event loop
    ctx["redis"] = redis.from_url(REDIS_URL, decode_responses=True)
    await ctx["redis"].ping()
    logger.info("✅ Redis connected")
    
    # Create task queue with existing connection
    ctx["task_queue"] = TaskQueue(redis_url=REDIS_URL)
    ctx["task_queue"]._redis = ctx["redis"]  # Reuse connection
    logger.info("✅ Task queue ready")
    
    # Import workflow components
    try:
        from dependencies import WorkflowManager, settings
        
        ctx["workflow"] = WorkflowManager.get_workflow()
        ctx["executor"] = WorkflowManager.get_executor()
        ctx["settings"] = settings
        
        logger.info("✅ Workflow components loaded")
    except ImportError as e:
        logger.warning(f"Could not import workflow components: {e}")
        ctx["workflow"] = None
        ctx["executor"] = None
        ctx["settings"] = None
    
    logger.info("✅ Worker initialized")


async def shutdown(ctx: dict) -> None:
    """arq shutdown hook - runs when worker stops."""
    if ctx.get("redis"):
        await ctx["redis"].close()
    logger.info("👋 Worker shutdown")


# -----------------------------------------------------------------------------
# Progress Emitter for Worker
# -----------------------------------------------------------------------------

class WorkerEmitter:
    """
    Emitter that publishes progress to Redis pub/sub.
    
    FastAPI subscribes and forwards to WebSocket clients.
    
    Uses Redis connection directly to avoid event loop issues.
    """
    
    PROGRESS_CHANNEL = "task:{task_id}:progress"
    
    def __init__(self, redis_conn: redis.Redis, task_id: str, workflow_id: str):
        self._redis = redis_conn
        self.task_id = task_id
        self.workflow_id = workflow_id
    
    async def emit(self, event_type: str, stage: str, message: str, data: Optional[dict] = None):
        """Emit progress event via Redis pub/sub."""
        import json
        channel = self.PROGRESS_CHANNEL.format(task_id=self.task_id)
        payload = {
            "type": event_type,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if data:
            payload["data"] = data
        
        await self._redis.publish(channel, json.dumps(payload))
    
    async def stage_start(self, stage: str, description: str = ""):
        await self.emit("stage_start", stage, description)
    
    async def stage_end(self, stage: str, success: bool = True):
        status = "completed" if success else "failed"
        await self.emit("stage_end", stage, status)
    
    async def rag_search_start(self, stage: str):
        await self.emit("rag_search_start", stage, "Starting RAG search...")
    
    async def rag_query(self, stage: str, query: str, iteration: int, chunks_found: int, relevance: float = None):
        await self.emit("rag_query", stage, f"Query {iteration}: {chunks_found} chunks", {
            "query": query,
            "iteration": iteration,
            "chunks_found": chunks_found,
            "relevance": relevance,
        })
    
    async def rag_search_end(self, stage: str, total_iterations: int, total_chunks: int, 
                             final_relevance: float, accepted: bool):
        await self.emit("rag_search_end", stage, "RAG search complete", {
            "total_iterations": total_iterations,
            "total_chunks": total_chunks,
            "final_relevance": final_relevance,
            "accepted": accepted,
        })
    
    async def code_generated(self, stage: str, code: str, line_count: int):
        await self.emit("code_generated", stage, f"Generated {line_count} lines")
    
    async def execution_result(self, stage: str, success: bool, output: str = ""):
        status = "success" if success else "error"
        await self.emit("execution_result", stage, status, {"output": output[:500]})
    
    async def error(self, stage: str, error: str):
        await self.emit("error", stage, error)


# -----------------------------------------------------------------------------
# Main Task Function
# -----------------------------------------------------------------------------

async def execute_workflow(ctx: dict, task_id: str) -> Dict[str, Any]:
    """
    Execute a workflow task.
    
    This is the main arq job function. It runs in the worker process,
    completely separate from FastAPI.
    
    ctx contains Redis connection and workflow components from startup().
    """
    redis_conn: redis.Redis = ctx["redis"]
    task_queue: TaskQueue = ctx["task_queue"]
    workflow = ctx.get("workflow")
    executor = ctx.get("executor")
    settings = ctx.get("settings")
    
    # Load task info from Redis
    task = await task_queue.get_task(task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    
    logger.info(f"🚀 Starting workflow: {task.workflow_id} (task: {task_id})")
    
    # Create emitter with Redis connection from ctx
    emitter = WorkerEmitter(redis_conn, task_id, task.workflow_id)
    
    # Update status to running
    await task_queue.update_task(task_id, status=TaskStatus.RUNNING, progress=5)
    await emitter.emit("progress", "init", "Workflow starting...")
    
    async def check_cancelled() -> bool:
        """Check if we should stop."""
        if await task_queue.is_cancelled(task_id):
            await emitter.emit("progress", "cancel", "Cancellation requested...")
            return True
        return False
    
    try:
        # Checkpoint 1: Init
        if await check_cancelled():
            await task_queue.acknowledge_cancel(task_id)
            return {"cancelled": True}
        
        # Check if workflow is available
        if not workflow:
            raise RuntimeError("Workflow not initialized. Check worker startup logs.")
        
        # Initialize dependencies
        await task_queue.update_task(task_id, current_node="init", progress=10)
        
        from dependencies import (
            RAGManager,
            OutputManagerRegistry,
        )
        
        output_mgr = OutputManagerRegistry.get_output_manager(task.workflow_id)
        
        # Checkpoint 2: RAG init
        if await check_cancelled():
            await task_queue.acknowledge_cancel(task_id)
            return {"cancelled": True}
        
        rag = None
        if task.rag_enabled:
            await task_queue.update_task(task_id, current_node="rag_init", progress=15)
            rag = await RAGManager.get_rag(task.workflow_id)
        
        # Checkpoint 3: Pre-execution
        if await check_cancelled():
            await task_queue.acknowledge_cancel(task_id)
            return {"cancelled": True}
        
        await task_queue.update_task(task_id, current_node="preparing", progress=20)
        
        # Build dependencies
        from langchain.messages import HumanMessage
        
        deps = {
            "executor": executor,
            "output_manager": output_mgr,
            "rag": rag,
            "llm": settings.DEFAULT_MODEL,
            "code_llm": settings.CODE_MODEL,
            "vision_llm": settings.VISION_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
            "max_retries": 3,
            "emitter": emitter,
            "workflow_id": task.workflow_id,
            "task_id": task_id,
            "check_cancelled": check_cancelled,
            "user": task.username,
        }
        
        initial_state = {
            "messages": [HumanMessage(content=task.query)],
            "data_path": task.data_path or "",
            "stage_name": task.stage_name,
            "workflow_id": task.workflow_id,
            "web_search_enabled": task.web_search_enabled,
            "rag_enabled": task.rag_enabled,
        }
        
        logger.info(f"📊 Executing: data_path={task.data_path}, rag={task.rag_enabled}")
        await emitter.stage_start("workflow", "Executing workflow...")
        await task_queue.update_task(task_id, current_node="executing", progress=25)
        
        # Main execution
        result = await workflow.ainvoke(initial_state, context=deps)
        
        # Checkpoint 4: Post-execution
        if await check_cancelled():
            await task_queue.acknowledge_cancel(task_id)
            return {"cancelled": True}
        
        summary = result.get("summary", "")
        action = result.get("action", "unknown")
        error = result.get("error", "")
        
        if error:
            await emitter.error("workflow", error)
            await task_queue.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=error,
                progress=100,
            )
        else:
            await task_queue.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                result_summary=summary,
                progress=100,
            )
        
        await emitter.stage_end("workflow", success=not error)
        
        # Publish completion event
        await emitter.emit("done", "workflow", "completed" if not error else "error", {
            "summary": summary,
            "action": action,
        })
        
        logger.info(f"✅ Workflow completed: {task.workflow_id}")
        
        return {
            "summary": summary,
            "action": action,
            "error": error,
        }
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        
        await emitter.error("workflow", str(e))
        await task_queue.update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e),
        )
        
        raise


# -----------------------------------------------------------------------------
# Periodic Tasks
# -----------------------------------------------------------------------------

async def cleanup_old_tasks(ctx: dict) -> int:
    """Clean up completed tasks older than 7 days."""
    task_queue = ctx.get("task_queue")
    if not task_queue:
        return 0
    
    # Implementation would scan and delete old tasks
    logger.info("🧹 Running task cleanup...")
    return 0


# -----------------------------------------------------------------------------
# Worker Settings
# -----------------------------------------------------------------------------

class WorkerSettings:
    """arq worker configuration."""
    
    functions = [execute_workflow]
    
    cron_jobs = [
        cron(cleanup_old_tasks, hour=3, minute=0),  # Run at 3am daily
    ]
    
    on_startup = startup
    on_shutdown = shutdown
    
    redis_settings = get_redis_settings()
    
    # Worker config
    max_jobs = 4  # Max concurrent jobs per worker
    job_timeout = 3600  # 1 hour max per job
    keep_result = 3600  # Keep results for 1 hour
    
    # Health check
    health_check_interval = 30


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    subprocess.run(["arq", "worker.WorkerSettings"])