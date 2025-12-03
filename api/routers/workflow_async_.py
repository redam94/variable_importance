"""
Workflow Router - Async execution with emitter injection.

Provides:
- POST /workflow/run-async - Start workflow in background with WebSocket updates
- GET /workflow/status/{task_id} - Check workflow task status
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Annotated

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from loguru import logger
from langchain.messages import HumanMessage

from schemas import WorkflowRequest, WorkflowStatus
from dependencies import (
    RAGManager,
    WorkflowManager,
    OutputManagerRegistry,
    settings,
)
from auth import get_current_active_user, User
from emitter import create_emitter, EventType
from routers.websocket import manager as ws_manager


router = APIRouter(prefix="/workflow", tags=["Workflow"])

_running_tasks: Dict[str, Dict[str, Any]] = {}


async def execute_workflow_with_emitter(
    task_id: str,
    workflow_id: str,
    query: str,
    model: Optional[str],
    stage_name: str,
    data_path: Optional[str],
    web_search_enabled: bool,
    rag_enabled: bool,
    username: str,
):
    """
    Execute workflow in background with emitter for progress updates.
    
    The emitter is injected into the workflow deps, allowing nodes
    to emit RAG search events, code generation updates, etc.
    """
    emitter = create_emitter(workflow_id, task_id, ws_manager)

    _running_tasks[task_id] = {
        "status": "running",
        "workflow_id": workflow_id,
        "started_at": datetime.now(),
        "current_node": None,
        "progress": 0,
    }

    try:
        await emitter.emit(EventType.PROGRESS, "init", "Workflow starting...")

        workflow = WorkflowManager.get_workflow()
        if not workflow:
            raise RuntimeError("Workflow not available")

        executor = WorkflowManager.get_executor()
        output_mgr = OutputManagerRegistry.get_output_manager(workflow_id)
        
        # Only initialize RAG if enabled
        rag = None
        if rag_enabled:
            rag = await RAGManager.get_rag(workflow_id)

        deps = {
            "executor": executor,
            "output_manager": output_mgr,
            "rag": rag,
            "llm": model or settings.DEFAULT_MODEL,
            "code_llm": settings.CODE_MODEL,
            "vision_llm": settings.VISION_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
            "max_retries": 3,
            "emitter": emitter,
            "workflow_id": workflow_id,
            "user": username,
        }

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "data_path": data_path or "",
            "stage_name": stage_name,
            "workflow_id": workflow_id,
            "web_search_enabled": web_search_enabled,
            "rag_enabled": rag_enabled,
        }

        logger.info(f"🚀 Running async workflow: {workflow_id}")
        logger.info(f"   data_path: {data_path}")
        logger.info(f"   rag_enabled: {rag_enabled}, web_search_enabled: {web_search_enabled}")

        await emitter.stage_start("workflow", "Starting workflow...")

        result = await workflow.ainvoke(initial_state, context=deps)

        summary = result.get("summary", "")
        action = result.get("action", "unknown")
        error = result.get("error", "")

        if error:
            await emitter.error("workflow", error)

        _running_tasks[task_id]["status"] = "completed"
        _running_tasks[task_id]["progress"] = 100
        _running_tasks[task_id]["completed_at"] = datetime.now()

        await emitter.stage_end("workflow", success=not error)

        await ws_manager.send_to_workflow(workflow_id, {
            "type": "done",
            "task_id": task_id,
            "status": "completed" if not error else "error",
            "summary": summary,
            "action": action,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(f"✅ Async workflow completed: {workflow_id}")

    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()

        _running_tasks[task_id]["status"] = "failed"
        _running_tasks[task_id]["error"] = str(e)

        await emitter.error("workflow", str(e))

        await ws_manager.send_to_workflow(workflow_id, {
            "type": "error",
            "task_id": task_id,
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        })


@router.post(
    "/run-async",
    response_model=WorkflowStatus,
    summary="Start workflow asynchronously",
)
async def run_workflow_async(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> WorkflowStatus:
    """
    Start workflow execution in background.

    Returns task_id immediately. Connect to WebSocket at
    /ws/workflow/{workflow_id} to receive real-time updates.

    Frontend flow:
    1. POST /workflow/run-async -> get task_id
    2. Connect WebSocket /ws/workflow/{workflow_id}
    3. Receive events: stage_start, rag_query, rag_search_end, etc.
    4. Poll /workflow/status/{task_id} for completion
    """
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    workflow_id = request.workflow_id

    _running_tasks[task_id] = {
        "status": "pending",
        "workflow_id": workflow_id,
        "created_at": datetime.now(),
        "progress": 0,
    }

    logger.info(f"📋 Queuing workflow: {workflow_id}, data_path: {request.data_path}")

    background_tasks.add_task(
        execute_workflow_with_emitter,
        task_id=task_id,
        workflow_id=workflow_id,
        query=request.query,
        model=None,
        stage_name=request.stage_name or "main",
        data_path=request.data_path,
        web_search_enabled=request.web_search_enabled,
        rag_enabled=request.rag_enabled,
        username=current_user.username,
    )

    return WorkflowStatus(
        task_id=task_id,
        workflow_id=workflow_id,
        stage_name=request.stage_name or "main",
        status="pending",
        progress=0,
    )


@router.get(
    "/status/{task_id}",
    response_model=WorkflowStatus,
    summary="Check workflow task status",
)
async def get_task_status(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> WorkflowStatus:
    """Get status of a running or completed workflow task."""
    if task_id not in _running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _running_tasks[task_id]

    elapsed = None
    if task.get("started_at"):
        end_time = task.get("completed_at", datetime.now())
        elapsed = (end_time - task["started_at"]).total_seconds()

    return WorkflowStatus(
        task_id=task_id,
        workflow_id=task["workflow_id"],
        stage_name="main",
        status=task["status"],
        current_node=task.get("current_node"),
        progress=task.get("progress", 0),
        elapsed_seconds=elapsed,
    )