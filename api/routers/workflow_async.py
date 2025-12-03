"""
Workflow Async Router - Enqueues jobs to arq worker.

Jobs execute in a separate worker process, not blocking FastAPI.

Endpoints:
- POST /workflow/run-async - Enqueue workflow job
- GET /workflow/status/{task_id} - Get task status
- POST /workflow/cancel/{task_id} - Cancel task
- GET /workflow/tasks/{workflow_id} - List workflow tasks
- GET /workflow/active - List all active tasks
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from schemas import WorkflowRequest, WorkflowStatus
from auth import get_current_active_user, User
from task_queue import get_task_queue, TaskQueue, TaskStatus


router = APIRouter(prefix="/workflow", tags=["Workflow Async"])


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@router.post(
    "/run-async",
    response_model=WorkflowStatus,
    summary="Start workflow asynchronously",
)
async def run_workflow_async(
    request: WorkflowRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    task_queue: TaskQueue = Depends(get_task_queue),
) -> WorkflowStatus:
    """
    Enqueue workflow for background execution.

    The workflow runs in a SEPARATE WORKER PROCESS, not blocking FastAPI.
    
    Returns task_id immediately. Use:
    - GET /workflow/status/{task_id} to poll status
    - WebSocket /ws/workflow/{workflow_id} for real-time updates
    - POST /workflow/cancel/{task_id} to cancel
    
    Requires worker to be running:
        arq worker.WorkerSettings
    """
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    
    # Enqueue job - returns immediately
    task = await task_queue.enqueue_workflow(
        task_id=task_id,
        workflow_id=request.workflow_id,
        query=request.query,
        username=current_user.username,
        stage_name=request.stage_name or "main",
        data_path=request.data_path,
        web_search_enabled=request.web_search_enabled,
        rag_enabled=request.rag_enabled,
    )
    
    logger.info(f"📋 Enqueued: {request.workflow_id} -> {task_id}")
    
    return WorkflowStatus(
        task_id=task_id,
        workflow_id=request.workflow_id,
        stage_name=request.stage_name or "main",
        status=task.status.value,
        progress=0,
    )


@router.get(
    "/status/{task_id}",
    response_model=WorkflowStatus,
    summary="Get task status",
)
async def get_task_status(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    task_queue: TaskQueue = Depends(get_task_queue),
) -> WorkflowStatus:
    """Get current status of a workflow task."""
    task = await task_queue.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Calculate elapsed time
    elapsed = None
    if task.started_at:
        start = datetime.fromisoformat(task.started_at)
        end = datetime.fromisoformat(task.completed_at) if task.completed_at else datetime.now()
        elapsed = (end - start).total_seconds()
    
    return WorkflowStatus(
        task_id=task_id,
        workflow_id=task.workflow_id,
        stage_name=task.stage_name,
        status=task.status.value,
        current_node=task.current_node,
        progress=task.progress,
        elapsed_seconds=elapsed,
    )


@router.post(
    "/cancel/{task_id}",
    summary="Cancel a running task",
)
async def cancel_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    task_queue: TaskQueue = Depends(get_task_queue),
) -> dict:
    """
    Request cancellation of a task.
    
    For queued tasks: removes from queue immediately.
    For running tasks: sets cancel flag, worker stops at next checkpoint.
    """
    task = await task_queue.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Ownership check
    if task.username and task.username != current_user.username:
        raise HTTPException(status_code=403, detail="Cannot cancel another user's task")
    
    success = await task_queue.request_cancel(task_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in state: {task.status.value}"
        )
    
    return {
        "task_id": task_id,
        "status": "cancelling",
        "message": "Cancellation requested. Task will stop at next checkpoint.",
    }


@router.get(
    "/tasks/{workflow_id}",
    summary="List tasks for workflow",
)
async def list_workflow_tasks(
    workflow_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    task_queue: TaskQueue = Depends(get_task_queue),
) -> list[dict]:
    """Get all tasks for a workflow, sorted newest first."""
    tasks = await task_queue.get_workflow_tasks(workflow_id)
    
    return [
        {
            "task_id": t.task_id,
            "status": t.status.value,
            "progress": t.progress,
            "current_node": t.current_node,
            "created_at": t.created_at,
            "started_at": t.started_at,
            "completed_at": t.completed_at,
            "error": t.error,
            "result_summary": t.result_summary,
            "username": t.username,
        }
        for t in tasks
    ]


@router.get(
    "/active",
    summary="List all active tasks",
)
async def list_active_tasks(
    current_user: Annotated[User, Depends(get_current_active_user)],
    task_queue: TaskQueue = Depends(get_task_queue),
) -> list[dict]:
    """Get all running/pending/queued tasks."""
    tasks = await task_queue.get_active_tasks()
    
    return [
        {
            "task_id": t.task_id,
            "workflow_id": t.workflow_id,
            "status": t.status.value,
            "progress": t.progress,
            "current_node": t.current_node,
            "created_at": t.created_at,
            "username": t.username,
        }
        for t in tasks
    ]


@router.get(
    "/result/{task_id}",
    summary="Get task result",
)
async def get_task_result(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    task_queue: TaskQueue = Depends(get_task_queue),
) -> dict:
    """Get result of a completed task."""
    task = await task_queue.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        raise HTTPException(
            status_code=400,
            detail=f"Task not complete. Status: {task.status.value}"
        )
    
    return {
        "task_id": task_id,
        "workflow_id": task.workflow_id,
        "status": task.status.value,
        "result_summary": task.result_summary,
        "error": task.error,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
    }