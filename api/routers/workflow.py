"""
Workflow Router - Endpoints for workflow execution.

Provides:
- POST /workflow/run - Execute workflow and return latest message
- GET /workflow/status/{task_id} - Check workflow status
- GET /workflow/stages/{workflow_id} - List stages for workflow
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Annotated
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from loguru import logger

from langchain.messages import HumanMessage, AIMessage

from schemas import (
    WorkflowRequest,
    WorkflowResponse,
    WorkflowMessage,
    WorkflowStatus,
    ErrorResponse,
)
from dependencies import (
    RAGManager,
    WorkflowManager,
    OutputManagerRegistry,
    settings,
)
from api.routers.auth_route import (
    get_current_user_optional,
    get_current_active_user,
    User,
)


router = APIRouter(prefix="/workflow", tags=["Workflow"])


# Task tracking
_running_tasks: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================


@router.post(
    "/run",
    response_model=WorkflowResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Execute workflow with query",
    description="Run the data analysis workflow with the given query and return the result.",
)
async def run_workflow(
    request: WorkflowRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> WorkflowResponse:
    """
    Execute the workflow and return the most recent message.

    This endpoint:
    1. Initializes RAG context for the workflow
    2. Runs the LangGraph workflow (gather → plan → execute/answer → summarize)
    3. Returns the summary or error message

    Requires authentication.
    """
    workflow = WorkflowManager.get_workflow()
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not available")

    executor = WorkflowManager.get_executor()
    output_mgr = OutputManagerRegistry.get_output_manager(request.workflow_id)
    rag = await RAGManager.get_rag(request.workflow_id)

    # Build initial state
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "data_path": request.data_path or "",
        "stage_name": request.stage_name,
        "workflow_id": request.workflow_id,
        "web_search_enabled": request.web_search_enabled,
    }

    # Build dependencies
    deps = {
        "executor": executor,
        "output_manager": output_mgr,
        "rag": rag,
        "llm": settings.DEFAULT_MODEL,
        "code_llm": settings.CODE_MODEL,
        "vision_llm": settings.VISION_MODEL,
        "base_url": settings.OLLAMA_BASE_URL,
        "max_retries": 3,
    }

    try:
        logger.info(f"🚀 Running workflow: {request.workflow_id}/{request.stage_name}")
        logger.info(f"   Query: {request.query[:100]}...")

        start_time = datetime.now()

        # Execute workflow
        result = await workflow.ainvoke(initial_state, context=deps)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Workflow completed in {elapsed:.2f}s")

        # Extract results
        summary = result.get("summary", "")
        action = result.get("action", "unknown")
        code = result.get("code", "")
        error = result.get("error", "")

        # Get plots if any
        plots = []
        context = result.get("context", {})
        if isinstance(context, dict):
            plots = context.get("plots", [])

        # Build response message
        if error:
            message_content = (
                f"Error: {error}\n\n{summary}" if summary else f"Error: {error}"
            )
            status = "error"
        else:
            message_content = summary or "Workflow completed"
            status = "completed"

        return WorkflowResponse(
            workflow_id=request.workflow_id,
            stage_name=request.stage_name,
            status=status,
            message=WorkflowMessage(
                role="assistant",
                content=message_content,
                timestamp=datetime.now(),
                metadata={"action": action, "elapsed_seconds": elapsed},
            ),
            action_taken=action,
            code_executed=code if code else None,
            plots=plots if plots else None,
            summary=summary if summary else None,
            error=error if error else None,
        )

    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()

        return WorkflowResponse(
            workflow_id=request.workflow_id,
            stage_name=request.stage_name,
            status="error",
            message=WorkflowMessage(
                role="assistant",
                content=f"Workflow execution failed: {str(e)}",
                timestamp=datetime.now(),
            ),
            error=str(e),
        )


# =============================================================================
# ASYNC WORKFLOW (Background Task)
# =============================================================================


@router.post(
    "/run-async",
    response_model=WorkflowStatus,
    summary="Start workflow asynchronously",
    description="Start workflow in background and return task ID for status polling.",
)
async def run_workflow_async(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> WorkflowStatus:
    """
    Start workflow execution in background.

    Returns task ID for polling status via /workflow/status/{task_id}

    Requires authentication.
    """
    task_id = f"{request.workflow_id}_{request.stage_name}_{datetime.now().strftime('%H%M%S%f')}"

    # Initialize task tracking
    _running_tasks[task_id] = {
        "workflow_id": request.workflow_id,
        "stage_name": request.stage_name,
        "status": "starting",
        "current_node": None,
        "progress": 0.0,
        "start_time": datetime.now(),
        "result": None,
        "error": None,
    }

    async def run_in_background():
        try:
            _running_tasks[task_id]["status"] = "running"

            # Run workflow (reuse logic from run_workflow)
            workflow = WorkflowManager.get_workflow()
            executor = WorkflowManager.get_executor()
            output_mgr = OutputManagerRegistry.get_output_manager(request.workflow_id)
            rag = await RAGManager.get_rag(request.workflow_id)

            initial_state = {
                "messages": [HumanMessage(content=request.query)],
                "data_path": request.data_path or "",
                "stage_name": request.stage_name,
                "workflow_id": request.workflow_id,
                "web_search_enabled": request.web_search_enabled,
            }

            deps = {
                "executor": executor,
                "output_manager": output_mgr,
                "rag": rag,
                "llm": settings.DEFAULT_MODEL,
                "base_url": settings.OLLAMA_BASE_URL,
            }

            result = await workflow.ainvoke(initial_state, context=deps)

            _running_tasks[task_id]["status"] = "completed"
            _running_tasks[task_id]["result"] = result
            _running_tasks[task_id]["progress"] = 1.0

        except Exception as e:
            _running_tasks[task_id]["status"] = "error"
            _running_tasks[task_id]["error"] = str(e)
            logger.error(f"Background task {task_id} failed: {e}")

    # Schedule background execution
    background_tasks.add_task(run_in_background)

    return WorkflowStatus(
        workflow_id=request.workflow_id,
        stage_name=request.stage_name,
        status="starting",
        progress=0.0,
    )


@router.get(
    "/status/{task_id}",
    response_model=WorkflowStatus,
    summary="Get workflow task status",
)
async def get_workflow_status(task_id: str) -> WorkflowStatus:
    """Get status of a background workflow task."""
    if task_id not in _running_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _running_tasks[task_id]
    elapsed = (datetime.now() - task["start_time"]).total_seconds()

    return WorkflowStatus(
        workflow_id=task["workflow_id"],
        stage_name=task["stage_name"],
        status=task["status"],
        current_node=task.get("current_node"),
        progress=task["progress"],
        elapsed_seconds=elapsed,
    )


@router.get(
    "/result/{task_id}",
    response_model=WorkflowResponse,
    summary="Get completed workflow result",
)
async def get_workflow_result(task_id: str) -> WorkflowResponse:
    """Get result of a completed background workflow task."""
    if task_id not in _running_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _running_tasks[task_id]

    if task["status"] not in ("completed", "error"):
        raise HTTPException(
            status_code=400, detail=f"Task not complete. Status: {task['status']}"
        )

    if task["status"] == "error":
        return WorkflowResponse(
            workflow_id=task["workflow_id"],
            stage_name=task["stage_name"],
            status="error",
            message=WorkflowMessage(
                role="assistant",
                content=f"Error: {task['error']}",
                timestamp=datetime.now(),
            ),
            error=task["error"],
        )

    result = task["result"]

    return WorkflowResponse(
        workflow_id=task["workflow_id"],
        stage_name=task["stage_name"],
        status="completed",
        message=WorkflowMessage(
            role="assistant",
            content=result.get("summary", "Completed"),
            timestamp=datetime.now(),
        ),
        action_taken=result.get("action"),
        code_executed=result.get("code"),
        summary=result.get("summary"),
    )


# =============================================================================
# WORKFLOW INFO
# =============================================================================


@router.get(
    "/stages/{workflow_id}",
    summary="List workflow stages",
    description="Get all stages for a workflow with file counts.",
)
async def list_workflow_stages(workflow_id: str) -> list:
    """List all stages for a workflow."""
    from pathlib import Path

    workflow_dir = Path(settings.RESULTS_DIR) / workflow_id
    if not workflow_dir.exists():
        return []

    stages = []
    for stage_dir in sorted(workflow_dir.iterdir()):
        if not stage_dir.is_dir():
            continue

        stage_info = {
            "stage_name": stage_dir.name,
            "path": str(stage_dir),
            "has_plots": False,
            "plot_count": 0,
            "has_code": False,
            "code_count": 0,
        }

        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plots = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
            stage_info["has_plots"] = len(plots) > 0
            stage_info["plot_count"] = len(plots)

        code_files = list(stage_dir.glob("*.py"))
        stage_info["has_code"] = len(code_files) > 0
        stage_info["code_count"] = len(code_files)

        stages.append(stage_info)

    return stages


@router.get(
    "/list",
    summary="List all workflows",
)
async def list_workflows() -> list:
    """List all available workflows."""
    from pathlib import Path

    results_dir = Path(settings.RESULTS_DIR)
    if not results_dir.exists():
        return []

    workflows = []
    for workflow_dir in sorted(results_dir.iterdir()):
        if workflow_dir.is_dir():
            stage_count = len([d for d in workflow_dir.iterdir() if d.is_dir()])
            workflows.append(
                {
                    "workflow_id": workflow_dir.name,
                    "stage_count": stage_count,
                }
            )

    return workflows
