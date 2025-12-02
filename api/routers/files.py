"""
Files Router - Serve files from workflow results.

Provides:
- GET /files/{workflow_id}/images - List all images for a workflow
- GET /files/{workflow_id}/image?path=... - Serve a specific image
- GET /files/{workflow_id}/stages - List stages with file info
"""

from pathlib import Path
from typing import Annotated, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from dependencies import settings
from auth import get_current_active_user, User


router = APIRouter(prefix="/files", tags=["Files"])


# =============================================================================
# SCHEMAS
# =============================================================================


class ImageInfo(BaseModel):
    """Image file metadata."""
    filename: str
    path: str  # Relative path from workflow root
    stage: str
    url: str   # API URL to fetch the image
    size_bytes: int
    

class StageFiles(BaseModel):
    """Files in a workflow stage."""
    stage_name: str
    images: List[ImageInfo]
    data_files: List[str]
    code_files: List[str]


class WorkflowImages(BaseModel):
    """All images for a workflow."""
    workflow_id: str
    total_images: int
    images: List[ImageInfo]


# =============================================================================
# HELPERS
# =============================================================================


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
ALLOWED_DATA_EXTENSIONS = {".csv", ".json", ".xlsx", ".parquet", ".tsv"}


def _get_results_base() -> Path:
    """Get the absolute path to results directory."""
    # Handle relative path by resolving from current working directory
    results_path = Path(settings.RESULTS_DIR)
    if not results_path.is_absolute():
        results_path = Path.cwd() / results_path
    return results_path.resolve()


def _get_workflow_dir(workflow_id: str) -> Path:
    """Get and validate workflow directory exists."""
    results_base = _get_results_base()
    workflow_dir = results_base / workflow_id
    
    logger.info(f"Looking for workflow at: {workflow_dir}")
    
    if not workflow_dir.exists():
        # Log available workflows for debugging
        if results_base.exists():
            available = [d.name for d in results_base.iterdir() if d.is_dir()]
            logger.warning(f"Workflow '{workflow_id}' not found. Available: {available}")
        else:
            logger.warning(f"Results directory does not exist: {results_base}")
        
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    
    return workflow_dir


def _is_safe_path(base: Path, path: Path) -> bool:
    """Ensure path doesn't escape the base directory."""
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


# =============================================================================
# DEBUG ENDPOINT
# =============================================================================


@router.get(
    "/debug/paths",
    summary="Debug path resolution (dev only)",
)
async def debug_paths(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> dict:
    """Show resolved paths for debugging 404 issues."""
    results_base = _get_results_base()
    
    workflows = []
    if results_base.exists():
        for d in results_base.iterdir():
            if d.is_dir():
                # Count images
                image_count = sum(
                    1 for f in d.rglob("*") 
                    if f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                )
                workflows.append({
                    "name": d.name,
                    "path": str(d),
                    "image_count": image_count,
                })
    
    return {
        "cwd": str(Path.cwd()),
        "settings_results_dir": settings.RESULTS_DIR,
        "resolved_results_base": str(results_base),
        "results_exists": results_base.exists(),
        "workflows": workflows,
    }


# =============================================================================
# IMAGE ENDPOINTS
# =============================================================================


@router.get(
    "/{workflow_id}/images",
    response_model=WorkflowImages,
    summary="List all images for a workflow",
)
async def list_workflow_images(
    workflow_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    stage: Optional[str] = Query(None, description="Filter by stage name"),
) -> WorkflowImages:
    """
    List all image files in a workflow's results directory.
    
    Returns image metadata including API URLs for fetching each image.
    Optionally filter by stage name.
    """
    workflow_dir = _get_workflow_dir(workflow_id)
    images: List[ImageInfo] = []
    
    # Iterate through stage directories
    for stage_dir in sorted(workflow_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        
        # Filter by stage if specified
        if stage and stage not in stage_dir.name:
            continue
        
        # Check plots subdirectory
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            for img_file in sorted(plots_dir.iterdir()):
                if img_file.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    relative_path = img_file.relative_to(workflow_dir)
                    images.append(ImageInfo(
                        filename=img_file.name,
                        path=str(relative_path),
                        stage=stage_dir.name,
                        url=f"/files/{workflow_id}/image?path={relative_path}",
                        size_bytes=img_file.stat().st_size,
                    ))
        
        # Also check for images directly in stage dir
        for img_file in stage_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                relative_path = img_file.relative_to(workflow_dir)
                images.append(ImageInfo(
                    filename=img_file.name,
                    path=str(relative_path),
                    stage=stage_dir.name,
                    url=f"/files/{workflow_id}/image?path={relative_path}",
                    size_bytes=img_file.stat().st_size,
                ))
    
    logger.info(f"📸 Listed {len(images)} images for workflow: {workflow_id}")
    
    return WorkflowImages(
        workflow_id=workflow_id,
        total_images=len(images),
        images=images,
    )


@router.get(
    "/{workflow_id}/image",
    summary="Get a specific image",
    response_class=FileResponse,
)
async def get_image(
    workflow_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    path: str = Query(..., description="Relative path to image file"),
) -> FileResponse:
    """
    Serve an image file from workflow results.
    
    The path should be relative to the workflow directory.
    Example: `?path=analysis/plots/correlation_heatmap.png`
    """
    workflow_dir = _get_workflow_dir(workflow_id)
    
    # Construct full path
    full_path = workflow_dir / path
    
    # Security: ensure path doesn't escape workflow directory
    if not _is_safe_path(workflow_dir, full_path):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    
    # Validate it's an allowed image type
    if full_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not an allowed image type")
    
    # Determine media type
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }
    media_type = media_types.get(full_path.suffix.lower(), "application/octet-stream")
    
    return FileResponse(
        path=str(full_path),
        media_type=media_type,
        filename=full_path.name,
    )


# =============================================================================
# STAGE FILE LISTING
# =============================================================================


@router.get(
    "/{workflow_id}/stages",
    response_model=List[StageFiles],
    summary="List all stages with file info",
)
async def list_stages_with_files(
    workflow_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> List[StageFiles]:
    """
    List all stages in a workflow with categorized file information.
    
    Returns images, data files, and code files for each stage.
    """
    workflow_dir = _get_workflow_dir(workflow_id)
    stages: List[StageFiles] = []
    
    for stage_dir in sorted(workflow_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        
        images: List[ImageInfo] = []
        data_files: List[str] = []
        code_files: List[str] = []
        
        # Recursively find files
        for file_path in stage_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            relative_path = file_path.relative_to(workflow_dir)
            suffix = file_path.suffix.lower()
            
            if suffix in ALLOWED_IMAGE_EXTENSIONS:
                images.append(ImageInfo(
                    filename=file_path.name,
                    path=str(relative_path),
                    stage=stage_dir.name,
                    url=f"/api/files/{workflow_id}/image?path={relative_path}",
                    size_bytes=file_path.stat().st_size,
                ))
            elif suffix in ALLOWED_DATA_EXTENSIONS:
                data_files.append(str(relative_path))
            elif suffix == ".py":
                code_files.append(str(relative_path))
        
        stages.append(StageFiles(
            stage_name=stage_dir.name,
            images=images,
            data_files=data_files,
            code_files=code_files,
        ))
    
    return stages


# =============================================================================
# FILE DOWNLOAD (generic)
# =============================================================================


@router.get(
    "/{workflow_id}/download",
    summary="Download any file from workflow",
    response_class=FileResponse,
)
async def download_file(
    workflow_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    path: str = Query(..., description="Relative path to file"),
) -> FileResponse:
    """
    Download any file from workflow results.
    
    Supports images, CSV, JSON, code files, etc.
    Example: `?path=analysis/data/results.csv`
    """
    workflow_dir = _get_workflow_dir(workflow_id)
    full_path = workflow_dir / path
    
    if not _is_safe_path(workflow_dir, full_path):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    
    return FileResponse(
        path=str(full_path),
        filename=full_path.name,
        media_type="application/octet-stream",
    )