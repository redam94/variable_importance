"""
PowerPoint Presentation Models

Pydantic schemas for structured presentation generation.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PlotInfo(BaseModel):
    """Information about an available plot."""
    name: str
    path: str
    description: Optional[str] = None


class SlideContent(BaseModel):
    """Content for a single slide."""
    slide_number: int
    title: str = Field(description="Slide title (keep concise)")
    layout: Literal["title", "content", "two_column", "image", "multi_image", "summary"] = Field(
        description="Slide layout type"
    )
    bullet_points: List[str] = Field(
        default_factory=list,
        description="Bullet points or key messages (3-5 max)"
    )
    speaker_notes: str = Field(
        default="",
        description="1-3 sentence speaker notes"
    )
    body_text: Optional[str] = Field(
        default=None,
        description="Optional paragraph text instead of bullets"
    )


class PlotAssignment(BaseModel):
    """LLM-determined plot assignment for a slide."""
    slide_number: int
    plot_names: List[str] = Field(
        default_factory=list,
        description="Plot filenames to include on this slide (can be empty)"
    )
    reasoning: str = Field(
        description="Why these plots match this slide's content"
    )


class PlotAssignmentPlan(BaseModel):
    """Complete plan for assigning plots to slides."""
    assignments: List[PlotAssignment] = Field(
        description="Plot assignments for each slide"
    )


class PresentationOutline(BaseModel):
    """High-level presentation structure."""
    title: str = Field(description="Presentation title")
    subtitle: Optional[str] = Field(default=None, description="Optional subtitle")
    color_primary: str = Field(
        default="1F4E79",
        description="Primary hex color (no #)"
    )
    color_accent: str = Field(
        default="2E86AB",
        description="Accent hex color (no #)"
    )
    slides: List[SlideContent] = Field(
        description="Ordered list of slides"
    )


class GeneratedPresentation(BaseModel):
    """Result of presentation generation."""
    output_path: str
    slide_count: int
    plots_used: List[str]
    success: bool
    error: Optional[str] = None