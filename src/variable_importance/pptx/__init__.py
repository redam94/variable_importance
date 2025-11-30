"""
PowerPoint Generation Module

Generate presentations from analysis results using:
- LLM-based content planning
- Intelligent plot-to-slide matching  
- python-pptx for PPTX creation
"""

from .models import (
    PresentationOutline,
    SlideContent,
    PlotInfo,
    PlotAssignment,
    GeneratedPresentation,
)
from .agent import PresentationAgent
from .builder import PresentationBuilder

__all__ = [
    "PresentationAgent",
    "PresentationBuilder",
    "PresentationOutline",
    "SlideContent",
    "PlotInfo",
    "PlotAssignment",
    "GeneratedPresentation",
]