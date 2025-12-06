"""
RevealJS Presentation Models

Pydantic schemas for slide generation, style guides, and agent outputs.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# SLIDE CONTENT MODELS
# =============================================================================


class SlideContent(BaseModel):
    """Content for a single Reveal.js slide."""
    
    slide_number: int
    title: Optional[str] = Field(default=None, description="Slide title (h2)")
    subtitle: Optional[str] = Field(default=None, description="Slide subtitle (h3)")
    layout: Literal[
        "title",        # Title slide with centered content
        "content",      # Standard content with bullets
        "two_column",   # Side-by-side columns
        "image",        # Large image with caption
        "code",         # Code block with syntax highlighting
        "quote",        # Blockquote slide
        "section",      # Section divider
        "summary",      # Key takeaways
    ] = Field(default="content", description="Slide layout type")
    
    content: List[str] = Field(
        default_factory=list,
        description="Main content items (bullets, paragraphs)"
    )
    code_block: Optional[str] = Field(
        default=None,
        description="Code snippet for code layout"
    )
    code_language: str = Field(
        default="python",
        description="Language for syntax highlighting"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to image (will be base64 embedded)"
    )
    image_caption: Optional[str] = Field(
        default=None,
        description="Caption for image"
    )
    speaker_notes: str = Field(
        default="",
        description="Speaker notes (not shown in presentation)"
    )
    fragment_style: Optional[Literal["fade-in", "fade-up", "zoom-in", "none"]] = Field(
        default="fade-in",
        description="Animation style for content reveal"
    )
    background_color: Optional[str] = Field(
        default=None,
        description="Override background color for this slide"
    )


class PresentationOutline(BaseModel):
    """Complete presentation structure."""
    
    title: str = Field(description="Presentation title")
    subtitle: Optional[str] = Field(default=None, description="Presentation subtitle")
    author: Optional[str] = Field(default=None, description="Author name")
    date: Optional[str] = Field(default=None, description="Presentation date")
    
    theme: Literal[
        "black", "white", "league", "beige", "sky",
        "night", "serif", "simple", "solarized", "moon", "dracula"
    ] = Field(default="black", description="Reveal.js theme")
    
    transition: Literal[
        "none", "fade", "slide", "convex", "concave", "zoom"
    ] = Field(default="slide", description="Slide transition effect")
    
    color_primary: str = Field(
        default="667eea",
        description="Primary accent color (hex without #)"
    )
    color_secondary: str = Field(
        default="764ba2",
        description="Secondary accent color (hex without #)"
    )
    
    slides: List[SlideContent] = Field(
        description="Ordered list of slides"
    )
    
    custom_css: Optional[str] = Field(
        default=None,
        description="Additional custom CSS"
    )


# =============================================================================
# STYLE GUIDE MODELS
# =============================================================================


class StyleRule(BaseModel):
    """A single style rule or preference."""
    
    rule_id: str = Field(description="Unique identifier")
    category: Literal[
        "colors", "typography", "layout", "animations",
        "code_style", "content", "branding", "general"
    ] = Field(description="Style category")
    
    rule_text: str = Field(
        description="Natural language description of the style rule"
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Example applications of this rule"
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="How strictly to apply this rule"
    )
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source: Literal["user", "agent", "default"] = Field(
        default="user",
        description="Who created this rule"
    )


class StyleGuide(BaseModel):
    """Complete style guide for a user."""
    
    user_id: str = Field(description="User identifier")
    name: str = Field(default="Default Style Guide")
    description: str = Field(default="")
    
    rules: List[StyleRule] = Field(default_factory=list)
    
    # Quick access defaults
    default_theme: str = Field(default="black")
    default_transition: str = Field(default="slide")
    default_colors: Dict[str, str] = Field(
        default_factory=lambda: {
            "primary": "667eea",
            "secondary": "764ba2",
            "accent": "f093fb"
        }
    )
    default_font_heading: str = Field(default="Source Sans Pro")
    default_font_body: str = Field(default="Source Sans Pro")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# AGENT OUTPUT MODELS
# =============================================================================


class SlideAssignment(BaseModel):
    """Plot/image assignment for a slide."""
    
    slide_number: int
    image_paths: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Why these images fit this slide")


class ImageAssignmentPlan(BaseModel):
    """Complete plan for assigning images to slides."""
    
    assignments: List[SlideAssignment]


class GeneratedPresentation(BaseModel):
    """Result of presentation generation."""
    
    output_path: str
    html_content: str = Field(description="Complete standalone HTML")
    slide_count: int
    images_embedded: List[str] = Field(default_factory=list)
    success: bool
    error: Optional[str] = None
    
    # Generation metadata
    theme_used: str = Field(default="black")
    style_rules_applied: List[str] = Field(default_factory=list)
    generation_time_seconds: float = Field(default=0.0)


# =============================================================================
# RAG DOCUMENT MODELS
# =============================================================================


class RevealJSSnippet(BaseModel):
    """A code snippet or example from RevealJS documentation."""
    
    title: str
    description: str
    code: str
    category: Literal[
        "setup", "slides", "fragments", "backgrounds",
        "transitions", "themes", "plugins", "api", "events"
    ]
    tags: List[str] = Field(default_factory=list)
    url: Optional[str] = Field(default=None, description="Source URL")