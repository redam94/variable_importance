"""
RevealJS Presentation Module

Generate standalone Reveal.js HTML presentations from workflow context.

Components:
- RevealJSAgent: LangGraph agent orchestrating generation
- RevealJSDocsRAG: Shared singleton for RevealJS documentation
- StyleMemory: Per-user style guide with semantic search
- RevealJSBuilder: Standalone HTML generator
- WebSearchClient: Web search for supplementary context

Features:
- Agent-based context gathering with RAG tools
- Per-slide verification against RAG to prevent hallucination
- Web search integration for supplementary information
- Plot analysis from RAG for accurate visualization descriptions
- Style guide learning from user feedback

Usage:
    from variable_importance.revealjs import create_revealjs_presentation
    
    result = await create_revealjs_presentation(
        topic="Q3 Analysis Results",
        workflow_id="my_workflow",
        stage_name="analysis",
        user_id="user123",
        enable_web_search=True,  # Enable web search for extra context
    )
    
    # result.html_content contains standalone HTML
    # result.output_path has the saved file path
"""

from .models import (
    SlideContent,
    PresentationOutline,
    StyleRule,
    StyleGuide,
    GeneratedPresentation,
    RevealJSSnippet,
    ImageAssignmentPlan,
    SlideAssignment,
)

from .docs_rag import (
    RevealJSDocsRAG,
    get_revealjs_docs_rag,
)

from .style_memory import (
    StyleMemory,
    get_style_memory,
)

from .builder import RevealJSBuilder

from .agent import (
    RevealJSAgent,
    create_revealjs_presentation,
    WebSearchClient,
    get_web_search_client,
)


__all__ = [
    # Main agent
    "RevealJSAgent",
    "create_revealjs_presentation",
    
    # Builder
    "RevealJSBuilder",
    
    # RAG systems
    "RevealJSDocsRAG",
    "get_revealjs_docs_rag",
    "StyleMemory",
    "get_style_memory",
    
    # Web search
    "WebSearchClient",
    "get_web_search_client",
    
    # Models
    "SlideContent",
    "PresentationOutline",
    "StyleRule",
    "StyleGuide",
    "GeneratedPresentation",
    "RevealJSSnippet",
    "ImageAssignmentPlan",
    "SlideAssignment",
]