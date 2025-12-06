"""
RevealJS Presentation Agent

LangGraph workflow with:
- Context gathering agent with RAG query tools
- Presentation planning with structured output
- Validation loop ensuring complete, valid RevealJS output
- Style guide integration and learning
"""

import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, TypedDict
from datetime import datetime
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from .models import (
    PresentationOutline,
    SlideContent,
    GeneratedPresentation,
    ImageAssignmentPlan,
)
from .docs_rag import get_revealjs_docs_rag
from .style_memory import get_style_memory
from .builder import RevealJSBuilder


# =============================================================================
# STATE DEFINITIONS
# =============================================================================


class RevealJSState(TypedDict):
    """State for the RevealJS generation workflow."""
    
    # Input
    topic: str
    workflow_id: str
    stage_name: str
    num_slides: int
    custom_instructions: str
    
    # Context gathered by agent
    workflow_context: str
    revealjs_snippets: str
    style_context: str
    plot_analyses: str  # Detailed plot descriptions from RAG
    available_images: List[Dict[str, str]]
    
    # Planning outputs
    outline: Optional[PresentationOutline]
    image_assignments: Dict[int, str]
    
    # Per-slide verification
    slide_contexts: Dict[int, str]  # slide_number -> retrieved context
    unsupported_slides: List[int]   # slides without sufficient evidence
    available_topics: List[str]     # topics actually found in RAG
    verification_passed: bool
    
    # Validation
    validation_errors: List[str]
    validation_passed: bool
    iteration_count: int
    max_iterations: int
    
    # Output
    html_content: str
    result: Optional[GeneratedPresentation]
    
    # Messages for agent
    messages: List[BaseMessage]


# =============================================================================
# REVEALJS AGENT
# =============================================================================


class RevealJSAgent:
    """
    LangGraph workflow agent for generating Reveal.js presentations.
    
    Workflow:
    1. gather_context: Agent uses tools to query RAGs
    2. plan_presentation: Create outline with LLM
    3. match_images: Assign images to slides
    4. build_html: Generate standalone HTML
    5. validate: Check HTML validity and content completeness
    6. (loop back to plan if validation fails)
    """
    
    def __init__(
        self,
        llm_model: str = "qwen3:30b",
        base_url: str = "http://100.91.155.118:11434",
        workflow_rag=None,
        output_manager=None,
        user_id: str = "default",
        progress_callback: Optional[Callable[[str], None]] = None,
        max_iterations: int = 3,
    ):
        self.llm_model = llm_model
        self.base_url = base_url
        self.workflow_rag = workflow_rag
        self.output_manager = output_manager
        self.user_id = user_id
        self.progress_callback = progress_callback
        self.max_iterations = max_iterations
        
        # Initialize LLMs
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.3,
            base_url=base_url
        )
        
        self.validator_llm = ChatOllama(
            model=llm_model,
            temperature=0.1,
            base_url=base_url
        )
        
        # Initialize RAGs and memory
        self.docs_rag = get_revealjs_docs_rag()
        self.style_memory = get_style_memory(user_id)
        
        # Builder
        self.builder = RevealJSBuilder(progress_callback=progress_callback)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _emit(self, message: str):
        """Emit progress update."""
        logger.info(f"🎨 RevealJS: {message}")
        if self.progress_callback:
            self.progress_callback(message)
    
    # =========================================================================
    # TOOLS FOR CONTEXT GATHERING AGENT
    # =========================================================================
    
    def _create_rag_tools(self):
        """Create tools for the context gathering agent."""
        
        workflow_rag = self.workflow_rag
        docs_rag = self.docs_rag
        style_memory = self.style_memory
        output_manager = self.output_manager
        
        @tool
        def query_workflow_context(query: str) -> str:
            """
            Query the workflow RAG for analysis results, summaries, and findings.
            
            Args:
                query: Search query for relevant context
            
            Returns:
                Relevant context from the workflow
            """
            if not workflow_rag or not workflow_rag.enabled:
                return "No workflow context available."
            
            try:
                type_list = None #[t.strip() for t in doc_types.split(",")] if doc_types else None
                
                results = workflow_rag.query_relevant_context(
                    query=query,
                    doc_types=type_list,
                    n_results=15,
                )
                
                if not results:
                    return f"No results found for query: {query}"
                
                context_parts = []
                for r in results:
                    doc = r.get("document", "")
                    metadata = r.get("metadata", {})
                    doc_type = metadata.get("type", "unknown")
                    stage = metadata.get("stage_name", "")
                    
                    context_parts.append(f"[{doc_type} - {stage}]\n{doc}\n")
                
                return "\n---\n".join(context_parts)
                
            except Exception as e:
                return f"Error querying workflow context: {e}"
        
        @tool
        def query_revealjs_docs(query: str, category: str = "") -> str:
            """
            Query RevealJS documentation for code snippets and best practices.
            
            Args:
                query: Search query (e.g., "two column layout", "code highlighting")
                category: Optional category filter (setup, slides, fragments, backgrounds, transitions, themes, code)
            
            Returns:
                Relevant RevealJS code snippets and documentation
            """
            if not docs_rag.enabled:
                return "RevealJS documentation not available."
            
            results = docs_rag.query(
                query=query,
                category=category if category else None,
                n_results=5
            )
            
            if not results:
                return f"No snippets found for: {query}"
            
            snippets = []
            for r in results:
                doc = r.get("document", "")
                metadata = r.get("metadata", {})
                title = metadata.get("title", "Snippet")
                snippets.append(f"### {title}\n{doc}")
            
            return "\n\n".join(snippets)
        
        @tool
        def get_style_preferences(query: str = "") -> str:
            """
            Get user's style preferences and rules for presentation design.
            
            Args:
                query: Optional query to find most relevant style rules
            
            Returns:
                Formatted style guide with user preferences
            """
            return style_memory.get_formatted_context(query if query else None)
        
        @tool
        def query_plot_analyses(query: str = "") -> str:
            """
            Query for plot/chart analyses from the RAG system.
            
            Returns detailed descriptions of what each visualization shows,
            including insights, trends, and key findings from each plot.
            
            Args:
                query: Optional search query to filter plots (e.g., "correlation", "distribution")
            
            Returns:
                Detailed plot analyses with descriptions
            """
            if not workflow_rag or not workflow_rag.enabled:
                return "No workflow RAG available."
            
            try:
                search_query = query if query else "plot chart visualization figure analysis"
                
                results = workflow_rag.query_relevant_context(
                    query=search_query,
                    doc_types=["plot_analysis"],
                    n_results=20,
                )
                
                if not results:
                    return "No plot analyses found in RAG."
                
                output_parts = ["## Plot Analyses from RAG\n"]
                
                for r in results:
                    metadata = r.get("metadata", {})
                    doc = r.get("document", "")
                    
                    plot_name = metadata.get("plot_name", "Unknown plot")
                    stage = metadata.get("stage_name", "")
                    
                    output_parts.append(f"### {plot_name}")
                    if stage:
                        output_parts.append(f"Stage: {stage}")
                    output_parts.append(doc)
                    output_parts.append("")
                
                return "\n".join(output_parts)
                
            except Exception as e:
                return f"Error querying plot analyses: {e}"
        
        @tool
        def list_available_images(stage_name: str) -> str:
            """
            List available images/plots from a workflow stage with their descriptions.
            
            Args:
                stage_name: Name of the stage to search for images
            
            Returns:
                List of available images with descriptions from plot analysis
            """
            if not output_manager:
                return "No output manager available."
            
            try:
                stage_dir = output_manager.get_stage_dir(stage_name)
                plots_dir = stage_dir / "plots"
                
                images = []
                if plots_dir.exists():
                    for ext in ["*.png", "*.jpg", "*.jpeg", "*.svg"]:
                        for img_file in plots_dir.glob(ext):
                            images.append({
                                "name": img_file.name,
                                "path": str(img_file),
                            })
                
                if not images:
                    return f"No images found in stage: {stage_name}"
                
                # Try to get descriptions from RAG
                plot_descriptions = {}
                if workflow_rag and workflow_rag.enabled:
                    try:
                        results = workflow_rag.query_relevant_context(
                            query="plot chart visualization figure analysis",
                            doc_types=["plot_analysis"],
                            n_results=30,
                        )
                        
                        for r in results:
                            metadata = r.get("metadata", {})
                            doc = r.get("document", "")
                            plot_name = metadata.get("plot_name", "")
                            
                            if plot_name:
                                # Extract analysis
                                if "Analysis:" in doc:
                                    analysis = doc.split("Analysis:")[-1].strip()
                                else:
                                    analysis = doc
                                
                                # Truncate
                                if len(analysis) > 300:
                                    analysis = analysis[:300] + "..."
                                
                                plot_descriptions[plot_name] = analysis
                                plot_descriptions[Path(plot_name).stem] = analysis
                    except Exception as e:
                        logger.warning(f"Could not get plot descriptions: {e}")
                
                # Build output with descriptions
                output_lines = [f"Available images in {stage_name}:"]
                for img in images:
                    desc = plot_descriptions.get(
                        img["name"],
                        plot_descriptions.get(
                            Path(img["name"]).stem,
                            "No description available"
                        )
                    )
                    output_lines.append(f"\n- {img['name']}:")
                    output_lines.append(f"  {desc}")
                
                return "\n".join(output_lines)
                
            except Exception as e:
                return f"Error listing images: {e}"
        
        return [query_workflow_context, query_revealjs_docs, get_style_preferences, query_plot_analyses, list_available_images]
    
    # =========================================================================
    # WORKFLOW NODES
    # =========================================================================
    
    async def _gather_context(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Use agent with tools to gather comprehensive context.
        """
        self._emit("Gathering context with RAG agent...")
        
        tools = self._create_rag_tools()
        
        
        
        system_prompt = """You are a research assistant gathering context for a presentation.

Your goal is to thoroughly research the topic and gather ALL relevant information.

INSTRUCTIONS:
1. Query the workflow context multiple times with different queries to get comprehensive coverage
2. Search for: documents, summaries, analysis results, key findings, data insights, conclusions
3. Query for plot analyses to understand what each visualization shows
4. Get RevealJS documentation for relevant layouts and features
5. Check user style preferences
6. List available images with their descriptions

Be thorough - make multiple queries to ensure you capture all important information.
Plot analyses are especially important for matching visualizations to content."""

        agent = create_agent(
            model=self.llm,
            tools=tools,
            middleware=[TodoListMiddleware()],
            system_prompt=system_prompt
        )
        user_message = f"""Research and gather all context for a presentation about: "{state['topic']}"

Workflow ID: {state['workflow_id']}
Stage: {state['stage_name']}

Make multiple queries to the workflow context to gather:
1. Main findings and results
2. Key data points and statistics  
3. Methodology and approach
4. Conclusions and recommendations
5. Additional insights and context

IMPORTANT - Query for plot analyses to understand visualizations:
6. Use query_plot_analyses to get detailed descriptions of all charts/plots
7. This will help match the right images to the right slides

Also get:
- Relevant RevealJS code snippets for layouts
- User style preferences
- List of available images with their descriptions

Be comprehensive - query multiple times with different search terms."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        
        try:
            result = await agent.ainvoke({"messages": messages})
            
            final_messages = result.get("messages", [])
            
            # Collect all tool outputs
            workflow_context_parts = []
            revealjs_parts = []
            plot_analysis_parts = []
            
            for msg in final_messages:
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    # Categorize based on content
                    if "## Plot Analyses" in content or "plot_analysis" in content.lower():
                        plot_analysis_parts.append(content)
                    elif any(marker in content.lower() for marker in ["[summary", "[code_execution", "[plot_analysis", "---"]):
                        workflow_context_parts.append(content)
                    elif "###" in content and any(kw in content.lower() for kw in ["slide", "reveal", "layout", "transition"]):
                        revealjs_parts.append(content)
            
            # Get final AI response as fallback
            final_response = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_response = msg.content
                    break
            
            state["workflow_context"] = "\n\n".join(workflow_context_parts) if workflow_context_parts else final_response
            state["revealjs_snippets"] = "\n\n".join(revealjs_parts)
            state["style_context"] = self.style_memory.get_formatted_context()
            state["plot_analyses"] = "\n\n".join(plot_analysis_parts)
            state["messages"] = final_messages
            
            self._emit(f"Gathered {len(workflow_context_parts)} context chunks, {len(plot_analysis_parts)} plot analyses")
            
        except Exception as e:
            logger.error(f"Context gathering failed: {e}")
            state["workflow_context"] = self._direct_workflow_query(state["topic"], state["workflow_id"])
            state["revealjs_snippets"] = self._direct_docs_query("presentation slides layouts")
            state["style_context"] = self.style_memory.get_formatted_context()
            state["plot_analyses"] = self._direct_plot_analyses_query(state["workflow_id"])
        
        # Get available images with RAG descriptions
        state["available_images"] = await self._get_images_with_descriptions(
            stage_name=state["stage_name"],
            workflow_id=state["workflow_id"],
        )
        
        return state
    
    def _direct_workflow_query(self, topic: str, workflow_id: str) -> str:
        """Fallback direct RAG query."""
        if not self.workflow_rag or not self.workflow_rag.enabled:
            return "No workflow context available."
        
        results = self.workflow_rag.query_relevant_context(
            query=topic,
            workflow_id=workflow_id,
            n_results=20,
        )
        
        if not results:
            return "No context found."
        
        parts = []
        for r in results:
            doc = r.get("document", "")
            metadata = r.get("metadata", {})
            parts.append(f"[{metadata.get('type', 'unknown')}]\n{doc}")
        
        return "\n---\n".join(parts)
    
    def _direct_docs_query(self, query: str) -> str:
        """Fallback direct docs query."""
        results = self.docs_rag.query(query, n_results=5)
        return "\n\n".join(r.get("document", "") for r in results)
    
    def _direct_plot_analyses_query(self, workflow_id: str) -> str:
        """Fallback direct query for plot analyses."""
        if not self.workflow_rag or not self.workflow_rag.enabled:
            return ""
        
        try:
            results = self.workflow_rag.query_relevant_context(
                query="plot chart visualization figure analysis",
                workflow_id=workflow_id,
                doc_types=["plot_analysis"],
                n_results=20,
            )
            
            if not results:
                return ""
            
            parts = ["## Plot Analyses\n"]
            for r in results:
                metadata = r.get("metadata", {})
                doc = r.get("document", "")
                plot_name = metadata.get("plot_name", "Unknown")
                parts.append(f"### {plot_name}\n{doc}\n")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Plot analyses query failed: {e}")
            return ""
    
    def _get_available_images(self, stage_name: str) -> List[Dict[str, str]]:
        """Get available images from workflow output with RAG descriptions."""
        images = []
        
        if not self.output_manager:
            return images
        
        try:
            stage_dir = self.output_manager.get_stage_dir(stage_name)
            plots_dir = stage_dir / "plots"
            
            if plots_dir.exists():
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.svg"]:
                    for img_file in plots_dir.glob(ext):
                        images.append({
                            "name": img_file.name,
                            "path": str(img_file),
                            "description": img_file.stem.replace("_", " ").title()
                        })
        except Exception as e:
            logger.warning(f"Error getting images: {e}")
        
        return images
    
    async def _get_images_with_descriptions(
        self, 
        stage_name: str, 
        workflow_id: str
    ) -> List[Dict[str, str]]:
        """
        Get available images enriched with descriptions from RAG plot_analysis.
        
        Queries RAG for plot_analysis documents to get detailed descriptions
        of what each plot shows.
        """
        # First get the file list
        images = self._get_available_images(stage_name)
        
        if not images:
            return images
        
        if not self.workflow_rag or not self.workflow_rag.enabled:
            return images
        
        self._emit(f"Enriching {len(images)} images with RAG descriptions...")
        
        # Query RAG for plot analyses
        try:
            results = self.workflow_rag.query_relevant_context(
                query="plot chart visualization analysis figure",
                workflow_id=workflow_id,
                doc_types=["plot_analysis"],
                n_results=30,
            )
            
            if not results:
                # Fallback: try querying by each plot name
                results = []
                for img in images:
                    plot_results = self.workflow_rag.query_relevant_context(
                        query=img["name"],
                        workflow_id=workflow_id,
                        doc_types=["plot_analysis"],
                        n_results=3,
                    )
                    results.extend(plot_results)
            
            # Build lookup from plot name to description
            plot_descriptions: Dict[str, str] = {}
            
            for r in results:
                metadata = r.get("metadata", {})
                doc = r.get("document", "")
                
                plot_name = metadata.get("plot_name", "")
                
                if plot_name:
                    # Extract the analysis part from the document
                    if "Analysis:" in doc:
                        analysis = doc.split("Analysis:")[-1].strip()
                    else:
                        analysis = doc
                    
                    # Truncate to reasonable length
                    if len(analysis) > 500:
                        analysis = analysis[:500] + "..."
                    
                    plot_descriptions[plot_name] = analysis
                    
                    # Also try matching without extension
                    base_name = Path(plot_name).stem
                    plot_descriptions[base_name] = analysis
            
            # Enrich images with descriptions
            enriched_images = []
            for img in images:
                description = plot_descriptions.get(
                    img["name"],
                    plot_descriptions.get(
                        Path(img["name"]).stem,
                        img["description"]  # fallback to filename-based
                    )
                )
                
                enriched_images.append({
                    "name": img["name"],
                    "path": img["path"],
                    "description": description,
                })
                
                if description != img["description"]:
                    self._emit(f"  📊 {img['name']}: Found RAG description")
            
            logger.info(f"📊 Enriched {len(enriched_images)} images with RAG descriptions")
            return enriched_images
            
        except Exception as e:
            logger.error(f"Failed to enrich images with RAG descriptions: {e}")
            return images
    
    async def _plan_presentation(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Plan presentation structure using gathered context.
        """
        self._emit(f"Planning presentation (iteration {state['iteration_count'] + 1})...")
        
        # Build feedback from previous iterations
        feedback_parts = []
        
        # Validation errors from previous iteration
        if state["validation_errors"]:
            feedback_parts.append(f"""
VALIDATION ERRORS FROM PREVIOUS ATTEMPT:
{chr(10).join('- ' + e for e in state['validation_errors'])}""")
        
        # Verification feedback - what topics ARE available
        if state.get("unsupported_slides"):
            feedback_parts.append(f"""
CONTENT VERIFICATION FAILED:
The following slide numbers had NO supporting evidence in the knowledge base and were removed:
{state['unsupported_slides']}

IMPORTANT: Only create slides about topics that exist in the workflow context below.
DO NOT make up information. If a topic isn't in the context, don't include it.""")
        
        if state.get("available_topics"):
            feedback_parts.append(f"""
VERIFIED AVAILABLE TOPICS (these have supporting evidence):
{chr(10).join('- ' + t for t in state['available_topics'])}

Focus your presentation on these verified topics.""")
        
        validation_feedback = "\n".join(feedback_parts)
        
        style_defaults = self.style_memory.get_defaults()
        
        prompt = f"""Create a presentation outline for: "{state['topic']}"

TARGET: {state['num_slides']} slides (including title and summary)

WORKFLOW CONTEXT (from RAG - summarize INFORMATION relevant to the topic):
{state['workflow_context'][:5000]}

PLOT ANALYSES (available visualizations and what they show):
{state.get('plot_analyses', 'No plot analyses available')[:2000]}

AVAILABLE IMAGES:
{chr(10).join(f"- {img['name']}: {img.get('description', '')[:100]}" for img in state.get('available_images', [])[:10])}

STYLE GUIDE:
{state['style_context']}

AVAILABLE LAYOUTS:
- title: Opening slide with title, subtitle, author
- content: Standard bullets with optional image
- two_column: Side-by-side content comparison
- image: Large image with caption
- code: Code block with syntax highlighting
- quote: Blockquote slide
- section: Section divider
- summary: Key takeaways

REVEALJS BEST PRACTICES:
{state['revealjs_snippets'][:2000]}

{f'ADDITIONAL INSTRUCTIONS: {state["custom_instructions"]}' if state.get("custom_instructions") else ''}
{validation_feedback}

CRITICAL REQUIREMENTS:
1. ONLY include information that appears in the WORKFLOW CONTEXT above
2. DO NOT make up statistics, findings, or claims not in the context
3. Every bullet point must be traceable to the source material
4. If information isn't available, use fewer slides rather than inventing content
5. Use PLOT ANALYSES to create slides that discuss visualizations accurately
6. Match slide content to available images based on their descriptions
7. Use varied layouts for visual interest
8. Keep bullet points concise (max 10 words each)
9. Include speaker notes citing which part of context supports each point

The presentation must be GROUNDED in the provided context. No hallucination.
When discussing a plot, use the analysis description to accurately describe what it shows."""

        structured_llm = self.llm.with_structured_output(PresentationOutline)
        
        try:
            outline = await asyncio.to_thread(
                structured_llm.invoke,
                [
                    SystemMessage(content="You are an expert presentation designer. Create comprehensive slide decks that fully capture all source material."),
                    HumanMessage(content=prompt)
                ]
            )
            
            outline.theme = style_defaults.get("theme", outline.theme)
            outline.transition = style_defaults.get("transition", outline.transition)
            
            colors = style_defaults.get("colors", {})
            if colors.get("primary"):
                outline.color_primary = colors["primary"]
            if colors.get("secondary"):
                outline.color_secondary = colors["secondary"]
            
            state["outline"] = outline
            self._emit(f"Planned {len(outline.slides)} slides")
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            state["outline"] = PresentationOutline(
                title=state["topic"],
                theme=style_defaults.get("theme", "black"),
                slides=[
                    SlideContent(slide_number=1, title=state["topic"], layout="title"),
                    SlideContent(slide_number=2, title="Overview", layout="content", 
                                content=["Planning failed - please retry"]),
                ]
            )
        
        return state
    
    async def _verify_slide_content(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Verify each slide has supporting evidence in RAG.
        
        For each slide:
        1. Query RAG with slide-specific terms
        2. Check if sufficient supporting content exists
        3. Store retrieved context per slide
        4. Flag slides without evidence
        """
        self._emit("Verifying slide content against RAG...")
        
        outline = state["outline"]
        if not outline:
            state["verification_passed"] = False
            return state
        
        slide_contexts: Dict[int, str] = {}
        unsupported_slides: List[int] = []
        available_topics: List[str] = []
        
        for slide in outline.slides:
            # Skip title and section slides - they don't need RAG backing
            if slide.layout in ["title", "section"]:
                slide_contexts[slide.slide_number] = ""
                continue
            
            # Build search query from slide content
            search_terms = []
            if slide.title:
                search_terms.append(slide.title)
            search_terms.extend(slide.content[:3])  # First 3 bullet points
            
            query = " ".join(search_terms)
            
            self._emit(f"  Verifying slide {slide.slide_number}: {slide.title or 'Untitled'}")
            
            # Query RAG for supporting content
            context = await self._query_slide_context(
                query=query,
                workflow_id=state["workflow_id"],
            )
            
            # Evaluate if slide has sufficient support
            support_score = await self._evaluate_slide_support(
                slide=slide,
                retrieved_context=context,
            )
            
            slide_contexts[slide.slide_number] = context
            
            if support_score < 0.5:
                unsupported_slides.append(slide.slide_number)
                self._emit(f"    ⚠️ Insufficient evidence (score: {support_score:.2f})")
            else:
                self._emit(f"    ✓ Verified (score: {support_score:.2f})")
                # Extract topics that ARE supported
                if slide.title:
                    available_topics.append(slide.title)
        
        state["slide_contexts"] = slide_contexts
        state["unsupported_slides"] = unsupported_slides
        state["available_topics"] = available_topics
        
        # Decide if we need to replan
        total_content_slides = len([s for s in outline.slides if s.layout not in ["title", "section"]])
        unsupported_ratio = len(unsupported_slides) / max(total_content_slides, 1)
        
        if unsupported_ratio > 0.3:  # More than 30% unsupported
            self._emit(f"Too many unsupported slides ({len(unsupported_slides)}/{total_content_slides}), will replan")
            state["verification_passed"] = False
        else:
            # Remove unsupported slides from outline
            if unsupported_slides:
                self._emit(f"Removing {len(unsupported_slides)} unsupported slides")
                outline.slides = [
                    s for s in outline.slides 
                    if s.slide_number not in unsupported_slides
                ]
                # Renumber slides
                for i, slide in enumerate(outline.slides):
                    slide.slide_number = i + 1
                state["outline"] = outline
            
            state["verification_passed"] = True
        
        return state
    
    async def _query_slide_context(self, query: str, workflow_id: str) -> str:
        """Query RAG for slide-specific content."""
        if not self.workflow_rag or not self.workflow_rag.enabled:
            return ""
        
        try:
            results = self.workflow_rag.query_relevant_context(
                query=query,
                workflow_id=workflow_id,
                n_results=10,
            )
            
            if not results:
                return ""
            
            context_parts = []
            for r in results:
                doc = r.get("document", "")
                if doc:
                    context_parts.append(doc)
            
            return "\n---\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Slide context query failed: {e}")
            return ""
    
    async def _evaluate_slide_support(
        self,
        slide: SlideContent,
        retrieved_context: str,
    ) -> float:
        """
        Evaluate how well retrieved context supports the slide content.
        
        Returns score from 0.0 (no support) to 1.0 (fully supported).
        """
        if not retrieved_context:
            return 0.0
        
        # For summary slides, be more lenient
        if slide.layout == "summary":
            return 0.7
        
        slide_content = f"{slide.title or ''} {' '.join(slide.content)}"
        
        if not slide_content.strip():
            return 1.0  # Empty slides are fine
        
        prompt = f"""Evaluate if the retrieved context supports the slide content.

SLIDE CONTENT:
Title: {slide.title or 'None'}
Points: {chr(10).join('- ' + p for p in slide.content)}

RETRIEVED CONTEXT:
{retrieved_context[:2000]}

Score from 0.0 to 1.0:
- 1.0: All claims in slide are directly supported by context
- 0.7: Most claims supported, minor inferences acceptable
- 0.5: Some support but significant gaps
- 0.3: Minimal support, mostly unsupported claims
- 0.0: No support found

Respond with ONLY a JSON object: {{"score": 0.X, "reason": "brief explanation"}}"""

        try:
            response = await asyncio.to_thread(
                self.validator_llm.invoke,
                [
                    SystemMessage(content="Evaluate if context supports slide content. Be strict - unsupported claims should score low."),
                    HumanMessage(content=prompt)
                ]
            )
            
            content = response.content
            
            import json
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return float(data.get("score", 0.5))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Support evaluation failed: {e}")
            return 0.5
    
    def _should_replan(self, state: RevealJSState) -> str:
        """
        Conditional edge: Decide whether to replan after verification.
        """
        if state["verification_passed"]:
            return "continue"
        
        if state["iteration_count"] >= state["max_iterations"]:
            self._emit("Max iterations reached, continuing with available content")
            # Remove unsupported slides anyway
            outline = state["outline"]
            if outline and state["unsupported_slides"]:
                outline.slides = [
                    s for s in outline.slides 
                    if s.slide_number not in state["unsupported_slides"]
                ]
                for i, slide in enumerate(outline.slides):
                    slide.slide_number = i + 1
                state["outline"] = outline
            return "continue"
        
        return "replan"
    
    async def _ground_slide_content(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Refine each slide's content using the retrieved context.
        
        Ensures bullet points are derived from actual RAG content,
        not LLM hallucination.
        """
        self._emit("Grounding slide content in retrieved context...")
        
        outline = state["outline"]
        slide_contexts = state.get("slide_contexts", {})
        
        if not outline:
            return state
        
        refined_slides = []
        
        for slide in outline.slides:
            # Skip title/section slides
            if slide.layout in ["title", "section"]:
                refined_slides.append(slide)
                continue
            
            context = slide_contexts.get(slide.slide_number, "")
            
            if not context:
                # No context - keep slide but mark in notes
                slide.speaker_notes = "Note: Limited source context for this slide."
                refined_slides.append(slide)
                continue
            
            # Use LLM to extract grounded content from context
            refined_slide = await self._refine_slide_from_context(slide, context)
            refined_slides.append(refined_slide)
        
        outline.slides = refined_slides
        state["outline"] = outline
        
        self._emit(f"Grounded {len(refined_slides)} slides in RAG context")
        return state
    
    async def _refine_slide_from_context(
        self,
        slide: SlideContent,
        context: str,
    ) -> SlideContent:
        """
        Refine a slide's content to be grounded in retrieved context.
        """
        prompt = f"""Refine this slide's content using ONLY information from the provided context.

CURRENT SLIDE:
Title: {slide.title}
Layout: {slide.layout}
Content: {chr(10).join('- ' + p for p in slide.content)}

RETRIEVED CONTEXT (source of truth):
{context[:3000]}

INSTRUCTIONS:
1. Keep the same title and layout
2. Rewrite bullet points using ONLY facts from the context
3. Include specific numbers, percentages, or findings from context
4. Remove any claims not supported by context
5. Add speaker notes citing the source
6. Keep bullets concise (max 10 words each)
7. Maximum 5 bullet points

Return a JSON object:
{{
    "title": "{slide.title}",
    "content": ["bullet1", "bullet2", ...],
    "speaker_notes": "Context sources: ..."
}}"""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [
                    SystemMessage(content="Extract slide content strictly from provided context. No hallucination."),
                    HumanMessage(content=prompt)
                ]
            )
            
            content = response.content
            
            import json
            json_match = re.search(r'\{[^{}]*"content"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    slide.content = data.get("content", slide.content)
                    slide.speaker_notes = data.get("speaker_notes", slide.speaker_notes)
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.error(f"Slide refinement failed: {e}")
        
        return slide
    
    async def _match_images(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Match available images to slides using RAG descriptions.
        """
        images = state["available_images"]
        outline = state["outline"]
        
        if not images or not outline:
            state["image_assignments"] = {}
            return state
        
        self._emit(f"Matching {len(images)} images to slides...")
        
        # Build slide descriptions
        slide_descriptions = []
        for slide in outline.slides:
            content_preview = ', '.join(slide.content[:3]) if slide.content else 'No content'
            desc = f"Slide {slide.slide_number}: '{slide.title}' ({slide.layout})\n  Content: {content_preview}"
            slide_descriptions.append(desc)
        
        # Build image descriptions with RAG context
        image_descriptions = []
        for img in images:
            # Use the enriched description from RAG
            desc = img.get('description', img['name'])
            # Truncate long descriptions for the prompt
            if len(desc) > 200:
                desc = desc[:200] + "..."
            image_descriptions.append(f"- {img['name']}:\n  {desc}")
        
        prompt = f"""Match images to slides based on semantic relevance.

SLIDES:
{chr(10).join(slide_descriptions)}

AVAILABLE IMAGES (with descriptions from analysis):
{chr(10).join(image_descriptions)}

RULES:
1. Title slides (layout="title") should NOT have images
2. Summary slides typically don't need images
3. Slides with layout="image" MUST have exactly 1 image
4. Content slides can have 0-1 images
5. Each image can only be used ONCE
6. Match based on:
   - Topic alignment between slide content and image description
   - Data relevance (e.g., correlation plot for correlation discussion)
   - Visual storytelling flow

Return assignments for slides that should have images."""

        structured_llm = self.llm.with_structured_output(ImageAssignmentPlan)
        
        try:
            plan = await asyncio.to_thread(
                structured_llm.invoke,
                [
                    SystemMessage(content="Match visualizations to presentation slides based on semantic relevance. Use the image descriptions to make informed matches."),
                    HumanMessage(content=prompt)
                ]
            )
            
            assignments = {}
            used_images = set()
            image_lookup = {img["name"]: img["path"] for img in images}
            
            for assignment in plan.assignments:
                for img_path in assignment.image_paths:
                    img_name = Path(img_path).name
                    
                    if img_name in used_images:
                        continue
                    
                    if img_name in image_lookup:
                        assignments[assignment.slide_number] = image_lookup[img_name]
                        used_images.add(img_name)
                        self._emit(f"  Slide {assignment.slide_number} ← {img_name}")
                        break
            
            state["image_assignments"] = assignments
            self._emit(f"Assigned {len(assignments)} images to slides")
            
        except Exception as e:
            logger.error(f"Image matching failed: {e}")
            state["image_assignments"] = {}
        
        return state
    
    async def _build_html(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Build the HTML presentation.
        """
        self._emit("Building HTML presentation...")
        
        outline = state["outline"]
        if not outline:
            state["html_content"] = ""
            return state
        
        result = self.builder.build(
            outline=outline,
            image_assignments=state["image_assignments"],
            output_path=None,
        )
        
        state["html_content"] = result.html_content
        return state
    
    async def _validate(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Validate the presentation for:
        1. Valid RevealJS HTML structure
        2. Complete coverage of source material
        """
        self._emit("Validating presentation...")
        
        errors = []
        html = state["html_content"]
        outline = state["outline"]
        workflow_context = state["workflow_context"]
        
        # =====================================================================
        # HTML STRUCTURE VALIDATION
        # =====================================================================
        
        if '<div class="reveal">' not in html:
            errors.append("Missing Reveal.js container: <div class='reveal'>")
        
        if '<div class="slides">' not in html:
            errors.append("Missing slides container: <div class='slides'>")
        
        section_count = html.count("<section")
        if section_count < 2:
            errors.append(f"Too few slides: found {section_count}, expected at least 2")
        
        if "Reveal.initialize" not in html:
            errors.append("Missing Reveal.initialize() call")
        
        open_sections = html.count("<section")
        close_sections = html.count("</section>")
        if open_sections != close_sections:
            errors.append(f"Mismatched section tags: {open_sections} open, {close_sections} close")
        
        if "reveal.css" not in html and "reveal.min.css" not in html:
            errors.append("Missing Reveal.js CSS")
        
        if "reveal.js" not in html and "reveal.min.js" not in html:
            errors.append("Missing Reveal.js JavaScript")
        
        # =====================================================================
        # CONTENT COMPLETENESS VALIDATION (LLM-based)
        # =====================================================================
        
        if not errors:
            self._emit("Checking content completeness...")
            
            slide_texts = []
            if outline:
                for slide in outline.slides:
                    slide_text = f"{slide.title or ''} {' '.join(slide.content)} {slide.speaker_notes}"
                    slide_texts.append(slide_text)
            
            presentation_content = " ".join(slide_texts)
            
            validation_prompt = f"""Analyze if this presentation adequately covers the source material.

SOURCE MATERIAL (from analysis):
{workflow_context[:4000]}

PRESENTATION CONTENT:
{presentation_content[:3000]}

Check for:
1. Are ALL major findings from the source included?
2. Are key statistics and data points mentioned?
3. Are conclusions and recommendations covered?
4. Is any significant information missing?

Respond with a JSON object:
{{
    "is_complete": true/false,
    "missing_topics": ["topic1", "topic2"],
    "coverage_score": 0-100,
    "suggestions": ["suggestion1", "suggestion2"]
}}

Be strict - the presentation should comprehensively cover the source material."""

            try:
                response = await asyncio.to_thread(
                    self.validator_llm.invoke,
                    [
                        SystemMessage(content="You are a strict content validator. Check if presentations fully cover their source material."),
                        HumanMessage(content=validation_prompt)
                    ]
                )
                
                content = response.content
                
                import json
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    try:
                        validation_data = json.loads(json_match.group())
                        
                        if not validation_data.get("is_complete", True):
                            missing = validation_data.get("missing_topics", [])
                            if missing:
                                errors.append(f"Missing topics: {', '.join(missing[:5])}")
                            
                            coverage = validation_data.get("coverage_score", 100)
                            if coverage < 70:
                                errors.append(f"Low content coverage: {coverage}% (need 70%+)")
                            
                            suggestions = validation_data.get("suggestions", [])
                            for s in suggestions[:3]:
                                errors.append(f"Suggestion: {s}")
                                
                    except json.JSONDecodeError:
                        logger.warning("Could not parse validation JSON")
                        
            except Exception as e:
                logger.error(f"Content validation failed: {e}")
        
        # =====================================================================
        # UPDATE STATE
        # =====================================================================
        
        state["validation_errors"] = errors
        state["validation_passed"] = len(errors) == 0
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        if errors:
            self._emit(f"Validation failed with {len(errors)} errors")
            for e in errors[:5]:
                self._emit(f"  - {e}")
        else:
            self._emit("Validation passed ✓")
        
        return state
    
    def _should_retry(self, state: RevealJSState) -> str:
        """
        Conditional edge: Decide whether to retry or finish.
        """
        if state["validation_passed"]:
            return "finalize"
        
        if state["iteration_count"] >= state["max_iterations"]:
            self._emit(f"Max iterations ({state['max_iterations']}) reached, finalizing anyway")
            return "finalize"
        
        self._emit(f"Retrying (iteration {state['iteration_count'] + 1})...")
        return "retry"
    
    async def _finalize(self, state: RevealJSState) -> RevealJSState:
        """
        Node: Save final output and create result.
        """
        self._emit("Finalizing presentation...")
        
        outline = state["outline"]
        html_content = state["html_content"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_manager:
            output_dir = self.output_manager.workflow_dir
        else:
            output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"presentation_{timestamp}.html")
        
        Path(output_path).write_text(html_content, encoding="utf-8")
        self._emit(f"Saved: {output_path}")
        
        state["result"] = GeneratedPresentation(
            output_path=output_path,
            html_content=html_content,
            slide_count=len(outline.slides) + 1 if outline else 0,
            images_embedded=list(state["image_assignments"].values()),
            success=True,
            theme_used=outline.theme if outline else "black",
            style_rules_applied=[
                r["metadata"].get("rule_id", "")
                for r in self.style_memory.get_all_rules()
            ],
        )
        
        return state
    
    # =========================================================================
    # WORKFLOW GRAPH
    # =========================================================================
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        workflow = StateGraph(RevealJSState)
        
        workflow.add_node("gather_context", self._gather_context)
        workflow.add_node("plan_presentation", self._plan_presentation)
        workflow.add_node("verify_slide_content", self._verify_slide_content)
        workflow.add_node("ground_slide_content", self._ground_slide_content)
        workflow.add_node("match_images", self._match_images)
        workflow.add_node("build_html", self._build_html)
        workflow.add_node("validate", self._validate)
        workflow.add_node("finalize", self._finalize)
        
        workflow.set_entry_point("gather_context")
        workflow.add_edge("gather_context", "plan_presentation")
        workflow.add_edge("plan_presentation", "verify_slide_content")
        
        # Conditional: replan if too many slides lack evidence
        workflow.add_conditional_edges(
            "verify_slide_content",
            self._should_replan,
            {
                "replan": "plan_presentation",
                "continue": "ground_slide_content",
            }
        )
        
        workflow.add_edge("ground_slide_content", "match_images")
        workflow.add_edge("match_images", "build_html")
        workflow.add_edge("build_html", "validate")
        
        # Conditional: retry validation if failed
        workflow.add_conditional_edges(
            "validate",
            self._should_retry,
            {
                "retry": "plan_presentation",
                "finalize": "finalize",
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def create_presentation(
        self,
        topic: str,
        workflow_id: str,
        stage_name: str,
        num_slides: int = 6,
        custom_instructions: str = "",
        theme: Optional[str] = None,
        transition: Optional[str] = None,
        color_primary: Optional[str] = None,
        color_secondary: Optional[str] = None,
    ) -> GeneratedPresentation:
        """
        Create a Reveal.js presentation using the full workflow.
        """
        self._emit("Starting presentation generation workflow...")
        
        if theme:
            self.style_memory.guide.default_theme = theme
        if transition:
            self.style_memory.guide.default_transition = transition
        if color_primary or color_secondary:
            colors = self.style_memory.guide.default_colors.copy()
            if color_primary:
                colors["primary"] = color_primary.lstrip("#")
            if color_secondary:
                colors["secondary"] = color_secondary.lstrip("#")
            self.style_memory.guide.default_colors = colors
        
        initial_state: RevealJSState = {
            "topic": topic,
            "workflow_id": workflow_id,
            "stage_name": stage_name,
            "num_slides": num_slides,
            "custom_instructions": custom_instructions,
            "workflow_context": "",
            "revealjs_snippets": "",
            "style_context": "",
            "plot_analyses": "",
            "available_images": [],
            "outline": None,
            "image_assignments": {},
            "slide_contexts": {},
            "unsupported_slides": [],
            "available_topics": [],
            "verification_passed": False,
            "validation_errors": [],
            "validation_passed": False,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "html_content": "",
            "result": None,
            "messages": [],
        }
        
        final_state = await self.workflow.ainvoke(initial_state)
        
        result = final_state.get("result")
        if not result:
            return GeneratedPresentation(
                output_path="",
                html_content="",
                slide_count=0,
                images_embedded=[],
                success=False,
                error="Workflow failed to produce result",
            )
        
        return result
    
    def add_style_rule(
        self,
        rule_text: str,
        category: str,
        examples: Optional[List[str]] = None,
        priority: str = "medium",
        source: str = "user",
    ) -> Optional[str]:
        """Add a new style rule to the user's style guide."""
        return self.style_memory.add_rule(
            rule_text=rule_text,
            category=category,
            examples=examples,
            priority=priority,
            source=source,
        )
    
    async def learn_from_feedback(self, feedback: str, presentation_context: str):
        """Agent learns from user feedback and updates style guide."""
        self._emit("Learning from feedback...")
        
        prompt = f"""Analyze this feedback and extract style rules.

FEEDBACK: {feedback}

CONTEXT: {presentation_context}

Extract 0-3 concrete style rules. Format each as:
- Category: (colors/typography/layout/animations/code_style/content/branding/general)
- Rule: (specific instruction)
- Priority: (high/medium/low)"""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [
                    SystemMessage(content="Extract actionable style rules from feedback."),
                    HumanMessage(content=prompt)
                ]
            )
            
            content = response.content
            rule_pattern = re.compile(
                r'Category:\s*(\w+).*?Rule:\s*(.+?)(?=Priority:|Category:|$).*?Priority:\s*(\w+)',
                re.IGNORECASE | re.DOTALL
            )
            
            for category, rule_text, priority in rule_pattern.findall(content):
                category = category.lower().strip()
                rule_text = rule_text.strip()
                priority = priority.lower().strip()
                
                valid_categories = ["colors", "typography", "layout", "animations", 
                                   "code_style", "content", "branding", "general"]
                if category in valid_categories:
                    self.add_style_rule(
                        rule_text=rule_text,
                        category=category,
                        priority=priority if priority in ["high", "medium", "low"] else "medium",
                        source="agent",
                    )
                    self._emit(f"Learned: {rule_text[:50]}...")
                    
        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
    
    def get_style_stats(self) -> Dict[str, Any]:
        """Get statistics about the user's style guide."""
        return self.style_memory.get_stats()
    
    def export_style_guide(self) -> Dict[str, Any]:
        """Export the user's complete style guide."""
        return self.style_memory.export_guide()
    
    def import_style_guide(self, guide_data: Dict[str, Any], merge: bool = True):
        """Import a style guide."""
        self.style_memory.import_guide(guide_data, merge=merge)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


async def create_revealjs_presentation(
    topic: str,
    workflow_id: str,
    stage_name: str,
    workflow_rag=None,
    output_manager=None,
    user_id: str = "default",
    num_slides: int = 6,
    custom_instructions: str = "",
    llm_model: str = "qwen3:30b",
    base_url: str = "http://100.91.155.118:11434",
    progress_callback: Optional[Callable[[str], None]] = None,
    max_iterations: int = 3,
    **kwargs,
) -> GeneratedPresentation:
    """
    Convenience function to create a Reveal.js presentation.
    """
    agent = RevealJSAgent(
        llm_model=llm_model,
        base_url=base_url,
        workflow_rag=workflow_rag,
        output_manager=output_manager,
        user_id=user_id,
        progress_callback=progress_callback,
        max_iterations=max_iterations,
    )
    
    return await agent.create_presentation(
        topic=topic,
        workflow_id=workflow_id,
        stage_name=stage_name,
        num_slides=num_slides,
        custom_instructions=custom_instructions,
        **kwargs,
    )