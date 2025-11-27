"""
PowerPoint Generation Agent

LangGraph-style workflow that:
1. Plans presentation structure using LLM
2. Matches plots to slides using LLM analysis
3. Builds PPTX with python-pptx
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from loguru import logger

from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage

from .models import (
    PresentationOutline,
    SlideContent,
    PlotInfo,
    PlotAssignment,
    PlotAssignmentPlan,
    GeneratedPresentation,
)


class PresentationAgent:
    """
    Agent for generating PowerPoint presentations from analysis results.
    
    Workflow:
    1. Gather context (RAG, plots, findings)
    2. Plan presentation structure
    3. Match plots to slides using LLM
    4. Build PPTX using python-pptx
    """
    
    def __init__(
        self,
        llm_model: str = "qwen3:30b",
        base_url: str = "http://100.91.155.118:11434",
        rag=None,
        output_manager=None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm_model = llm_model
        self.base_url = base_url
        self.rag = rag
        self.output_manager = output_manager
        self.progress_callback = progress_callback
        
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.3,
            base_url=base_url
        )
    
    def _emit(self, message: str):
        """Emit progress update."""
        logger.info(f"📊 PPTX: {message}")
        if self.progress_callback:
            self.progress_callback(message)
    
    def _get_available_plots(self, stage_name: str) -> List[PlotInfo]:
        """Get list of available plots from the stage directory."""
        plots = []
        
        if not self.output_manager:
            return plots
        
        try:
            stage_dir = self.output_manager.get_stage_dir(stage_name)
            plots_dir = stage_dir / "plots"
            
            if plots_dir.exists():
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    for plot_file in plots_dir.glob(ext):
                        plots.append(PlotInfo(
                            name=plot_file.name,
                            path=str(plot_file),
                            description=plot_file.stem.replace("_", " ").title()
                        ))
            
            logger.info(f"📊 Found {len(plots)} plots in {stage_name}")
        except Exception as e:
            logger.warning(f"Error getting plots: {e}")
        
        return plots
    
    def _get_rag_context(self, query: str, workflow_id: str) -> str:
        """Get relevant context from RAG."""
        if not self.rag or not self.rag.enabled:
            return ""
        
        try:
            context = self.rag.get_context_summary(
                query=query,
                workflow_id=workflow_id,
                max_tokens=2000
            )
            return context
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            return ""
    
    async def plan_presentation(
        self,
        topic: str,
        workflow_id: str,
        stage_name: str,
        num_slides: int = 6,
        custom_instructions: str = "",
    ) -> PresentationOutline:
        """
        Plan the presentation structure using LLM.
        """
        self._emit(f"Planning presentation: {topic}")
        
        # Gather context
        rag_context = self._get_rag_context(
            f"analysis findings results summary {topic}",
            workflow_id
        )
        
        plots = self._get_available_plots(stage_name)
        plot_list = "\n".join([f"- {p.name}: {p.description}" for p in plots])
        
        self._emit(f"Found {len(plots)} plots and {len(rag_context)} chars of context")
        
        prompt = f"""Create a presentation outline for: "{topic}"

ANALYSIS CONTEXT:
{rag_context[:3000] if rag_context else "No prior analysis context available."}

AVAILABLE PLOTS:
{plot_list if plots else "No plots available."}

REQUIREMENTS:
- Create exactly {num_slides} slides
- First slide should be title slide (layout="title")
- Last slide should be summary/conclusion (layout="summary")
- Middle slides should present findings
- Use layout="image" for slides that focus on a single visualization
- Use layout="multi_image" for comparison slides with 2-3 images
- Use layout="content" for text-heavy explanation slides
- Use layout="two_column" for side-by-side comparisons
- Keep bullet points to 3-5 per slide maximum
- Be specific and data-driven in the content

{f"ADDITIONAL INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}

Return the structured outline."""

        structured_llm = self.llm.with_structured_output(PresentationOutline)
        
        try:
            outline = structured_llm.invoke([
                SystemMessage(content="You are a presentation designer. Create clear, data-driven slide structures based on the analysis context provided."),
                HumanMessage(content=prompt)
            ])
            
            self._emit(f"Planned {len(outline.slides)} slides")
            return outline
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return PresentationOutline(
                title=topic,
                slides=[
                    SlideContent(slide_number=1, title=topic, layout="title"),
                    SlideContent(slide_number=2, title="Overview", layout="content", 
                                bullet_points=["Analysis results pending"]),
                ]
            )
    
    async def match_plots_to_slides(
        self,
        outline: PresentationOutline,
        plots: List[PlotInfo],
        workflow_id: str,
    ) -> Dict[int, List[str]]:
        """
        Use LLM to intelligently match plots to slides based on content.
        
        Returns:
            Dict mapping slide_number -> list of plot filenames
        """
        if not plots:
            self._emit("No plots available to match")
            return {}
        
        self._emit(f"Matching {len(plots)} plots to {len(outline.slides)} slides...")
        
        # Get additional context about plots from RAG (plot analyses)
        plot_context = self._get_rag_context(
            "plot analysis visualization chart graph",
            workflow_id
        )
        
        # Build slide descriptions
        slide_descriptions = []
        for slide in outline.slides:
            desc = f"Slide {slide.slide_number}: \"{slide.title}\" (layout={slide.layout})"
            if slide.bullet_points:
                desc += f"\n  Content: {'; '.join(slide.bullet_points[:3])}"
            slide_descriptions.append(desc)
        
        # Build plot descriptions
        plot_descriptions = []
        for p in plots:
            plot_descriptions.append(f"- {p.name}: {p.description}")
        
        prompt = f"""Match the available plots to the most appropriate slides.

SLIDES:
{chr(10).join(slide_descriptions)}

AVAILABLE PLOTS:
{chr(10).join(plot_descriptions)}

PLOT ANALYSIS CONTEXT (from previous analysis):
{plot_context[:1500] if plot_context else "No plot analysis available."}

RULES:
1. Match plots based on semantic relevance to slide content
2. Title slides (slide 1) should NOT have plots
3. Summary slides typically don't need plots
4. Slides with layout="image" should have exactly 1 plot
5. Slides with layout="multi_image" should have 2-3 plots
6. Slides with layout="content" or "two_column" can have 0-2 plots
7. Each plot can only be used ONCE across all slides
8. Not every slide needs a plot
9. Match based on: topic alignment, data relationships, visual storytelling

For each slide, list which plots (if any) should appear on it."""

        structured_llm = self.llm.with_structured_output(PlotAssignmentPlan)
        
        try:
            plan = structured_llm.invoke([
                SystemMessage(content="You are an expert at matching data visualizations to presentation content. Match plots to slides based on semantic relevance."),
                HumanMessage(content=prompt)
            ])
            
            # Convert to dict and validate (each plot used only once)
            assignments = {}
            used_plots = set()
            
            for assignment in plan.assignments:
                valid_plots = []
                for plot_name in assignment.plot_names:
                    if plot_name not in used_plots:
                        # Verify plot exists
                        if any(p.name == plot_name for p in plots):
                            valid_plots.append(plot_name)
                            used_plots.add(plot_name)
                
                if valid_plots:
                    assignments[assignment.slide_number] = valid_plots
                    self._emit(f"Slide {assignment.slide_number}: {', '.join(valid_plots)}")
            
            self._emit(f"Matched {len(used_plots)} plots to slides")
            return assignments
            
        except Exception as e:
            logger.error(f"Plot matching failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    async def create_presentation(
        self,
        topic: str,
        workflow_id: str,
        stage_name: str,
        num_slides: int = 6,
        custom_instructions: str = "",
        output_path: Optional[str] = None,
    ) -> GeneratedPresentation:
        """
        Full pipeline: plan → match plots → build presentation.
        """
        self._emit("Starting presentation generation...")
        
        # Step 1: Plan structure
        outline = await self.plan_presentation(
            topic=topic,
            workflow_id=workflow_id,
            stage_name=stage_name,
            num_slides=num_slides,
            custom_instructions=custom_instructions,
        )
        
        # Step 2: Get plots and match to slides
        plots = self._get_available_plots(stage_name)
        plot_assignments = await self.match_plots_to_slides(
            outline=outline,
            plots=plots,
            workflow_id=workflow_id,
        )
        
        # Create plot path lookup
        plot_paths = {p.name: p.path for p in plots}
        
        # Step 3: Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.output_manager:
                output_dir = self.output_manager.workflow_dir
            else:
                output_dir = Path("results")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"presentation_{timestamp}.pptx")
        
        # Step 4: Build PPTX
        from .builder import PresentationBuilder
        
        builder = PresentationBuilder(progress_callback=self.progress_callback)
        
        result = builder.build(
            outline=outline,
            plot_assignments=plot_assignments,
            plot_paths=plot_paths,
            output_path=output_path,
        )
        
        return result