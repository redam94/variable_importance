from langgraph.runtime import Runtime
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from loguru import logger
from pathlib import Path
import base64

from .config import DefaultConfig
from .state import State, ExecutionDeps

DEFAULT_CONFIG = DefaultConfig()

def check_existing_outputs(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """Check for existing outputs from previous executions."""
    output_manager = runtime.context["output_manager"]
    stage_name = state.get("stage_name", "")
    
    existing = {
        "has_plots": False,
        "plots": [],
        "has_data": False,
        "data_files": [],
        "has_console_output": False,
        "console_output": [],
        "has_previous_code": False,
        "previous_code": None
    }
    
    if not stage_name:
        return {"existing_outputs": existing, "skip_execution": False}
    
    try:
        stage_dir = output_manager.get_stage_dir(stage_name)
        
        # Check for plots
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
            if plot_files:
                existing["has_plots"] = True
                existing["plots"] = [str(p) for p in plot_files]
                logger.info(f"📊 Found {len(plot_files)} existing plots")
        
        # Check for data files
        data_dir = stage_dir / "data"
        if data_dir.exists():
            data_files = list(data_dir.glob("*.csv"))
            if data_files:
                existing["has_data"] = True
                existing["data_files"] = [str(d) for d in data_files]
        
        # Check console output
        console_files = sorted(list(stage_dir.glob("console_output_*.txt")), reverse=True)
        if console_files:
            for console_file in console_files:
                if console_file.stat().st_size > 0:
                    break
                existing["has_console_output"] = True
                with open(console_file, 'r') as f:
                    existing["console_output"].append(f.read())
                
    except Exception as e:
        logger.warning(f"⚠️ Error checking existing outputs: {e}")
    
    return {"existing_outputs": existing, "skip_execution": False}


def analyze_plots_with_vision_llm(state: State, runtime) -> dict:
    """Analyze plots using vision LLM."""
    output_manager = runtime.context["output_manager"]
    stage_name = state.get("stage_name", "")
    rag = runtime.context.get("rag")
    try:
        stage_dir = output_manager.get_stage_dir(stage_name)
        
        # Check for plots
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))

    except Exception as e:
        logger.warning(f"⚠️ Error accessing plots for analysis: {e}")
        plot_files = []
                
    if not plot_files:
        logger.info("No plots to analyze")
        return {"plot_analyses": []}
    
    logger.info(f"🔍 Analyzing {len(plot_files)} plots...")
    plot_cache = runtime.context.get("plot_cache")
    vision_llm = ChatOllama(
        model="qwen3-vl:30b",
        temperature=0,
        base_url=DEFAULT_CONFIG.base_url
    )
    
    analyses = []
    
    for plot_path in plot_files:
        try:
            plot_name = Path(plot_path).name
            
            # Check cache
            if plot_cache:
                cached = plot_cache.get(plot_path)
                
                if cached:
                    logger.info(f"✅ Cache hit for plot: {cached}")
                    analyses.append(cached)
                    logger.info(f"⚡ Used cached analysis: {plot_name}")
                    
                    continue
            
            # Analyze with vision LLM
            with open(plot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            system_prompt = f"""
            You are an expert data scientist. Analyze the provided visualization plot and describe key insights, trends, and any anomalies you observe.
            Provide insights in a clear and concise manner, suitable for a layperson audience.
            Be thorough and describe all relevant aspects of the visualization.
            - If the plot shows trends over time, describe those trends.
            - If the plot has legends, axes, or labels, interpret them.
            """
            response = vision_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"},
                    {"type": "text", "text": "Analyze this plot."}
                ])
            ])
            
            analysis = {
                "plot_path": plot_path,
                "plot_name": plot_name,
                "analysis": response.content
            }
            analyses.append(analysis)
            
            if plot_cache:
                plot_cache.set(plot_path, analysis)

            if rag and rag.enabled:
                rag.add_plot_analysis(
                    plot_name=plot_name,
                    plot_path=str(plot_path),
                    analysis=response.content,
                    stage_name=stage_name,
                    workflow_id=state.get("workflow_id", "default")
                )
            

            logger.info(f"✅ Analyzed: {plot_name}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {plot_path}: {e}")
    
    return {"plot_analyses": analyses}