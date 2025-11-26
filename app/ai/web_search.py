"""
Web Search - Analytics methodology research using crawl4ai.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    logger.warning("crawl4ai not installed")


@dataclass
class SearchResult:
    url: str
    title: str
    content: str
    score: float = 0.0


@dataclass
class WebSearchContext:
    query: str
    results: List[SearchResult] = field(default_factory=list)
    error: Optional[str] = None


# Curated documentation sources
SOURCES = {
    "sklearn": "https://scikit-learn.org/stable/user_guide.html",
    "pandas": "https://pandas.pydata.org/docs/user_guide/index.html",
    "seaborn": "https://seaborn.pydata.org/tutorial.html",
    "statsmodels": "https://www.statsmodels.org/stable/user-guide.html",
    "matplotlib": "https://matplotlib.org/stable/tutorials/index.html",
}


def select_sources(query: str, max_sources: int = 2) -> List[str]:
    """Select relevant sources based on query keywords."""
    q = query.lower()
    selected = []
    
    if any(k in q for k in ["model", "classify", "regress", "cluster", "predict"]):
        selected.append(SOURCES["sklearn"])
    if any(k in q for k in ["dataframe", "pandas", "merge", "groupby", "clean"]):
        selected.append(SOURCES["pandas"])
    if any(k in q for k in ["plot", "visual", "chart", "graph", "heatmap"]):
        selected.append(SOURCES["seaborn"])
    if any(k in q for k in ["statistic", "hypothesis", "anova", "ttest"]):
        selected.append(SOURCES["statsmodels"])
    
    # Fill with defaults if needed
    if not selected:
        selected = [SOURCES["pandas"], SOURCES["seaborn"]]
    
    return selected[:max_sources]


def extract_relevant(markdown: str, query: str, max_len: int = 1500) -> str:
    """Extract relevant sections from markdown."""
    if not markdown:
        return ""
    
    sections = markdown.split("\n\n")
    query_words = set(query.lower().split())
    
    scored = []
    for section in sections:
        if len(section.strip()) < 30:
            continue
        
        s_lower = section.lower()
        score = sum(1 for w in query_words if w in s_lower)
        
        # Boost code blocks
        if "```" in section or "def " in section:
            score += 3
        if "example" in s_lower:
            score += 2
            
        scored.append((score, section))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    parts, length = [], 0
    for _, section in scored:
        if length + len(section) > max_len:
            break
        parts.append(section)
        length += len(section)
    
    return "\n\n".join(parts)


async def search_analytics_methods(query: str, max_results: int = 2) -> WebSearchContext:
    """Search documentation for analytics methodology guidance."""
    ctx = WebSearchContext(query=query)
    
    if not AVAILABLE:
        ctx.error = "crawl4ai not installed"
        return ctx
    
    sources = select_sources(query, max_results)
    logger.info(f"🌐 Searching {len(sources)} sources...")
    
    try:
        browser_cfg = BrowserConfig(headless=True, verbose=False)
        crawler_cfg = CrawlerRunConfig(word_count_threshold=100)
        
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            for url in sources:
                try:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=crawler_cfg),
                        timeout=20
                    )
                    
                    if result.success and result.markdown:
                        content = extract_relevant(result.markdown, query)
                        if content:
                            ctx.results.append(SearchResult(
                                url=url,
                                title=result.metadata.get("title", url),
                                content=content,
                                score=len(content) / 1000
                            ))
                            logger.info(f"✅ Got content from {url}")
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout: {url}")
                except Exception as e:
                    logger.warning(f"Error crawling {url}: {e}")
                    
    except Exception as e:
        ctx.error = str(e)
        logger.error(f"Search failed: {e}")
    
    return ctx