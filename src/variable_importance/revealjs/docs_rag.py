"""
RevealJS Documentation RAG

Shared singleton RAG containing:
- RevealJS code snippets
- Best practices
- Theme configurations
- Plugin examples

Pre-populated with curated content on first initialization.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("⚠️ ChromaDB not available")


class RevealJSDocsRAG:
    """
    Singleton RAG for RevealJS documentation and best practices.
    
    Shared across all users - contains static reference content.
    Pre-populated with curated snippets on first initialization.
    """
    
    _instance: Optional["RevealJSDocsRAG"] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        persist_directory: str = "cache/revealjs_docs_rag",
        collection_name: str = "revealjs_docs",
    ):
        if self._initialized:
            return
        
        if not CHROMADB_AVAILABLE:
            logger.error("❌ ChromaDB not available. RevealJS docs RAG disabled.")
            self.enabled = False
            self._initialized = True
            return
        
        self.enabled = True
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Populate with default content if empty
        if self.collection.count() == 0:
            self._populate_default_content()
        
        logger.info(f"📚 RevealJS DocsRAG initialized: {self.collection.count()} snippets")
        self._initialized = True
    
    @classmethod
    def get_instance(cls, **kwargs) -> "RevealJSDocsRAG":
        """Get or create singleton instance."""
        return cls(**kwargs)
    
    def _generate_id(self, content: str, category: str) -> str:
        """Generate unique ID for a snippet."""
        hash_input = f"{category}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add_snippet(
        self,
        title: str,
        description: str,
        code: str,
        category: str,
        tags: Optional[List[str]] = None,
        url: Optional[str] = None,
    ):
        """Add a code snippet to the RAG."""
        if not self.enabled:
            return
        
        doc_text = f"""Title: {title}
Category: {category}
Description: {description}

Code:
```
{code}
```

Tags: {', '.join(tags or [])}"""
        
        metadata = {
            "title": title,
            "category": category,
            "tags": ", ".join(tags or []),
            "url": url or "",
            "type": "snippet",
            "added_at": datetime.now().isoformat(),
        }
        
        doc_id = self._generate_id(code, category)
        
        try:
            self.collection.upsert(
                documents=[doc_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.debug(f"📝 Added snippet: {title}")
        except Exception as e:
            logger.error(f"Failed to add snippet: {e}")
    
    def query(
        self,
        query: str,
        category: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query for relevant snippets.
        
        Args:
            query: Search query
            category: Filter by category (optional)
            n_results: Max results to return
            
        Returns:
            List of matching snippets with metadata
        """
        if not self.enabled:
            return []
        
        try:
            where_filter = None
            if category:
                where_filter = {"category": category}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            snippets = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    snippets.append({
                        "document": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
            
            logger.debug(f"🔍 Found {len(snippets)} snippets for: {query[:50]}...")
            return snippets
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_by_category(self, category: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all snippets in a category."""
        if not self.enabled:
            return []
        
        try:
            results = self.collection.get(
                where={"category": category},
                limit=limit
            )
            
            snippets = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    snippets.append({
                        "document": doc,
                        "metadata": results["metadatas"][i]
                    })
            
            return snippets
        except Exception as e:
            logger.error(f"Get by category failed: {e}")
            return []
    
    def _populate_default_content(self):
        """Populate with curated RevealJS documentation."""
        logger.info("📚 Populating RevealJS documentation RAG...")
        
        snippets = self._get_default_snippets()
        
        for snippet in snippets:
            self.add_snippet(**snippet)
        
        logger.info(f"✅ Added {len(snippets)} default snippets")
    
    def _get_default_snippets(self) -> List[Dict[str, Any]]:
        """Return curated RevealJS snippets."""
        return [
            # === SETUP ===
            {
                "title": "Basic HTML Structure",
                "description": "Minimal HTML structure for a Reveal.js presentation",
                "category": "setup",
                "tags": ["html", "structure", "basic"],
                "code": """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Presentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/black.css">
</head>
<body>
    <div class="reveal">
        <div class="slides">
            <section>Slide 1</section>
            <section>Slide 2</section>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
    <script>Reveal.initialize();</script>
</body>
</html>"""
            },
            {
                "title": "Reveal.js Initialization Options",
                "description": "Common configuration options for Reveal.initialize()",
                "category": "setup",
                "tags": ["config", "options", "initialize"],
                "code": """Reveal.initialize({
    // Display controls in the bottom right corner
    controls: true,
    // Display a presentation progress bar
    progress: true,
    // Display the page number of the current slide
    slideNumber: true,
    // Push each slide change to browser history
    history: true,
    // Enable keyboard shortcuts for navigation
    keyboard: true,
    // Enable slide overview mode
    overview: true,
    // Vertical centering of slides
    center: true,
    // Enable touch navigation on touch devices
    touch: true,
    // Transition style: none/fade/slide/convex/concave/zoom
    transition: 'slide',
    // Transition speed: default/fast/slow
    transitionSpeed: 'default',
    // Parallax background image
    parallaxBackgroundImage: '',
    // Number of slides away from current that are visible
    viewDistance: 3,
});"""
            },
            
            # === SLIDES ===
            {
                "title": "Vertical Slides (Nested)",
                "description": "Create vertical slide stacks for sub-topics",
                "category": "slides",
                "tags": ["vertical", "nested", "navigation"],
                "code": """<section>
    <section>Horizontal Slide 1 - Vertical 1</section>
    <section>Horizontal Slide 1 - Vertical 2</section>
    <section>Horizontal Slide 1 - Vertical 3</section>
</section>
<section>
    <section>Horizontal Slide 2 - Vertical 1</section>
    <section>Horizontal Slide 2 - Vertical 2</section>
</section>"""
            },
            {
                "title": "Two Column Layout",
                "description": "Side-by-side content using CSS grid or flexbox",
                "category": "slides",
                "tags": ["layout", "columns", "grid"],
                "code": """<section>
    <h2>Two Column Layout</h2>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2em;">
        <div>
            <h3>Left Column</h3>
            <ul>
                <li>Point one</li>
                <li>Point two</li>
            </ul>
        </div>
        <div>
            <h3>Right Column</h3>
            <ul>
                <li>Point three</li>
                <li>Point four</li>
            </ul>
        </div>
    </div>
</section>"""
            },
            {
                "title": "Image with Caption",
                "description": "Display image with styled caption below",
                "category": "slides",
                "tags": ["image", "caption", "figure"],
                "code": """<section>
    <h2>Image Slide</h2>
    <figure>
        <img src="image.png" alt="Description" style="max-height: 400px;">
        <figcaption style="font-size: 0.6em; color: #888;">
            Figure 1: Image caption here
        </figcaption>
    </figure>
</section>"""
            },
            
            # === FRAGMENTS ===
            {
                "title": "Fragment Animations",
                "description": "Reveal content step by step with fragments",
                "category": "fragments",
                "tags": ["animation", "reveal", "step"],
                "code": """<section>
    <h2>Fragment Examples</h2>
    <p class="fragment">Appears first</p>
    <p class="fragment fade-in">Fade in</p>
    <p class="fragment fade-up">Fade up</p>
    <p class="fragment fade-in-then-out">Fade in, then out</p>
    <p class="fragment highlight-red">Highlight red</p>
    <p class="fragment highlight-current-blue">Blue only while current</p>
</section>"""
            },
            {
                "title": "Fragment Order Control",
                "description": "Control the order fragments appear using data-fragment-index",
                "category": "fragments",
                "tags": ["order", "index", "sequence"],
                "code": """<section>
    <p class="fragment" data-fragment-index="3">Appears third</p>
    <p class="fragment" data-fragment-index="1">Appears first</p>
    <p class="fragment" data-fragment-index="2">Appears second</p>
</section>"""
            },
            
            # === BACKGROUNDS ===
            {
                "title": "Slide Background Colors",
                "description": "Set custom background colors per slide",
                "category": "backgrounds",
                "tags": ["color", "background", "style"],
                "code": """<section data-background-color="#667eea">
    <h2>Purple Background</h2>
</section>
<section data-background-gradient="linear-gradient(to bottom, #667eea, #764ba2)">
    <h2>Gradient Background</h2>
</section>"""
            },
            {
                "title": "Background Image",
                "description": "Full-bleed background images with options",
                "category": "backgrounds",
                "tags": ["image", "background", "cover"],
                "code": """<section data-background-image="image.jpg"
         data-background-size="cover"
         data-background-position="center"
         data-background-opacity="0.5">
    <h2>Image Background</h2>
</section>"""
            },
            
            # === TRANSITIONS ===
            {
                "title": "Per-Slide Transitions",
                "description": "Override transition for specific slides",
                "category": "transitions",
                "tags": ["transition", "animation", "effect"],
                "code": """<section data-transition="zoom">
    <h2>Zoom Transition</h2>
</section>
<section data-transition="fade">
    <h2>Fade Transition</h2>
</section>
<section data-transition="slide-in fade-out">
    <h2>Mixed: Slide In, Fade Out</h2>
</section>
<section data-transition-speed="fast">
    <h2>Fast Transition</h2>
</section>"""
            },
            
            # === CODE HIGHLIGHTING ===
            {
                "title": "Code Block with Highlight.js",
                "description": "Syntax highlighted code with line numbers",
                "category": "code",
                "tags": ["code", "syntax", "highlight"],
                "code": """<section>
    <h2>Code Example</h2>
    <pre><code class="language-python" data-trim data-line-numbers>
def hello_world():
    message = "Hello, Reveal.js!"
    print(message)
    return message

if __name__ == "__main__":
    hello_world()
    </code></pre>
</section>"""
            },
            {
                "title": "Code Line Highlighting",
                "description": "Highlight specific lines in code blocks",
                "category": "code",
                "tags": ["highlight", "lines", "focus"],
                "code": """<section>
    <pre><code data-line-numbers="1|3-4|6-8">
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Process
df['new_col'] = df['col1'] * 2
result = df.groupby('category').mean()
    </code></pre>
</section>"""
            },
            
            # === THEMES ===
            {
                "title": "Custom Theme CSS",
                "description": "Override theme variables for custom styling",
                "category": "themes",
                "tags": ["css", "custom", "variables"],
                "code": """.reveal {
    --r-background-color: #1a1a2e;
    --r-main-color: #eee;
    --r-heading-color: #667eea;
    --r-link-color: #764ba2;
    --r-link-color-hover: #f093fb;
    --r-selection-background-color: #667eea;
    --r-heading-font: 'Source Sans Pro', sans-serif;
    --r-main-font: 'Source Sans Pro', sans-serif;
    --r-heading-text-transform: none;
}

.reveal h1, .reveal h2, .reveal h3 {
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}"""
            },
            {
                "title": "Available Built-in Themes",
                "description": "List of themes included with Reveal.js",
                "category": "themes",
                "tags": ["themes", "list", "builtin"],
                "code": """<!-- Available themes (replace 'black' with theme name) -->
<link rel="stylesheet" href="dist/theme/black.css">

<!-- Options:
- black (default) - Black background, white text, blue links
- white - White background, black text, blue links  
- league - Gray background, white text, blue links
- beige - Beige background, dark text, brown links
- sky - Blue background, thin dark text, blue links
- night - Black background, thick white text, orange links
- serif - Cappuccino background, gray text, brown links
- simple - White background, black text, blue links
- solarized - Cream-colored background, dark green text
- moon - Dark blue background
- dracula - Dark purple background
-->"""
            },
            
            # === SPEAKER NOTES ===
            {
                "title": "Speaker Notes",
                "description": "Add presenter notes (press 'S' to view)",
                "category": "slides",
                "tags": ["notes", "speaker", "presenter"],
                "code": """<section>
    <h2>Slide with Notes</h2>
    <p>Audience sees this content.</p>
    
    <aside class="notes">
        Speaker notes go here. Press 'S' to open speaker view.
        - Remember to mention X
        - Demo Y at this point
        - Transition to next topic
    </aside>
</section>"""
            },
            
            # === BEST PRACTICES ===
            {
                "title": "Responsive Images",
                "description": "Best practice for embedding images that scale properly",
                "category": "slides",
                "tags": ["responsive", "image", "best-practice"],
                "code": """<section>
    <h2>Responsive Image</h2>
    <img src="chart.png" 
         alt="Chart description"
         style="max-width: 80%; max-height: 60vh; object-fit: contain;">
</section>"""
            },
            {
                "title": "Data Visualization Slide",
                "description": "Best layout for presenting charts and graphs",
                "category": "slides",
                "tags": ["chart", "visualization", "data"],
                "code": """<section>
    <h2 style="margin-bottom: 0.5em;">Key Findings</h2>
    <div style="display: flex; align-items: center; gap: 2em;">
        <div style="flex: 2;">
            <img src="chart.png" style="max-height: 450px;">
        </div>
        <div style="flex: 1; text-align: left; font-size: 0.8em;">
            <h4>Insights</h4>
            <ul>
                <li class="fragment">Point one</li>
                <li class="fragment">Point two</li>
                <li class="fragment">Point three</li>
            </ul>
        </div>
    </div>
</section>"""
            },
            {
                "title": "Quote Slide",
                "description": "Styled blockquote for impactful quotes",
                "category": "slides",
                "tags": ["quote", "blockquote", "citation"],
                "code": """<section>
    <blockquote style="background: rgba(255,255,255,0.05); 
                       padding: 1em 2em; 
                       border-left: 4px solid #667eea;
                       font-style: italic;">
        "The best way to predict the future is to invent it."
        <footer style="font-size: 0.7em; margin-top: 0.5em; font-style: normal;">
            — Alan Kay
        </footer>
    </blockquote>
</section>"""
            },
            {
                "title": "Standalone HTML Export",
                "description": "Structure for self-contained HTML with embedded resources",
                "category": "setup",
                "tags": ["export", "standalone", "embed"],
                "code": """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Presentation</title>
    <!-- Embed CSS directly for standalone -->
    <style>
        /* Reveal.js core CSS inlined here */
        /* Theme CSS inlined here */
        /* Custom styles */
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            <!-- Slides here -->
        </div>
    </div>
    <!-- Reveal.js script inlined or from CDN -->
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
    <script>Reveal.initialize({hash: true});</script>
</body>
</html>"""
            },
        ]


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_docs_rag_instance: Optional[RevealJSDocsRAG] = None


def get_revealjs_docs_rag(**kwargs) -> RevealJSDocsRAG:
    """Get the shared RevealJS documentation RAG instance."""
    global _docs_rag_instance
    if _docs_rag_instance is None:
        _docs_rag_instance = RevealJSDocsRAG(**kwargs)
    return _docs_rag_instance