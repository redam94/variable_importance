"""
RevealJS HTML Builder

Generates standalone HTML presentations with:
- Embedded CSS and JS from CDN
- Base64 embedded images
- Custom theming
- Speaker notes
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
from loguru import logger

from .models import PresentationOutline, SlideContent, GeneratedPresentation


class RevealJSBuilder:
    """
    Builds standalone Reveal.js HTML presentations.
    
    Features:
    - CDN-based resources for portability
    - Base64 image embedding for true standalone
    - Custom CSS injection
    - All slide layouts supported
    - Speaker notes included
    """
    
    REVEALJS_VERSION = "5.1.0"
    HIGHLIGHT_VERSION = "11.9.0"
    
    CDN_BASE = f"https://cdn.jsdelivr.net/npm/reveal.js@{REVEALJS_VERSION}"
    HIGHLIGHT_CDN = f"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/{HIGHLIGHT_VERSION}"
    
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        self.progress_callback = progress_callback
    
    def _emit(self, message: str):
        """Emit progress update."""
        logger.info(f"🎨 Builder: {message}")
        if self.progress_callback:
            self.progress_callback(message)
    
    def _embed_image(self, image_path: str) -> Optional[str]:
        """Convert image to base64 data URL."""
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        try:
            suffix = path.suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(suffix, "image/png")
            
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            
            return f"data:{mime_type};base64,{data}"
            
        except Exception as e:
            logger.error(f"Failed to embed image {image_path}: {e}")
            return None
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
    
    def _build_title_slide(self, outline: PresentationOutline) -> str:
        """Build the title slide HTML."""
        parts = [f'<section data-background-gradient="linear-gradient(135deg, #{outline.color_primary}, #{outline.color_secondary})">']
        parts.append(f'    <h1 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{self._escape_html(outline.title)}</h1>')
        
        if outline.subtitle:
            parts.append(f'    <h3 style="color: rgba(255,255,255,0.9);">{self._escape_html(outline.subtitle)}</h3>')
        
        if outline.author or outline.date:
            parts.append('    <p style="margin-top: 2em; color: rgba(255,255,255,0.7);">')
            if outline.author:
                parts.append(f'        {self._escape_html(outline.author)}')
            if outline.author and outline.date:
                parts.append('        <br>')
            if outline.date:
                parts.append(f'        {self._escape_html(outline.date)}')
            parts.append('    </p>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_content_slide(
        self,
        slide: SlideContent,
        image_data: Optional[str] = None,
        primary_color: str = "667eea",
    ) -> str:
        """Build a standard content slide."""
        parts = []
        
        # Slide opening with optional background
        if slide.background_color:
            parts.append(f'<section data-background-color="#{slide.background_color}">')
        else:
            parts.append('<section>')
        
        # Title
        if slide.title:
            parts.append(f'    <h2>{self._escape_html(slide.title)}</h2>')
        
        if slide.subtitle:
            parts.append(f'    <h3 style="color: #{primary_color};">{self._escape_html(slide.subtitle)}</h3>')
        
        # Content with optional image
        if image_data and slide.content:
            # Side-by-side layout
            parts.append('    <div style="display: flex; align-items: flex-start; gap: 2em;">')
            parts.append('        <div style="flex: 1; text-align: left;">')
            parts.append('            <ul>')
            for item in slide.content:
                frag = f'class="fragment {slide.fragment_style}"' if slide.fragment_style != "none" else ""
                parts.append(f'                <li {frag}>{self._escape_html(item)}</li>')
            parts.append('            </ul>')
            parts.append('        </div>')
            parts.append('        <div style="flex: 1;">')
            parts.append(f'            <img src="{image_data}" style="max-height: 400px; max-width: 100%;">')
            if slide.image_caption:
                parts.append(f'            <p style="font-size: 0.6em; color: #888;">{self._escape_html(slide.image_caption)}</p>')
            parts.append('        </div>')
            parts.append('    </div>')
        elif image_data:
            # Image only
            parts.append(f'    <img src="{image_data}" style="max-height: 500px; max-width: 90%;">')
            if slide.image_caption:
                parts.append(f'    <p style="font-size: 0.6em; color: #888;">{self._escape_html(slide.image_caption)}</p>')
        elif slide.content:
            # Content only
            parts.append('    <ul>')
            for item in slide.content:
                frag = f'class="fragment {slide.fragment_style}"' if slide.fragment_style != "none" else ""
                parts.append(f'        <li {frag}>{self._escape_html(item)}</li>')
            parts.append('    </ul>')
        
        # Speaker notes
        if slide.speaker_notes:
            parts.append(f'    <aside class="notes">{self._escape_html(slide.speaker_notes)}</aside>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_two_column_slide(
        self,
        slide: SlideContent,
        primary_color: str = "667eea",
    ) -> str:
        """Build a two-column slide."""
        parts = ['<section>']
        
        if slide.title:
            parts.append(f'    <h2>{self._escape_html(slide.title)}</h2>')
        
        # Split content into two columns
        mid = len(slide.content) // 2
        left_content = slide.content[:mid] if slide.content else []
        right_content = slide.content[mid:] if slide.content else []
        
        parts.append('    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2em; text-align: left;">')
        
        # Left column
        parts.append('        <div>')
        parts.append('            <ul>')
        for item in left_content:
            frag = f'class="fragment {slide.fragment_style}"' if slide.fragment_style != "none" else ""
            parts.append(f'                <li {frag}>{self._escape_html(item)}</li>')
        parts.append('            </ul>')
        parts.append('        </div>')
        
        # Right column
        parts.append('        <div>')
        parts.append('            <ul>')
        for item in right_content:
            frag = f'class="fragment {slide.fragment_style}"' if slide.fragment_style != "none" else ""
            parts.append(f'                <li {frag}>{self._escape_html(item)}</li>')
        parts.append('            </ul>')
        parts.append('        </div>')
        
        parts.append('    </div>')
        
        if slide.speaker_notes:
            parts.append(f'    <aside class="notes">{self._escape_html(slide.speaker_notes)}</aside>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_image_slide(
        self,
        slide: SlideContent,
        image_data: Optional[str] = None,
    ) -> str:
        """Build a large image slide."""
        parts = ['<section>']
        
        if slide.title:
            parts.append(f'    <h2>{self._escape_html(slide.title)}</h2>')
        
        if image_data:
            parts.append(f'    <img src="{image_data}" style="max-height: 550px; max-width: 95%;">')
            if slide.image_caption:
                parts.append(f'    <p style="font-size: 0.6em; color: #888; margin-top: 0.5em;">{self._escape_html(slide.image_caption)}</p>')
        else:
            parts.append('    <p style="color: #888;">Image not available</p>')
        
        if slide.speaker_notes:
            parts.append(f'    <aside class="notes">{self._escape_html(slide.speaker_notes)}</aside>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_code_slide(
        self,
        slide: SlideContent,
        primary_color: str = "667eea",
    ) -> str:
        """Build a code block slide."""
        parts = ['<section>']
        
        if slide.title:
            parts.append(f'    <h2>{self._escape_html(slide.title)}</h2>')
        
        if slide.code_block:
            lang = slide.code_language or "python"
            # Don't escape code - it needs to be literal
            parts.append(f'    <pre><code class="language-{lang}" data-trim data-line-numbers>')
            parts.append(slide.code_block)
            parts.append('    </code></pre>')
        
        if slide.content:
            parts.append('    <p style="font-size: 0.7em; text-align: left; margin-top: 1em;">')
            for item in slide.content:
                parts.append(f'        {self._escape_html(item)}<br>')
            parts.append('    </p>')
        
        if slide.speaker_notes:
            parts.append(f'    <aside class="notes">{self._escape_html(slide.speaker_notes)}</aside>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_quote_slide(
        self,
        slide: SlideContent,
        primary_color: str = "667eea",
    ) -> str:
        """Build a blockquote slide."""
        parts = ['<section>']
        
        if slide.title:
            parts.append(f'    <h2>{self._escape_html(slide.title)}</h2>')
        
        if slide.content:
            quote_text = slide.content[0] if slide.content else ""
            attribution = slide.content[1] if len(slide.content) > 1 else ""
            
            parts.append(f'''    <blockquote style="background: rgba(255,255,255,0.05); 
                       padding: 1em 2em; 
                       border-left: 4px solid #{primary_color};
                       font-style: italic;
                       text-align: left;">
        "{self._escape_html(quote_text)}"''')
            
            if attribution:
                parts.append(f'''        <footer style="font-size: 0.7em; margin-top: 0.5em; font-style: normal; text-align: right;">
            — {self._escape_html(attribution)}
        </footer>''')
            
            parts.append('    </blockquote>')
        
        if slide.speaker_notes:
            parts.append(f'    <aside class="notes">{self._escape_html(slide.speaker_notes)}</aside>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_section_slide(
        self,
        slide: SlideContent,
        primary_color: str = "667eea",
        secondary_color: str = "764ba2",
    ) -> str:
        """Build a section divider slide."""
        parts = [f'<section data-background-gradient="linear-gradient(135deg, #{primary_color}, #{secondary_color})">']
        
        if slide.title:
            parts.append(f'    <h1 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{self._escape_html(slide.title)}</h1>')
        
        if slide.subtitle:
            parts.append(f'    <h3 style="color: rgba(255,255,255,0.8);">{self._escape_html(slide.subtitle)}</h3>')
        
        if slide.speaker_notes:
            parts.append(f'    <aside class="notes">{self._escape_html(slide.speaker_notes)}</aside>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_summary_slide(
        self,
        slide: SlideContent,
        primary_color: str = "667eea",
    ) -> str:
        """Build a summary/takeaways slide."""
        parts = [f'<section data-background-color="#{primary_color}">']
        
        title = slide.title or "Key Takeaways"
        parts.append(f'    <h2 style="color: white;">{self._escape_html(title)}</h2>')
        
        if slide.content:
            parts.append('    <ul style="list-style: none; padding: 0;">')
            for item in slide.content:
                frag = f'class="fragment {slide.fragment_style}"' if slide.fragment_style != "none" else ""
                parts.append(f'        <li {frag} style="color: white; margin: 0.5em 0; font-size: 1.2em;">✓ {self._escape_html(item)}</li>')
            parts.append('    </ul>')
        
        if slide.speaker_notes:
            parts.append(f'    <aside class="notes">{self._escape_html(slide.speaker_notes)}</aside>')
        
        parts.append('</section>')
        return "\n".join(parts)
    
    def _build_slide(
        self,
        slide: SlideContent,
        image_paths: Dict[int, str],
        outline: PresentationOutline,
    ) -> str:
        """Build a single slide based on its layout."""
        image_data = None
        if slide.image_path:
            image_data = self._embed_image(slide.image_path)
        elif slide.slide_number in image_paths:
            image_data = self._embed_image(image_paths[slide.slide_number])
        
        layout = slide.layout
        primary = outline.color_primary
        secondary = outline.color_secondary
        
        if layout == "title":
            # Title slides handled separately in build()
            return self._build_content_slide(slide, image_data, primary)
        elif layout == "two_column":
            return self._build_two_column_slide(slide, primary)
        elif layout == "image":
            return self._build_image_slide(slide, image_data)
        elif layout == "code":
            return self._build_code_slide(slide, primary)
        elif layout == "quote":
            return self._build_quote_slide(slide, primary)
        elif layout == "section":
            return self._build_section_slide(slide, primary, secondary)
        elif layout == "summary":
            return self._build_summary_slide(slide, primary)
        else:  # content
            return self._build_content_slide(slide, image_data, primary)
    
    def _get_custom_css(self, outline: PresentationOutline) -> str:
        """Generate custom CSS for the presentation."""
        primary = outline.color_primary
        secondary = outline.color_secondary
        
        css = f"""
/* Custom theme overrides */
.reveal {{
    --r-heading-color: #{primary};
    --r-link-color: #{secondary};
    --r-link-color-hover: #{primary};
    --r-selection-background-color: #{primary};
}}

.reveal h1, .reveal h2, .reveal h3 {{
    text-transform: none;
}}

.reveal ul {{
    text-align: left;
}}

.reveal li {{
    margin: 0.5em 0;
}}

.reveal pre {{
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    border-radius: 8px;
}}

.reveal blockquote {{
    box-shadow: none;
}}

/* Gradient text for headers on dark backgrounds */
.reveal section[data-background-gradient] h1,
.reveal section[data-background-gradient] h2 {{
    background: none;
    -webkit-background-clip: unset;
    -webkit-text-fill-color: white;
}}
"""
        
        if outline.custom_css:
            css += f"\n/* User custom CSS */\n{outline.custom_css}"
        
        return css
    
    def build(
        self,
        outline: PresentationOutline,
        image_assignments: Optional[Dict[int, str]] = None,
        output_path: Optional[str] = None,
    ) -> GeneratedPresentation:
        """
        Build the complete standalone HTML presentation.
        
        Args:
            outline: Presentation structure and content
            image_assignments: Mapping of slide_number -> image_path
            output_path: Where to save the HTML file
            
        Returns:
            GeneratedPresentation with HTML content and metadata
        """
        start_time = datetime.now()
        self._emit("Building Reveal.js presentation...")
        
        image_assignments = image_assignments or {}
        images_embedded = []
        
        try:
            # Build slides HTML
            slides_html = []
            
            # First slide is always the title
            slides_html.append(self._build_title_slide(outline))
            
            # Build remaining slides
            for slide in outline.slides:
                if slide.layout == "title" and slide.slide_number == 1:
                    continue  # Skip if it's the title slide we already added
                
                self._emit(f"Building slide {slide.slide_number}: {slide.title or 'Untitled'}")
                
                slide_html = self._build_slide(slide, image_assignments, outline)
                slides_html.append(slide_html)
                
                # Track embedded images
                if slide.slide_number in image_assignments:
                    images_embedded.append(image_assignments[slide.slide_number])
                if slide.image_path:
                    images_embedded.append(slide.image_path)
            
            # Assemble full HTML
            custom_css = self._get_custom_css(outline)
            
            html_content = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._escape_html(outline.title)}</title>
    
    <!-- Reveal.js CSS -->
    <link rel="stylesheet" href="{self.CDN_BASE}/dist/reveal.css">
    <link rel="stylesheet" href="{self.CDN_BASE}/dist/theme/{outline.theme}.css">
    
    <!-- Highlight.js for code -->
    <link rel="stylesheet" href="{self.HIGHLIGHT_CDN}/styles/monokai.min.css">
    
    <style>
{custom_css}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
{chr(10).join('            ' + line for slide in slides_html for line in slide.split(chr(10)))}
        </div>
    </div>

    <!-- Reveal.js -->
    <script src="{self.CDN_BASE}/dist/reveal.js"></script>
    <script src="{self.CDN_BASE}/plugin/notes/notes.js"></script>
    <script src="{self.CDN_BASE}/plugin/highlight/highlight.js"></script>
    
    <script>
        Reveal.initialize({{
            hash: true,
            slideNumber: true,
            transition: '{outline.transition}',
            plugins: [ RevealNotes, RevealHighlight ]
        }});
    </script>
</body>
</html>"""
            
            # Save to file if path provided
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(html_content, encoding="utf-8")
                self._emit(f"Saved: {output_path}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return GeneratedPresentation(
                output_path=output_path or "",
                html_content=html_content,
                slide_count=len(outline.slides) + 1,  # +1 for title
                images_embedded=images_embedded,
                success=True,
                theme_used=outline.theme,
                generation_time_seconds=elapsed,
            )
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return GeneratedPresentation(
                output_path=output_path or "",
                html_content="",
                slide_count=0,
                images_embedded=[],
                success=False,
                error=str(e),
            )