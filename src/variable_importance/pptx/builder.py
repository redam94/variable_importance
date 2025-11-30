"""
PowerPoint Builder using python-pptx

Features:
- Pure python-pptx implementation (no external dependencies)
- Smart image fitting that maintains aspect ratio
- Multi-image layouts with automatic positioning
- Custom color themes
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable
from loguru import logger

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

from .models import PresentationOutline, SlideContent, GeneratedPresentation


class PresentationBuilder:
    """
    Builds PowerPoint presentations using python-pptx.
    
    Handles:
    - Multiple layout types
    - Smart image positioning and scaling
    - Multi-image slides
    - Color theming
    """
    
    # Slide dimensions (16:9 aspect ratio)
    SLIDE_WIDTH = Inches(13.333)
    SLIDE_HEIGHT = Inches(7.5)
    
    # Content zones
    MARGIN = Inches(0.5)
    TITLE_HEIGHT = Inches(1.0)
    CONTENT_TOP = Inches(1.5)
    CONTENT_WIDTH = Inches(12.333)  # SLIDE_WIDTH - 2*MARGIN
    CONTENT_HEIGHT = Inches(5.5)    # Available for content
    
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        self.progress_callback = progress_callback
    
    def _emit(self, message: str):
        """Emit progress update."""
        logger.info(f"📊 Builder: {message}")
        if self.progress_callback:
            self.progress_callback(message)
    
    def _hex_to_rgb(self, hex_color: str) -> RGBColor:
        """Convert hex color string to RGBColor."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return RGBColor(r, g, b)
    
    def _get_image_dimensions(self, image_path: str) -> tuple[int, int]:
        """Get image dimensions in pixels."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            logger.warning(f"Could not read image dimensions: {e}")
            return (800, 600)  # Default fallback
    
    def _calculate_image_fit(
        self,
        img_width: int,
        img_height: int,
        max_width: float,
        max_height: float,
    ) -> tuple[float, float]:
        """
        Calculate image dimensions to fit within bounds while maintaining aspect ratio.
        
        Returns:
            (width, height) in inches as floats
        """
        img_aspect = img_width / img_height
        box_aspect = max_width / max_height
        
        if img_aspect > box_aspect:
            # Image is wider than box - constrain by width
            final_width = max_width
            final_height = max_width / img_aspect
        else:
            # Image is taller than box - constrain by height
            final_height = max_height
            final_width = max_height * img_aspect
        
        return (final_width, final_height)
    
    def _add_title_slide(
        self,
        prs: Presentation,
        outline: PresentationOutline,
        primary_color: RGBColor,
    ):
        """Add title slide."""
        slide_layout = prs.slide_layouts[6]  # Blank layout
        slide = prs.slides.add_slide(slide_layout)
        
        # Title
        title_box = slide.shapes.add_textbox(
            self.MARGIN, Inches(2.5),
            self.CONTENT_WIDTH, Inches(1.5)
        )
        title_frame = title_box.text_frame
        title_para = title_frame.paragraphs[0]
        title_para.text = outline.title
        title_para.font.size = Pt(44)
        title_para.font.bold = True
        title_para.font.color.rgb = primary_color
        title_para.alignment = PP_ALIGN.CENTER
        
        # Subtitle
        if outline.subtitle:
            subtitle_box = slide.shapes.add_textbox(
                self.MARGIN, Inches(4.2),
                self.CONTENT_WIDTH, Inches(1.0)
            )
            subtitle_frame = subtitle_box.text_frame
            subtitle_para = subtitle_frame.paragraphs[0]
            subtitle_para.text = outline.subtitle
            subtitle_para.font.size = Pt(24)
            subtitle_para.font.color.rgb = RGBColor(100, 100, 100)
            subtitle_para.alignment = PP_ALIGN.CENTER
        
        # Bottom accent line
        line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(4), Inches(5.5),
            Inches(5.333), Inches(0.05)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = primary_color
        line.line.fill.background()
    
    def _add_content_slide(
        self,
        prs: Presentation,
        content: SlideContent,
        images: List[str],
        primary_color: RGBColor,
        accent_color: RGBColor,
    ):
        """Add a content slide with optional images."""
        slide_layout = prs.slide_layouts[6]  # Blank
        slide = prs.slides.add_slide(slide_layout)
        
        # Title bar background
        title_bg = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(0), Inches(0),
            self.SLIDE_WIDTH, Inches(1.2)
        )
        title_bg.fill.solid()
        title_bg.fill.fore_color.rgb = primary_color
        title_bg.line.fill.background()
        
        # Title text
        title_box = slide.shapes.add_textbox(
            self.MARGIN, Inches(0.3),
            self.CONTENT_WIDTH, Inches(0.8)
        )
        title_frame = title_box.text_frame
        title_para = title_frame.paragraphs[0]
        title_para.text = content.title
        title_para.font.size = Pt(32)
        title_para.font.bold = True
        title_para.font.color.rgb = RGBColor(255, 255, 255)
        
        # Determine layout based on images
        num_images = len(images)
        
        if num_images == 0:
            # Full width text
            self._add_text_content(
                slide, content,
                left=self.MARGIN,
                top=self.CONTENT_TOP,
                width=self.CONTENT_WIDTH,
                height=self.CONTENT_HEIGHT,
                accent_color=accent_color,
            )
        elif num_images == 1:
            # Text on left, image on right (or full image for image layout)
            if content.layout == "image":
                # Large centered image with minimal text
                self._add_single_image_layout(
                    slide, content, images[0], accent_color
                )
            else:
                # Split layout
                text_width = Inches(5.5)
                self._add_text_content(
                    slide, content,
                    left=self.MARGIN,
                    top=self.CONTENT_TOP,
                    width=text_width,
                    height=self.CONTENT_HEIGHT,
                    accent_color=accent_color,
                )
                self._add_image(
                    slide, images[0],
                    left=Inches(6.5),
                    top=self.CONTENT_TOP,
                    max_width=6.0,
                    max_height=5.2,
                )
        else:
            # Multiple images
            self._add_multi_image_layout(
                slide, content, images, accent_color
            )
        
        # Speaker notes
        if content.speaker_notes:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = content.speaker_notes
    
    def _add_text_content(
        self,
        slide,
        content: SlideContent,
        left: float,
        top: float,
        width: float,
        height: float,
        accent_color: RGBColor,
    ):
        """Add text content (bullets or body text) to slide."""
        text_box = slide.shapes.add_textbox(left, top, width, height)
        text_frame = text_box.text_frame
        text_frame.word_wrap = True
        
        if content.body_text:
            # Paragraph text
            para = text_frame.paragraphs[0]
            para.text = content.body_text
            para.font.size = Pt(18)
            para.font.color.rgb = RGBColor(50, 50, 50)
            para.line_spacing = 1.5
        elif content.bullet_points:
            # Bullet points
            for i, bullet in enumerate(content.bullet_points):
                if i == 0:
                    para = text_frame.paragraphs[0]
                else:
                    para = text_frame.add_paragraph()
                
                para.text = f"• {bullet}"
                para.font.size = Pt(20)
                para.font.color.rgb = RGBColor(50, 50, 50)
                para.space_after = Pt(12)
                para.level = 0
    
    def _add_single_image_layout(
        self,
        slide,
        content: SlideContent,
        image_path: str,
        accent_color: RGBColor,
    ):
        """Layout for slides focused on a single image."""
        # Small text area at bottom if there are bullet points
        if content.bullet_points:
            # Image takes most of the space
            self._add_image(
                slide, image_path,
                left=Inches(1.5),
                top=Inches(1.4),
                max_width=10.333,
                max_height=4.5,
            )
            
            # Compact bullet points at bottom
            text_box = slide.shapes.add_textbox(
                self.MARGIN, Inches(6.0),
                self.CONTENT_WIDTH, Inches(1.2)
            )
            text_frame = text_box.text_frame
            text_frame.word_wrap = True
            
            # Join bullets as a single line
            para = text_frame.paragraphs[0]
            para.text = " | ".join(content.bullet_points[:3])
            para.font.size = Pt(14)
            para.font.color.rgb = RGBColor(100, 100, 100)
            para.alignment = PP_ALIGN.CENTER
        else:
            # Full size image
            self._add_image(
                slide, image_path,
                left=Inches(1.0),
                top=Inches(1.4),
                max_width=11.333,
                max_height=5.8,
            )
    
    def _add_multi_image_layout(
        self,
        slide,
        content: SlideContent,
        images: List[str],
        accent_color: RGBColor,
    ):
        """Layout for slides with multiple images."""
        num_images = len(images)
        
        # Text on left (narrow)
        if content.bullet_points:
            self._add_text_content(
                slide, content,
                left=self.MARGIN,
                top=self.CONTENT_TOP,
                width=Inches(4.0),
                height=self.CONTENT_HEIGHT,
                accent_color=accent_color,
            )
            img_start_left = Inches(5.0)
            img_area_width = 7.833  # Remaining width
        else:
            img_start_left = self.MARGIN
            img_area_width = 12.333
        
        # Calculate image grid
        if num_images == 2:
            # Side by side
            img_width = (img_area_width - 0.5) / 2
            positions = [
                (img_start_left, self.CONTENT_TOP, img_width, 5.0),
                (img_start_left + Inches(img_width + 0.5), self.CONTENT_TOP, img_width, 5.0),
            ]
        elif num_images == 3:
            # 2 on top, 1 on bottom (or 1 top, 2 bottom)
            img_width = (img_area_width - 0.5) / 2
            positions = [
                (img_start_left, self.CONTENT_TOP, img_width, 2.4),
                (img_start_left + Inches(img_width + 0.5), self.CONTENT_TOP, img_width, 2.4),
                (img_start_left + Inches(img_width / 2), Inches(4.2), img_width, 2.8),
            ]
        else:
            # Grid layout for 4+
            cols = 2 if num_images <= 4 else 3
            rows = (num_images + cols - 1) // cols
            img_width = (img_area_width - 0.3 * (cols - 1)) / cols
            img_height = (5.0 - 0.3 * (rows - 1)) / rows
            
            positions = []
            for i in range(num_images):
                row = i // cols
                col = i % cols
                left = img_start_left + Inches(col * (img_width + 0.3))
                top = self.CONTENT_TOP + Inches(row * (img_height + 0.3))
                positions.append((left, top, img_width, img_height))
        
        # Add images
        for i, image_path in enumerate(images[:len(positions)]):
            left, top, max_w, max_h = positions[i]
            self._add_image(slide, image_path, left, top, max_w, max_h)
    
    def _add_image(
        self,
        slide,
        image_path: str,
        left: float,
        top: float,
        max_width: float,
        max_height: float,
    ):
        """Add an image to slide, scaled to fit within bounds."""
        try:
            # Get actual image dimensions
            img_w, img_h = self._get_image_dimensions(image_path)
            
            # Calculate fitted size
            fit_w, fit_h = self._calculate_image_fit(
                img_w, img_h, max_width, max_height
            )
            
            # Center within the available space
            center_left = left + Inches((max_width - fit_w) / 2)
            center_top = top + Inches((max_height - fit_h) / 2)
            
            # Add image
            slide.shapes.add_picture(
                image_path,
                center_left,
                center_top,
                width=Inches(fit_w),
                height=Inches(fit_h)
            )
            
            logger.debug(f"Added image: {Path(image_path).name} ({fit_w:.1f}x{fit_h:.1f} in)")
            
        except Exception as e:
            logger.error(f"Failed to add image {image_path}: {e}")
    
    def _add_summary_slide(
        self,
        prs: Presentation,
        content: SlideContent,
        primary_color: RGBColor,
        accent_color: RGBColor,
    ):
        """Add summary/conclusion slide."""
        slide_layout = prs.slide_layouts[6]  # Blank
        slide = prs.slides.add_slide(slide_layout)
        
        # Full background color
        background = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(0), Inches(0),
            self.SLIDE_WIDTH, self.SLIDE_HEIGHT
        )
        background.fill.solid()
        background.fill.fore_color.rgb = primary_color
        background.line.fill.background()
        
        # Send to back
        spTree = slide.shapes._spTree
        sp = background._element
        spTree.remove(sp)
        spTree.insert(2, sp)
        
        # Title
        title_box = slide.shapes.add_textbox(
            self.MARGIN, Inches(1.0),
            self.CONTENT_WIDTH, Inches(1.2)
        )
        title_frame = title_box.text_frame
        title_para = title_frame.paragraphs[0]
        title_para.text = content.title
        title_para.font.size = Pt(40)
        title_para.font.bold = True
        title_para.font.color.rgb = RGBColor(255, 255, 255)
        title_para.alignment = PP_ALIGN.CENTER
        
        # Key takeaways
        if content.bullet_points:
            text_box = slide.shapes.add_textbox(
                Inches(1.5), Inches(2.8),
                Inches(10.333), Inches(4.0)
            )
            text_frame = text_box.text_frame
            text_frame.word_wrap = True
            
            for i, bullet in enumerate(content.bullet_points):
                if i == 0:
                    para = text_frame.paragraphs[0]
                else:
                    para = text_frame.add_paragraph()
                
                para.text = f"✓ {bullet}"
                para.font.size = Pt(24)
                para.font.color.rgb = RGBColor(255, 255, 255)
                para.space_after = Pt(18)
                para.alignment = PP_ALIGN.LEFT
        
        # Speaker notes
        if content.speaker_notes:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = content.speaker_notes
    
    def build(
        self,
        outline: PresentationOutline,
        plot_assignments: Dict[int, List[str]],
        plot_paths: Dict[str, str],
        output_path: str,
    ) -> GeneratedPresentation:
        """
        Build the PowerPoint file.
        
        Args:
            outline: Presentation structure
            plot_assignments: Mapping of slide_number -> list of plot filenames
            plot_paths: Mapping of plot filename -> full path
            output_path: Where to save the PPTX
        """
        self._emit("Building PowerPoint file...")
        
        try:
            # Create presentation
            prs = Presentation()
            prs.slide_width = self.SLIDE_WIDTH
            prs.slide_height = self.SLIDE_HEIGHT
            
            # Parse colors
            primary = self._hex_to_rgb(outline.color_primary)
            accent = self._hex_to_rgb(outline.color_accent)
            
            plots_used = []
            
            for slide_content in outline.slides:
                slide_num = slide_content.slide_number
                layout = slide_content.layout
                
                # Get assigned images for this slide
                assigned_plots = plot_assignments.get(slide_num, [])
                image_paths = []
                for plot_name in assigned_plots:
                    if plot_name in plot_paths:
                        image_paths.append(plot_paths[plot_name])
                        plots_used.append(plot_name)
                
                self._emit(f"Slide {slide_num}: {slide_content.title} ({len(image_paths)} images)")
                
                if layout == "title":
                    self._add_title_slide(prs, outline, primary)
                elif layout == "summary":
                    self._add_summary_slide(prs, slide_content, primary, accent)
                else:
                    self._add_content_slide(
                        prs, slide_content, image_paths, primary, accent
                    )
            
            # Save
            prs.save(output_path)
            self._emit(f"Saved: {output_path}")
            
            return GeneratedPresentation(
                output_path=output_path,
                slide_count=len(outline.slides),
                plots_used=plots_used,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return GeneratedPresentation(
                output_path=output_path,
                slide_count=0,
                plots_used=[],
                success=False,
                error=str(e),
            )