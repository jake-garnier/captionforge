"""
Text Renderer - Renders styled text for video overlay.

Creates styled overlay text with:
- A bold accent colour (see style_presets.py)
- Drop shadow for readability
- Soft glow effect
- Clean, readable font
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Default colour palette (presets override these)
COLORS = {
    'hot_pink': (255, 105, 180),       # #FF69B4 - Primary text color
    'soft_pink': (255, 182, 193),      # #FFB6C1 - Glow color
    'deep_pink': (255, 20, 147),       # #FF1493 - Accent
    'white': (255, 255, 255),          # White for outline
    'shadow': (0, 0, 0),               # Black shadow
}


class TextRenderer:
    """Renders styled text frames for video overlay."""

    def __init__(
        self,
        font_path: Optional[str] = None,
        font_size: int = 56,
        text_color: Tuple[int, int, int] = COLORS['hot_pink'],
        shadow_color: Tuple[int, int, int] = COLORS['shadow'],
        shadow_offset: Tuple[int, int] = (3, 3),
        outline_color: Tuple[int, int, int] = COLORS['white'],
        outline_width: int = 2,
        glow_color: Tuple[int, int, int] = COLORS['soft_pink'],
        glow_radius: int = 5,
        padding_bottom: int = 150,
        max_width_ratio: float = 0.85,
    ):
        """
        Args:
            font_path: Path to TTF font file (uses system font if None)
            font_size: Base font size in pixels
            text_color: RGB tuple for main text color
            shadow_color: RGB tuple for drop shadow
            shadow_offset: (x, y) offset for shadow
            outline_color: RGB tuple for text outline
            outline_width: Width of outline in pixels
            glow_color: RGB tuple for glow effect
            glow_radius: Blur radius for glow
            padding_bottom: Pixels from bottom of frame
            max_width_ratio: Max text width as ratio of frame width
        """
        self.font_path = font_path
        self.font_size = font_size
        self.text_color = text_color
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.glow_color = glow_color
        self.glow_radius = glow_radius
        self.padding_bottom = padding_bottom
        self.max_width_ratio = max_width_ratio

        # Font cache
        self._font_cache = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get or create font at specified size."""
        if size in self._font_cache:
            return self._font_cache[size]

        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, size)
            else:
                # Try common system fonts - prefer clean, readable fonts
                font_names = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
                    '/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf',
                    'arial.ttf',
                    'Arial Bold.ttf',
                ]
                font = None
                for font_name in font_names:
                    try:
                        font = ImageFont.truetype(font_name, size)
                        break
                    except (IOError, OSError):
                        continue

                if font is None:
                    # Fallback to default
                    font = ImageFont.load_default()
                    logger.warning("Using default font - text may not look optimal")

        except Exception as e:
            logger.error(f"Error loading font: {e}")
            font = ImageFont.load_default()

        self._font_cache[size] = font
        return font

    def render(self, text: str, frame_size: Tuple[int, int]) -> np.ndarray:
        """
        Render styled text as RGBA numpy array.

        Args:
            text: Text to render
            frame_size: (width, height) of the video frame

        Returns:
            RGBA numpy array with transparent background
        """
        width, height = frame_size

        # Calculate max text width
        max_text_width = int(width * self.max_width_ratio)

        # Scale font size based on video dimensions (smaller for narrow videos)
        base_font_size = self._calculate_base_font_size(width, height)

        # Wrap text to fit within max width
        font_size, wrapped_lines = self._wrap_text_to_fit(text, max_text_width, base_font_size)
        font = self._get_font(font_size)
        logger.debug(f"Rendering {len(wrapped_lines)} lines at font size {font_size} for {width}x{height}")

        # Create transparent image
        img = Image.new('RGBA', frame_size, (0, 0, 0, 0))

        # Calculate total text block height
        line_height = font_size + 8  # Add spacing between lines
        total_height = line_height * len(wrapped_lines)

        # Start y position (bottom aligned with padding)
        # But ensure text doesn't go off the top of the screen
        start_y = height - total_height - self.padding_bottom
        if start_y < 20:  # Minimum 20px from top
            start_y = 20

        # Render each line
        for i, line in enumerate(wrapped_lines):
            # Get line size for centering
            line_bbox = self._get_text_bbox(line, font)
            line_width = line_bbox[2] - line_bbox[0]

            # Center horizontally
            x = (width - line_width) // 2
            y = start_y + (i * line_height)

            # Render layers for this line
            self._draw_glow(img, line, (x, y), font)
            self._draw_shadow(img, line, (x, y), font)
            self._draw_outline(img, line, (x, y), font)
            self._draw_text(img, line, (x, y), font)

        return np.array(img)

    def _calculate_base_font_size(self, width: int, height: int) -> int:
        """Calculate base font size based on video dimensions."""
        # Use the smaller dimension to scale font
        min_dim = min(width, height)

        # Scale: 1080p width -> 48px, scale proportionally
        # For narrow portrait videos, this gives smaller fonts
        scale_factor = min_dim / 1080
        scaled_size = int(self.font_size * scale_factor)

        # Clamp to reasonable range
        return max(24, min(scaled_size, self.font_size))

    def _wrap_text_to_fit(self, text: str, max_width: int, base_font_size: int, max_lines: int = 4) -> Tuple[int, list]:
        """
        Wrap text to fit within max_width, adjusting font size if needed.

        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            base_font_size: Starting font size
            max_lines: Maximum visual lines per slide (default 4)

        Returns:
            Tuple of (font_size, list_of_lines)
        """
        words = text.split()

        # Try different font sizes, starting from base
        for font_size in range(base_font_size, 16, -2):
            font = self._get_font(font_size)
            lines = self._wrap_words(words, font, max_width)

            # Check if all lines fit width AND we don't have too many lines
            all_fit = True
            if len(lines) > max_lines:
                all_fit = False
            else:
                for line in lines:
                    bbox = self._get_text_bbox(line, font)
                    if bbox[2] - bbox[0] > max_width:
                        all_fit = False
                        break

            if all_fit:
                return font_size, lines

        # Minimum size fallback
        font = self._get_font(16)
        lines = self._wrap_words(words, font, max_width)
        return 16, lines

    def _wrap_words(self, words: list, font: ImageFont.FreeTypeFont, max_width: int) -> list:
        """Wrap words into lines that fit within max_width."""
        lines = []
        current_line = []

        for word in words:
            # Test adding this word to current line
            test_line = ' '.join(current_line + [word])
            bbox = self._get_text_bbox(test_line, font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line.append(word)
            else:
                # Start new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        # Add last line
        if current_line:
            lines.append(' '.join(current_line))

        return lines if lines else [' '.join(words)]

    def _find_fitting_font_size(self, text: str, max_width: int) -> int:
        """Find the largest font size that fits within max_width."""
        # Start with configured size and reduce if needed
        for size in range(self.font_size, 20, -2):
            font = self._get_font(size)
            bbox = self._get_text_bbox(text, font)
            text_width = bbox[2] - bbox[0]

            if text_width <= max_width:
                return size

        return 20  # Minimum size

    def _get_text_bbox(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
        """Get bounding box for text."""
        # Create temporary image to measure text
        tmp = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(tmp)
        return draw.textbbox((0, 0), text, font=font)

    def _draw_glow(self, img: Image.Image, text: str, pos: Tuple[int, int], font: ImageFont.FreeTypeFont):
        """Draw soft glow behind text."""
        if self.glow_radius <= 0:
            return

        # Create glow layer
        glow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow)

        # Draw text for glow
        glow_color = (*self.glow_color, 180)  # Semi-transparent
        draw.text(pos, text, font=font, fill=glow_color)

        # Blur for glow effect
        glow = glow.filter(ImageFilter.GaussianBlur(radius=self.glow_radius))

        # Composite onto main image (use alpha mask to preserve existing content)
        img.paste(glow, (0, 0), glow)

    def _draw_shadow(self, img: Image.Image, text: str, pos: Tuple[int, int], font: ImageFont.FreeTypeFont):
        """Draw drop shadow."""
        draw = ImageDraw.Draw(img)

        shadow_pos = (
            pos[0] + self.shadow_offset[0],
            pos[1] + self.shadow_offset[1]
        )
        shadow_color = (*self.shadow_color, 180)  # Semi-transparent
        draw.text(shadow_pos, text, font=font, fill=shadow_color)

    def _draw_outline(self, img: Image.Image, text: str, pos: Tuple[int, int], font: ImageFont.FreeTypeFont):
        """Draw text outline for readability."""
        if self.outline_width <= 0:
            return

        draw = ImageDraw.Draw(img)
        outline_color = (*self.outline_color, 255)

        # Draw text at offset positions to create outline
        for dx in range(-self.outline_width, self.outline_width + 1):
            for dy in range(-self.outline_width, self.outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((pos[0] + dx, pos[1] + dy), text, font=font, fill=outline_color)

    def _draw_text(self, img: Image.Image, text: str, pos: Tuple[int, int], font: ImageFont.FreeTypeFont):
        """Draw main text."""
        draw = ImageDraw.Draw(img)
        text_color = (*self.text_color, 255)
        draw.text(pos, text, font=font, fill=text_color)

    def set_font(self, font_path: str):
        """Set a custom font file."""
        self.font_path = font_path
        self._font_cache.clear()

    def set_colors(
        self,
        text_color: Optional[Tuple[int, int, int]] = None,
        glow_color: Optional[Tuple[int, int, int]] = None,
        shadow_color: Optional[Tuple[int, int, int]] = None,
        outline_color: Optional[Tuple[int, int, int]] = None
    ):
        """Update color scheme."""
        if text_color:
            self.text_color = text_color
        if glow_color:
            self.glow_color = glow_color
        if shadow_color:
            self.shadow_color = shadow_color
        if outline_color:
            self.outline_color = outline_color
