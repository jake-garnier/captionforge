"""
Style Presets - bold, high-contrast text styles for video composition.

A small palette of saturated accent colours with a white outline and drop
shadow so overlays stay readable on any background footage.
"""

import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Color definitions - accent palette.
#
# Text-color rule: any color used as text_color must have WCAG relative
# luminance below ~0.35 so it stays readable against light BG content
# even with the white outline. The light-pink shades (soft_pink ~0.59,
# baby_pink ~0.62, lavender ~0.80) wash out on bright skies, snow and walls
# and have been removed from text-color use. They remain available as
# glow_color where the white outline still does the heavy lifting.
COLORS = {
    # Pinks usable as text_color (luminance < 0.35)
    'hot_pink': (255, 105, 180),    # 0.35
    'deep_pink': (255, 20, 147),    # 0.24
    'magenta': (255, 0, 255),       # 0.28
    'rose': (255, 102, 178),        # 0.34

    # Glow-only / accent (too light for text_color)
    'soft_pink': (255, 182, 193),   # 0.59 — glow only
    'white': (255, 255, 255),

    # Shadows/outlines
    'black': (0, 0, 0),
    'dark_pink': (139, 69, 98),
}

# Font paths - clean, readable fonts only
FONTS = {
    'dejavu_sans': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    'liberation_sans': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
}


@dataclass
class StylePreset:
    """A complete text style configuration."""
    name: str
    font_path: str
    font_size: int
    text_color: Tuple[int, int, int]
    outline_color: Tuple[int, int, int]
    outline_width: int
    shadow_color: Tuple[int, int, int]
    shadow_offset: Tuple[int, int]
    glow_color: Tuple[int, int, int]
    glow_radius: int
    padding_bottom: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'font_size': self.font_size,
            'text_color': self.text_color,
            'outline_color': self.outline_color,
            'outline_width': self.outline_width,
            'shadow_offset': list(self.shadow_offset),
            'glow_radius': self.glow_radius,
            'padding_bottom': self.padding_bottom,
        }


# Style presets - saturated accent colours with high readability
STYLE_PRESETS = [
    # Hot pink with white outline - classic, cute, very readable
    StylePreset(
        name='cute_pink',
        font_path=FONTS['dejavu_sans'],
        font_size=58,
        text_color=COLORS['hot_pink'],
        outline_color=COLORS['white'],
        outline_width=3,
        shadow_color=COLORS['black'],
        shadow_offset=(3, 3),
        glow_color=COLORS['soft_pink'],
        glow_radius=6,
        padding_bottom=120,
    ),

    # Deep pink with soft glow - bold and punchy
    StylePreset(
        name='bold_pink',
        font_path=FONTS['dejavu_sans'],
        font_size=58,
        text_color=COLORS['deep_pink'],
        outline_color=COLORS['white'],
        outline_width=3,
        shadow_color=COLORS['black'],
        shadow_offset=(3, 3),
        glow_color=COLORS['soft_pink'],
        glow_radius=5,
        padding_bottom=120,
    ),

    # NOTE: 'dreamy_pink' (text_color=soft_pink) was removed because the
    # light pink (luminance 0.59) was washing out on bright BG content
    # despite the white outline. If we ever want a "soft" style back,
    # switch to a darker pink and keep the dreamy font/glow combo.

    # Rose pink - warm and pretty
    StylePreset(
        name='rose_glow',
        font_path=FONTS['dejavu_sans'],
        font_size=58,
        text_color=COLORS['rose'],
        outline_color=COLORS['white'],
        outline_width=3,
        shadow_color=COLORS['black'],
        shadow_offset=(3, 3),
        glow_color=COLORS['soft_pink'],
        glow_radius=6,
        padding_bottom=120,
    ),

    # Magenta with pink glow - vibrant and loud
    StylePreset(
        name='magenta_glow',
        font_path=FONTS['dejavu_sans'],
        font_size=58,
        text_color=COLORS['magenta'],
        outline_color=COLORS['white'],
        outline_width=3,
        shadow_color=COLORS['dark_pink'],
        shadow_offset=(3, 3),
        glow_color=COLORS['soft_pink'],
        glow_radius=6,
        padding_bottom=120,
    ),
]


def get_random_style() -> StylePreset:
    """Get a random style preset."""
    return random.choice(STYLE_PRESETS)


def get_random_style_different_from(current_style_name: str) -> StylePreset:
    """Get a random style that's different from the current one."""
    available = [s for s in STYLE_PRESETS if s.name != current_style_name]
    if not available:
        available = STYLE_PRESETS
    return random.choice(available)


def get_style_by_name(name: str) -> StylePreset:
    """Get a specific style by name."""
    for style in STYLE_PRESETS:
        if style.name == name:
            return style
    return STYLE_PRESETS[0]  # Default to first if not found


def list_style_names() -> list:
    """List all available style names."""
    return [s.name for s in STYLE_PRESETS]
