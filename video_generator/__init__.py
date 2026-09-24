"""
Video Generator Module

Composes caption videos by overlaying styled text on background videos.
Text is displayed at natural reading pace using the style presets in style_presets.py.
"""

from .chunker import TextChunker
from .timing import TimingEngine
from .renderer import TextRenderer
from .composer import VideoComposer
from .style_presets import (
    StylePreset,
    STYLE_PRESETS,
    get_random_style,
    get_random_style_different_from,
    get_style_by_name,
    list_style_names,
)

__all__ = [
    'TextChunker',
    'TimingEngine',
    'TextRenderer',
    'VideoComposer',
    'StylePreset',
    'STYLE_PRESETS',
    'get_random_style',
    'get_random_style_different_from',
    'get_style_by_name',
    'list_style_names',
]
