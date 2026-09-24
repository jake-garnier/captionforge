"""
Video Composer - Composes final video with text overlay.

Uses MoviePy to overlay timed text chunks on background video.
"""

import os
import tempfile
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips
)
import numpy as np

from .chunker import TextChunker
from .timing import TimingEngine, TimedChunk
from .renderer import TextRenderer

logger = logging.getLogger(__name__)


@dataclass
class CompositionResult:
    """Result of video composition."""
    success: bool
    output_path: Optional[str]
    duration: float
    error: Optional[str] = None
    caption_chunks: int = 0


class VideoComposer:
    """Composes caption videos from background video and text."""

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        timing: Optional[TimingEngine] = None,
        renderer: Optional[TextRenderer] = None,
        output_dir: str = "/data/generated_videos",
    ):
        """
        Args:
            chunker: Text chunker instance (creates default if None)
            timing: Timing engine instance (creates default if None)
            renderer: Text renderer instance (creates default if None)
            output_dir: Directory for output videos
        """
        self.chunker = chunker or TextChunker()
        self.timing = timing or TimingEngine()
        self.renderer = renderer or TextRenderer()
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def compose(
        self,
        background_path: str,
        caption: str,
        output_filename: Optional[str] = None,
    ) -> CompositionResult:
        """
        Compose a caption video.

        Args:
            background_path: Path to background video file
            caption: Caption text to overlay
            output_filename: Output filename (auto-generated if None)

        Returns:
            CompositionResult with success status and output path
        """
        try:
            # Validate inputs
            if not os.path.exists(background_path):
                return CompositionResult(
                    success=False,
                    output_path=None,
                    duration=0,
                    error=f"Background video not found: {background_path}"
                )

            if not caption or not caption.strip():
                return CompositionResult(
                    success=False,
                    output_path=None,
                    duration=0,
                    error="Caption is empty"
                )

            # Load background video
            logger.info(f"Loading background video: {background_path}")
            bg_video = VideoFileClip(background_path)
            bg_duration = bg_video.duration
            frame_size = (bg_video.w, bg_video.h)

            logger.info(f"Background video: {frame_size[0]}x{frame_size[1]}, {bg_duration:.1f}s")

            # Chunk the caption
            chunks = self.chunker.chunk(caption)
            if not chunks:
                bg_video.close()
                return CompositionResult(
                    success=False,
                    output_path=None,
                    duration=0,
                    error="No text chunks generated from caption"
                )

            # Calculate timing
            timed_chunks = self.timing.calculate_timings(chunks)
            total_caption_duration = timed_chunks[-1].end_time

            # Check if caption fits in video
            if total_caption_duration > bg_duration:
                bg_video.close()
                return CompositionResult(
                    success=False,
                    output_path=None,
                    duration=total_caption_duration,
                    error=f"Caption too long ({total_caption_duration:.1f}s) for video ({bg_duration:.1f}s)",
                    caption_chunks=len(chunks)
                )

            # Create text overlay clips
            logger.info(f"Creating {len(timed_chunks)} text overlays")
            text_clips = []

            for i, timed_chunk in enumerate(timed_chunks):
                # Render text frame
                logger.info(f"Rendering chunk {i}: '{timed_chunk.text[:50]}...' for frame {frame_size}")
                text_frame = self.renderer.render(timed_chunk.text, frame_size)

                # Create clip from frame
                text_clip = (
                    ImageClip(text_frame)
                    .set_start(timed_chunk.start_time)
                    .set_duration(timed_chunk.duration)
                    .set_position('center')
                )
                text_clips.append(text_clip)

            # Trim background to caption duration (no looping, just use what we need)
            bg_trimmed = bg_video.subclip(0, total_caption_duration)

            # Composite all clips
            logger.info("Compositing video...")
            final = CompositeVideoClip([bg_trimmed] + text_clips)

            # Generate output path
            if output_filename is None:
                output_filename = f"caption_video_{os.urandom(8).hex()}.mp4"

            output_path = os.path.join(self.output_dir, output_filename)

            # Write output with web-compatible settings for Patreon/browser playback
            logger.info(f"Writing output: {output_path}")
            final.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=bg_video.fps,
                preset='medium',
                threads=4,
                logger=None,  # Suppress moviepy progress bar
                # Web-compatible settings for inline playback on Patreon/browsers
                ffmpeg_params=[
                    '-pix_fmt', 'yuv420p',      # Required for browser playback
                    '-profile:v', 'main',       # H.264 Main profile (widely compatible)
                    '-level', '4.0',            # H.264 level 4.0 (supports 1080p)
                    '-movflags', '+faststart',  # Move moov atom to start for streaming
                    '-crf', '23',               # Good quality/size balance
                ]
            )

            # Cleanup
            final.close()
            bg_video.close()
            for clip in text_clips:
                clip.close()

            logger.info(f"Video created successfully: {output_path}")

            return CompositionResult(
                success=True,
                output_path=output_path,
                duration=total_caption_duration,
                caption_chunks=len(chunks)
            )

        except Exception as e:
            logger.error(f"Error composing video: {e}")
            return CompositionResult(
                success=False,
                output_path=None,
                duration=0,
                error=str(e)
            )

    def preview_timing(self, caption: str) -> dict:
        """
        Preview timing calculations without rendering.

        Args:
            caption: Caption text

        Returns:
            Dictionary with timing information
        """
        chunks = self.chunker.chunk(caption)
        timed_chunks = self.timing.calculate_timings(chunks)

        return {
            'chunks': [
                {
                    'text': tc.text,
                    'start': tc.start_time,
                    'end': tc.end_time,
                    'duration': tc.duration
                }
                for tc in timed_chunks
            ],
            'total_duration': timed_chunks[-1].end_time if timed_chunks else 0,
            'chunk_count': len(chunks)
        }

    def check_compatibility(self, background_path: str, caption: str) -> dict:
        """
        Check if a caption is compatible with a background video.

        Args:
            background_path: Path to background video
            caption: Caption text

        Returns:
            Dictionary with compatibility info
        """
        try:
            # Get video duration
            bg_video = VideoFileClip(background_path)
            bg_duration = bg_video.duration
            frame_size = (bg_video.w, bg_video.h)
            bg_video.close()

            # Calculate caption duration
            chunks = self.chunker.chunk(caption)
            timed_chunks = self.timing.calculate_timings(chunks)
            caption_duration = timed_chunks[-1].end_time if timed_chunks else 0

            compatible = caption_duration <= bg_duration

            return {
                'compatible': compatible,
                'video_duration': bg_duration,
                'caption_duration': caption_duration,
                'chunk_count': len(chunks),
                'frame_size': frame_size,
                'margin': bg_duration - caption_duration if compatible else 0
            }

        except Exception as e:
            return {
                'compatible': False,
                'error': str(e)
            }

    def compose_multi(
        self,
        background_paths: list,
        caption: str,
        output_filename: Optional[str] = None,
    ) -> CompositionResult:
        """
        Compose a caption video using multiple concatenated background videos.

        This allows longer captions to be composed by seamlessly joining
        multiple shorter background videos.

        Args:
            background_paths: List of paths to background video files (in order)
            caption: Caption text to overlay
            output_filename: Output filename (auto-generated if None)

        Returns:
            CompositionResult with success status and output path
        """
        if not background_paths:
            return CompositionResult(
                success=False,
                output_path=None,
                duration=0,
                error="No background videos provided"
            )

        # If only one video, use the standard compose method
        if len(background_paths) == 1:
            return self.compose(background_paths[0], caption, output_filename)

        try:
            # Validate all inputs exist
            for path in background_paths:
                if not os.path.exists(path):
                    return CompositionResult(
                        success=False,
                        output_path=None,
                        duration=0,
                        error=f"Background video not found: {path}"
                    )

            if not caption or not caption.strip():
                return CompositionResult(
                    success=False,
                    output_path=None,
                    duration=0,
                    error="Caption is empty"
                )

            # Load all background videos
            logger.info(f"Loading {len(background_paths)} background videos for concatenation")
            bg_clips = []
            total_bg_duration = 0
            frame_size = None

            for i, path in enumerate(background_paths):
                clip = VideoFileClip(path)
                bg_clips.append(clip)
                total_bg_duration += clip.duration
                logger.info(f"  Video {i+1}: {clip.w}x{clip.h}, {clip.duration:.1f}s")

                # Use first video's frame size as reference
                if frame_size is None:
                    frame_size = (clip.w, clip.h)

            logger.info(f"Total concatenated duration: {total_bg_duration:.1f}s")

            # Chunk the caption
            chunks = self.chunker.chunk(caption)
            if not chunks:
                for clip in bg_clips:
                    clip.close()
                return CompositionResult(
                    success=False,
                    output_path=None,
                    duration=0,
                    error="No text chunks generated from caption"
                )

            # Calculate timing
            timed_chunks = self.timing.calculate_timings(chunks)
            total_caption_duration = timed_chunks[-1].end_time

            # Check if caption fits in concatenated video
            if total_caption_duration > total_bg_duration:
                for clip in bg_clips:
                    clip.close()
                return CompositionResult(
                    success=False,
                    output_path=None,
                    duration=total_caption_duration,
                    error=f"Caption too long ({total_caption_duration:.1f}s) for videos ({total_bg_duration:.1f}s)",
                    caption_chunks=len(chunks)
                )

            # Concatenate background videos
            logger.info("Concatenating background videos...")
            bg_concatenated = concatenate_videoclips(bg_clips, method="compose")

            # Create text overlay clips
            logger.info(f"Creating {len(timed_chunks)} text overlays")
            text_clips = []

            for i, timed_chunk in enumerate(timed_chunks):
                # Render text frame
                text_frame = self.renderer.render(timed_chunk.text, frame_size)

                # Create clip from frame
                text_clip = (
                    ImageClip(text_frame)
                    .set_start(timed_chunk.start_time)
                    .set_duration(timed_chunk.duration)
                    .set_position('center')
                )
                text_clips.append(text_clip)

            # Trim concatenated background to caption duration
            bg_trimmed = bg_concatenated.subclip(0, total_caption_duration)

            # Composite all clips
            logger.info("Compositing video...")
            final = CompositeVideoClip([bg_trimmed] + text_clips)

            # Generate output path
            if output_filename is None:
                output_filename = f"caption_video_{os.urandom(8).hex()}.mp4"

            output_path = os.path.join(self.output_dir, output_filename)

            # Write output with web-compatible settings
            logger.info(f"Writing output: {output_path}")
            final.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=bg_clips[0].fps,
                preset='medium',
                threads=4,
                logger=None,
                ffmpeg_params=[
                    '-pix_fmt', 'yuv420p',
                    '-profile:v', 'main',
                    '-level', '4.0',
                    '-movflags', '+faststart',
                    '-crf', '23',
                ]
            )

            # Cleanup
            final.close()
            bg_concatenated.close()
            for clip in bg_clips:
                clip.close()
            for clip in text_clips:
                clip.close()

            logger.info(f"Multi-background video created successfully: {output_path}")

            return CompositionResult(
                success=True,
                output_path=output_path,
                duration=total_caption_duration,
                caption_chunks=len(chunks)
            )

        except Exception as e:
            logger.error(f"Error composing multi-background video: {e}")
            return CompositionResult(
                success=False,
                output_path=None,
                duration=0,
                error=str(e)
            )


def append_outro_to_video(
    composed_video_path: str,
    bg_path: Optional[str],
    bg_offset_seconds: float,
    outro_text: str,
    output_path: str,
    renderer: Optional[TextRenderer] = None,
    timing: Optional[TimingEngine] = None,
    chunker: Optional[TextChunker] = None,
) -> CompositionResult:
    """
    Append a CTA/watermark outro segment to an already-composed caption video.

    The outro is concatenated after the existing video. Its background is the
    leftover frames of the original BG starting at bg_offset_seconds (i.e. the
    footage that was trimmed off when the caption ended). If the leftover is
    shorter than the outro text needs, the remainder is padded with a black
    clip the same size as the composed video. If bg_path is None or has no
    leftover, the entire outro is rendered on black.

    The outro text is chunked, timed, and rendered with the same TextRenderer
    used for the main caption so styling matches.

    Args:
        composed_video_path: Path to the already-composed mp4 (caption baked in).
        bg_path: Original background video path. None => pure-black outro.
        bg_offset_seconds: Where the caption ended in the original BG. Frames
            after this point are the outro background source.
        outro_text: CTA/watermark text to render.
        output_path: Where to write the new mp4. Will be overwritten.
        renderer: TextRenderer (creates default if None — must match main pass).
        timing: TimingEngine (creates default if None).
        chunker: TextChunker (creates default if None).

    Returns:
        CompositionResult with success, output_path, total duration.
    """
    chunker = chunker or TextChunker()
    timing = timing or TimingEngine()
    renderer = renderer or TextRenderer()

    if not os.path.exists(composed_video_path):
        return CompositionResult(False, None, 0, error=f"Composed video missing: {composed_video_path}")
    if not outro_text or not outro_text.strip():
        return CompositionResult(False, None, 0, error="Outro text is empty")

    composed_clip: Optional[VideoFileClip] = None
    bg_clip: Optional[VideoFileClip] = None
    outro_bg: Optional[CompositeVideoClip] = None
    final: Optional[CompositeVideoClip] = None
    text_clip: Optional[ImageClip] = None
    bg_tail: Optional[VideoFileClip] = None
    black_pad: Optional[ColorClip] = None

    try:
        composed_clip = VideoFileClip(composed_video_path)
        frame_size = (composed_clip.w, composed_clip.h)
        fps = composed_clip.fps

        # How long the outro needs to be on screen.
        # The TimingEngine reading-pace math gives ~3-4s for a short 9-word
        # CTA, but a CTA isn't a story chunk — readers need a brief moment
        # to register the value prop (Patreon, free captions, name). Floor
        # at 4s with a small 0.5s linger so the last word doesn't snap away.
        # User found 7s too long — better to have viewers re-watch than
        # bounce off a slow ending.
        OUTRO_MIN_DURATION = 4.0
        OUTRO_LINGER_PAD = 0.5
        outro_chunks = chunker.chunk(outro_text)
        outro_timed = timing.calculate_timings(outro_chunks)
        outro_read_time = outro_timed[-1].end_time if outro_timed else 0
        outro_duration = max(OUTRO_MIN_DURATION, outro_read_time + OUTRO_LINGER_PAD)
        if outro_duration <= 0:
            return CompositionResult(False, None, 0, error="Outro duration came out to 0")

        # Build the outro background: leftover BG + black pad as needed.
        leftover_clip = None
        leftover_duration = 0.0
        if bg_path and os.path.exists(bg_path):
            bg_clip = VideoFileClip(bg_path)
            if bg_clip.duration and bg_clip.duration > bg_offset_seconds:
                leftover_duration = min(bg_clip.duration - bg_offset_seconds, outro_duration)
                if leftover_duration > 0.05:  # ignore microscopic tails
                    bg_tail = bg_clip.subclip(bg_offset_seconds, bg_offset_seconds + leftover_duration)
                    # Resize tail to match composed video frame_size if necessary.
                    if (bg_tail.w, bg_tail.h) != frame_size:
                        bg_tail = bg_tail.resize(newsize=frame_size)
                    leftover_clip = bg_tail

        pad_duration = outro_duration - leftover_duration
        if pad_duration < 0:
            pad_duration = 0
        if pad_duration > 0:
            black_pad = ColorClip(size=frame_size, color=(0, 0, 0), duration=pad_duration)

        # Compose the outro background: tail then black, or just black.
        if leftover_clip and black_pad:
            outro_bg_base = concatenate_videoclips([leftover_clip, black_pad], method="compose")
        elif leftover_clip:
            outro_bg_base = leftover_clip
        else:
            # Pure-black fallback
            outro_bg_base = ColorClip(size=frame_size, color=(0, 0, 0), duration=outro_duration)
            black_pad = outro_bg_base  # tracked for cleanup

        # Render outro text onto a single image and overlay for the full outro
        # duration. We chunked above only to compute timing; the visual treatment
        # is "show the whole CTA the entire time" — short, single-line CTAs read
        # best as one persistent overlay.
        text_frame = renderer.render(outro_text, frame_size)
        text_clip = (
            ImageClip(text_frame)
            .set_duration(outro_duration)
            .set_position('center')
        )

        outro_segment = CompositeVideoClip([outro_bg_base, text_clip], size=frame_size)
        outro_segment = outro_segment.set_duration(outro_duration)

        final = concatenate_videoclips([composed_clip, outro_segment], method="compose")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(
            f"Writing outro-appended video: {output_path} "
            f"(composed={composed_clip.duration:.1f}s + outro={outro_duration:.1f}s, "
            f"leftover_bg={leftover_duration:.1f}s, black_pad={pad_duration:.1f}s)"
        )
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            preset='medium',
            threads=4,
            logger=None,
            ffmpeg_params=[
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'main',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-crf', '23',
            ],
        )

        return CompositionResult(
            success=True,
            output_path=output_path,
            duration=composed_clip.duration + outro_duration,
            caption_chunks=len(outro_chunks),
        )

    except Exception as e:
        logger.error(f"Error appending outro to video: {e}")
        return CompositionResult(False, None, 0, error=str(e))
    finally:
        for c in (final, outro_bg, text_clip, bg_tail, black_pad, bg_clip, composed_clip):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass
