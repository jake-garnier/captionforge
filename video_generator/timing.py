"""
Timing Engine - Calculates display duration for text chunks.

Uses reading pace calculations to determine how long each chunk
should be displayed on screen for comfortable reading.
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TimedChunk:
    """A text chunk with timing information."""
    text: str
    start_time: float
    end_time: float
    duration: float


class TimingEngine:
    """Calculates timing for text chunk display."""

    def __init__(
        self,
        words_per_minute: int = 150,
        min_duration: float = 2.0,
        punctuation_pause: float = 0.3,
        paragraph_pause: float = 0.5,
        transition_pause: float = 0.3
    ):
        """
        Args:
            words_per_minute: Target reading speed (default 150 WPM - comfortable for captions)
            min_duration: Minimum display time for any chunk in seconds (2s for short lines)
            punctuation_pause: Extra pause after sentence-ending punctuation
            paragraph_pause: Extra pause between major sections
            transition_pause: Brief pause between slides for visual transition
        """
        self.words_per_minute = words_per_minute
        self.words_per_second = words_per_minute / 60.0
        self.min_duration = min_duration
        self.punctuation_pause = punctuation_pause
        self.paragraph_pause = paragraph_pause
        self.transition_pause = transition_pause

    def calculate_timings(self, chunks: List[str], start_offset: float = 0.0) -> List[TimedChunk]:
        """
        Calculate start/end times for each text chunk.

        Args:
            chunks: List of text chunks to time
            start_offset: Time offset to start from (default 0)

        Returns:
            List of TimedChunk objects with timing info
        """
        timed_chunks = []
        current_time = start_offset

        for i, chunk in enumerate(chunks):
            duration = self._calculate_duration(chunk)

            timed_chunk = TimedChunk(
                text=chunk,
                start_time=current_time,
                end_time=current_time + duration,
                duration=duration
            )
            timed_chunks.append(timed_chunk)

            # Add transition pause between slides (not after the last one)
            current_time += duration
            if i < len(chunks) - 1:
                current_time += self.transition_pause

        return timed_chunks

    def _calculate_duration(self, chunk: str) -> float:
        """
        Calculate display duration for a single chunk.

        Args:
            chunk: The text chunk

        Returns:
            Duration in seconds
        """
        words = chunk.split()
        word_count = len(words)

        # Base duration from word count
        base_duration = word_count / self.words_per_second

        # Add pause for sentence-ending punctuation
        if chunk.rstrip()[-1:] in '.!?':
            base_duration += self.punctuation_pause

        # Add slight pause for other major punctuation
        elif chunk.rstrip()[-1:] in ';:':
            base_duration += self.punctuation_pause * 0.5

        # Ensure minimum duration
        return max(base_duration, self.min_duration)

    def get_total_duration(self, chunks: List[str]) -> float:
        """
        Get total duration needed for all chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Total duration in seconds
        """
        timed = self.calculate_timings(chunks)
        if not timed:
            return 0.0
        return timed[-1].end_time

    def fits_in_duration(self, chunks: List[str], max_duration: float) -> bool:
        """
        Check if chunks fit within a time limit.

        Args:
            chunks: List of text chunks
            max_duration: Maximum allowed duration in seconds

        Returns:
            True if chunks fit, False otherwise
        """
        total = self.get_total_duration(chunks)
        return total <= max_duration
