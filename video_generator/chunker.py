"""
Text Chunker - Splits captions into readable display chunks.

Two modes:
1. Line-based: Each line in the caption becomes a slide (for pre-formatted captions)
2. Auto-split: Splits text at natural sentence breaks (for raw text)

Prioritizes keeping complete sentences together for better readability.
"""

import re
from typing import List


class TextChunker:
    """Splits caption text into display-friendly chunks."""

    def __init__(self, max_words: int = 14, min_words: int = 5, respect_lines: bool = True):
        """
        Args:
            max_words: Maximum words per chunk (default 14). At 14 words a chunk
                wraps to ~2 lines on the swipe-tab overlay, which is the
                intended visual density for read-while-watching captions.
                Was previously 22 — produced 4-line walls of text that
                covered the video and overwhelmed the viewer.
            min_words: Minimum words to avoid tiny chunks (default 5)
            respect_lines: If True, split by line breaks; if False, auto-split (default True)
        """
        self.max_words = max_words
        self.min_words = min_words
        self.respect_lines = respect_lines

    def chunk(self, text: str) -> List[str]:
        """
        Split text into readable chunks.

        Args:
            text: The caption text to split

        Returns:
            List of text chunks for display
        """
        if not text or not text.strip():
            return []

        # Check if text has intentional line breaks (pre-formatted)
        if self.respect_lines and self._has_line_breaks(text):
            return self._chunk_by_lines(text)

        # Fall back to auto-splitting for raw text
        return self._chunk_auto(text)

    def _has_line_breaks(self, text: str) -> bool:
        """Check if text has intentional line breaks."""
        # Look for newlines that indicate pre-formatted text
        # Double newlines (paragraph breaks) or consistent single newlines
        return '\n' in text

    def _chunk_by_lines(self, text: str) -> List[str]:
        """Split text by line breaks, then enforce max_words on each line.

        Lines from the LLM frequently contain multiple sentences smashed
        into one paragraph (e.g. "On hands and knees, motivation, let him see
        ... His hand grips your hip hard ..."). Without enforcing
        max_words here, those become 50+ word slides that wallpaper the
        video. We treat newlines as a *minimum* split signal and apply
        the auto-chunking sub-splitter to any line that's still too long.
        """
        # Split by one or more newlines
        lines = re.split(r'\n+', text)

        chunks = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Normalize internal whitespace
            line = re.sub(r'\s+', ' ', line)

            if len(line.split()) <= self.max_words:
                chunks.append(line)
                continue

            # Line is too long — fall back to sentence/clause splitting
            # (same path the auto chunker uses).
            for sentence in self._split_sentences(line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence.split()) <= self.max_words:
                    chunks.append(sentence)
                else:
                    chunks.extend(self._split_by_clauses(sentence))

        # Merge anything tiny that came out of the splitter, same as
        # _chunk_auto does — keeps single-word "Motivation." trailers from
        # being their own slide.
        chunks = self._merge_tiny_chunks(chunks)
        return chunks

    def _chunk_auto(self, text: str) -> List[str]:
        """Auto-split text at natural reading breaks."""
        # Clean the text
        text = self._clean_text(text)

        # First split by sentences
        sentences = self._split_sentences(text)

        # Then split long sentences into smaller chunks
        chunks = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            word_count = len(sentence.split())

            if word_count <= self.max_words:
                chunks.append(sentence)
            else:
                # Split long sentence by clauses
                sub_chunks = self._split_by_clauses(sentence)
                chunks.extend(sub_chunks)

        # Merge tiny chunks with neighbors
        chunks = self._merge_tiny_chunks(chunks)

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    # Words that commonly start with capitals mid-sentence (don't split before these)
    _NO_SPLIT_WORDS = {'I', "I'm", "I'll", "I've", "I'd"}

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling dialogue quotes properly.
        Keeps quoted dialogue with its attribution together.

        Also detects implicit sentence boundaries where punctuation is missing
        (e.g., "finish the set Then rest" splits at "set | Then").
        """
        sentences = []
        current = ""
        i = 0
        in_quote = False

        while i < len(text):
            char = text[i]
            current += char

            # Track quote state
            if char == '"':
                in_quote = not in_quote

            # Check for sentence end (. ! ?) followed by space
            if char in '.!?' and not in_quote:
                # Look ahead for space + capital or end of text
                if i + 1 >= len(text):
                    # End of text
                    sentences.append(current.strip())
                    current = ""
                elif i + 2 < len(text) and text[i + 1] == ' ':
                    # Check if next char after space is capital (new sentence)
                    next_char = text[i + 2]
                    if next_char.isupper() or next_char == '"':
                        sentences.append(current.strip())
                        current = ""
                        i += 1  # Skip the space

            # Detect implicit sentence boundary: lowercase + space + Uppercase
            # without any preceding punctuation (LLM omitted the period)
            elif (char == ' ' and not in_quote and
                  i >= 1 and i + 1 < len(text) and
                  text[i - 1].islower() and
                  text[i + 1].isupper()):

                # Check the next word isn't in our exception list
                next_word_end = i + 1
                while next_word_end < len(text) and (text[next_word_end].isalpha() or text[next_word_end] == "'"):
                    next_word_end += 1
                next_word = text[i + 1:next_word_end]

                if next_word not in self._NO_SPLIT_WORDS:
                    # Split here - the space is the boundary
                    # Remove trailing space from current chunk
                    current = current.rstrip()
                    if current:
                        sentences.append(current)
                    current = ""

            i += 1

        # Add remaining text
        if current.strip():
            sentences.append(current.strip())

        return sentences

    def _split_by_clauses(self, sentence: str) -> List[str]:
        """
        Split a long sentence at natural clause boundaries.
        Prefers splitting after complete thoughts, not mid-phrase.
        Only splits if sentence exceeds max_words.
        """
        word_count = len(sentence.split())

        # If sentence fits, don't split it
        if word_count <= self.max_words:
            return [sentence]

        # For very long sentences, find the best split points
        # Prefer splitting at: semicolons, then colons, then "and"/"but", then commas
        # Avoid splitting inside quotes

        # Try semicolons first (strongest clause separator)
        if '; ' in sentence:
            parts = sentence.split('; ')
            return self._recombine_parts(parts, '; ')

        # Try splitting at coordinating conjunctions (and/but/or). Only do
        # this if BOTH halves end up at least min_words wide — otherwise
        # we orphan stubs like "On hands" when the first "and" appears
        # very early in a long sentence (the prior version produced
        # "On hands" / "and knees, motivation, let him see..." seams).
        # Walk all matches and pick the first one that yields a balanced
        # split; fall through to comma split if none qualifies.
        conj_pattern = r',?\s+(and|but|or)\s+'
        for conj_match in re.finditer(conj_pattern, sentence):
            split_pos = conj_match.start()
            part1 = sentence[:split_pos].strip()
            part2 = sentence[conj_match.start():].strip()
            if (
                len(part1.split()) >= self.min_words
                and len(part2.split()) >= self.min_words
            ):
                chunks = []
                if len(part1.split()) <= self.max_words:
                    chunks.append(part1)
                else:
                    chunks.extend(self._force_split(part1))
                if len(part2.split()) <= self.max_words:
                    chunks.append(part2)
                else:
                    chunks.extend(self._force_split(part2))
                return chunks

        # Last resort: split at commas, but try to keep phrases together
        if ', ' in sentence:
            parts = sentence.split(', ')
            return self._recombine_parts(parts, ', ')

        # If no good split points, force split by words (phrase-aware)
        return self._force_split(sentence)

    def _recombine_parts(self, parts: List[str], separator: str) -> List[str]:
        """Recombine split parts to stay under max_words while keeping related phrases together."""
        chunks = []
        current_chunk = ""

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Add separator back except for first part
            if current_chunk:
                test_chunk = current_chunk + separator + part
            else:
                test_chunk = part

            word_count = len(test_chunk.split())

            if word_count <= self.max_words:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk)

                # Check if this part alone is too long
                if len(part.split()) > self.max_words:
                    chunks.extend(self._force_split(part))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # Words that typically start a new phrase/clause - prefer splitting BEFORE these
    _SPLIT_BEFORE_WORDS = {
        'and', 'but', 'or', 'so', 'yet', 'while', 'when', 'where', 'as',
        'if', 'because', 'since', 'until', 'before', 'after', 'although',
        'though', 'even', 'just', 'then', 'now', 'like', 'with',
    }

    def _force_split(self, text: str) -> List[str]:
        """
        Force split text that's too long, preferring natural phrase boundaries.

        Instead of splitting at arbitrary word counts, looks for natural break
        points near the max_words boundary: commas, conjunctions, prepositions.
        """
        words = text.split()
        if len(words) <= self.max_words:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            remaining = len(words) - start

            # If remaining words fit in one chunk, take them all
            if remaining <= self.max_words:
                chunks.append(' '.join(words[start:]))
                break

            # Find the best split point near max_words
            # Search backwards from max_words for a natural break
            best_split = start + self.max_words  # default: hard split at max

            # Search window: from 60% to 100% of max_words
            search_start = start + max(1, int(self.max_words * 0.6))
            search_end = start + self.max_words

            for pos in range(search_end, search_start - 1, -1):
                if pos >= len(words):
                    continue

                word = words[pos].lower().rstrip('.,!?;:')

                # Prefer splitting BEFORE conjunction/preposition words
                if word in self._SPLIT_BEFORE_WORDS:
                    best_split = pos
                    break

                # Also prefer splitting after a comma
                if pos > 0 and words[pos - 1].endswith(','):
                    best_split = pos
                    break

            chunks.append(' '.join(words[start:best_split]))
            start = best_split

        return chunks

    # Sentence-terminating punctuation. A chunk that ends in any of these
    # (or these followed by a closing quote) is a sentence boundary and
    # MUST NOT be merged across — doing so produces seams like
    # "...by the bed. On hands" which read like the chunker is broken.
    _SENTENCE_END = ('.', '!', '?')

    def _is_sentence_end(self, chunk: str) -> bool:
        """True if chunk ends a complete sentence (allowing closing quote)."""
        s = chunk.rstrip()
        if not s:
            return False
        # Strip trailing closing quote so '..."' still counts as boundary.
        if s[-1] in ('"', "'", '”', '’'):
            s = s[:-1].rstrip()
        return bool(s) and s[-1] in self._SENTENCE_END

    def _starts_sentence(self, chunk: str) -> bool:
        """True if chunk starts a new sentence — capital first letter.

        Also true if the chunk starts with a quote then capital ("Coach said...").
        We use this to refuse forward-merging a sentence-ending chunk into
        a chunk that's the start of a new one.
        """
        s = chunk.lstrip()
        if not s:
            return False
        # Skip opening quotes
        if s[0] in ('"', "'", '“', '‘'):
            s = s[1:].lstrip()
        return bool(s) and s[0].isupper()

    def _merge_tiny_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too small with neighbors, never crossing
        sentence boundaries.

        Sentence boundaries are sacred: a chunk ending in . ! ? cannot be
        merged with a chunk that starts a new sentence, even if both are
        tiny and the combined fits within max_words. Crossing the boundary
        produces unreadable seams like "...forgotten by the bed. On hands"
        where the LLM had a clean two-sentence structure.

        Strategy (each step requires no boundary crossing):
          1. Forward-merge if combined ≤ max_words AND the previous-end /
             next-start aren't both sentence boundaries.
          2. Backward-merge as fallback, same constraint.
          3. Leave the chunk alone — better a small chunk than a seam.
        """
        if len(chunks) <= 1:
            return chunks

        merged: List[str] = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]
            word_count = len(chunk.split())

            if word_count < self.min_words:
                # Forward merge first (chunk + next).
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    crosses_boundary = self._is_sentence_end(chunk) and self._starts_sentence(next_chunk)
                    combined = f"{chunk} {next_chunk}"
                    if not crosses_boundary and len(combined.split()) <= self.max_words:
                        merged.append(combined)
                        i += 2
                        continue
                # Backward merge as fallback (previous + chunk).
                if merged:
                    prev = merged[-1]
                    crosses_boundary = self._is_sentence_end(prev) and self._starts_sentence(chunk)
                    combined = f"{prev} {chunk}"
                    if not crosses_boundary and len(combined.split()) <= self.max_words:
                        merged[-1] = combined
                        i += 1
                        continue

            merged.append(chunk)
            i += 1

        return merged
