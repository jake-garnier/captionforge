"""
Post-processing pipeline for cleaning OCR-extracted captions
Fixes common OCR errors, removes duplicates, and cleans up text
Now with LLM-powered intelligent post-processing (Mistral-7B-Instruct-v0.3)
"""
import re
import os
from typing import List, Set, Optional
import logging
from wordsegment import load, segment
import torch

logger = logging.getLogger(__name__)

# Load word segmentation model (only once)
try:
    load()
    _WORDSEGMENT_LOADED = True
except Exception as e:
    logger.warning(f"Failed to load wordsegment: {e}")
    _WORDSEGMENT_LOADED = False

# Load LLM for intelligent post-processing (only once, lazy loading)
_LLM_MODEL = None
_LLM_TOKENIZER = None
_LLM_ENABLED = os.getenv("LLM_POSTPROCESS_ENABLED", "true").lower() == "true"
_LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
_LLM_LAST_USED = None  # Timestamp for auto-unload


def _load_llm(force: bool = False):
    """
    Lazy load LLM model (only when first needed).

    Args:
        force: If True, load LLM even when LLM_POSTPROCESS_ENABLED=false.
               Used by batch refinement tasks that need LLM regardless of inline setting.
    """
    global _LLM_MODEL, _LLM_TOKENIZER, _LLM_LAST_USED
    import time

    # Update last used timestamp
    _LLM_LAST_USED = time.time()

    if _LLM_MODEL is not None:
        return _LLM_MODEL, _LLM_TOKENIZER

    if not _LLM_ENABLED and not force:
        logger.info("LLM post-processing disabled via environment variable")
        return None, None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading LLM model: {_LLM_MODEL_NAME} (4-bit quantized)")

        # Use 4-bit quantization for memory efficiency (~3-4GB VRAM)
        # Each worker sees only its assigned GPU as device 0 via CUDA_VISIBLE_DEVICES
        # Use eager attention since RTX 2080 Ti (Turing) doesn't support FlashAttention
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        _LLM_TOKENIZER = AutoTokenizer.from_pretrained(_LLM_MODEL_NAME)
        _LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            _LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map={"": 0},  # Use worker's visible GPU (CUDA_VISIBLE_DEVICES remaps to 0)
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Disable FlashAttention for Turing GPU compatibility
        )

        logger.info(f"LLM loaded successfully. VRAM usage: ~3-4GB (4-bit quantized)")
        return _LLM_MODEL, _LLM_TOKENIZER

    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        logger.warning("Falling back to rule-based post-processing only")
        return None, None


class CaptionPostProcessor:
    """
    Post-processes raw OCR captions to improve quality
    """

    def __init__(self):
        # Common OCR character substitution patterns
        # NOTE: OCR systematically misreads C as G - need G→C fixes
        self.char_substitutions = {
            r'\bI([0-9])': r'1\1',  # I followed by digit -> 1
            r'\b([0-9])I\b': r'\g<1>1',  # Digit followed by I -> 1
            r'\bO([0-9])': r'0\1',  # O followed by digit -> 0
            r'\b([0-9])O\b': r'\g<1>0',  # Digit followed by O -> 0
        }

        # G→C substitutions for common OCR errors (OCR misreads C as G)
        self.g_to_c_words = {
            # Common words with C misread as G
            r'\bGOME\b': 'COME', r'\bGome\b': 'Come', r'\bgome\b': 'come',
            r'\bGOULD\b': 'COULD', r'\bGould\b': 'Could', r'\bgould\b': 'could',
            r'\bGAN\b': 'CAN', r'\bGan\b': 'Can', r'\bgan\b': 'can',
            r'\bGANT\b': 'CANT', r'\bGan\'t\b': 'Can\'t', r'\bgan\'t\b': 'can\'t',
            r'\bGOUNT\b': 'COUNT', r'\bGount\b': 'Count', r'\bgount\b': 'count',
            r'\bGONSIDER\b': 'CONSIDER', r'\bGonsider\b': 'Consider', r'\bgonsider\b': 'consider',
            r'\bGOMPLETE\b': 'COMPLETE', r'\bGomplete\b': 'Complete', r'\bgomplete\b': 'complete',
            r'\bGLICKED\b': 'CLICKED', r'\bGlicked\b': 'Clicked', r'\bglicked\b': 'clicked',
            r'\bGLOTHES\b': 'CLOTHES', r'\bGlothes\b': 'Clothes', r'\bglothes\b': 'clothes',
            r'\bGOULDN\'T\b': 'COULDN\'T', r'\bGouldn\'t\b': 'Couldn\'t', r'\bgouldn\'t\b': 'couldn\'t',
            r'\bGONFRONT\b': 'CONFRONT', r'\bGonfront\b': 'Confront', r'\bgonfront\b': 'confront',
            r'\bGOVERING\b': 'COVERING', r'\bGovering\b': 'Covering', r'\bgovering\b': 'covering',
            r'\bGAPTIONS?\b': 'CAPTIONS', r'\bGaptions?\b': 'Captions', r'\bgaptions?\b': 'captions',
            r'\bGRYING\b': 'CRYING', r'\bGrying\b': 'Crying', r'\bgrying\b': 'crying',
            r'\bGONNECTION\b': 'CONNECTION', r'\bGonnection\b': 'Connection',
            r'\bGHOICES?\b': 'CHOICES', r'\bGhoices?\b': 'Choices', r'\bghoices?\b': 'choices',
            r'\bGHRISTMAS\b': 'CHRISTMAS', r'\bGhristmas\b': 'Christmas', r'\bghristmas\b': 'christmas',
            r'\bGHALLENGE\b': 'CHALLENGE', r'\bGhallenge\b': 'Challenge', r'\bghallenge\b': 'challenge',
            r'\bGHEGK\b': 'CHECK', r'\bGheck\b': 'Check', r'\bgheck\b': 'check',
            r'\bGARE\b': 'CARE', r'\bGare\b': 'Care', r'\bgare\b': 'care',
            r'\bGOZY\b': 'COZY', r'\bGozy\b': 'Cozy', r'\bgozy\b': 'cozy',
            r'\bGOUGH\b': 'COUCH', r'\bGouch\b': 'Couch', r'\bgouch\b': 'couch',
            r'\bGOAGH\b': 'COACH', r'\bGoach\b': 'Coach', r'\bgoach\b': 'coach',
            r'\bGARDIO\b': 'CARDIO', r'\bGardio\b': 'Cardio', r'\bgardio\b': 'cardio',
            r'\bGOOK\b': 'COOK', r'\bGook\b': 'Cook', r'\bgook\b': 'cook',
            r'\bGHEF\b': 'CHEF', r'\bGhef\b': 'Chef', r'\bghef\b': 'chef',
            r'\bGONSISTENCY\b': 'CONSISTENCY', r'\bGonsistency\b': 'Consistency', r'\bgonsistency\b': 'consistency',
            r'\bGONVERTING\b': 'CONVERTING', r'\bGonverting\b': 'Converting',
        }

        # URLs, watermarks, and promotional patterns to remove from OCR text.
        # Kept generic (platforms, URL shapes, editor exports, CTA phrasing);
        # new candidates surface via scripts/audit_watermark_phrases.py.
        self.spam_patterns = [
            # === URLs and Domains ===
            r'patreon\.com/\S+',
            r'ko-fi\.com/\S+',
            r'buymeacoffee\.com/\S+',
            r'linktr\.ee/\S+',
            r'tr\.ee/\S+',
            r'youtube\.com/\S+',
            r'youtu\.be/\S+',
            r'tiktok\.com/\S+',
            r'instagram\.com/\S+',
            r'www\.\S+',
            r'https?://\S+',  # Generic URLs
            r'https?://',  # Partial URLs without domains

            # === Social Media ===
            r'twitter\s+@\S+',
            r'@[A-Za-z0-9_]+',  # Social media handles
            r'u/[A-Za-z0-9_]+',  # Reddit usernames
            r'TikTok',  # TikTok watermark
            r'[Ii]nsta(?:gram)?\s*[:@]\s*\S+',  # Instagram handles
            r'[Tt]umblr\s*[:@]\s*\S+',  # Tumblr handles
            r'[Dd]iscord\s*[:@]?\s*\S+',  # Discord invites
            r'[Yy]ou[Tt]ube\s*[:@]\s*\S+',  # YouTube channel callouts
            r'\d+[Kk]\s*[Ff]ollowers',  # Follower counts
            r'[Ll]ike\s*&?\s*[Ss]hare',  # Engagement prompts
            r'Part\s*\d+\s*on\s*my\s*(?:Twitter|TikTok|YouTube|Instagram):?',  # Part X on my <platform>:
            r'Send\s+a\s+Chat',  # Snapchat UI

            # === Editor / stock-footage watermarks ===
            r'CapCut',
            r'CyberLink\s*PowerDirector',  # Video editor watermark
            r'[Ii]n[Ss]hot',  # InShot video editor
            r'[Cc]lideo',  # Clideo video editor
            r'[Kk]apwing',  # Kapwing video editor
            r'[Kk]ine[Mm]aster',  # KineMaster video editor
            r'[Ff]ilmora',  # Filmora video editor
            r'[Cc]anva',  # Canva design tool
            r'[Pp]ics[Aa]rt',  # PicsArt editor
            r'[Vv]eed\.io',  # VEED editor
            r'[Ii]n[Vv]ideo',  # InVideo editor
            r'[Ll]inktree',  # Linktree
            r'[Ll]inktr\.ee\S*',  # Linktr.ee URLs
            r'[Ss]hutterstock',  # Stock footage watermark
            r'[Gg]etty\s*[Ii]mages',
            r'[Ss]toryblocks',
            r'[Pp]exels',
            r'[Pp]ixabay',

            # === PATREON Watermarks (COMMON) ===
            r'PATREON:?\s*',  # PATREON: at line start
            r'Patreon:?\s*Caption\s*\w+',  # Patreon: Caption 4f
            r'Hundreds\s+of\s+videos\s+on\s+Patreon\.?',
            r'GET\s+MORE\s+EXCLUSIVE\s+[CG]ONTENT\s+ON\s+PATREON.*?(?:\.|$)',
            r'Join\s+my\s+Patreon\.?',
            r'Check\s+out\s+my\s+Patreon\.?',
            r'supporting\s+me\s+on\s+Patreon/\S+',
            r'Want\s+more\s+long\s+videos\s+like\s+this\s+one\.?\s*Join\s+my\s+Patreon\.?',
            r'Full\s+video\s+available\s+(?:here|on)\b.*?(?:\.|$)',
            r'BLACK\s*FRIDAY\s*DEAL:.*?(?:DECEMBER|$)',  # Promotional
            r'[CG]HECK\s+OUT\s+MY\s+Pinned\s+POST\s+FOR\s+EXCLUSIVE\s+[CG]APTIONS',

            # === Generic Creator Patterns ===
            r'[Cc]aption[_\s]*[Mm]aker\S*',
            r'[Cc]aption\s+(?:by|from|made by|created by)\s+\S+',
            r'[Cc]aptions?\s*[:@]\s*\S+',  # "Captions: @handle"
            r'[Gg]aptions?\S*',  # Common OCR variant of "captions"
            r'[Mm]ade\s+[Bb]y\s+\S+',
            r'[Cc]reated\s+[Bb]y\s+\S+',
            r'[Ee]dited\s+[Bb]y\s+\S+',
            r'[Cc]redits?\s*(?:to|:)\s*\S+',
            r'[Ff]ollow\s+[Mm]e\s+\S*',
            r'[Ss]ubscribe\s+\S*',

            # === Call-to-Action / Promotional Spam ===
            r'Unlock\s+\d+\+\s*[Ee]x[ec]lusive\s*captions?.*?(?:join|$)',
            r'Link\s+in\s+bio',
            r'See\s+full\s+episodes\s+(?:&|and)\s+exclusive\s+content\s+only\s+on',
            r'Enjoyed\s+this\s+caption\.?\s*Join\s+\d+\+\s+(?:members|supporters).*?Patreon/\S+',
            r'Liked\s+This\s+Caption\.?\s*Unlock.*?(?:bio|join)',
            r'[Ee]njoyed?\s+this\s+caption',
            r'[Ll]iked?\s+this\s+caption',
            r'[Ff]ollow\s+for\s+more',
            r'[Mm]ore\s+captions?\s+(?:on|at)\s+\S+',
            r'[Ff]ind\s+(?:more|over)\s+\d*\s*captions?\s+(?:on|at)',
            r'\d+\+?\s*(?:exclusive\s+)?captions?\s+(?:on|at)\b',
            r'[Ee]xclusive\s+captions?',
            r'[Gg]et\s+full\s+access',
            r'[Aa]ccess\s+to\s+(?:all|over|\d+)',
            r'[Pp]romo(?:tion(?:al)?)?\s*code',
            r'[Dd]iscount',
            r'[Ff]ree\s+trial',
            r'[Ss]ign\s+up',
            r'[Cc]lick\s+(?:here|the\s+link|below)',
            r'[Dd]m\s+(?:me|us)\s+for\b',
            r'[Bb]uy\s+me\s+a\s+coffee',

            # === Mangled/Partial URLs from OCR ===
            r'[Tt]r[\.e]+/\S*',
            r'ee/\w+\s*\S*',  # linktr.ee slug with the domain OCR-dropped
            r'\S*\.[Cc][Oo][Mm]/\S*',  # any ".com/slug" fragment

            # === OCR Coordinate Artifacts ===
            r'\(\d+,\s*\d+\),?\s*\(\d+,\s*\d+\)',  # (123,456),(789,012)
            r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]',  # [x1, y1, x2, y2]
            r'\b\d{3,},\s*\d{3,}\b',  # Large coordinate pairs
            r'bbox:?\s*\[?\d+',

            # === AI-Generated Noise ===
            r'^AI\s+Generate\s*$',  # AI Generate label
            r'This\s+is\s+a\s+(?:video|screenshot)\s+of\s+.*?(?:\.|$)',  # AI descriptions
            r'The\s+text\s+in\s+the\s+image\s+is\s+not\s+clear.*?(?:\.|$)',

            # === UI/State Artifacts (Snapchat, etc.) ===
            r'State:\s*(?:In\s+progress|[CG]OMPLETE)',
            r'YOU\s+SCREEN\s+RECORDED\.?',
            r'YOU\s+REPLAYED\s+A\s+SNAP\.?',
            r'Tap\s+to\s+view',
            r'Hold\s+to\s+replay\s+again',
            r'Try\s+Lens',

            # === Misc Noise ===
            r'Best\s+Only\s*$',  # Truncated watermark at end
        ]

        # Common word splits that should be joined
        self.word_joins = {
            r'\b(any) (one)\b': r'\1one',
            r'\b(some) (one)\b': r'\1one',
            r'\b(no) (one)\b': r'\1one',
            r'\b(to) (day)\b': r'\1day',
            r'\b(to) (night)\b': r'\1night',
            r'\b(with) (out)\b': r'\1out',
            r'\b(your) (self)\b': r'\1self',
            r'\b(my) (self)\b': r'\1self',
            r'\b(him) (self)\b': r'\1self',
            r'\b(her) (self)\b': r'\1self',
        }

    def remove_duplicates(self, text: str, preserve_delimiters: bool = True) -> str:
        """
        Remove duplicate content both within and ACROSS slides.
        OCR often captures the same text multiple times from similar frames.

        Cross-slide deduplication strategy:
        1. Deduplicate within each slide first
        2. Compare each slide to previously seen slides
        3. If a slide is 80%+ similar to a previous slide, skip it entirely
        4. Also track individual phrases across slides to catch partial duplicates

        Args:
            text: Input text (may contain *|* delimiters)
            preserve_delimiters: Keep slide structure intact (recommended: True)

        Returns:
            Deduplicated text
        """
        if not text or not text.strip():
            return text

        # Split by slide delimiter to preserve structure
        if preserve_delimiters and ' *|* ' in text:
            slides = text.split(' *|* ')
            processed_slides = []
            seen_slide_hashes = set()  # Track normalized slide content
            seen_phrases_global = set()  # Track phrases across ALL slides

            for slide in slides:
                if not slide.strip():
                    continue

                # First: deduplicate within this slide
                deduplicated_slide = self._deduplicate_slide(slide)

                if not deduplicated_slide.strip():
                    continue

                # Second: check if this entire slide is duplicate of a previous one
                # Normalize for comparison (lowercase, collapse whitespace, remove punctuation)
                normalized_slide = self._normalize_for_comparison(deduplicated_slide)

                # Skip if we've seen this exact slide content before
                if normalized_slide in seen_slide_hashes:
                    logger.debug(f"Skipping duplicate slide: {deduplicated_slide[:50]}...")
                    continue

                # Check for high similarity (80%+ word overlap) with previous slides
                if self._is_similar_to_seen(normalized_slide, seen_slide_hashes, threshold=0.8):
                    logger.debug(f"Skipping similar slide: {deduplicated_slide[:50]}...")
                    continue

                # Third: remove phrases we've already seen in previous slides
                deduplicated_slide = self._remove_seen_phrases(
                    deduplicated_slide, seen_phrases_global
                )

                if not deduplicated_slide.strip():
                    continue

                # Add this slide's content to our tracking sets
                seen_slide_hashes.add(normalized_slide)
                self._add_phrases_to_seen(deduplicated_slide, seen_phrases_global)

                processed_slides.append(deduplicated_slide)

            # Rejoin slides with delimiter
            result = ' *|* '.join(processed_slides)
            logger.debug(f"Cross-slide dedup: {len(slides)} slides -> {len(processed_slides)} slides")
            return result
        else:
            # No delimiters, process as single block
            return self._deduplicate_slide(text)

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for duplicate comparison."""
        # Lowercase, collapse whitespace, remove punctuation
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        return normalized

    def _is_similar_to_seen(self, normalized_text: str, seen_hashes: set, threshold: float = 0.8) -> bool:
        """Check if text is highly similar to any previously seen text."""
        if not normalized_text or not seen_hashes:
            return False

        words_new = set(normalized_text.split())
        if not words_new:
            return False

        for seen in seen_hashes:
            words_seen = set(seen.split())
            if not words_seen:
                continue

            # Calculate Jaccard similarity
            intersection = len(words_new & words_seen)
            union = len(words_new | words_seen)

            if union > 0 and intersection / union >= threshold:
                return True

        return False

    def _remove_seen_phrases(self, text: str, seen_phrases: set) -> str:
        """Remove phrases that have already been seen in previous slides."""
        # Split into sentences
        sentences = re.split(r'([.!?…]+\s*)', text)
        result = []

        for sentence in sentences:
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue

            # Skip punctuation-only
            if re.match(r'^[.!?…\s]+$', sentence):
                if result and not re.match(r'^[.!?…\s]+$', result[-1]):
                    result.append(sentence)
                continue

            # Check if this sentence is already seen
            normalized = self._normalize_for_comparison(sentence_stripped)
            if normalized and len(normalized) > 10 and normalized not in seen_phrases:
                result.append(sentence)

        return ''.join(result).strip()

    def _add_phrases_to_seen(self, text: str, seen_phrases: set) -> None:
        """Add all phrases from text to the seen set."""
        # Split into sentences and add each
        sentences = re.split(r'[.!?…]+', text)
        for sentence in sentences:
            normalized = self._normalize_for_comparison(sentence)
            if normalized and len(normalized) > 10:
                seen_phrases.add(normalized)

    def _deduplicate_slide(self, slide_text: str) -> str:
        """
        Deduplicate text within a single slide
        """
        if not slide_text.strip():
            return ""

        # First pass: Remove exact duplicate lines
        lines = slide_text.split('\n')
        seen_lines = set()
        unique_lines = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Normalize for comparison (lowercase, collapse whitespace)
            normalized = ' '.join(line_stripped.lower().split())

            # Track occurrences - keep only first occurrence
            if normalized not in seen_lines:
                seen_lines.add(normalized)
                unique_lines.append(line_stripped)

        # Rejoin lines for this slide
        slide_text = ' '.join(unique_lines)

        # Second pass: Remove duplicate sentences/phrases
        phrases = re.split(r'([.!?…]+\s*)', slide_text)

        seen_phrases = set()
        result = []

        for i, phrase in enumerate(phrases):
            phrase_stripped = phrase.strip()
            if not phrase_stripped:
                continue

            # Skip punctuation-only segments
            if re.match(r'^[.!?…\s]+$', phrase):
                # Only add punctuation if we have content before it
                if result and not re.match(r'^[.!?…\s]+$', result[-1]):
                    result.append(phrase)
                continue

            # Normalize for comparison
            normalized = ' '.join(phrase_stripped.lower().split())

            # Skip if we've seen this phrase before in this slide
            if normalized not in seen_phrases:
                seen_phrases.add(normalized)
                result.append(phrase)

        slide_result = ''.join(result).strip()
        return slide_result

    def fix_character_errors(self, text: str) -> str:
        """
        Fix common OCR character recognition errors
        """
        result = text

        # Apply general character substitutions (I/1, O/0)
        for pattern, replacement in self.char_substitutions.items():
            result = re.sub(pattern, replacement, result)

        # Apply G→C substitutions for common words (OCR misreads C as G)
        for pattern, replacement in self.g_to_c_words.items():
            result = re.sub(pattern, replacement, result)

        return result

    def strip_leading_chars(self, text: str) -> str:
        """
        Strip problematic leading characters from captions.
        OCR sometimes captures leading > or " characters.
        """
        if not text:
            return text

        # Strip leading > character (common OCR artifact)
        result = re.sub(r'^>\s*', '', text)

        # Strip unbalanced leading quotes
        if result.startswith('"') and result.count('"') == 1:
            result = result[1:].strip()

        return result.strip()

    def consolidate_empty_segments(self, text: str) -> str:
        """
        Remove empty segments from delimited text.
        After spam removal, we can end up with *|* *|* sequences.
        """
        if not text or ' *|* ' not in text:
            return text

        # Split, filter empty segments, rejoin
        segments = text.split(' *|* ')
        non_empty = [s.strip() for s in segments if s.strip()]

        return ' *|* '.join(non_empty)

    def remove_spam(self, text: str) -> str:
        """
        Remove URLs, watermarks, and promotional content
        """
        result = text

        for pattern in self.spam_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        return result

    def fix_spacing(self, text: str) -> str:
        """
        Fix spacing issues around punctuation and words
        """
        # Remove spaces before punctuation
        result = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Add space after punctuation if missing
        result = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', result)

        # Fix common word splits
        for pattern, replacement in self.word_joins.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Remove multiple consecutive spaces
        result = re.sub(r'\s{2,}', ' ', result)

        return result.strip()

    def normalize_case(self, text: str) -> str:
        """
        Fix excessive ALL CAPS and mixed case issues
        Keep sentence case for better readability
        """
        # Split into sentences
        sentences = re.split(r'([.!?…]+)', text)

        result = []
        for i, sentence in enumerate(sentences):
            # Skip punctuation-only segments
            if not sentence.strip() or re.match(r'^[.!?…\s]+$', sentence):
                result.append(sentence)
                continue

            # Check if entire sentence is uppercase
            if sentence.isupper() and len(sentence.strip()) > 10:
                # Convert to title case, but preserve some common acronyms
                words = sentence.split()
                normalized_words = []

                for word in words:
                    # Keep short uppercase words as-is (likely acronyms)
                    if len(word) <= 3 and word.isupper():
                        normalized_words.append(word)
                    else:
                        normalized_words.append(word.capitalize())

                result.append(' '.join(normalized_words))
            else:
                result.append(sentence)

        return ''.join(result)

    def remove_fragments(self, text: str, min_length: int = 15) -> str:
        """
        Remove very short fragments that are likely OCR noise
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?…]+', text)

        # Keep only sentences longer than min_length characters
        cleaned = [s.strip() for s in sentences if len(s.strip()) >= min_length]

        return '. '.join(cleaned) + '.' if cleaned else text

    def extract_dialogue(self, text: str) -> str:
        """
        Try to identify and extract actual dialogue/caption text
        vs. metadata/UI elements
        """
        # Remove lines that are clearly UI elements (very short, all caps, etc.)
        lines = text.split('\n')

        content_lines = []
        for line in lines:
            line = line.strip()

            # Skip very short lines (likely UI elements)
            if len(line) < 10:
                continue

            # Skip lines that are all numbers/symbols
            if re.match(r'^[0-9\s\-+=%]+$', line):
                continue

            content_lines.append(line)

        return ' '.join(content_lines)

    def fix_concatenated_words(self, text: str) -> str:
        """
        Fix words that were concatenated due to OCR missing spaces
        Enhanced version with better heuristics and dictionary checking

        Examples:
            "itonce" -> "it once"
            "reallypaying" -> "really paying"
            "warmupfirst" -> "warm up first"
        """
        if not _WORDSEGMENT_LOADED:
            return text

        # Common concatenated word patterns specific to our domain
        common_fixes = {
            r'\b(really)(paying|good|bad|hard|soft)\b': r'\1 \2',
            r'\b(warm)(up)\b': r'\1 \2',
            r'\b(meal)(prep)\b': r'\1 \2',
            r'\b(at)(least)\b': r'\1 \2',
            r'\b(to)(day|night|morrow)\b': r'\1 \2',
            r'\b(you)(re)\b': r'\1\'\2',  # youre -> you're (but NOT your)
            r'\b(aren)(t)\b': r'\1\'\2',     # aren't
            r'\b(wasn)(t)\b': r'\1\'\2',     # wasn't
            r'\b(weren)(t)\b': r'\1\'\2',    # weren't
        }

        # Apply common fixes first
        result = text
        for pattern, replacement in common_fixes.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Split text into words for statistical segmentation
        words = result.split()
        fixed_words = []

        for word in words:
            # Skip short words (likely correct)
            if len(word) <= 5:  # Increased threshold from 4 to 5
                fixed_words.append(word)
                continue

            # Skip words with punctuation (likely correct)
            if any(char in word for char in '.,!?;:\'"'):
                fixed_words.append(word)
                continue

            # Skip words that are all uppercase (likely acronyms)
            if word.isupper():
                fixed_words.append(word)
                continue

            # Skip words that start with uppercase (likely proper nouns)
            if word[0].isupper() and len(word) > 1 and word[1:].islower():
                fixed_words.append(word)
                continue

            # Try to segment the word
            try:
                # Convert to lowercase for segmentation
                word_lower = word.lower()
                segments = segment(word_lower)

                # Only apply segmentation if:
                # 1. It found multiple parts
                # 2. Parts are reasonable length (no single letters except 'a' or 'i')
                # 3. Each segment looks like a real word
                if len(segments) > 1:
                    # Filter out single-letter segments except 'a' and 'i'
                    valid_segments = []
                    min_segment_length = 2  # Most segments should be at least 2 chars

                    for seg in segments:
                        if len(seg) == 1 and seg not in ['a', 'i']:
                            # Likely a bad segmentation, keep original word
                            valid_segments = None
                            break
                        # Prefer longer segments
                        if len(seg) >= min_segment_length or seg in ['a', 'i']:
                            valid_segments.append(seg)
                        else:
                            # Short segment that's not 'a' or 'i' - might be bad
                            valid_segments = None
                            break

                    if valid_segments and len(valid_segments) > 1:
                        # Additional check: average segment length should be reasonable
                        avg_len = sum(len(s) for s in valid_segments) / len(valid_segments)
                        if avg_len >= 2.5:  # Average segment length > 2.5 chars
                            # Preserve original capitalization pattern
                            if word[0].isupper():
                                valid_segments[0] = valid_segments[0].capitalize()
                            fixed_word = ' '.join(valid_segments)
                            fixed_words.append(fixed_word)
                            logger.debug(f"Segmented '{word}' -> '{fixed_word}'")
                        else:
                            fixed_words.append(word)
                    else:
                        fixed_words.append(word)
                else:
                    fixed_words.append(word)
            except Exception as e:
                logger.debug(f"Error segmenting word '{word}': {e}")
                fixed_words.append(word)

        return ' '.join(fixed_words)

    def _assess_caption_quality(self, text: str) -> float:
        """
        Assess caption quality to determine if LLM processing is needed
        Returns score 0-1 (0 = poor quality, 1 = excellent quality)
        """
        score = 1.0

        # Penalize excessive word repetition (OCR duplicates)
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # More than 50% repeated words
                score -= 0.4

        # Penalize lack of punctuation (unreadable run-on text)
        punct_count = text.count('.') + text.count('!') + text.count('?')
        if len(text) > 200 and punct_count < len(text) / 150:
            score -= 0.3

        # Penalize OCR artifacts (consecutive caps, weird character patterns)
        if re.search(r'[A-Z]{6,}', text):  # Long all-caps sequences
            score -= 0.2
        if re.search(r'\w{20,}', text):  # Extremely long words (OCR errors)
            score -= 0.2
        if re.search(r'[a-z][A-Z][a-z]', text):  # Mixed case within words
            score -= 0.1

        return max(0.0, score)

    def _llm_refine_caption(self, text: str, force: bool = False) -> str:
        """
        Use LLM to intelligently refine caption text
        Fixes errors that rule-based processing can't handle

        Args:
            text: The text to refine
            force: If True, load LLM even when LLM_POSTPROCESS_ENABLED=false
        """
        model, tokenizer = _load_llm(force=force)

        if model is None or tokenizer is None:
            logger.debug("LLM not available, skipping LLM refinement")
            return text

        try:
            # Craft a prompt that fixes OCR errors while PRESERVING all unique content
            prompt = f"""<s>[INST] You are cleaning OCR-extracted text from a video. Fix errors while PRESERVING ALL UNIQUE CONTENT.

CRITICAL - DO NOT REMOVE CONTENT:
1. PRESERVE every unique sentence/phrase - each slide may have different dialogue
2. The " *|* " delimiter separates different video slides - keep this structure
3. Only remove EXACT duplicates (word-for-word identical phrases)
4. Each slide likely has unique content - do NOT merge or summarize slides

FIX THESE OCR ERRORS:
- Character errors: I/1, O/0, rn/m, cl/d
- Missing punctuation and sentence breaks
- Spacing issues and run-on words
- Remove watermarks, URLs, @handles (but keep the dialogue around them)

DO NOT:
- Remove unique content even if it seems short or incomplete
- Summarize or combine different slides
- Add new content
- Rewrite the tone or change the wording

EXAMPLE:
Input: "Alarm at 5 *|* Nobody is goming to do it for you *|* One more mile *|* Then breakfast"
Output: Alarm at 5. *|* Nobody is coming to do it for you. *|* One more mile! *|* Then breakfast.

IMPORTANT: Output ONLY the cleaned text. No explanations, no "corrected version", no commentary.

Raw OCR text:
{text}

Cleaned caption: [/INST]"""

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.3,  # Low temperature for consistent, conservative edits
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode ONLY the generated tokens (not the input prompt)
            # inputs["input_ids"] is the tokenized prompt, we want only the new tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]  # Skip the prompt tokens
            cleaned = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Post-process: Remove any explanatory text the model might add
            # Sometimes model outputs "Corrected version:" or repeats the content
            for marker in ['Corrected version:', 'Corrected:', 'Fixed version:', 'Here is the cleaned']:
                if marker in cleaned:
                    # Take only the text before the marker (first version)
                    cleaned = cleaned.split(marker)[0].strip()
                    break

            # Remove surrounding quotes if present
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            elif cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1].strip()

            logger.debug(f"LLM refined caption: {len(text)} -> {len(cleaned)} chars")
            return cleaned

        except Exception as e:
            logger.error(f"LLM refinement failed: {e}")
            return text  # Fallback to rule-based result

    def process(self, text: str, aggressive: bool = False, return_all_versions: bool = False):
        """
        Run full post-processing pipeline

        Args:
            text: Raw OCR output
            aggressive: If True, apply more aggressive cleaning (may remove some content)
            return_all_versions: If True, return dict with all 3 processing stages + quality metrics

        Returns:
            If return_all_versions=True: dict with keys 'raw_ocr', 'rule_based', 'llm_refined', 'final', 'quality_metrics'
            If return_all_versions=False: str (cleaned caption text) - for backwards compatibility
        """
        if not text or len(text.strip()) < 5:
            if return_all_versions:
                return {
                    'raw_ocr': text,
                    'rule_based': text,
                    'llm_refined': None,
                    'final': text,
                    'quality_metrics': self._compute_quality_metrics(text, text)
                }
            return text

        logger.debug(f"Processing caption: {len(text)} chars")

        # Store raw OCR
        raw_ocr = text

        # Step 1: Strip leading characters (>, unbalanced quotes)
        result = self.strip_leading_chars(text)

        # Step 2: Remove spam (URLs, watermarks)
        result = self.remove_spam(result)

        # Step 3: Fix character recognition errors (including G→C)
        result = self.fix_character_errors(result)

        # Step 4: Consolidate empty segments (after spam removal)
        result = self.consolidate_empty_segments(result)

        # Step 5: Fix spacing issues
        result = self.fix_spacing(result)

        # Step 6: Remove duplicate phrases
        result = self.remove_duplicates(result)

        # Step 7: Extract dialogue (remove UI elements)
        if aggressive:
            result = self.extract_dialogue(result)

        # Step 8: Normalize case
        if aggressive:
            result = self.normalize_case(result)

        # Step 9: Remove very short fragments
        if aggressive:
            result = self.remove_fragments(result)

        # Step 10: Fix concatenated words (missing spaces)
        if aggressive:
            result = self.fix_concatenated_words(result)

        # Final cleanup - consolidate again and fix spacing
        result = self.consolidate_empty_segments(result)
        result = self.fix_spacing(result)

        # Store rule-based result (before LLM)
        rule_based = result

        # Step 11: LLM-powered intelligent refinement (always run if enabled)
        llm_refined = None
        if aggressive and _LLM_ENABLED:
            logger.info("Applying LLM refinement to caption")
            llm_result = self._llm_refine_caption(result)
            llm_refined = llm_result
            result = llm_result  # Use LLM result as final

        logger.debug(f"Processed caption: {len(result)} chars")

        # Return all versions if requested
        if return_all_versions:
            return {
                'raw_ocr': raw_ocr,
                'rule_based': rule_based,
                'llm_refined': llm_refined,  # None if LLM wasn't applied
                'final': result,  # Final result (either rule_based or llm_refined)
                'quality_metrics': self._compute_quality_metrics(raw_ocr, result)
            }

        return result

    def _compute_quality_metrics(self, raw_text: str, final_text: str) -> dict:
        """
        Compute quality metrics for caption processing.

        Returns dict with:
            compression_ratio: raw_length / final_length (higher = more dedup)
            slide_count_raw: Number of slides in raw OCR
            slide_count_final: Number of slides after processing
            unique_word_ratio: Unique words / total words (lower = more repetition)
        """
        metrics = {
            'compression_ratio': None,
            'slide_count_raw': None,
            'slide_count_final': None,
            'unique_word_ratio': None
        }

        if not raw_text:
            return metrics

        # Compression ratio
        raw_len = len(raw_text)
        final_len = len(final_text) if final_text else 0
        if final_len > 0:
            metrics['compression_ratio'] = round(raw_len / final_len, 2)

        # Slide counts
        metrics['slide_count_raw'] = raw_text.count(' *|* ') + 1 if raw_text else 0
        metrics['slide_count_final'] = final_text.count(' *|* ') + 1 if final_text else 0

        # Unique word ratio (on final text)
        if final_text:
            words = final_text.lower().split()
            if len(words) > 0:
                unique_words = set(words)
                metrics['unique_word_ratio'] = round(len(unique_words) / len(words), 3)

        return metrics


# Convenience function
def clean_caption(text: str, aggressive: bool = False) -> str:
    """
    Quick caption cleaning function

    Args:
        text: Raw OCR caption
        aggressive: Apply aggressive cleaning

    Returns:
        Cleaned caption
    """
    processor = CaptionPostProcessor()
    return processor.process(text, aggressive=aggressive)


# === LLM Model Management ===

def cleanup_llm():
    """Release GPU memory by unloading the LLM model.

    Properly frees CUDA memory by:
    1. Moving model to CPU (releases VRAM)
    2. Deleting all references
    3. Running garbage collection
    4. Synchronizing and emptying CUDA cache
    """
    global _LLM_MODEL, _LLM_TOKENIZER, _LLM_LAST_USED
    import gc

    if _LLM_MODEL is not None:
        try:
            # Move model to CPU first to release VRAM
            _LLM_MODEL.to('cpu')
        except Exception as e:
            logger.debug(f"Could not move LLM to CPU: {e}")
        del _LLM_MODEL
        _LLM_MODEL = None

    if _LLM_TOKENIZER is not None:
        del _LLM_TOKENIZER
        _LLM_TOKENIZER = None

    _LLM_LAST_USED = None

    # Force garbage collection to release Python references
    gc.collect()

    if torch.cuda.is_available():
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
        # Empty the cache
        torch.cuda.empty_cache()
        # Run GC again after cache clear
        gc.collect()

    logger.info("LLM (Mistral-7B) unloaded and GPU memory cleared")


def unload_llm_if_idle(idle_timeout_minutes: int = 5) -> bool:
    """
    Unload LLM if it hasn't been used recently.

    Args:
        idle_timeout_minutes: Minutes of inactivity before unloading (default: 5)

    Returns:
        True if model was unloaded, False otherwise
    """
    global _LLM_MODEL, _LLM_LAST_USED
    import time

    if _LLM_MODEL is None:
        return False  # Nothing to unload

    if _LLM_LAST_USED is None:
        return False  # No usage timestamp

    idle_seconds = time.time() - _LLM_LAST_USED
    idle_minutes = idle_seconds / 60

    if idle_minutes >= idle_timeout_minutes:
        logger.info(f"LLM (Mistral-7B) idle for {idle_minutes:.1f} minutes, unloading...")
        cleanup_llm()
        return True

    return False


def is_llm_loaded() -> bool:
    """Check if LLM is currently loaded."""
    return _LLM_MODEL is not None


def get_llm_status() -> dict:
    """Get current LLM status including idle time."""
    import time

    status = {
        "loaded": _LLM_MODEL is not None,
        "enabled": _LLM_ENABLED,
        "model_name": _LLM_MODEL_NAME,
        "last_used": None,
        "idle_seconds": None,
    }

    if _LLM_LAST_USED is not None:
        status["last_used"] = _LLM_LAST_USED
        status["idle_seconds"] = time.time() - _LLM_LAST_USED

    return status
