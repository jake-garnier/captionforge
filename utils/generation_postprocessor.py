"""
Post-processing utilities for LLM-generated captions.

Cleans up common issues from fine-tuned models:
- Prompt leakage
- Frame delimiters from training data
- Promotional spam (Patreon, Discord, etc.)
- OCR artifacts and gibberish

Also provides tag extraction for matching captions to background videos.
"""
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Patterns that indicate promotional spam (case-insensitive).
# These catch creator promo / watermark text that leaks from training data
# into generated captions. Keep them generic (platform names, URL shapes,
# CTA phrasing, editor watermarks) rather than per-creator — new candidates
# are surfaced by scripts/audit_watermark_phrases.py.
SPAM_PATTERNS = [
    # === Platforms & URLs ===
    r'\bpatreon\b',
    r'\bko-?fi\b',
    r'\bbuymeacoffee\b',
    r'\bdiscord\b',
    r'\blinktree\b',
    r'\blinktr\.ee\b',
    r'\breddit\.com\b',
    r'\btumblr\b',
    r'\btelegram\b',
    r'\bsnapchat\b',
    r'\btiktok\b',
    r'\binstagram\b',
    r'\byoutube\b',
    r'\btwitch\b',
    r'\bsubstack\b',
    r'\bgumroad\b',
    r'\broku\b',
    r'\bfire tv\b',
    r'\bapple tv\b',
    r'\bsmart tv\b',
    r'\bandroid tv\b',
    r'\.com/',
    r'https?://\S+',
    r'\[link[^\]]*\]',
    r'link in bio',
    r'@\w{3,}',  # Social media handles (min 3 chars to avoid false positives)

    # === Promotional / Paywall ===
    r'\bjoin now\b',
    r'\bjoin free\b',
    r'\bjoin today\b',
    r'\bjoin my\b',
    r'\bjoin \d+\+?\s*(fans|members|patrons|subscribers)',
    r'\bsubscribe\b',
    r'\bsubscription\b',
    r'\bcheck out my pinned\b',
    r'\bhundreds of videos\b',
    r'\bthousands of videos\b',
    r'\bpremium members?\b',
    r'\bupgrade to premium\b',
    r'\bmembership renews?\b',
    r'\bpatrons only\b',
    r'\bfull video (?:&|and) audio\b',
    r'\bfull-length video\b',
    r'\bget full access\b',
    r'\baccess to (?:all|over|\d+)',
    r'\bunlock\s+\d+\+?\s*(?:exclusive|captions?|videos?)',
    r'\bexclusive captions?\b',
    r'\bexclusive content\b',
    r'\bexclusive videos?\b',
    r'\bmore captions?\s+(?:on|at)\b',
    r'\bfind\s+(?:more|over)\s+\d*\s*captions?\b',
    r'\b\d+\+?\s*(?:exclusive\s+)?captions?\s+(?:on|at)\b',
    r'\bpromo(?:tion(?:al)?)?\s*code\b',
    r'\bdiscount\b',
    r'\bfree trial\b',
    r'\bsign up\b',
    r'\bclick (?:here|the link|below)\b',

    # === Creator Attributions ===
    r'\bcaption (?:by|from|made by|created by)\b',
    r'\bmade by\b',
    r'\bcreated by\b',
    r'\bcaptions?\s*(?:by|from)\s*[:@]?\s*\w*',
    r'\bedited\s*by\s*[:\-]?\s*\S+',
    r'\bcredits?\s*(?:to|:)\s*\S+',
    r'\b(?:ig|insta|tt|yt)\s*[:@]\s*\S+',

    # === Social CTAs ===
    r'\benjoyed?\s+this\s+caption\b',
    r'\bliked?\s+this\s+caption\b',
    r'\bfollow\s+(?:me|for|us)\b',
    r'\bfollow\s+(?:on|at|@)\b',
    r'\bshare\s+(?:this|with)\b',
    r'\blike\s*(?:&|and)\s*(?:share|subscribe|follow)\b',
    r'\bcomment\s+(?:below|what|if|your)\b',
    r'\blet\s+(?:me|us)\s+know\s+(?:in|what)\b',
    r'\bdm\s+(?:me|us)\s+for\b',

    # === Editor / stock-footage watermarks ===
    r'\bcaption\s*maker\b',
    r'\binshot\b',
    r'\bclideo\b',
    r'\bkapwing\b',
    r'\bcapcut\b',
    r'\bkinemaster\b',
    r'\bfilmora\b',
    r'\bcanva\b',
    r'\bpicsart\b',
    r'\bveed\.io\b',
    r'\binvideo\b',
    r'\bshutterstock\b',
    r'\bgetty\s*images\b',
    r'\bpexels\b',
    r'\bpixabay\b',
    r'\bstoryblocks\b',
    r'\bstock\s*footage\b',

    # === Prompt Leaks ===
    r'\bread all the text\b',
    r'\bread all text\b',
    r'\boutput only the text\b',
    r'\bpreserving the reading order\b',
    r'\btop to bottom,? left to right\b',
    r'\bdo not add (?:any )?(?:explanations|descriptions)\b',
    r'\bsecond person\b',
    r'\b40[\s-]*80 words\b',
    r'\b40[\s-]*100 words\b',
    r'\b2[\s-]*4 sentences\b',
    r'\bwrite a (?:short )?(?:motivational |fitness |cooking |travel )?caption\b',
    r'\bgenerate a caption\b',
    r'\bcaption:\s*$',
    r'\bbg source\b',
    r'\bcaption \d+[a-z]?\b',  # "Caption 4b", "Caption 1a" — slide markers
    r'\bcaption maker\b',

    # === Reddit-style call-to-action / promo tails ===
    r'\bwant the[\s.]*full[\s.]*story\b',
    r'\b\d+\+?\s*captions?\s*(?:&|and)\s*daily\b',
    r'\b\d+\+?\s*captions?\s*(?:&|and)\s*updates?\b',
    r'\bdaily updates?\b',
    r'\baccess to[\s.]*\(?[\s.]*\d+%?\b',  # "Access to. (20% off"
    r'\b\d+%\s*off\s*(?:code|with)\b',
    r'\b(?:promo|discount)\s*code\s*[:"]?\s*[A-Z0-9]+\b',
    r'\bcheck it out[\s.]*com\b',
    r'\bget longer,?\s*on\b',
    r'\bjoin for free\b',
    r'\bupgrade to (?:the )?full\b',
    r'\bget \d+\+?\s*captions?\b',

    # === Multi-choice quiz format (LoRA training-data leak) ===
    r'\bchoose[\s.]*your[\s.]*(?:path|fate|adventure)\b',
    r'\bedition\s*\(\s*\d+\s*options?\s*\)',
    r'^\s*[A-D]\s*\)\s',  # A) B) C) D) at line start

    # === URL-style watermarks ===
    # Generic shapes that catch the watermark forms observed in fitness/
    # travel/cooking OCR without needing per-creator maintenance. See
    # docs/watermark_audit.md for the audit that motivated these.
    r'\bpatreon\.com\s*/\s*\w+',
    r'\bko-?fi\.com\s*/\s*\w+',
    r'\bbuymeacoffee\.com\s*/\s*\w+',
    r'\byoutube\.com\s*/\s*@?\w+',
    r'\btiktok\.com\s*/\s*@\w+',
    r'\binstagram\.com\s*/\s*\w+',
    r'\b(?:patreon|kofi|telegram|discord|instagram|tiktok|youtube)\s*:\s*\S+',

    # === @handle / u-handle / r-sub watermarks ===
    r'@[A-Za-z][\w_]{3,}',
    r'\b/?u/[A-Za-z][\w_-]{2,}',
    r'\br/[A-Za-z][\w_-]{2,}',

    # === Membership-tier promo blocks ===
    r'(?:^|\s)\d+\+?\s*exclusive\s*(?:captions?|videos?|posts?)\b',
    r'\bsubscribe\s+for\s+full\s+(?:videos?|captions?|content)\b',
]

# Compile patterns for efficiency
SPAM_REGEX = re.compile('|'.join(SPAM_PATTERNS), re.IGNORECASE)

# Frame delimiter from OCR training data
FRAME_DELIMITER = '*|*'

# Common prompt prefixes that leak into output
PROMPT_PREFIXES = [
    'generate a caption:',
    'generate a caption',
    'write a caption:',
    'write a caption',
    'create a caption:',
    'create a caption',
]


def _fix_missing_punctuation(text: str) -> str:
    """
    Fix missing sentence-ending punctuation in LLM-generated text.

    Detects implicit sentence boundaries where a lowercase letter is followed
    by a space and a capital letter (indicating a new sentence) but no
    punctuation separates them.

    Example: "finish the set Then rest" -> "finish the set. Then rest"
    """
    if not text:
        return text

    # Common words that start with capitals mid-sentence (don't split before these).
    # Includes proper nouns, vocatives, honorifics, weekdays/months, and
    # contraction-stripped I-forms. All-uppercase acronyms (HIIT, PR, GPS, USA, etc.)
    # are handled separately via the `next_word.isupper()` check below.
    no_split_words = {
        'I', "I'm", "I'll", "I've", "I'd", 'Im', 'Ill', 'Ive', 'Id',
        # Honorifics / vocatives (these regularly appear mid-sentence in captions)
        'Coach', 'Chef', 'Mom', 'Dad', 'Sir', 'Madam', 'Maam', 'Doc', 'Doctor',
        'God', 'Jesus', 'Christ',
        # Demonyms that read as proper nouns mid-sentence
        'American', 'British', 'French', 'Italian', 'Japanese', 'Mexican', 'Thai',
        # Weekdays
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        # Months
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
    }

    result = []
    i = 0
    in_quote = False

    while i < len(text):
        char = text[i]

        # Track quote state
        if char == '"':
            in_quote = not in_quote

        result.append(char)

        # Look for pattern: lowercase_letter + space + UPPERCASE_letter
        # Only fix outside of quotes
        if (not in_quote and
                char == ' ' and
                i >= 1 and i + 1 < len(text) and
                text[i - 1].islower() and
                text[i + 1].isupper()):

            # Check the next word isn't in our exception list
            next_word_end = i + 1
            while next_word_end < len(text) and (text[next_word_end].isalpha() or text[next_word_end] == "'"):
                next_word_end += 1
            next_word = text[i + 1:next_word_end]

            # Skip if next word is an all-uppercase acronym (HIIT, PR, GPS, USA, HR, TV, etc.).
            # These are routinely embedded mid-sentence — inserting a period before them
            # was the dominant artifact in Stage 4 reviews (~18% of captions).
            is_all_caps_acronym = len(next_word) >= 2 and next_word.isupper()

            # Also check that the character before the space isn't already punctuation
            if (not is_all_caps_acronym
                    and next_word not in no_split_words
                    and text[i - 1] not in '.!?,;:'):
                # Insert a period before the space
                result[-1] = '. '

        i += 1

    return ''.join(result)


# Per-niche vocabulary normalisation. Small, deterministic spelling/term
# fixes applied after generic cleanup so captions read consistently within
# a niche (the LoRA copies whichever spelling variant dominated its training
# data). Order matters within a niche: longer/more-specific patterns first.
#
# Patterns are case-insensitive and word-boundary-aware; capitalisation of
# the replacement follows the leading character of the original match (so
# "Work-out" → "Workout", "sautee" → "saute"). Add a niche here when its
# corpus shows a recurring inconsistency; unknown niches are a no-op.
NICHE_TERM_REWRITES: dict[str, list[tuple[re.Pattern, str]]] = {
    "motivation": [
        (re.compile(r'\bself\s+discipline\b', re.IGNORECASE), 'self-discipline'),
        (re.compile(r'\bmind[\s-]set\b', re.IGNORECASE), 'mindset'),
    ],
    "fitness": [
        (re.compile(r'\bwork-out\b', re.IGNORECASE), 'workout'),
        (re.compile(r'\bdead[\s-]lift', re.IGNORECASE), 'deadlift'),
        (re.compile(r'\bpull[\s-]ups?\b', re.IGNORECASE), 'pullups'),
        (re.compile(r'\bpush[\s-]ups?\b', re.IGNORECASE), 'pushups'),
    ],
    "cooking": [
        (re.compile(r'\bsautee(d|ing|s)?\b', re.IGNORECASE), r'saute\1'),
        (re.compile(r'\bbar-?b-?q\b', re.IGNORECASE), 'barbecue'),
        (re.compile(r'\bmeal-prep', re.IGNORECASE), 'meal prep'),
    ],
    "travel": [
        (re.compile(r'\broad-?trip', re.IGNORECASE), 'road trip'),
        (re.compile(r'\bback[\s-]pack', re.IGNORECASE), 'backpack'),
        (re.compile(r'\bcheck[\s-]in\b', re.IGNORECASE), 'check-in'),
    ],
}


def _apply_niche_term_rewrites(text: str, niche: str | None) -> str:
    """
    Apply NICHE_TERM_REWRITES for `niche`. Match-case-aware: capitalizes the
    replacement when the matched token starts with an uppercase letter so
    sentence-initial occurrences keep their case. Unknown/None niche is a
    no-op.
    """
    for pattern, replacement in NICHE_TERM_REWRITES.get(niche or "", []):
        def _sub(m, repl=replacement):
            out = m.expand(repl)
            if m.group(0)[:1].isupper():
                return out[:1].upper() + out[1:]
            return out
        text = pattern.sub(_sub, text)
    return text


def clean_generated_caption(text: str, aggressive: bool = True, niche: str | None = None) -> str:
    """
    Clean a generated caption by removing artifacts and spam.

    Args:
        text: Raw generated caption text
        aggressive: If True, truncate at first spam pattern.
                   If False, only remove obvious artifacts.
        niche: Optional niche name. Applies that niche's entries from
                NICHE_TERM_REWRITES after the generic cleanup (spelling /
                term normalisation). Unknown values or None are no-ops.

    Returns:
        Cleaned caption text
    """
    if not text:
        return ""

    original_length = len(text)

    # Step 1: Remove prompt prefix if present
    text_lower = text.lower().strip()
    for prefix in PROMPT_PREFIXES:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()
            break

    # Step 2: Remove frame delimiters and clean up
    # Replace *|* with paragraph breaks, then clean up
    text = text.replace(FRAME_DELIMITER, '\n\n')

    # Step 3: Handle spam - truncate at first spam pattern if aggressive
    if aggressive:
        # Find first spam pattern
        match = SPAM_REGEX.search(text)
        if match:
            # Find the start of the sentence containing spam
            spam_pos = match.start()

            # Look backwards for sentence boundary (., !, ?, newline)
            sentence_start = spam_pos
            for i in range(spam_pos - 1, max(0, spam_pos - 200), -1):
                if text[i] in '.!?\n':
                    sentence_start = i + 1
                    break

            # Truncate before the spam sentence, but only if we keep meaningful content
            if sentence_start > 100:  # Keep at least 100 chars of good content
                text = text[:sentence_start].strip()
            elif spam_pos > 100:
                # If spam is after first 100 chars, truncate there
                text = text[:spam_pos].strip()
            else:
                # Spam is very early - try to find content AFTER the spam sentence
                # Look for the end of the spam sentence
                spam_end = len(text)
                for i in range(match.end(), min(len(text), match.end() + 200)):
                    if text[i] in '.!?\n':
                        spam_end = i + 1
                        break

                remaining = text[spam_end:].strip()
                if len(remaining) > 100:
                    # Use content after spam if substantial
                    text = remaining
                    logger.debug(f"Removed early spam, using remaining {len(remaining)} chars")
                # else: keep original text with spam - better than empty

    # Step 4: Clean up whitespace
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = text.strip()

    # Step 4b: Strip punctuation artifacts the renderer can't handle.
    # The hard-rule prompt and the judge BOTH tell the LLM not to use these,
    # but it ignores instructions roughly half the time — particularly the
    # literary "— like this — " construction. Stripping here is deterministic
    # and means the prompt/judge are belt-and-braces, not the load-bearing
    # defense.
    #
    # Em-dash (—), en-dash (–), and " - " hyphen used as a dash → comma.
    # Semicolons (;) → comma.
    # Curly quotes (' " ' ") → straight equivalents (then dropped because
    # the renderer mangles apostrophes too — see the apostrophe rule in the
    # prompt).
    text = text.replace('—', ',').replace('–', ',')
    text = re.sub(r'\s+-\s+', ', ', text)  # " - " spaced hyphen, leave word-internal hyphens alone
    text = text.replace(';', ',')
    # Normalize curly punctuation to ASCII so downstream renderer doesn't
    # fall back to glyphs it can't render.
    text = text.replace('‘', '').replace('’', '')  # ' '  -> drop (per apostrophe rule)
    text = text.replace('“', '"').replace('”', '"')  # " " -> "
    text = text.replace('…', '...')  # ellipsis char -> three dots
    # Apostrophe stripping (the renderer mangles them, prompt instructs the
    # LLM to omit, but we belt-and-brace here too). Only in contractions:
    # don't -> dont, you're -> youre, etc. We don't drop possessives
    # ("coach's") because dropping the apostrophe there breaks readability
    # ("coachs" reads fine actually). Drop all apostrophes:
    text = text.replace("'", "")
    # Collapse any double-comma or comma-space-comma artifacts the
    # replacements above can produce (e.g. "good — but" -> "good , but" ->
    # "good, but"; "tonight; bring" -> "tonight, bring"; back-to-back swaps
    # can sometimes leave ", ,").
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',(\S)', r', \1', text)
    text = re.sub(r' {2,}', ' ', text).strip()

    # Step 5: Fix missing sentence-ending punctuation
    # LLM sometimes omits punctuation between sentences, e.g.:
    # "Keep the pace steady You have got this" -> "Keep the pace steady. You have got this"
    # Detect: lowercase letter followed by space + capital letter (not inside quotes)
    # Insert a period at the boundary
    text = _fix_missing_punctuation(text)

    # Step 6: Remove incomplete final sentence (ends without punctuation)
    if text and text[-1] not in '.!?"\'':
        # Find last complete sentence
        last_punct = max(
            text.rfind('.'),
            text.rfind('!'),
            text.rfind('?'),
            text.rfind('"'),
        )
        # Only truncate if we keep >50% AND at least 50 chars
        if last_punct > len(text) * 0.5 and last_punct >= 50:
            text = text[:last_punct + 1]

    # Step 7: Remove gibberish patterns (repeated characters/words)
    # Remove words that are just repeated characters (aaaa, bbbb)
    text = re.sub(r'\b([a-zA-Z])\1{4,}\b', '', text)
    # Remove excessive repeated words
    text = re.sub(r'\b(\w+)(\s+\1){3,}\b', r'\1', text, flags=re.IGNORECASE)

    # Step 8: per-niche vocabulary normalisation (see NICHE_TERM_REWRITES).
    text = _apply_niche_term_rewrites(text, niche)

    # Final cleanup
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    cleaned_length = len(text)
    if original_length > 0:
        reduction = (1 - cleaned_length / original_length) * 100
        if reduction > 20:
            logger.debug(f"Caption cleaned: {original_length} -> {cleaned_length} chars ({reduction:.1f}% reduction)")

    return text


def clean_for_training(text: str) -> str:
    """
    Clean caption text before using it for training.
    More aggressive than generation cleaning - removes all spam.

    Args:
        text: Raw caption text from database

    Returns:
        Cleaned text suitable for training, or empty string if too contaminated
    """
    if not text:
        return ""

    # Remove frame delimiters first
    text = text.replace(FRAME_DELIMITER, ' ')

    # Remove all spam patterns (don't just truncate, remove entirely)
    # Find all spam and remove the sentences containing them
    lines = text.split('\n')
    clean_lines = []

    for line in lines:
        if not SPAM_REGEX.search(line):
            clean_lines.append(line)

    text = '\n'.join(clean_lines)

    # Also remove sentences with spam within paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = []

    for sentence in sentences:
        if not SPAM_REGEX.search(sentence):
            clean_sentences.append(sentence)

    text = ' '.join(clean_sentences)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # If more than 30% was spam, mark as too contaminated
    original_word_count = len(text.split())
    if original_word_count < 20:
        return ""  # Too short after cleaning

    return text


def calculate_quality_score(text: str) -> float:
    """
    Calculate a quality score for generated text (0-1).

    Factors:
    - Length (prefer 100-500 chars)
    - Has complete sentences
    - No spam patterns
    - No excessive repetition

    Returns:
        Score from 0.0 (poor) to 1.0 (excellent)
    """
    if not text:
        return 0.0

    score = 1.0

    # Length penalty
    length = len(text)
    if length < 50:
        score *= 0.3
    elif length < 100:
        score *= 0.7
    elif length > 800:
        score *= 0.8
    elif length > 1200:
        score *= 0.5

    # Spam penalty
    if SPAM_REGEX.search(text):
        score *= 0.4

    # Frame delimiter penalty
    if FRAME_DELIMITER in text:
        score *= 0.7

    # Incomplete sentence penalty
    if text and text[-1] not in '.!?"\'':
        score *= 0.8

    # Repetition penalty
    words = text.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            score *= 0.5
        elif unique_ratio < 0.7:
            score *= 0.8

    return round(score, 2)


# ==============================================================================
# ACTIVITY TAG EXTRACTION
# ==============================================================================
# Unified activity vocabulary shared between:
# 1. Caption text extraction (regex-based, this file)
# 2. Background video scene tagging (VLM-based, tasks/ml_tagging.py builds
#    its VALID_ACTIVITIES from this list)
#
# Tags are designed to be:
# - Visually distinguishable in video frames (the VLM can classify them)
# - Textually matchable in captions (regex can find them)
# - Non-overlapping (no synonym pairs like "jogging" vs "running")
# ==============================================================================

# Canonical set of activity tags — used by both caption extraction and VLM
# tagging. Composition scores a (caption, background) pair by the overlap
# between the caption's tags and the background's ml_tags["activities"].
UNIFIED_ACTIVITY_TAGS = [
    "cooking",      # at the stove: stirring, simmering, chopping, seasoning
    "baking",       # oven work: dough, batter, loaves, cookies
    "eating",       # tasting, plating, digging in
    "running",      # jogging, sprinting, treadmill, race distances
    "lifting",      # barbells, dumbbells, squats, reps
    "stretching",   # mobility, warm-up / cool-down, foam rolling
    "yoga",         # mat work, breathwork, meditation
    "climbing",     # bouldering, rope climbing, the wall
    "cycling",      # road / trail bikes, pedalling
    "swimming",     # pool laps, open water
    "hiking",       # trails, ridges, switchbacks
    "walking",      # strolling, on foot
    "driving",      # highway, road trip, van life
    "flying",       # flights, airports, take-off
    "talking",      # dialogue, conversation
    "dancing",      # choreography, moving to a beat
    "working",      # desk, laptop, office, shifts
    "reading",      # books, chapters
    "relaxing",     # resting, unwinding, slowing down
    "sightseeing",  # landmarks, exploring, wandering a city
    "camping",      # tents, campfires, nights outdoors
    "posing",       # flexing, progress pics, mirror shots
]

# Regex patterns that map caption text → unified activity tags.
# Format: tag_name -> list of regex patterns that indicate this activity.
ACTIVITY_TAG_PATTERNS = {
    "cooking": [
        r"\bcook(?:s|ed|ing)?\b", r"\bstir(?:s|red|ring)?\b", r"\bsimmer(?:s|ed|ing)?\b",
        r"\bsaut[eé](?:s|d|ing)?\b", r"\bchop(?:s|ped|ping)?\b", r"\bsear(?:s|ed|ing)?\b",
        r"\bthe (?:pan|skillet|stove|wok|burner)\b", r"\brecipe\b", r"\bseason(?:s|ed|ing)?\b",
        r"\bgarlic\b", r"\bonions?\b", r"\bsauce\b",
    ],
    "baking": [
        r"\bbak(?:e|es|ed|ing)\b", r"\bthe oven\b", r"\bdough\b", r"\bbatter\b",
        r"\bloa(?:f|ves)\b", r"\bsourdough\b", r"\bproof(?:s|ed|ing)?\b",
        r"\bcookies?\b", r"\bcakes?\b", r"\bpastry\b",
    ],
    "eating": [
        r"\beat(?:s|ing)?\b", r"\bfirst bite\b", r"\btast(?:e|es|ed|ing)\b",
        r"\bplat(?:e|es|ed|ing)\b", r"\bdig in\b", r"\bserv(?:e|es|ed|ing)\b",
    ],
    "running": [
        r"\brun(?:s|ning)?\b", r"\bjog(?:s|ged|ging)?\b", r"\bsprint(?:s|ed|ing)?\b",
        r"\bmiles?\b", r"\b5k\b", r"\b10k\b", r"\bmarathon\b", r"\btreadmill\b",
        r"\bstride\b", r"\bsplits?\b",
    ],
    "lifting": [
        r"\blift(?:s|ed|ing)?\b", r"\bdeadlift", r"\bsquat(?:s|ted|ting)?\b",
        r"\bbench(?:ed|ing)?\b", r"\bbarbell\b", r"\bdumbbells?\b", r"\breps?\b",
        r"\bpersonal record\b", r"\brack\b",
    ],
    "stretching": [
        r"\bstretch(?:es|ed|ing)?\b", r"\bmobility\b", r"\bfoam roll",
        r"\bwarm[\s-]?up\b", r"\bcool[\s-]?down\b", r"\bhamstrings?\b",
    ],
    "yoga": [
        r"\byoga\b", r"\bthe mat\b", r"\bdownward dog\b", r"\bsun salutation\b",
        r"\bbreathwork\b", r"\bmeditat(?:e|es|ed|ing|ion)\b",
    ],
    "climbing": [
        r"\bclimb(?:s|ed|ing)?\b", r"\bboulder(?:s|ed|ing)?\b", r"\bbelay\b",
        r"\bcrimp\b", r"\bthe wall\b", r"\bchalk\b",
    ],
    "cycling": [
        r"\bcycl(?:e|es|ed|ing)\b", r"\bbik(?:e|es|ed|ing)\b", r"\bpedal(?:s|ed|ing)?\b",
        r"\bsaddle\b", r"\bcadence\b",
    ],
    "swimming": [
        r"\bswim(?:s|ming)?\b", r"\blaps?\b", r"\bthe pool\b", r"\bfreestyle\b",
        r"\bopen water\b",
    ],
    "hiking": [
        r"\bhik(?:e|es|ed|ing)\b", r"\btrails?\b", r"\btrailhead\b", r"\bridge\b",
        r"\bswitchbacks?\b", r"\bsummit\b",
    ],
    "walking": [
        r"\bwalk(?:s|ed|ing)?\b", r"\bstroll(?:s|ed|ing)?\b", r"\bon foot\b",
    ],
    "driving": [
        r"\bdriv(?:e|es|ing)\b", r"\bdrove\b", r"\bhighway\b", r"\bthe road\b",
        r"\bsteering wheel\b", r"\bgas station\b", r"\broad trip\b", r"\bthe van\b",
    ],
    "flying": [
        r"\bfl(?:y|ies|ew|ying)\b", r"\bflight\b", r"\bplane\b", r"\bairport\b",
        r"\btake[\s-]?off\b", r"\bboarding\b", r"\bwindow seat\b",
    ],
    "talking": [
        r"\btalk(?:s|ed|ing)?\b", r"\bconversation\b", r"\bask(?:s|ed|ing)?\b",
        r"\btell(?:s|ing)?\b", r"\btold\b",
    ],
    "dancing": [
        r"\bdanc(?:e|es|ed|ing)\b", r"\bchoreograph", r"\bthe beat\b",
    ],
    "working": [
        r"\bdesk\b", r"\blaptop\b", r"\bthe office\b", r"\bshift\b",
        r"\bdeadline\b", r"\bemails?\b", r"\bspreadsheet\b",
    ],
    "reading": [
        r"\bread(?:s|ing)?\b", r"\bbooks?\b", r"\bchapter\b",
    ],
    "relaxing": [
        r"\brelax(?:es|ed|ing)?\b", r"\brest(?:s|ed|ing)?\b", r"\bunwind(?:s|ing)?\b",
        r"\bslow down\b", r"\bbreathe\b",
    ],
    "sightseeing": [
        r"\bsightsee", r"\blandmarks?\b", r"\bexplor(?:e|es|ed|ing)\b",
        r"\btour(?:s|ed|ing)?\b", r"\bwander(?:s|ed|ing)?\b", r"\bold town\b",
        r"\bmuseum\b", r"\bskyline\b",
    ],
    "camping": [
        r"\bcamp(?:s|ed|ing)?\b", r"\btent\b", r"\bcampfire\b", r"\bsleeping bag\b",
        r"\bunder the stars\b",
    ],
    "posing": [
        r"\bpos(?:e|es|ed|ing)\b", r"\bflex(?:es|ed|ing)?\b", r"\bmirror selfie\b",
        r"\bprogress pic\b",
    ],
}

# Compile patterns for efficiency
COMPILED_TAG_PATTERNS = {
    tag: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for tag, patterns in ACTIVITY_TAG_PATTERNS.items()
}


def extract_tags(text: str) -> List[str]:
    """
    Extract activity tags from caption text.

    Scans the text for keywords and phrases that map onto the unified
    activity vocabulary (UNIFIED_ACTIVITY_TAGS). Composition uses the
    overlap between these tags and a background's ml_tags["activities"]
    to pick a fitting clip.

    Args:
        text: Caption text to analyze

    Returns:
        List of matched tag names (lowercase, sorted)

    Example:
        >>> extract_tags("Tie your shoes before sunrise, run the loop twice, then stretch on the porch.")
        ['running', 'stretching']
    """
    if not text:
        return []

    matched_tags = set()
    text_lower = text.lower()

    for tag, patterns in COMPILED_TAG_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text_lower):
                matched_tags.add(tag)
                break  # Found a match for this tag, move to next tag

    result = sorted(list(matched_tags))

    if result:
        logger.debug(f"Extracted {len(result)} tags from caption: {result}")

    return result


# ==============================================================================
# LLM-BASED TAG EXTRACTION
# ==============================================================================
# Uses Mistral-7B to intelligently identify relevant tags from caption content.
# More accurate than regex for nuanced references and indirect descriptions.
# ==============================================================================

# All available tags for LLM to choose from (same as UNIFIED_ACTIVITY_TAGS)
AVAILABLE_TAGS = sorted(UNIFIED_ACTIVITY_TAGS)

# LLM tag extraction prompt template
TAG_EXTRACTION_PROMPT = """You are a tag extractor. Given a caption, identify which tags from the provided list apply to the content.

AVAILABLE TAGS:
{tags}

CAPTION:
{caption}

Return ONLY a JSON array of matching tags. Only include tags that clearly apply to the caption content.
If no tags match, return an empty array: []

Example output: ["running", "hiking"]

Your response (JSON array only):"""


def extract_tags_llm(text: str, model=None, tokenizer=None) -> List[str]:
    """
    Extract activity tags from caption text using LLM.

    Uses Mistral-7B to intelligently identify relevant tags based on
    semantic understanding rather than just keyword matching.

    Args:
        text: Caption text to analyze
        model: Pre-loaded LLM model (optional, will load if not provided)
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided)

    Returns:
        List of matched tag names (lowercase, sorted)
    """
    import json
    import torch

    if not text or len(text.strip()) < 10:
        return []

    # Load model if not provided
    if model is None or tokenizer is None:
        try:
            from scrapers.caption_postprocessor import _load_llm
            model, tokenizer = _load_llm(force=True)
            if model is None:
                logger.warning("LLM not available, falling back to regex extraction")
                return extract_tags(text)
        except Exception as e:
            logger.warning(f"Failed to load LLM for tag extraction: {e}")
            return extract_tags(text)

    try:
        # Build the prompt
        tags_list = ", ".join(AVAILABLE_TAGS)
        prompt = TAG_EXTRACTION_PROMPT.format(tags=tags_list, caption=text[:1500])  # Limit caption length

        # Format for Mistral instruct
        formatted_prompt = f"[INST] {prompt} [/INST]"

        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Tags list is short
                temperature=0.1,  # Low temperature for consistent output
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Parse JSON array from response
        # Try to extract JSON array even if there's extra text
        start_idx = response.find('[')
        end_idx = response.rfind(']')

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx + 1]
            tags = json.loads(json_str)

            # Validate tags are in our available list
            valid_tags = [t.lower() for t in tags if isinstance(t, str) and t.lower() in [a.lower() for a in AVAILABLE_TAGS]]
            result = sorted(list(set(valid_tags)))

            logger.debug(f"LLM extracted {len(result)} tags: {result}")
            return result
        else:
            logger.warning(f"LLM response not valid JSON array: {response[:200]}")
            # Fallback to regex
            return extract_tags(text)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM tag response as JSON: {e}")
        return extract_tags(text)
    except Exception as e:
        logger.error(f"Error in LLM tag extraction: {e}")
        return extract_tags(text)


def extract_tags_smart(text: str, use_llm: bool = True, model=None, tokenizer=None) -> List[str]:
    """
    Smart tag extraction that uses LLM when available, falls back to regex.

    Args:
        text: Caption text to analyze
        use_llm: Whether to attempt LLM-based extraction
        model: Pre-loaded LLM model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        List of matched tag names (lowercase, sorted)
    """
    if use_llm:
        return extract_tags_llm(text, model, tokenizer)
    return extract_tags(text)


# ==============================================================================
# LLM-BASED QUALITY SCORING
# ==============================================================================
# Uses Mistral-7B to score caption quality on multiple criteria.
# Returns a score from 1-100 based on technical quality, narrative, and engagement.
# ==============================================================================

# Stricter scoring prompt with spelled-out penalties
QUALITY_SCORING_PROMPT = """You are a STRICT caption quality evaluator. Score the following caption on a scale from 1 to 100.

SCORING CRITERIA:

1. **Technical Quality** (0-40 points):
   - Correct grammar, spelling, punctuation (questions MUST end with ?)
   - Proper sentence structure and formatting
   - NO ALL CAPS shouting (instant -15 points if present)
   - NO gibberish or nonsensical text (instant -20 points)
   - Clean formatting without artifacts

2. **Narrative Quality** (0-35 points):
   - Complete story with beginning, middle, and END
   - Logical flow that makes sense
   - NO abrupt endings mid-sentence (instant -10 points)
   - Characters and scenario are clear
   - Satisfying conclusion or cliffhanger (not just cut off)

3. **Engagement** (0-25 points):
   - Interesting hook that draws reader in
   - Emotional impact and immersion
   - Creative scenario (not generic/formulaic)
   - Appropriate pacing

AUTOMATIC DEDUCTIONS:
- ALL CAPS words (except emphasis): -15 points
- Missing question marks on questions: -5 points each
- Incomplete/cut-off ending: -10 points
- Gibberish or nonsense text: -20 points
- Repetitive phrases: -10 points

BE STRICT. Most captions should score 60-80. Only exceptional captions score 90+.
A score of 100 means PERFECT with zero issues.

CAPTION:
{caption}

Evaluate strictly. Your response MUST end with:
SCORE: [number]

Your evaluation:"""


def score_caption_llm(text: str, model=None, tokenizer=None) -> Optional[float]:
    """
    Score a caption's quality using LLM evaluation.

    Evaluates the caption on:
    - Grammar, spelling, sentence structure (0-30 points)
    - Engagement and appeal (0-35 points)
    - Story clarity and flow (0-25 points)

    Args:
        text: Caption text to score
        model: Pre-loaded LLM model (optional, will load if not provided)
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided)

    Returns:
        Quality score from 1-100, or None if scoring fails
    """
    import re
    import torch

    if not text or len(text.strip()) < 20:
        return None

    # Load model if not provided
    if model is None or tokenizer is None:
        try:
            from scrapers.caption_postprocessor import _load_llm
            model, tokenizer = _load_llm(force=True)
            if model is None:
                logger.warning("LLM not available for quality scoring")
                return None
        except Exception as e:
            logger.warning(f"Failed to load LLM for quality scoring: {e}")
            return None

    try:
        # Build the prompt
        prompt = QUALITY_SCORING_PROMPT.format(caption=text[:2000])  # Limit caption length

        # Format for Mistral instruct
        formatted_prompt = f"[INST] {prompt} [/INST]"

        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,  # Allow room for evaluation text
                temperature=0.3,  # Moderate temperature for consistent scoring
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Extract score from response
        # Look for "SCORE: XX" pattern
        score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            # Clamp to valid range
            score = max(1, min(100, score))
            logger.debug(f"LLM scored caption: {score}/100")
            return float(score)

        # Fallback: try to find any number at the end
        number_match = re.search(r'\b(\d{1,3})\b\s*$', response)
        if number_match:
            score = int(number_match.group(1))
            if 1 <= score <= 100:
                logger.debug(f"LLM scored caption (fallback): {score}/100")
                return float(score)

        logger.warning(f"Could not extract score from LLM response: {response[-200:]}")
        return None

    except Exception as e:
        logger.error(f"Error in LLM quality scoring: {e}")
        return None


def score_caption_batch(captions: list, model=None, tokenizer=None) -> dict:
    """
    Score multiple captions efficiently by reusing the loaded model.

    Args:
        captions: List of (caption_id, caption_text) tuples
        model: Pre-loaded LLM model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        Dict mapping caption_id -> score (or None if scoring failed)
    """
    # Load model once
    if model is None or tokenizer is None:
        try:
            from scrapers.caption_postprocessor import _load_llm
            model, tokenizer = _load_llm(force=True)
            if model is None:
                logger.warning("LLM not available for batch scoring")
                return {cid: None for cid, _ in captions}
        except Exception as e:
            logger.warning(f"Failed to load LLM for batch scoring: {e}")
            return {cid: None for cid, _ in captions}

    results = {}
    for caption_id, text in captions:
        score = score_caption_llm(text, model, tokenizer)
        results[caption_id] = score

    return results


# ==============================================================================
# IMPROVED SCORING SYSTEM
# ==============================================================================
# Combines pre-rejection filtering, LLM scoring, and rule-based deductions
# for more accurate quality assessment.
# ==============================================================================

# Literary-register flag words. When the LLM gets sampled too liberally
# (high temp / wide top_p / aggressive repetition_penalty) it falls out of
# the niche LoRA's plain Reddit register into base Mistral's "novel"
# register and starts emitting greeting-card prose: "a lazy stripe of
# sunlight", "trembling fingers", "a symphony of flavours", etc. Any single
# one of these in isolation is OK, but if a caption hits 2+ of them it's
# mode-shifted and needs to be re-sampled. The threshold is 2 because a
# niche LoRA does occasionally use one of these phrases legitimately, but
# plain Reddit captions almost never stack them.
LITERARY_FLAG_PATTERNS = [
    r'\b(blurs?|blurring)\s+into\b',          # "blurs into darkness"
    r'\bwhispers?\s+across\b',                 # "whispers across the valley"
    r'\bcascad(e|ing|es)\b',                   # "light cascading"
    r'\bglistens?\b|\bglistening\b',           # "glistening" anything
    r'\bshimmer(s|ing|ed)?\b',                  # "shimmering" anything
    r'\btrembling\s+(hands|voice)\b',
    r'\bdelicate\s+(fingers|dance|balance|patterns)\b',
    r'\bpracticed\s+(grace|ease|skill|pace)\b',
    r'\blazy\s+stripe\b|\blazy\s+afternoon\b',
    r'\byour\s+reflection\s+(catches|shows)\b',
    r'\bdances?\s+(across|over)\b',            # "shadows dance across"
    r'\bunder\s+the\s+harsh\s+glare\b',
    r'\bsavor\s+every\b',
    r'\bpainted\s+(on|across|with)\b',
    r'\bin\s+practiced\b',
    r'\bunder\s+the\s+moonlight\b',
    r'\bclock\s+(strikes|ticks)\s+(midnight|past|loudly)\b',
    r'\bsink\s+into\s+the\s+moment\b',
    r'\bsoul\s+awakens?\b',
    r'\bgolden\s+hour\s+bathes\b',
    r'\bsymphony\s+of\b',
    r'\btapestry\s+of\b',
    r'\btestament\s+to\b',
    r'\bembark\s+on\s+a\s+journey\b',
    r'\bbeckons?\b',
]
LITERARY_FLAG_REGEX = re.compile('|'.join(LITERARY_FLAG_PATTERNS), re.IGNORECASE)
LITERARY_FLAG_THRESHOLD = 2  # 2+ literary flags in one caption = reject


# Scene-invention patterns. The VLM tags a background's setting / activities
# / subjects / mood / camera, but never specific colours, brands, place
# names, or clock times — so a caption that asserts "red jacket" or "at
# 5:03 AM" is claiming something the BG can't be matched against. Reject
# these and force the model back to generic terms ("your jacket", "before
# sunrise"). Mirrors the SCENE INVENTION rule in the Stage 3 judge prompt.
_COLOURS = r'(?:red|pink|black|white|blue|green|purple|yellow|orange|grey|gray|crimson|neon)'
SCENE_INVENTION_PATTERNS = [
    # colour + gear/clothing/prop
    rf'\b{_COLOURS}\s+(?:jacket|hoodie|shirt|shorts|leggings|sneakers|shoes|apron|mat|helmet|backpack|tent|van|car|truck|bike|kayak|mug|bowl|plate)s?\b',
    # brand names the footage may not show
    r'\b(?:peloton|nike|adidas|lululemon|garmin|le\s*creuset|kitchenaid|vitamix|yeti|tesla|jeep|subaru|airbnb)\b',
    # named destinations
    r'\b(?:yosemite|zion|patagonia|bali|santorini|iceland|kyoto|tokyo|paris|lisbon|banff|machu\s*picchu)\b',
    # exact clock times
    r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b',
]
SCENE_INVENTION_REGEX = re.compile('|'.join(SCENE_INVENTION_PATTERNS), re.IGNORECASE)


def should_reject_caption(text: str) -> tuple:
    """
    Pre-filter to reject obviously low-quality captions before LLM scoring.

    Args:
        text: Caption text to evaluate

    Returns:
        Tuple of (should_reject: bool, reason: str or None)
    """
    if not text or len(text.strip()) < 30:
        return True, "Too short (< 30 chars)"

    # Count ALL CAPS words (4+ letters)
    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    if len(caps_words) > 8:
        return True, f"Too many ALL CAPS words ({len(caps_words)})"

    # Check for frame delimiters from training data
    if '*|*' in text or '*||*' in text:
        return True, "Contains training frame delimiters"

    # Check for excessive repetition
    words = text.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return True, f"Too repetitive (unique ratio: {unique_ratio:.2f})"

    # Check for gibberish patterns (random consonant clusters)
    gibberish_patterns = [
        r'\b[bcdfghjklmnpqrstvwxz]{5,}\b',  # 5+ consonants in a row
        r'\bGON[A-Z]{3,}\b',  # GONEASTERN, GONNOLY, etc.
        r'\bWGNA\b',
        r'\bGANO[A-Z]*\b',
    ]
    for pattern in gibberish_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, f"Contains gibberish pattern: {pattern}"

    # Check for spam patterns. Reddit-style training-data spam often lands at
    # the END of a caption (story first, promo last). Reject anywhere now —
    # the LoRA learned these patterns from training data and can't be trusted
    # to keep them out. The patterns are specific enough that false positives
    # are rare.
    spam_match = SPAM_REGEX.search(text)
    if spam_match:
        return True, f"Contains spam/promotional pattern: {spam_match.group(0)[:60]}"

    # Literary/greeting-card-mode detection. A single literary phrase is fine
    # (real captions occasionally have one), but stacking 2+ means the model
    # has fallen out of the plain Reddit register and is writing florid prose.
    literary_hits = [m.group(0) for m in LITERARY_FLAG_REGEX.finditer(text)]
    if len(literary_hits) >= LITERARY_FLAG_THRESHOLD:
        return True, f"Too many literary-register phrases ({len(literary_hits)}): {literary_hits[:3]}"

    # Scene invention. The BG tagger doesn't record colours / brands / place
    # names / clock times, so any caption asserting them can't be matched.
    # Force the model back to generic terms.
    invention_match = SCENE_INVENTION_REGEX.search(text)
    if invention_match:
        return True, f"Invents scene specifics ({invention_match.group(0)}); BG cannot be matched"

    return False, None


def apply_rule_based_deductions(text: str, base_score: float) -> tuple:
    """
    Apply rule-based deductions to LLM score for specific quality issues.

    Args:
        text: Caption text
        base_score: Initial score from LLM

    Returns:
        Tuple of (adjusted_score: float, deductions: list of (issue, points))
    """
    deductions = []
    score = base_score

    # 1. Missing question marks (-5 each, max -15)
    # Find sentences that look like questions but don't end with ?
    question_words = ['how', 'what', 'why', 'where', 'when', 'who', 'which',
                      'are you', 'do you', 'did you', 'have you', 'can you',
                      'will you', 'would you', 'could you', 'is it', 'is this',
                      'is that', 'was it', 'were you']

    sentences = re.split(r'(?<=[.!?])\s+', text)
    missing_q_count = 0
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        # Check if starts with question word but ends with period
        if any(sentence_lower.startswith(qw) for qw in question_words):
            if sentence.rstrip().endswith('.'):
                missing_q_count += 1

    if missing_q_count > 0:
        penalty = min(15, missing_q_count * 5)
        deductions.append((f"Missing ? on {missing_q_count} question(s)", penalty))
        score -= penalty

    # 2. ALL CAPS words (-5 each, max -20)
    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    if caps_words:
        penalty = min(20, len(caps_words) * 5)
        deductions.append((f"ALL CAPS words ({len(caps_words)})", penalty))
        score -= penalty

    # 2b. Word-by-word fragmentation (-15)
    # Catches OCR-style patterns like "One. More. Rep. You. Can."
    frag_matches = re.findall(r'(?:\b\w{1,6}\.\s+){3,}', text)
    if frag_matches:
        deductions.append(("Word-by-word fragmentation (OCR artifact)", 15))
        score -= 15

    # 3. Incomplete ending (-10)
    text_stripped = text.rstrip()
    if text_stripped and text_stripped[-1] not in '.!?"\'':
        deductions.append(("Incomplete ending (no punctuation)", 10))
        score -= 10
    elif text_stripped.endswith(','):
        deductions.append(("Ends with comma (cut off)", 8))
        score -= 8

    # 4. Emoticons/informal text (-10)
    if re.search(r';[\)\(]{1,}|:\)|:\(|:D|<3|\blol\b|\bhaha\b', text, re.IGNORECASE):
        deductions.append(("Contains emoticons/informal text", 10))
        score -= 10

    # 4b. Unicode emojis (-10)
    if re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF\U00002702-\U000027B0]', text):
        deductions.append(("Contains unicode emojis", 10))
        score -= 10

    # 5. Starts with quote + newline formatting issue (-5)
    if text.startswith('"') and len(text) > 2 and text[1:3] in ['\n', '\r\n']:
        deductions.append(("Poor formatting (starts with quote+newline)", 5))
        score -= 5

    # 6. Very short captions (-5 to -15)
    if len(text) < 100:
        deductions.append(("Very short caption", 10))
        score -= 10
    elif len(text) < 150:
        deductions.append(("Short caption", 5))
        score -= 5

    # 7. Frame delimiters still present (-10)
    if '*|*' in text or '*||*' in text:
        deductions.append(("Contains frame delimiters", 10))
        score -= 10

    # 8. Spam/promotional content (-10 to -20)
    spam_matches = list(SPAM_REGEX.finditer(text))
    if spam_matches:
        # Spam in first 30% is worse (-20), later spam is less severe (-10)
        earliest_pos = min(m.start() for m in spam_matches) / max(len(text), 1)
        if earliest_pos < 0.3:
            deductions.append((f"Early spam/promotional content ({len(spam_matches)} match(es))", 20))
            score -= 20
        else:
            deductions.append((f"Spam/promotional content ({len(spam_matches)} match(es))", 10))
            score -= 10

    # 9. Training data contamination patterns (-15)
    contamination_patterns = [
        (r'\bcaptions?\s*:\s*\d+\s*/\s*\d+', "Slide numbering (e.g. 'Captions: 07/14')"),
        (r'\bcaptions?\s*:\s*\d+-\d+\s*words', "Prompt leakage (word count instruction)"),
        (r'\bcaptions?\s*:\s*\d+-\d+\s*words.*MAX\)', "Prompt leakage (MAX constraint)"),
        (r'\bread all (?:the )?text in (?:the )?image', "OCR instruction leakage"),
        (r'\d+\+?\s*captions?\s+(?:on|at|available)', "Caption count reference"),
        (r'\bcaptions?\s*by\s*\.?\s*$', "Empty creator attribution (watermark)"),
        (r'\bcaptions?\s*by\s+\w+', "Creator attribution"),
        (r'\bnew ones? added daily\b', "Update schedule text"),
        (r'\bwant the full story\b', "Teaser/paywall text"),
        (r'\bcom/\w{3,}', "URL fragment"),
        (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', "Date stamp"),
    ]
    for pattern, description in contamination_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            deductions.append((f"Training data contamination: {description}", 15))
            score -= 15
            break  # Only penalize once for contamination

    # Clamp score to valid range
    score = max(0, min(100, score))

    return score, deductions


def score_caption_strict(text: str, model=None, tokenizer=None) -> dict:
    """
    Comprehensive caption scoring with pre-filtering and rule-based adjustments.

    This is the recommended scoring function that combines:
    1. Pre-rejection filter for obvious garbage
    2. LLM-based scoring with stricter prompt
    3. Rule-based deductions for specific issues

    Args:
        text: Caption text to score
        model: Pre-loaded LLM model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        Dict with:
        - rejected: bool - Whether caption was pre-rejected
        - reject_reason: str or None - Reason for rejection
        - llm_score: float or None - Raw LLM score
        - final_score: float - Final adjusted score (0 if rejected)
        - deductions: list - List of (issue, points) deductions applied
    """
    result = {
        'rejected': False,
        'reject_reason': None,
        'llm_score': None,
        'final_score': 0.0,
        'deductions': []
    }

    # Step 1: Pre-rejection filter
    should_reject, reason = should_reject_caption(text)
    if should_reject:
        result['rejected'] = True
        result['reject_reason'] = reason
        logger.debug(f"Caption rejected: {reason}")
        return result

    # Step 2: LLM scoring
    llm_score = score_caption_llm(text, model, tokenizer)
    if llm_score is None:
        # LLM failed, use simple rule-based score as fallback
        llm_score = calculate_quality_score(text) * 100

    result['llm_score'] = llm_score

    # Step 3: Apply rule-based deductions
    final_score, deductions = apply_rule_based_deductions(text, llm_score)
    result['final_score'] = final_score
    result['deductions'] = deductions

    if deductions:
        total_deduction = sum(d[1] for d in deductions)
        logger.debug(f"Score adjusted: {llm_score} -> {final_score} (-{total_deduction} from {len(deductions)} issues)")

    return result


def score_caption_batch_strict(captions: list, model=None, tokenizer=None) -> dict:
    """
    Score multiple captions using the strict scoring system.

    Args:
        captions: List of (caption_id, caption_text) tuples
        model: Pre-loaded LLM model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        Dict mapping caption_id -> score result dict
    """
    # Load model once for efficiency
    if model is None or tokenizer is None:
        try:
            from scrapers.caption_postprocessor import _load_llm
            model, tokenizer = _load_llm(force=True)
        except Exception as e:
            logger.warning(f"Failed to load LLM for batch scoring: {e}")
            model, tokenizer = None, None

    results = {}
    rejected_count = 0

    for caption_id, text in captions:
        result = score_caption_strict(text, model, tokenizer)
        results[caption_id] = result

        if result['rejected']:
            rejected_count += 1

    if rejected_count > 0:
        logger.info(f"Batch scoring: {rejected_count}/{len(captions)} captions pre-rejected")

    return results


def get_final_score(text: str, model=None, tokenizer=None) -> float:
    """
    Convenience function to get just the final score.

    Args:
        text: Caption text to score
        model: Pre-loaded LLM model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        Final quality score (0-100), 0 if rejected
    """
    result = score_caption_strict(text, model, tokenizer)
    return result['final_score']


# ==============================================================================
# REDDIT TITLE GENERATION
# ==============================================================================
# Generates catchy, short titles for Reddit posts based on caption content.
# Titles should be engaging, safe for any subreddit, and < 300 chars.
# ==============================================================================

TITLE_GENERATION_PROMPT = """Generate a short, catchy Reddit post title for this video caption.

RULES:
- Title MUST be under 150 characters (shorter is better)
- Be punchy and curiosity-driven, make people want to click
- Capture the main theme or scenario
- Use first person ("I", "my") or second person ("you", "your") perspective
- Do NOT use hashtags, emojis, or ALL CAPS
- Do NOT include promotional text or links
- Do NOT use generic titles like "Great caption" or "Check this out"
- Can use ellipsis (...) to create intrigue

EXAMPLES OF GOOD TITLES:
- "The 5 AM alarm hits different when you know why you set it"
- "What finally fixed my sourdough crust"
- "Nobody tells you this about your first solo trip..."
- "The rep you skip is the one you needed"
- "I stopped waiting for motivation and this is what happened"

CAPTION:
{caption}

Generate ONE title only. Output ONLY the title text, nothing else.

TITLE:"""


def generate_title(caption_text: str, model=None, tokenizer=None) -> Optional[str]:
    """
    Generate a catchy Reddit post title for a caption.

    Uses the same LLM that generated the caption to create an appropriate,
    engaging title that captures the essence of the content.

    Args:
        caption_text: The full caption text to summarize
        model: Pre-loaded LLM model (optional, will load if not provided)
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided)

    Returns:
        Generated title string (< 300 chars), or None if generation fails
    """
    import torch

    if not caption_text or len(caption_text.strip()) < 20:
        return None

    # Load model if not provided
    if model is None or tokenizer is None:
        try:
            from scrapers.caption_postprocessor import _load_llm
            model, tokenizer = _load_llm(force=True)
            if model is None:
                logger.warning("LLM not available for title generation")
                return None
        except Exception as e:
            logger.warning(f"Failed to load LLM for title generation: {e}")
            return None

    try:
        # Build the prompt with truncated caption
        prompt = TITLE_GENERATION_PROMPT.format(caption=caption_text[:1500])

        # Format for Mistral instruct
        formatted_prompt = f"[INST] {prompt} [/INST]"

        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,  # Titles should be short
                temperature=0.7,  # Some creativity
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        title = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean up the title
        title = clean_generated_title(title)

        if title and 10 <= len(title) <= 300:
            logger.debug(f"Generated title ({len(title)} chars): {title[:50]}...")
            return title
        else:
            logger.warning(f"Generated title invalid length ({len(title) if title else 0}): {title[:100] if title else 'empty'}")
            return None

    except Exception as e:
        logger.error(f"Error in title generation: {e}")
        return None


def clean_generated_title(title: str) -> str:
    """
    Clean up a generated title by removing artifacts and ensuring quality.

    Args:
        title: Raw generated title

    Returns:
        Cleaned title string
    """
    if not title:
        return ""

    # Remove common LLM output artifacts
    title = title.strip()

    # Remove OCR frame delimiters (*|*) - take only first segment as title
    if '*|*' in title:
        segments = [s.strip() for s in title.split('*|*') if s.strip()]
        # Use first meaningful segment (skip if too short)
        for seg in segments:
            if len(seg) >= 20:
                title = seg
                break
        else:
            title = segments[0] if segments else title

    # Remove quotes if the entire title is wrapped in them
    if (title.startswith('"') and title.endswith('"')) or \
       (title.startswith("'") and title.endswith("'")):
        title = title[1:-1].strip()

    # Remove "TITLE:" prefix if present
    title_lower = title.lower()
    for prefix in ['title:', 'title -', 'title=', 'reddit title:']:
        if title_lower.startswith(prefix):
            title = title[len(prefix):].strip()
            title_lower = title.lower()

    # Remove promotional/spam patterns (Patreon, Ko-fi, link-in-bio, etc.)
    spam_patterns = [
        r'(?:find|get|see)\s+(?:more|over)?\s*\d*\s*captions?\s+(?:on|at)\s+patreon',
        r'patreon\.?\s*com\s*/?\s*\w+',
        r'ko-?fi\.?\s*com\s*/?\s*\w+',
        r'join\s+(?:my\s+)?(?:patreon|discord|newsletter)',
        r'link\s+in\s+bio',
        r'follow\s+(?:me\s+)?(?:on|at|@)',
        r'custom\s+captions?\s+(?:based\s+on|from)\s+(?:requests?|patrons?)',
        r'buy\s+me\s+a\s+coffee',
    ]
    for pattern in spam_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()

    # Remove newlines and extra whitespace
    title = ' '.join(title.split())

    # Remove hashtags
    title = re.sub(r'#\w+', '', title).strip()

    # Remove emojis (common unicode ranges)
    title = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]', '', title).strip()

    # Remove promotional patterns
    title = re.sub(r'\[link.*?\]', '', title, flags=re.IGNORECASE).strip()
    title = re.sub(r'\(link.*?\)', '', title, flags=re.IGNORECASE).strip()

    # Clean up trailing/leading punctuation that might be left over
    title = title.strip(' .,;:-')

    # Truncate to max 80 chars with intelligent boundary detection
    MAX_TITLE_LENGTH = 80

    if len(title) > MAX_TITLE_LENGTH:
        # Strategy 1: Try to find a natural sentence boundary (. ? !) within limit
        truncated = title[:MAX_TITLE_LENGTH]

        # Look for last sentence-ending punctuation
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')

        # Find the best natural break point (closest to the limit)
        best_break = max(last_period, last_question, last_exclaim)

        if best_break > MAX_TITLE_LENGTH * 0.5:  # At least 50% of max length
            # Use the natural sentence break
            title = title[:best_break + 1].strip()
        else:
            # Strategy 2: No good sentence break, truncate at word boundary with ellipsis
            last_space = truncated.rfind(' ')
            if last_space > MAX_TITLE_LENGTH * 0.6:  # At least 60% to avoid very short titles
                title = truncated[:last_space].rstrip('.,!?;:- ') + "..."
            else:
                # Strategy 3: Just hard truncate with ellipsis
                title = truncated[:MAX_TITLE_LENGTH - 3].rstrip() + "..."

    # Final cleanup
    title = title.strip()

    return title


def generate_title_batch(captions: list, model=None, tokenizer=None) -> dict:
    """
    Generate titles for multiple captions efficiently.

    Args:
        captions: List of (caption_id, caption_text) tuples
        model: Pre-loaded LLM model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        Dict mapping caption_id -> generated title (or None if failed)
    """
    # Load model once for efficiency
    if model is None or tokenizer is None:
        try:
            from scrapers.caption_postprocessor import _load_llm
            model, tokenizer = _load_llm(force=True)
            if model is None:
                logger.warning("LLM not available for batch title generation")
                return {cid: None for cid, _ in captions}
        except Exception as e:
            logger.warning(f"Failed to load LLM for batch title generation: {e}")
            return {cid: None for cid, _ in captions}

    results = {}
    success_count = 0

    for caption_id, text in captions:
        title = generate_title(text, model, tokenizer)
        results[caption_id] = title
        if title:
            success_count += 1

    logger.info(f"Batch title generation: {success_count}/{len(captions)} successful")
    return results
