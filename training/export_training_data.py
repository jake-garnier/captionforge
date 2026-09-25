"""
Export caption data from PostgreSQL to JSONL format for LLM training

Features:
- Cleans captions to remove spam, promotional content, OCR artifacts
- Removes frame delimiters (*|*) that shouldn't be in generated output
- Filters by minimum quality standards
- Exports both raw and cleaned versions for comparison
"""
import json
import os
import re
from database.db import get_db_context
from database.models import ScrapedCaption

# Spam patterns to remove from training data.
# These prevent creator promo / watermark text from contaminating LoRA
# fine-tuning. Kept generic (platform names, URL shapes, CTA phrasing,
# editor exports); new candidates surface via scripts/audit_watermark_phrases.py.
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
    r'[a-z]+\.com/',
    r'https?://\S+',
    r'\[link[^\]]*\]',
    r'link in bio',
    r'@[a-zA-Z0-9_]{3,}',  # Social media handles

    # === Promotional / Paywall ===
    r'\bjoin now\b',
    r'\bjoin free\b',
    r'\bjoin the\b',
    r'\bjoin today\b',
    r'\bjoin my\b',
    r'\bjoin \d+\+?\s*(fans|members|patrons|subscribers)',
    r'\bsubscribe\b',
    r'\bcheck out my pinned\b',
    r'\bhundreds of videos\b',
    r'\bthousands of videos\b',
    r'\bpremium members?\b',
    r'\bupgrade to premium\b',
    r'\bmembership renews?\b',
    r'\bpatrons only\b',
    r'\bdaily live shows?\b',
    r'\bcustom requests?\b',
    r'\bfull video and audio\b',
    r'\bfull videos? (&|and) audio\b',
    r'\bsource material\b',
    r'\bexclusive captions?\b',
    r'\bexclusive content\b',
    r'\bexclusive videos?\b',
    r'\bget full access\b',
    r'\baccess to (?:all|over|\d+)',
    r'\bunlock\s+\d+\+?\s*(?:exclusive|captions?|videos?)',
    r'\bmore captions?\s+(?:on|at)\b',
    r'\bfind\s+(?:more|over)\s+\d*\s*captions?\b',
    r'\b\d+\+?\s*(?:exclusive\s+)?captions?\s+(?:on|at)\b',
    r'\bpromo(?:tion(?:al)?)?\s*code\b',
    r'\bfree trial\b',
    r'\bsign up\b',
    r'\bclick (?:here|the link|below)\b',

    # === Creator Attributions ===
    r'\bcaption (?:by|from|made by|created by)\b',
    r'\bmade by\b',
    r'\bcreated by\b',
    r'\bedited\s*by\s*[:\-]?\s*\S+',
    r'\bcredits?\s*(?:to|:)\s*\S+',
    r'\b(?:ig|insta|tt|yt)\s*[:@]\s*\S+',

    # === Social CTAs ===
    r'\benjoyed?\s+this\s+caption\b',
    r'\bliked?\s+this\s+caption\b',
    r'\bfollow\s+(?:me|for|us)\b',
    r'\bfollow\s+(?:on|at|@)\b',
    r'\blike\s*(?:&|and)\s*(?:share|subscribe|follow)\b',
    r'\bpm\s+me\b',
    r'\bdm\s+me\b',

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

    # === URL-style watermarks ===
    # Generic shapes that catch common watermark forms without needing
    # per-creator maintenance. New candidates surface via
    # scripts/audit_watermark_phrases.py.
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
    r'(?:^|\s)\d+\+?\s*exclusive\s*captions?\s*(?:categories|stories)?',
    r'\bsubscribe\s+for\s+full\s+(?:videos?|captions?|content)\b',
]

SPAM_REGEX = re.compile('|'.join(SPAM_PATTERNS), re.IGNORECASE)


def clean_caption_for_training(text: str) -> str:
    """
    Clean caption text for training by removing spam and artifacts.

    Returns empty string if text is too contaminated to use.
    """
    if not text:
        return ""

    original = text

    # Step 1: Remove frame delimiters - join text naturally
    # Replace *|* with space (these are frame boundaries from OCR)
    text = text.replace(' *|* ', ' ')
    text = text.replace('*|*', ' ')

    # Step 2: Remove lines/sentences that contain spam
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if not SPAM_REGEX.search(line):
            clean_lines.append(line)
    text = '\n'.join(clean_lines)

    # Step 3: Also check sentence-level for inline spam
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = []
    for sentence in sentences:
        if not SPAM_REGEX.search(sentence):
            clean_sentences.append(sentence)
    text = ' '.join(clean_sentences)

    # Step 4: Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Step 5: Remove gibberish (repeated characters, nonsense)
    # Words with >4 repeated chars
    text = re.sub(r'\b\w*(.)\1{4,}\w*\b', '', text)
    # Clean up resulting double spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Step 6: Quality gate - reject if too short or too much was spam
    if len(text) < 25:
        return ""

    # If we removed more than 50% of the content, it's too contaminated
    if len(text) < len(original) * 0.5:
        return ""

    return text


def export_niche_training_data(
    niche: str,
    output_path: str,
    min_upvotes: int = 100,
    min_length: int = 100
) -> dict:
    """
    Export training data for a specific niche category.

    Args:
        niche: Niche category (e.g., "motivation", "fitness", "cooking", "travel")
        output_path: Path to write JSONL file
        min_upvotes: Minimum upvote threshold for quality filtering
        min_length: Minimum caption length in characters

    Returns:
        Dict with export statistics:
        {
            "niche": str,
            "total_found": int,
            "exported": int,
            "skipped_short": int,
            "skipped_spam": int,
            "output_path": str
        }
    """
    from config.automation_config import get_automation_config

    config = get_automation_config()
    subreddits = config.get_subreddits_for_niche(niche)

    if not subreddits:
        raise ValueError(f"Unknown niche: {niche}. Available: {list(config.niches.keys())}")

    stats = {
        "niche": niche,
        "subreddits": subreddits,
        "total_found": 0,
        "exported": 0,
        "skipped_short": 0,
        "skipped_spam": 0,
        "skipped_low_upvotes": 0,
        "skipped_rejected": 0,
        "output_path": output_path,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with get_db_context() as db:
        # Query captions from niche's subreddits with LLM-refined text
        # Exclude captions rejected during training review
        captions = db.query(ScrapedCaption).filter(
            ScrapedCaption.llm_refined_text != None,
            ScrapedCaption.llm_refined_text != "",
            ScrapedCaption.source_subreddit.in_(subreddits),
            ScrapedCaption.training_status != "rejected",
        ).all()

        stats["total_found"] = len(captions)

        with open(output_path, "w") as f:
            for caption in captions:
                # Use fixed text if caption was reviewed and corrected
                if caption.training_status == "fixed" and caption.training_fixed_text:
                    raw_text = caption.training_fixed_text.strip()
                else:
                    raw_text = caption.llm_refined_text.strip()
                upvotes = caption.upvotes or 0

                # Filter by upvotes
                if upvotes < min_upvotes:
                    stats["skipped_low_upvotes"] += 1
                    continue

                # Filter by length
                if len(raw_text) < min_length:
                    stats["skipped_short"] += 1
                    continue

                # Clean the text
                cleaned_text = clean_caption_for_training(raw_text)

                if not cleaned_text:
                    stats["skipped_spam"] += 1
                    continue

                # Write training example
                example = {
                    "id": caption.id,
                    "text": cleaned_text,
                    "upvotes": upvotes,
                    "subreddit": caption.source_subreddit,
                    "niche": niche,
                }
                f.write(json.dumps(example) + "\n")
                stats["exported"] += 1

    return stats


def get_niche_caption_count(niche: str, min_upvotes: int = 100) -> int:
    """
    Get count of available training captions for a niche.

    Args:
        niche: Niche category
        min_upvotes: Minimum upvote threshold

    Returns:
        Number of available captions
    """
    from config.automation_config import get_automation_config

    config = get_automation_config()
    subreddits = config.get_subreddits_for_niche(niche)

    if not subreddits:
        return 0

    with get_db_context() as db:
        count = db.query(ScrapedCaption).filter(
            ScrapedCaption.llm_refined_text != None,
            ScrapedCaption.llm_refined_text != "",
            ScrapedCaption.source_subreddit.in_(subreddits),
            ScrapedCaption.upvotes >= min_upvotes
        ).count()

    return count


def main():
    print("Exporting caption dataset from database...")
    print("Cleaning mode: ENABLED (removing spam/promotional content)")

    # Create output directory
    os.makedirs("training/data", exist_ok=True)

    stats = {
        "total": 0,
        "exported": 0,
        "skipped_short": 0,
        "skipped_spam": 0,
        "cleaned": 0,
    }

    with get_db_context() as db:
        # Get all captions with LLM-refined text
        captions = db.query(ScrapedCaption).filter(
            ScrapedCaption.llm_refined_text != None,
            ScrapedCaption.llm_refined_text != ""
        ).all()

        stats["total"] = len(captions)
        print(f"Found {len(captions)} captions with LLM-refined text")

        # Export to JSONL (cleaned version)
        output_path = "training/data/captions_all.jsonl"
        output_raw_path = "training/data/captions_raw.jsonl"

        with open(output_path, "w") as f_clean, open(output_raw_path, "w") as f_raw:
            for caption in captions:
                raw_text = caption.llm_refined_text.strip()

                # Skip very short captions
                if len(raw_text) < 100:
                    stats["skipped_short"] += 1
                    continue

                # Clean the text
                cleaned_text = clean_caption_for_training(raw_text)

                if not cleaned_text:
                    stats["skipped_spam"] += 1
                    continue

                # Track if cleaning changed anything
                if cleaned_text != raw_text:
                    stats["cleaned"] += 1

                # Write raw version (for comparison/debugging)
                raw_example = {
                    "id": caption.id,
                    "text": raw_text,
                    "upvotes": caption.upvotes or 0,
                    "subreddit": caption.source_subreddit,
                }
                f_raw.write(json.dumps(raw_example) + "\n")

                # Write cleaned version (for training)
                example = {
                    "id": caption.id,
                    "text": cleaned_text,
                    "upvotes": caption.upvotes or 0,
                    "subreddit": caption.source_subreddit,
                    "length": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                }
                f_clean.write(json.dumps(example) + "\n")
                stats["exported"] += 1

        print(f"\n✓ Exported cleaned data to {output_path}")
        print(f"✓ Exported raw data to {output_raw_path}")

        # Print statistics
        with open(output_path, "r") as f:
            data = [json.loads(line) for line in f]

        print(f"\n{'='*50}")
        print(f"Export Statistics:")
        print(f"{'='*50}")
        print(f"  Total in database: {stats['total']}")
        print(f"  Exported (clean):  {stats['exported']}")
        print(f"  Skipped (short):   {stats['skipped_short']}")
        print(f"  Skipped (spam):    {stats['skipped_spam']}")
        print(f"  Modified by cleaning: {stats['cleaned']}")
        print(f"\nDataset Statistics:")
        if data:
            print(f"  Total captions: {len(data)}")
            print(f"  Average length: {sum(d['length'] for d in data) / len(data):.0f} chars")
            print(f"  Average words: {sum(d['word_count'] for d in data) / len(data):.0f} words")
        else:
            print("  No data exported!")


if __name__ == "__main__":
    main()
