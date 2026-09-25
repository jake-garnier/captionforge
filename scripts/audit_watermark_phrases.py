"""
Audit suspicious phrases in raw OCR text per niche to find candidate
watermark / creator patterns to add to:
  - training/export_training_data.py SPAM_PATTERNS
  - utils/generation_postprocessor.py SPAM_PATTERNS

Why raw OCR: the rule-based and LLM stages already strip many watermarks,
so pre-existing patterns under-count. Raw OCR is what we want to filter
when training, so it's where contamination is highest.

Strategy: for each niche (mapped via the subreddit list in
config/automation_config.py NicheConfig.subreddits, plus a hand-tuned
override below to catch the actual scrape sources), find n-gram phrases
that:
  - appear at least N times across all captions for that niche
  - look like brand/creator names (capitalized words, "captions" suffix,
    @handle markers, etc.) by passing one of several heuristic regexes
  - are not already covered by the existing SPAM patterns
The script then prints the top suspicious phrases per niche for human
triage.

Run inside the api container:
    docker-compose exec -T api python scripts/audit_watermark_phrases.py

Tunable knobs are at the top of the file.
"""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "/app")

from sqlalchemy import text  # noqa: E402

from database.db import engine  # noqa: E402

# Tunables
MIN_COUNT = 5  # phrase must appear at least this many times to surface
TOP_N = 60     # show this many top phrases per niche

# Subreddit -> niche, matching the subreddit→niche derivation we used
# elsewhere. We lowercase both sides at compare time.
NICHE_SUBREDDITS = {
    "motivation": ["getmotivated", "motivation", "quotes", "decidingtobebetter"],
    "fitness": ["fitness", "bodyweightfitness", "xxfitness", "running"],
    "cooking": ["cooking", "recipes", "mealprepsunday", "eatcheapandhealthy"],
    "travel": ["travel", "solotravel", "digitalnomad", "backpacking"],
}

# Heuristics: a phrase is "suspicious" if it matches any of these.
# Skewed toward false-positives — we triage by hand at the end.
SUSPICIOUS_PATTERNS = [
    re.compile(r"\b[A-Z][a-z]+\s+(clips?|edits?|studio|media|films?|productions?|creations?)\b", re.I),
    re.compile(r"\b(?:edited|filmed|made)\s+by\b", re.I),
    re.compile(r"@\w{3,}"),
    re.compile(r"\.com|\.net|\.io|\.tv|patreon|ko-?fi|telegram|tiktok|youtube|instagram|reddit\b", re.I),
    re.compile(r"\b[A-Z][A-Z]+[a-z]+|[a-z]+[A-Z][A-Z]+\b"),  # CamelCase-ish
    re.compile(r"\bsubscribe\b|\bfollow (?:me|us)\b|\blink in bio\b", re.I),
]

# Phrases we already filter — used to suppress already-known watermarks
# from the audit output so we focus on net-new candidates.
ALREADY_KNOWN = {
    p.lower()
    for p in [
        # editors (already in both SPAM_PATTERNS lists)
        "inshot", "clideo", "kapwing", "capcut", "kinemaster", "filmora",
        "canva", "picsart", "veed.io", "invideo", "caption maker",
        # stock-footage / licensing watermarks
        "shutterstock", "getty images", "pexels", "pixabay", "storyblocks",
        "stock footage",
        # platforms
        "patreon", "ko-fi", "buymeacoffee", "linktree", "tiktok", "youtube",
        "instagram", "twitch", "discord", "telegram",
    ]
}

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def is_known(phrase: str) -> bool:
    p = normalize(phrase)
    if p in ALREADY_KNOWN:
        return True
    # Substring match on multi-word watermarks
    for known in ALREADY_KNOWN:
        if known in p or p in known:
            return True
    return False


def is_suspicious(phrase: str) -> bool:
    return any(rx.search(phrase) for rx in SUSPICIOUS_PATTERNS)


def extract_phrases(text_blob: str, n_min: int = 2, n_max: int = 4):
    """Yield n-gram phrases of length 2..4 words from text."""
    # Tokenize on whitespace, keep punctuation attached so brand names like
    # "captions:by" surface intact.
    tokens = text_blob.split()
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if 4 <= len(phrase) <= 60:
                yield phrase


def audit_niche(niche: str, subreddits: list[str]) -> Counter:
    counts: Counter = Counter()
    sql = text(
        """
        SELECT raw_ocr_text
        FROM scraped_captions sc
        JOIN videos v ON v.id = sc.video_id
        WHERE LOWER(v.source_subreddit) = ANY(:subs)
          AND sc.raw_ocr_text IS NOT NULL
          AND length(sc.raw_ocr_text) > 0
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"subs": [s.lower() for s in subreddits]})
        for (raw,) in rows:
            for phrase in extract_phrases(raw):
                if not is_suspicious(phrase):
                    continue
                if is_known(phrase):
                    continue
                counts[normalize(phrase)] += 1
    return counts


def main() -> int:
    print("# Watermark audit — top suspicious phrases per niche\n")
    print(f"_Min count: {MIN_COUNT}, top {TOP_N} per niche, raw OCR text only._\n")
    for niche, subs in NICHE_SUBREDDITS.items():
        print(f"## {niche}\n")
        counts = audit_niche(niche, subs)
        ranked = [
            (phrase, count) for phrase, count in counts.most_common()
            if count >= MIN_COUNT
        ][:TOP_N]
        if not ranked:
            print("_(no candidates above threshold)_\n")
            continue
        print("| count | candidate phrase |")
        print("|---|---|")
        for phrase, count in ranked:
            print(f"| {count} | `{phrase}` |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
