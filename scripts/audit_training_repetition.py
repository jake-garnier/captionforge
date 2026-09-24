"""
Same shape as audit_caption_repetition.py but pulls the *training corpus* —
the LLM-refined captions in scraped_captions that the LoRA was/will be
trained on, filtered the same way export_training_data.py does.

Tells us whether the over-used phrases in generation are baked into the
training data, or are LoRA / decoding artifacts.

Usage:
    docker-compose exec -T api python scripts/audit_training_repetition.py
"""
import re
import sys
from collections import Counter

sys.path.insert(0, "/app")

from sqlalchemy import text  # noqa: E402

from database.db import engine  # noqa: E402

# Mirrors export_training_data.export_niche_training_data filters:
MIN_UPVOTES = 100
MIN_LENGTH = 100

# Match config.automation_config NICHE_CONFIGS subreddits
NICHE_SUBREDDITS = {
    "motivation": ["getmotivated", "motivation", "quotes", "decidingtobebetter"],
    "fitness": ["fitness", "bodyweightfitness", "xxfitness", "running"],
    "cooking": ["cooking", "recipes", "mealprepsunday", "eatcheapandhealthy"],
    "travel": ["travel", "solotravel", "digitalnomad", "backpacking"],
}

MIN_NGRAM = 3
MAX_NGRAM = 6
TOP_N = 30
MIN_PCT_THRESHOLD = 1.0

WORD_RE = re.compile(r"[a-z']+")


def tokenize(s: str) -> list[str]:
    return WORD_RE.findall(s.lower())


def ngram_counts(captions: list[str], n: int) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for cap in captions:
        toks = tokenize(cap)
        if len(toks) < n:
            continue
        total += 1
        seen = set()
        for i in range(len(toks) - n + 1):
            gram = " ".join(toks[i : i + n])
            if gram in seen:
                continue
            seen.add(gram)
            counts[gram] += 1
    return counts, total


STOPWORDS = {
    "the","a","an","you","your","of","is","are","be","to",
    "and","or","but","as","at","in","on","for","with","by",
    "that","this","it","its","i","me","my","we","he","she",
    "his","her","they","them","their","do","does","did",
    "have","has","had","not","no","yes","s","t","m","re","ve",
}


def main() -> int:
    print(f"# Training-corpus repetition audit\n")
    print(f"_min_upvotes={MIN_UPVOTES}, min_length={MIN_LENGTH}, training_status != rejected._\n")
    sql = text("""
        SELECT source_subreddit, llm_refined_text
        FROM scraped_captions
        WHERE llm_refined_text IS NOT NULL
          AND llm_refined_text != ''
          AND coalesce(training_status, '') != 'rejected'
          AND coalesce(upvotes, 0) >= :min_up
          AND length(llm_refined_text) >= :min_len
          AND lower(source_subreddit) = ANY(:subs)
    """)
    all_subs = [s.lower() for fl in NICHE_SUBREDDITS.values() for s in fl]
    by_niche: dict[str, list[str]] = {}
    sub_to_niche = {s.lower(): f for f, sl in NICHE_SUBREDDITS.items() for s in sl}
    with engine.connect() as conn:
        for sub, cap in conn.execute(sql, {"min_up": MIN_UPVOTES, "min_len": MIN_LENGTH, "subs": all_subs}):
            fet = sub_to_niche.get((sub or "").lower())
            if not fet:
                continue
            by_niche.setdefault(fet, []).append(cap)

    for niche, caps in sorted(by_niche.items(), key=lambda kv: -len(kv[1])):
        print(f"## {niche} — {len(caps)} training captions\n")
        for n in range(MIN_NGRAM, MAX_NGRAM + 1):
            counts, total = ngram_counts(caps, n)
            if not total:
                continue
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            kept = []
            for gram, c in ranked:
                pct = 100.0 * c / total
                if pct < MIN_PCT_THRESHOLD:
                    break
                content = [t for t in gram.split() if t not in STOPWORDS]
                if not content:
                    continue
                kept.append((gram, c, pct))
                if len(kept) >= TOP_N:
                    break
            if not kept:
                print(f"### {n}-grams: no hits above {MIN_PCT_THRESHOLD}%\n")
                continue
            print(f"### {n}-grams (top {len(kept)})\n")
            print("| count | % of captions | n-gram |")
            print("|---:|---:|---|")
            for gram, c, pct in kept:
                print(f"| {c} | {pct:.1f}% | `{gram}` |")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
