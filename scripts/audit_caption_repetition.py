"""
Quantify phrase-level repetition across the recently-generated caption pool.

For each niche:
  - Pull the last 30 days of caption_text from generated_captions
  - Tokenize, lowercase, strip punctuation
  - Count n-gram frequencies (3-, 4-, 5-, 6-word windows)
  - Surface the n-grams that appear in >= MIN_PCT of captions for that niche
    (these are the "every caption sounds the same" markers)

Usage (inside the api container):
    docker-compose exec -T api python scripts/audit_caption_repetition.py
"""
import re
import sys
from collections import Counter

sys.path.insert(0, "/app")

from sqlalchemy import text  # noqa: E402

from database.db import engine  # noqa: E402

# Knobs
DAYS_BACK = 30
MIN_NGRAM = 3
MAX_NGRAM = 6
TOP_N = 30                    # show top N hits per (niche, ngram-size)
MIN_PCT_THRESHOLD = 1.0       # only show n-grams in >= this % of captions

WORD_RE = re.compile(r"[a-z']+")


def tokenize(s: str) -> list[str]:
    return WORD_RE.findall(s.lower())


def ngram_counts(captions: list[str], n: int) -> tuple[Counter, int]:
    """Returns (Counter of ngram -> docs containing it, total docs)."""
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


def main() -> int:
    sql = text(
        """
        SELECT niche, caption_text
        FROM generated_captions
        WHERE generated_at > NOW() - INTERVAL ':days days'
          AND caption_text IS NOT NULL
          AND length(caption_text) > 20
        """.replace(":days", str(DAYS_BACK))
    )
    by_niche: dict[str, list[str]] = {}
    with engine.connect() as conn:
        for niche, caption in conn.execute(sql):
            by_niche.setdefault(niche or "(null)", []).append(caption)

    print(f"# Caption repetition audit (last {DAYS_BACK} days)\n")
    for niche, caps in sorted(by_niche.items(), key=lambda kv: -len(kv[1])):
        print(f"## {niche} — {len(caps)} captions\n")
        for n in range(MIN_NGRAM, MAX_NGRAM + 1):
            counts, total = ngram_counts(caps, n)
            if not total:
                continue
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            # Skip super-common stopword grams; keep ones that are in
            # >= MIN_PCT_THRESHOLD% of captions and aren't just function words.
            kept = []
            for gram, c in ranked:
                pct = 100.0 * c / total
                if pct < MIN_PCT_THRESHOLD:
                    break
                # Cheap stopword filter — skip pure function-word ngrams.
                toks = gram.split()
                content_toks = [t for t in toks if t not in {
                    "the","a","an","you","your","of","is","are","be","to",
                    "and","or","but","as","at","in","on","for","with","by",
                    "that","this","it","its","i","me","my","we","he","she",
                    "his","her","they","them","their","do","does","did",
                    "have","has","had","not","no","yes","s","t","m","re","ve",
                }]
                if not content_toks:
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
