"""
Sample link flairs from each subreddit we crosspost to, across all four
niches, by pulling the 100 most-recent posts from the public Reddit JSON
listing endpoint (no auth required).

Run locally — no Docker, no creds:
    python3 scripts/list_subreddit_flairs.py

Output: markdown table to stdout. Copy/paste into docs/flairs_audit.md and
update PostponeConfig.subreddit_flairs in config/automation_config.py if any
flairs have changed.

Why not PRAW? `reddit_accounts` table is empty so there are no creds to use,
and Reddit's `/api/link_flair_v2.json` requires USER_REQUIRED auth. Sampling
real posts proxies the answer — flairs that appear on real posts are the
flairs the sub accepts.
"""
import json
import sys
import urllib.request
import urllib.error

SUBREDDITS_BY_NICHE = {
    "motivation": ["GetMotivated", "Motivation", "quotes", "DecidingToBeBetter"],
    "fitness": ["Fitness", "bodyweightfitness", "xxfitness", "running"],
    "cooking": ["Cooking", "recipes", "MealPrepSunday", "EatCheapAndHealthy"],
    "travel": ["travel", "solotravel", "digitalnomad", "backpacking"],
}

def fetch_flairs(sub: str, limit: int = 100):
    req = urllib.request.Request(
        f"https://www.reddit.com/r/{sub}/new.json?limit={limit}",
        headers={"User-Agent": "captionforge-flair-audit/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    posts = data.get("data", {}).get("children", [])
    counts: dict[str, int] = {}
    for p in posts:
        flair = p["data"].get("link_flair_text")
        if flair:
            counts[flair] = counts.get(flair, 0) + 1
    return len(posts), counts


def main() -> int:
    print("| Niche | Subreddit | Posts sampled | Flairs found (count) |")
    print("|---|---|---|---|")
    for niche, subs in SUBREDDITS_BY_NICHE.items():
        for sub in subs:
            try:
                n, flairs = fetch_flairs(sub)
            except urllib.error.HTTPError as e:
                print(f"| {niche} | r/{sub} | — | _HTTP {e.code}_ |")
                continue
            except Exception as e:  # noqa: BLE001
                print(f"| {niche} | r/{sub} | — | _{type(e).__name__}: {e}_ |")
                continue
            if not flairs:
                print(f"| {niche} | r/{sub} | {n} | _none — flair not used_ |")
                continue
            cell = "<br>".join(
                f"`{name}` ({count})"
                for name, count in sorted(flairs.items(), key=lambda x: -x[1])
            )
            print(f"| {niche} | r/{sub} | {n} | {cell} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
