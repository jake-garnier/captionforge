"""
Niche rules for background video selection.

Two layers, both keyed by niche name:

1. NICHE_SUBREDDITS — allow-list of source subreddits per niche. A background
   video is eligible for a niche only if it was scraped from one of that
   niche's subreddits (stored on BackgroundVideo.searched_tag). This is the
   coarse gate; `is_subreddit_compatible` is what every legacy call site uses.

2. NICHE_TAG_RULES — required / forbidden / preferred conditions evaluated
   against the VLM scene tags on BackgroundVideo.ml_tags (schema documented
   in tasks/ml_tagging.py): subjects[].type, activities, setting, mood,
   camera, text_on_screen.
     - required:  EVERY clause must hold, otherwise the BG is rejected
     - forbidden: if ANY clause holds the BG is rejected
     - preferred: soft; each clause that holds adds one point to a ranking
                  score (see `score_tag_preferences`)

Clause format: a dict whose keys are all ANDed together. Supported keys:
     {"setting": ["kitchen", "home"]}             setting is in the list
     {"mood": ["calm", "inspiring"]}              mood is in the list
     {"camera": ["drone"]}                        camera is in the list
     {"activities_any": ["running", "lifting"]}   at least one activity in list
     {"subject_types_any": ["food"]}              at least one subject.type in list
     {"subject_types_none": ["person", "group"]}  no subject.type in list
     {"text_on_screen": False}                    exact boolean match
     {"any_of": [clause, clause, ...]}            OR over sub-clauses

Keep the rules small and legible. If a niche's matched BGs are wrong in a
specific way, the fix is usually one new clause here — not a bespoke filter
at a call site.
"""
from typing import Any, Dict, List, Optional, Tuple


# Map of niche -> list of allowed source subreddits for background matching.
# Names are illustrative; edit to taste when adding a niche.
NICHE_SUBREDDITS: Dict[str, List[str]] = {
    "motivation": [
        "naturegifs", "oddlysatisfying", "aviation", "hiking",
    ],
    "fitness": [
        "gymmotivation", "running", "climbing", "calisthenic",
    ],
    "cooking": [
        "gifrecipes", "foodvideos", "baking", "cooking",
    ],
    "travel": [
        "travelvideos", "dronevideos", "citybreaks", "roadtrip", "vandwellers",
    ],
}


# Activities that count as "a workout" for the fitness niche.
WORKOUT_ACTIVITIES: List[str] = [
    "running", "lifting", "stretching", "yoga", "climbing", "cycling",
    "swimming", "hiking",
]


# Per-niche tag rules. See the module docstring for clause semantics.
NICHE_TAG_RULES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "motivation": {
        # Quote overlays need a clean plate — any burned-in text competes
        # with the rendered caption.
        "required": [],
        "forbidden": [
            {"text_on_screen": True},
            {"mood": ["funny"]},
        ],
        "preferred": [
            {"mood": ["inspiring", "calm"]},
            {"subject_types_any": ["landscape"]},
            {"camera": ["drone", "timelapse"]},
            {"setting": ["outdoors"]},
        ],
    },
    "fitness": {
        "required": [
            {"activities_any": WORKOUT_ACTIVITIES},
        ],
        "forbidden": [
            {"setting": ["kitchen"]},
        ],
        "preferred": [
            {"setting": ["gym", "outdoors"]},
            {"mood": ["energetic", "inspiring"]},
            {"subject_types_any": ["person", "group"]},
        ],
    },
    "cooking": {
        "required": [
            {"any_of": [
                {"setting": ["kitchen"]},
                {"subject_types_any": ["food"]},
            ]},
        ],
        "forbidden": [
            {"activities_any": ["driving", "flying", "swimming", "climbing"]},
        ],
        "preferred": [
            {"activities_any": ["cooking", "baking", "eating"]},
            {"mood": ["cozy", "calm"]},
            {"camera": ["static", "handheld"]},
        ],
    },
    "travel": {
        "required": [
            {"any_of": [
                {"setting": ["outdoors", "city", "beach", "road"]},
                {"activities_any": ["driving", "flying", "sightseeing", "hiking", "camping"]},
            ]},
        ],
        "forbidden": [
            {"setting": ["gym", "kitchen"]},
        ],
        "preferred": [
            {"camera": ["drone"]},
            {"setting": ["outdoors", "city"]},
            {"mood": ["inspiring", "dramatic", "calm"]},
            {"subject_types_any": ["landscape", "vehicle"]},
        ],
    },
}


# ---------------------------------------------------------------------------
# Subreddit allow-list layer
# ---------------------------------------------------------------------------

def get_subreddits_for_niche(niche: str) -> List[str]:
    """Get allowed background subreddits for a niche."""
    return NICHE_SUBREDDITS.get(niche.lower(), [])


def get_niche_for_subreddit(subreddit: str) -> Optional[str]:
    """Find which niche a background subreddit belongs to."""
    sub_lower = subreddit.lower()
    for niche, subs in NICHE_SUBREDDITS.items():
        if sub_lower in [s.lower() for s in subs]:
            return niche
    return None


def is_subreddit_compatible(subreddit: str, niche: str) -> bool:
    """Check if a background subreddit is on a niche's allow-list."""
    if not subreddit or not niche:
        return False
    return subreddit.lower() in [s.lower() for s in get_subreddits_for_niche(niche)]


def list_niches() -> List[str]:
    """List all niches with defined rules."""
    return list(NICHE_SUBREDDITS.keys())


# ---------------------------------------------------------------------------
# Scene-tag rule layer
# ---------------------------------------------------------------------------

def get_tag_rules(niche: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return the {required, forbidden, preferred} rule groups for a niche."""
    return NICHE_TAG_RULES.get(niche.lower(), {"required": [], "forbidden": [], "preferred": []})


def _subject_types(ml_tags: Dict[str, Any]) -> List[str]:
    subjects = ml_tags.get("subjects") or []
    out = []
    for s in subjects:
        if isinstance(s, dict) and s.get("type"):
            out.append(str(s["type"]).lower())
        elif isinstance(s, str):
            out.append(s.lower())
    return out


def _clause_holds(clause: Dict[str, Any], ml_tags: Dict[str, Any]) -> bool:
    """Evaluate one clause against a tag dict. All keys are ANDed."""
    if "any_of" in clause:
        if not any(_clause_holds(sub, ml_tags) for sub in clause["any_of"]):
            return False

    activities = [str(a).lower() for a in (ml_tags.get("activities") or [])]
    subject_types = _subject_types(ml_tags)

    for key, expected in clause.items():
        if key == "any_of":
            continue
        if key in ("setting", "mood", "camera"):
            value = str(ml_tags.get(key) or "").lower()
            if value not in [str(v).lower() for v in expected]:
                return False
        elif key == "activities_any":
            if not any(a in activities for a in expected):
                return False
        elif key == "subject_types_any":
            if not any(t in subject_types for t in expected):
                return False
        elif key == "subject_types_none":
            if any(t in subject_types for t in expected):
                return False
        elif key == "text_on_screen":
            if bool(ml_tags.get("text_on_screen", False)) != bool(expected):
                return False
        else:
            raise ValueError(f"Unknown rule clause key: {key}")
    return True


def _describe_clause(clause: Dict[str, Any]) -> str:
    if "any_of" in clause:
        return " OR ".join(_describe_clause(c) for c in clause["any_of"])
    return ", ".join(f"{k}={v}" for k, v in clause.items())


def check_tag_rules(ml_tags: Optional[Dict[str, Any]], niche: str) -> Tuple[bool, List[str]]:
    """
    Evaluate a niche's required/forbidden rules against a BG's ml_tags.

    Returns (passes, reasons). `reasons` lists the clauses that caused a
    rejection so callers can log why a BG was skipped. Untagged BGs pass
    trivially — the tag layer can only judge what it can see; callers that
    need tags should filter `ml_tags IS NOT NULL` themselves.
    """
    if not niche:
        return True, []
    if not ml_tags or not isinstance(ml_tags, dict):
        return True, []

    rules = get_tag_rules(niche)
    reasons: List[str] = []

    for clause in rules.get("required", []):
        if not _clause_holds(clause, ml_tags):
            reasons.append(f"required not met: {_describe_clause(clause)}")

    for clause in rules.get("forbidden", []):
        if _clause_holds(clause, ml_tags):
            reasons.append(f"forbidden: {_describe_clause(clause)}")

    return (not reasons), reasons


def score_tag_preferences(ml_tags: Optional[Dict[str, Any]], niche: str) -> int:
    """Count how many of a niche's preferred clauses a BG satisfies (0..N)."""
    if not niche or not ml_tags or not isinstance(ml_tags, dict):
        return 0
    rules = get_tag_rules(niche)
    return sum(1 for clause in rules.get("preferred", []) if _clause_holds(clause, ml_tags))
