"""
Automation Configuration for Pipeline Orchestrator

Defines niche categories, training thresholds, generation settings,
composition matching, and daily quotas.

The four niches shipped here (motivation, fitness, cooking, travel) are
examples — each one is just a NicheConfig entry, so adding a niche is a
matter of adding another entry plus a background allow-list in
config/niche_rules.py.
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class NicheConfig:
    """Configuration for a single niche category."""
    name: str
    subreddits: List[str]          # Caption-source subreddits scraped for training data
    keywords: List[str]            # Generic content keywords for prompt grounding / search
    enabled: bool = True
    daily_video_target: int = 3
    generation_prompt: str = "Generate a caption:"  # Default prompt, can be customized per niche
    generation_temperature: float = 0.8  # Lower = more focused, higher = more creative
    generation_repetition_penalty: float = 1.25  # Higher = less repetition
    postpone_reddit_username: Optional[str] = None  # Override per-niche Postpone Reddit account
    # Optional CTA text appended as an outro segment when publishing to Reddit
    # only (not Patreon/Telegram, since the CTA promotes Patreon). Set per
    # niche; leave None to skip the outro.
    reddit_outro_text: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training trigger thresholds.

    Auto-training is permanently disabled — training runs are manual only,
    triggered via POST /training/trigger/{niche}. The threshold fields below
    are retained because they're referenced by the (dormant) selector logic
    in pipeline_orchestrator.get_next_niche_needing_training, which now
    short-circuits to None unconditionally.
    """
    skip_training: bool = True  # Permanent — auto-training removed. Do not flip.
    min_new_captions_since_last_train: int = 500
    min_total_captions: int = 500
    retrain_interval_hours: int = 168
    quality_score_threshold: int = 50
    train_niches_sequentially: bool = True


@dataclass
class GenerationConfig:
    """Generation trigger settings."""
    auto_generate_after_training: bool = True
    captions_per_batch: int = 100  # Per-niche batch size (increased for better yield)
    min_quality_score_for_composition: int = 80  # Only compose top quality
    max_pending_before_pause: int = 200  # Per-niche pending limit


@dataclass
class CompositionConfig:
    """Composition matching settings."""
    auto_compose_approved_captions: bool = True
    min_captions_for_batch: int = 10  # Per-niche minimum to start
    min_tag_overlap: int = 1  # At least 1 matching tag required
    prefer_high_view_backgrounds: bool = True
    exclude_recently_used_backgrounds: bool = True
    background_reuse_cooldown_hours: int = 8760  # Don't reuse for 1 year


@dataclass
class QuotaConfig:
    """Daily quota settings."""
    videos_per_niche_per_day: int = 3
    reset_hour_utc: int = 0  # Reset at midnight UTC
    pause_when_quota_met: bool = True


@dataclass
class GPUConfig:
    """GPU allocation settings."""
    training_gpu: int = 0  # GPU for training (larger VRAM)
    generation_gpu: int = 0  # GPU for generation (needs LoRA)
    pause_extraction_for_training: bool = True
    pause_extraction_for_generation: bool = True


@dataclass
class ScheduleConfig:
    """Optional schedule windows."""
    training_allowed_hours: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    respect_schedule: bool = False  # Set True to enforce windows


@dataclass
class PostponeConfig:
    """Postpone scheduling settings."""
    reddit_username: str = "captionforge_clips"  # Default connected Reddit account in Postpone (exact case from socialAccounts)
    default_posting_hour_utc: int = 18  # 6 PM UTC = 1 PM EST
    default_posting_minute: int = 0
    default_stagger_minutes: int = 10  # Minutes between subreddit posts (min 10)
    # Per-subreddit flair text (subreddit name -> flair text).
    # Picks come from sampling the 100 most-recent posts on each sub with
    # scripts/list_subreddit_flairs.py. Match flair text EXACTLY including
    # brackets, case, and trailing spaces. Only subs that require a flair
    # need an entry.
    subreddit_flairs: Dict[str, str] = field(default_factory=lambda: {
        # motivation
        "getmotivated": "[Video]",
        "motivation": "Video",
        # fitness
        "bodyweightfitness": "Video",
        "running": "Video",
        # cooking
        "recipes": "Recipe",
        "cooking": "Video",
        # travel
        "travel": "Video",
        "solotravel": "Video",
    })


@dataclass
class PatreonScheduleConfig:
    """Patreon scheduled-publish settings (one post per niche per day)."""
    default_posting_hour_utc: int = 18  # 6 PM UTC = 1 PM EST
    default_posting_minute: int = 0


@dataclass
class TelegramScheduleConfig:
    """Telegram scheduled-publish settings (one post per niche per day)."""
    default_posting_hour_utc: int = 18
    default_posting_minute: int = 0


class AutomationConfig:
    """
    Main automation configuration.

    All settings can be overridden via environment variables.
    """

    def __init__(self):
        self._niches: Dict[str, NicheConfig] = {}
        self._training = TrainingConfig()
        self._generation = GenerationConfig()
        self._composition = CompositionConfig()
        self._quotas = QuotaConfig()
        self._gpu = GPUConfig()
        self._schedule = ScheduleConfig()
        self._postpone = PostponeConfig()
        self._patreon_schedule = PatreonScheduleConfig()
        self._telegram_schedule = TelegramScheduleConfig()

        self._load_defaults()
        self._load_from_env()

    def _load_defaults(self):
        """Load default niche definitions."""
        # Short captions for video overlay.
        # Target: 40-80 words that fit ~60 second videos at 150 WPM.
        # Focus: punchy, direct, specific — not long stories.
        motivation_prompt = """Write a short motivational caption (40-80 words MAX). Second person ("you").

RULES: 2-4 sentences. Every sentence ends with punctuation (. or ! or ?). No URLs or promotions.

STYLE: Mix a blunt truth with encouragement and one concrete next action. Use vivid, specific scenarios — not generic. Vary your opening line and word choices. Include at least one unexpected detail or reframe.

Caption:"""

        fitness_prompt = """Write a short fitness caption (40-80 words MAX). Second person ("you") — you are the one training.

RULES: 2-4 sentences. Every sentence ends with punctuation (. or ! or ?). No URLs or promotions.

STYLE: Mix a practical training cue, a reason it matters, and a push to finish the set. Use vivid, specific scenarios — not generic. Vary your opening line and word choices. Each caption should land on a fresh detail; do not lean on a small set of stock phrases.

Caption:"""

        cooking_prompt = """Write a short cooking caption (40-80 words MAX). Second person ("you") — you are the one at the stove.

RULES: 2-4 sentences. Every sentence ends with punctuation (. or ! or ?). No URLs or promotions.

STYLE: One concrete technique or ingredient tip, why it works, and what the result tastes or looks like. Sensory and specific — heat, texture, smell, timing. Vary your opening line and word choices. Each caption should land on a fresh detail; do not lean on a small set of stock phrases.

Caption:"""

        travel_prompt = """Write a short travel caption (40-80 words MAX). Second person ("you").

RULES: 2-4 sentences. Every sentence ends with punctuation (. or ! or ?). No URLs or promotions.

STYLE: Wanderlust with a practical edge — a specific place, moment, or route, one sensory detail, and a nudge to actually go. Every caption should phrase the pull of travel differently. Use vivid, specific scenarios — not generic. Vary your opening line and word choices. Each caption should land on a fresh detail; do not lean on a small set of stock phrases.

Caption:"""

        self._niches = {
            "motivation": NicheConfig(
                name="motivation",
                subreddits=["GetMotivated", "Motivation", "quotes", "DecidingToBeBetter"],
                keywords=["motivation", "discipline", "mindset", "habits"],
                enabled=True,
                # Daily quota: 30 composed videos per day. Once we hit it the
                # orchestrator stops dispatching new BG-first cycles for motivation
                # and waits for tomorrow.
                daily_video_target=30,
                generation_prompt=motivation_prompt,
                generation_temperature=0.85,  # Slightly higher for variety in short captions
                generation_repetition_penalty=1.5,  # Higher to reduce overused phrases
                postpone_reddit_username="motivation_clips",
                reddit_outro_text="Daily motivation clips — more on our Patreon",
            ),
            "fitness": NicheConfig(
                name="fitness",
                subreddits=["Fitness", "bodyweightfitness", "xxfitness", "running"],
                keywords=["fitness", "workout", "training", "gym"],
                enabled=True,
                daily_video_target=30,
                generation_prompt=fitness_prompt,
                generation_temperature=0.85,
                generation_repetition_penalty=1.5,
                postpone_reddit_username="fitness_clips",
                # TODO: fill in once a Patreon page is live for fitness
                reddit_outro_text=None,
            ),
            "cooking": NicheConfig(
                name="cooking",
                subreddits=["Cooking", "recipes", "MealPrepSunday", "EatCheapAndHealthy"],
                keywords=["cooking", "recipe", "kitchen", "meal prep"],
                enabled=True,
                daily_video_target=30,
                generation_prompt=cooking_prompt,
                generation_temperature=0.85,
                generation_repetition_penalty=1.5,
                postpone_reddit_username="cooking_clips",
                # TODO: fill in once a Patreon page is live for cooking
                reddit_outro_text=None,
            ),
            "travel": NicheConfig(
                name="travel",
                subreddits=["travel", "solotravel", "digitalnomad", "backpacking"],
                keywords=["travel", "wanderlust", "road trip", "backpacking"],
                enabled=True,
                daily_video_target=30,
                generation_prompt=travel_prompt,
                generation_temperature=0.85,
                generation_repetition_penalty=1.5,
                postpone_reddit_username="travel_clips",
                # TODO: fill in once a Patreon page is live for travel
                reddit_outro_text=None,
            ),
        }

    def _load_from_env(self):
        """Override settings from environment variables."""
        # Training auto-trigger settings have been removed — auto-training is
        # permanently disabled and these env vars are intentionally not honored
        # any more (skip_training is locked True). Use POST /training/vastai/
        # {niche} for manual runs.

        # Generation settings
        if val := os.environ.get("AUTOMATION_CAPTIONS_PER_BATCH"):
            self._generation.captions_per_batch = int(val)
        if val := os.environ.get("AUTOMATION_MIN_QUALITY_SCORE"):
            self._generation.min_quality_score_for_composition = int(val)

        # Composition settings
        if val := os.environ.get("AUTOMATION_BACKGROUND_COOLDOWN_HOURS"):
            self._composition.background_reuse_cooldown_hours = int(val)

        # Quota settings
        if val := os.environ.get("AUTOMATION_VIDEOS_PER_NICHE_DAY"):
            self._quotas.videos_per_niche_per_day = int(val)

        # GPU settings (training_gpu retained for any legacy references but
        # no longer settable from env — training is manual-only via Vast.ai)
        if val := os.environ.get("AUTOMATION_GENERATION_GPU"):
            self._gpu.generation_gpu = int(val)

        # Schedule settings
        if os.environ.get("AUTOMATION_RESPECT_SCHEDULE", "").lower() == "true":
            self._schedule.respect_schedule = True

        # Postpone settings
        if val := os.environ.get("POSTPONE_REDDIT_USERNAME"):
            self._postpone.reddit_username = val
        if val := os.environ.get("POSTPONE_POSTING_HOUR_UTC"):
            self._postpone.default_posting_hour_utc = int(val)
        if val := os.environ.get("POSTPONE_POSTING_MINUTE"):
            self._postpone.default_posting_minute = int(val)
        if val := os.environ.get("POSTPONE_STAGGER_MINUTES"):
            self._postpone.default_stagger_minutes = max(10, int(val))

        # Patreon schedule overrides
        if val := os.environ.get("PATREON_POSTING_HOUR_UTC"):
            self._patreon_schedule.default_posting_hour_utc = int(val)
        if val := os.environ.get("PATREON_POSTING_MINUTE"):
            self._patreon_schedule.default_posting_minute = int(val)

        # Telegram schedule overrides
        if val := os.environ.get("TELEGRAM_POSTING_HOUR_UTC"):
            self._telegram_schedule.default_posting_hour_utc = int(val)
        if val := os.environ.get("TELEGRAM_POSTING_MINUTE"):
            self._telegram_schedule.default_posting_minute = int(val)

    @property
    def niches(self) -> Dict[str, NicheConfig]:
        return self._niches

    @property
    def training(self) -> TrainingConfig:
        return self._training

    @property
    def generation(self) -> GenerationConfig:
        return self._generation

    @property
    def composition(self) -> CompositionConfig:
        return self._composition

    @property
    def quotas(self) -> QuotaConfig:
        return self._quotas

    @property
    def gpu(self) -> GPUConfig:
        return self._gpu

    @property
    def schedule(self) -> ScheduleConfig:
        return self._schedule

    @property
    def postpone(self) -> PostponeConfig:
        return self._postpone

    @property
    def patreon_schedule(self) -> PatreonScheduleConfig:
        return self._patreon_schedule

    @property
    def telegram_schedule(self) -> TelegramScheduleConfig:
        return self._telegram_schedule

    def get_enabled_niches(self) -> List[NicheConfig]:
        """Get list of enabled niche configs."""
        return [f for f in self._niches.values() if f.enabled]

    def get_niche(self, name: str) -> Optional[NicheConfig]:
        """Get niche config by name."""
        return self._niches.get(name)

    def get_subreddits_for_niche(self, niche_name: str) -> List[str]:
        """Get caption-source subreddit list for a niche."""
        niche = self._niches.get(niche_name)
        return niche.subreddits if niche else []

    def get_keywords_for_niche(self, niche_name: str) -> List[str]:
        """Get content keyword list for a niche."""
        niche = self._niches.get(niche_name)
        return niche.keywords if niche else []

    def get_niche_for_subreddit(self, subreddit: str) -> Optional[str]:
        """Find which niche a caption-source subreddit belongs to."""
        subreddit_lower = subreddit.lower()
        for niche_name, config in self._niches.items():
            if subreddit_lower in [s.lower() for s in config.subreddits]:
                return niche_name
        return None

    def to_dict(self) -> dict:
        """Export config as dictionary (for API responses)."""
        return {
            "niches": {
                name: {
                    "subreddits": f.subreddits,
                    "keywords": f.keywords,
                    "enabled": f.enabled,
                    "daily_video_target": f.daily_video_target,
                    "generation_prompt": f.generation_prompt[:100] + "..." if len(f.generation_prompt) > 100 else f.generation_prompt,
                    "generation_temperature": f.generation_temperature,
                    "generation_repetition_penalty": f.generation_repetition_penalty,
                }
                for name, f in self._niches.items()
            },
            "training": {
                "skip_training": self._training.skip_training,
                "min_new_captions_since_last_train": self._training.min_new_captions_since_last_train,
                "min_total_captions": self._training.min_total_captions,
                "retrain_interval_hours": self._training.retrain_interval_hours,
                "quality_score_threshold": self._training.quality_score_threshold,
                "train_niches_sequentially": self._training.train_niches_sequentially,
            },
            "generation": {
                "auto_generate_after_training": self._generation.auto_generate_after_training,
                "captions_per_batch": self._generation.captions_per_batch,
                "min_quality_score_for_composition": self._generation.min_quality_score_for_composition,
                "max_pending_before_pause": self._generation.max_pending_before_pause,
            },
            "composition": {
                "auto_compose_approved_captions": self._composition.auto_compose_approved_captions,
                "min_captions_for_batch": self._composition.min_captions_for_batch,
                "min_tag_overlap": self._composition.min_tag_overlap,
                "prefer_high_view_backgrounds": self._composition.prefer_high_view_backgrounds,
                "exclude_recently_used_backgrounds": self._composition.exclude_recently_used_backgrounds,
                "background_reuse_cooldown_hours": self._composition.background_reuse_cooldown_hours,
            },
            "quotas": {
                "videos_per_niche_per_day": self._quotas.videos_per_niche_per_day,
                "reset_hour_utc": self._quotas.reset_hour_utc,
                "pause_when_quota_met": self._quotas.pause_when_quota_met,
            },
            "gpu": {
                "training_gpu": self._gpu.training_gpu,
                "generation_gpu": self._gpu.generation_gpu,
                "pause_extraction_for_training": self._gpu.pause_extraction_for_training,
                "pause_extraction_for_generation": self._gpu.pause_extraction_for_generation,
            },
            "schedule": {
                "training_allowed_hours": self._schedule.training_allowed_hours,
                "respect_schedule": self._schedule.respect_schedule,
            },
        }


# Global config instance
_config: Optional[AutomationConfig] = None


def get_automation_config() -> AutomationConfig:
    """Get or create the global automation config instance."""
    global _config
    if _config is None:
        _config = AutomationConfig()
    return _config


def reload_config() -> AutomationConfig:
    """Force reload of configuration (useful after env changes)."""
    global _config
    _config = AutomationConfig()
    return _config
