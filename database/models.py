"""
Database models using SQLAlchemy ORM
"""
from sqlalchemy import Column, Integer, String, Text, Float, BigInteger, Boolean, TIMESTAMP, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()


class Video(Base):
    """Videos table - stores downloaded video metadata"""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    source_url = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    file_hash = Column(String(64), unique=True, index=True)
    duration_seconds = Column(Integer)
    resolution = Column(String(50))  # Increased from 20 to 50 to handle floating point resolutions
    file_size_bytes = Column(BigInteger)
    source_subreddit = Column(String(100), index=True)
    source_post_id = Column(String(50), unique=True, index=True)
    media_type = Column(String(20), default="video", index=True)  # video, image, gallery, gif
    gallery_image_count = Column(Integer, nullable=True)  # For galleries: number of images
    upvotes = Column(Integer, default=0, index=True)  # Reddit post upvotes (for sorting/filtering)
    last_upvote_check = Column(TIMESTAMP(timezone=True), nullable=True)  # When we last updated upvotes
    download_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    processing_status = Column(String(20), default="pending", index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    gallery_added_at = Column(TIMESTAMP(timezone=True), nullable=True)  # When video was added to media gallery

    # Relationships
    analysis = relationship("VideoAnalysis", back_populates="video", uselist=False, cascade="all, delete-orphan")
    scraped_captions = relationship("ScrapedCaption", back_populates="video", cascade="all, delete-orphan")
    generated_captions = relationship("GeneratedCaption", back_populates="video", cascade="all, delete-orphan")
    publish_queue_entries = relationship("PublishQueue", back_populates="video", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Video(id={self.id}, post_id={self.source_post_id}, status={self.processing_status})>"


class VideoAnalysis(Base):
    """AI analysis results for videos"""
    __tablename__ = "video_analysis"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, unique=True)
    scene_description = Column(Text)
    detected_actions = Column(JSON)  # Array of detected actions
    detected_tags = Column(JSON)  # Array of tags
    confidence_score = Column(Float)
    analyzed_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    video = relationship("Video", back_populates="analysis")

    def __repr__(self):
        return f"<VideoAnalysis(video_id={self.video_id}, confidence={self.confidence_score})>"


class ScrapedCaption(Base):
    """Captions scraped from Reddit posts (training data)"""
    __tablename__ = "scraped_captions"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    caption_text = Column(Text, nullable=False)  # Final processed caption (kept for backwards compatibility)
    raw_ocr_text = Column(Text, nullable=True)  # Original OCR output before any processing
    rule_based_text = Column(Text, nullable=True)  # After rule-based post-processing but before LLM
    llm_refined_text = Column(Text, nullable=True)  # After LLM refinement (if applied)
    source_subreddit = Column(String(100), index=True)
    upvotes = Column(Integer)
    scraped_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Quality metrics (for analysis and training data filtering)
    compression_ratio = Column(Float, nullable=True)  # raw_ocr_length / final_length (higher = more dedup)
    slide_count_raw = Column(Integer, nullable=True)  # Number of *|* delimited slides in raw OCR
    slide_count_final = Column(Integer, nullable=True)  # Number of slides after processing
    unique_word_ratio = Column(Float, nullable=True)  # Unique words / total words (lower = more repetition)
    extraction_metadata = Column(JSON, nullable=True)  # Extractor stats (frames processed, dedup %, etc.)

    # Training review (populated by caption review script)
    training_status = Column(String(20), default="pending", index=True)  # pending, approved, rejected, fixed
    training_fixed_text = Column(Text, nullable=True)  # Corrected text when status='fixed'
    training_rejection_reason = Column(Text, nullable=True)  # Why caption was rejected

    # Relationships
    video = relationship("Video", back_populates="scraped_captions")

    def __repr__(self):
        return f"<ScrapedCaption(id={self.id}, video_id={self.video_id})>"


class GeneratedCaption(Base):
    """Generated captions from LLM"""
    __tablename__ = "generated_captions"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=True)  # Optional - can be standalone
    generation_job_id = Column(Integer, ForeignKey("generation_jobs.id"), nullable=True)  # Link to batch job
    caption_text = Column(Text, nullable=False)
    generated_title = Column(String(300), nullable=True)  # LLM-generated Reddit post title
    llm_model = Column(String(100))  # Increased for longer model names
    generation_prompt = Column(Text)
    quality_score = Column(Float)
    status = Column(String(20), default="pending_review", index=True)  # pending_review, approved, rejected, published
    is_favorite = Column(Boolean, default=False)  # User can mark favorites
    tags = Column(JSON, default=list)  # Activity tags extracted from caption content
    niche = Column(String(50), nullable=True, index=True)  # motivation, fitness, cooking, travel
    generated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Generation parameters (for reproducibility)
    temperature = Column(Float, nullable=True)
    top_p = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)

    # LLM judge (Stage 3 of the quality pipeline (README)). Mistral-Small scores
    # each caption on grammar/flow/bg_consistency/appeal 1-10 and lists
    # specific issues. judge_pass = all axes >= 7.
    judge_scores = Column(JSON, nullable=True)        # {"grammar": 8, "flow": 9, "bg_consistency": 7, "appeal": 8, "overall": 8}
    judge_issues = Column(JSON, nullable=True)        # ["ends with rhetorical question", ...]
    judge_pass = Column(Boolean, nullable=True)
    judge_status = Column(String(20), nullable=True, index=True)  # NULL/pending/judged/failed
    judge_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    video = relationship("Video", back_populates="generated_captions")
    generation_job = relationship("GenerationJob", back_populates="captions")
    publish_queue = relationship("PublishQueue", back_populates="caption")

    def __repr__(self):
        return f"<GeneratedCaption(id={self.id}, video_id={self.video_id}, status={self.status})>"


class CaptionCandidate(Base):
    """
    Stage 2 of the quality pipeline (README, "What it does"): BG-first generation.

    Each row is one candidate caption generated SPECIFICALLY for a given
    background video. We pick the BG first, then generate K candidates
    targeted at its subjects / activities / setting / mood,
    judge each one inline with Stage 3, and the winner gets composed.

    This replaces the old "generate captions blind, then post-hoc match
    them to whatever BG happens to be available" flow.
    """
    __tablename__ = "caption_candidates"

    id = Column(Integer, primary_key=True, index=True)
    background_video_id = Column(Integer, ForeignKey("background_videos.id"), nullable=False, index=True)
    niche = Column(String(50), nullable=False, index=True)
    prompt_version = Column(String(20), nullable=False)  # bg_first_v1, etc.

    caption_text = Column(Text, nullable=False)
    llm_model = Column(String(100), nullable=True)
    generation_temperature = Column(Float, nullable=True)

    # Stage 3 judge results (inline at generation time)
    judge_scores = Column(JSON, nullable=True)        # {"grammar":8,"flow":9,"bg_consistency":7,"appeal":8,"overall":8}
    judge_issues = Column(JSON, nullable=True)        # ["..."]
    judge_pass = Column(Boolean, nullable=True)
    judge_overall = Column(Integer, nullable=True, index=True)  # Denormalized for fast ORDER BY
    judge_status = Column(String(20), nullable=True)  # judged / failed / pending

    # winner | candidate | discarded
    # winner: highest-scoring judge-pass candidate for this BG, used for composition
    # candidate: judged but not selected (yet)
    # discarded: judge-fail or worse — won't be considered
    status = Column(String(20), default="candidate", index=True)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    background_video = relationship("BackgroundVideo")

    def __repr__(self):
        return f"<CaptionCandidate(id={self.id}, bg={self.background_video_id}, status={self.status}, overall={self.judge_overall})>"


class GenerationJob(Base):
    """
    Batch caption generation job tracking.
    Each job generates multiple captions using a trained model.
    """
    __tablename__ = "generation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=False)

    # Job identification
    job_name = Column(String(200), nullable=False)
    celery_task_id = Column(String(100), nullable=True, index=True)

    # Generation configuration
    num_captions = Column(Integer, nullable=False)  # How many captions to generate
    prompt = Column(Text, default="Generate a caption:")
    temperature = Column(Float, default=0.9)
    top_p = Column(Float, default=0.95)
    max_new_tokens = Column(Integer, default=300)
    repetition_penalty = Column(Float, default=1.15)

    # Progress tracking
    status = Column(String(30), default="pending", index=True)
    # Status values: pending, queued, running, completed, failed, cancelled
    progress_percent = Column(Float, default=0.0)
    captions_generated = Column(Integer, default=0)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    model = relationship("TrainedModel", back_populates="generation_jobs")
    captions = relationship("GeneratedCaption", back_populates="generation_job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<GenerationJob(id={self.id}, name={self.job_name}, status={self.status}, progress={self.captions_generated}/{self.num_captions})>"


class PublishQueue(Base):
    """Queue for publishing content"""
    __tablename__ = "publish_queue"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    caption_id = Column(Integer, ForeignKey("generated_captions.id"), nullable=False)
    target_subreddit = Column(String(100))
    target_platform = Column(String(20))  # 'reddit' or 'patreon'
    scheduled_time = Column(TIMESTAMP(timezone=True))
    published_time = Column(TIMESTAMP(timezone=True), nullable=True)
    status = Column(String(20), default="queued", index=True)
    post_url = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    video = relationship("Video", back_populates="publish_queue_entries")
    caption = relationship("GeneratedCaption", back_populates="publish_queue")

    def __repr__(self):
        return f"<PublishQueue(id={self.id}, status={self.status}, platform={self.target_platform})>"


class ScrapingProgress(Base):
    """Track scraping progress for incremental/resumable scraping"""
    __tablename__ = "scraping_progress"

    id = Column(Integer, primary_key=True, index=True)
    subreddit = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(String(255), nullable=True)  # Optional description for the subreddit
    batch_size = Column(Integer, default=25)  # Number of posts to fetch per scrape

    # Progressive depth scraping stage
    # Progression: top_all -> top_year -> top_month -> top_week -> top_day -> new
    scrape_stage = Column(String(20), default="top_all", index=True)

    last_post_id = Column(String(50), nullable=True)  # Last successfully scraped post ID
    last_post_score = Column(Integer, nullable=True)  # Score of last scraped post
    last_pagination_url = Column(Text, nullable=True)  # Reddit "next" button URL to resume pagination
    posts_scraped = Column(Integer, default=0)  # Total posts examined (includes duplicates, 404s)
    videos_downloaded = Column(Integer, default=0)  # Actual unique videos successfully stored
    target_videos = Column(Integer, default=500)  # Target number of unique videos to collect
    videos_failed = Column(Integer, default=0)  # Videos that failed download (404s, duplicates)
    target_min_score = Column(Integer, default=300)  # Stop scraping below this score (25 for /new)
    scraping_active = Column(Boolean, default=True)  # Whether to continue scraping
    last_scrape_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<ScrapingProgress(subreddit={self.subreddit}, videos={self.videos_downloaded}/{self.target_videos}, posts={self.posts_scraped})>"


# NOTE: TaskLog model removed - SSE streaming was unreliable
# Use docker-compose logs or Flower dashboard for log visibility


class HangIncident(Base):
    """
    Records of worker hang incidents for pattern analysis and debugging.

    Captures comprehensive diagnostics when a hang is detected to help identify
    root causes and prevent future occurrences.
    """
    __tablename__ = "hang_incidents"

    id = Column(Integer, primary_key=True, index=True)

    # When the hang was detected
    detected_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)

    # Task information
    task_name = Column(String(200), nullable=True)  # e.g., 'extract_caption_task'
    task_id = Column(String(100), nullable=True)  # Celery task UUID
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=True)  # Video being processed

    # Heartbeat info at time of hang
    last_heartbeat_stage = Column(String(100), nullable=True)  # e.g., 'ocr_frame_45'
    last_heartbeat_at = Column(TIMESTAMP(timezone=True), nullable=True)
    heartbeat_age_seconds = Column(Float, nullable=True)  # Seconds since last heartbeat

    # Task duration
    task_duration_seconds = Column(Float, nullable=True)  # How long task was running

    # Video metadata (for pattern analysis)
    video_file_size_mb = Column(Float, nullable=True)
    video_duration_seconds = Column(Integer, nullable=True)
    video_resolution = Column(String(50), nullable=True)
    video_subreddit = Column(String(100), nullable=True, index=True)

    # System state at hang
    gpu_memory_used_mb = Column(Integer, nullable=True)
    gpu_memory_total_mb = Column(Integer, nullable=True)
    gpu_utilization_percent = Column(Integer, nullable=True)
    gpu_processes = Column(JSON, nullable=True)  # List of processes using GPU
    system_memory_used_mb = Column(Integer, nullable=True)
    system_memory_percent = Column(Float, nullable=True)
    cpu_percent = Column(Float, nullable=True)

    # Worker process info
    worker_pid = Column(Integer, nullable=True)
    worker_memory_mb = Column(Float, nullable=True)
    worker_threads = Column(Integer, nullable=True)
    worker_open_files = Column(Integer, nullable=True)

    # Full diagnostics dump (JSON for flexibility)
    full_diagnostics = Column(JSON, nullable=True)

    # Recovery action taken
    recovery_action = Column(String(50), nullable=True)  # 'manual', 'auto_restart', 'none'
    recovery_successful = Column(Boolean, nullable=True)

    # Optional notes
    notes = Column(Text, nullable=True)

    # Relationships
    video = relationship("Video", foreign_keys=[video_id])

    def __repr__(self):
        return f"<HangIncident(id={self.id}, task={self.task_name}, video_id={self.video_id}, detected_at={self.detected_at})>"


class TrainedModel(Base):
    """
    Trained LoRA models for caption generation.
    Each model is trained on data from specific subreddit(s).
    """
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)  # User-friendly name
    description = Column(Text, nullable=True)

    # Base model info
    base_model = Column(String(200), nullable=False)  # e.g., 'mistralai/Mistral-7B-Instruct-v0.3'
    adapter_path = Column(Text, nullable=False)  # Path to LoRA adapter weights

    # Training data source
    source_subreddits = Column(JSON, nullable=False)  # List of subreddits used for training
    training_samples = Column(Integer, nullable=False)  # Number of samples used
    min_upvotes_filter = Column(Integer, default=0)  # Upvote filter used during training

    # Training hyperparameters (stored for reproducibility)
    hyperparameters = Column(JSON, nullable=True)  # lora_rank, learning_rate, epochs, etc.

    # Training metrics
    final_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    training_duration_seconds = Column(Integer, nullable=True)

    # Status
    status = Column(String(20), default="ready", index=True)  # ready, loading, loaded, error
    is_loaded = Column(Boolean, default=False)  # Currently loaded in GPU memory
    last_loaded_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # GPU assignment
    target_gpu = Column(Integer, default=0)  # Which GPU this model was trained on / should load to

    # Niche tracking (for per-niche automation)
    niche = Column(String(50), nullable=True, index=True)  # motivation, fitness, cooking, travel

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="model", cascade="all, delete-orphan")
    generation_jobs = relationship("GenerationJob", back_populates="model", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TrainedModel(id={self.id}, name={self.name}, base={self.base_model}, gpu={self.target_gpu})>"


class TrainingJob(Base):
    """
    Training job tracking with progress reporting.
    Each job trains a new model or retrains an existing one.
    """
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=True)  # Null until model created

    # Job identification
    job_name = Column(String(200), nullable=False)
    celery_task_id = Column(String(100), nullable=True, index=True)

    # Training configuration
    base_model = Column(String(200), nullable=False)
    source_subreddits = Column(JSON, nullable=False)  # List of subreddits to use
    min_upvotes = Column(Integer, default=0)
    min_caption_length = Column(Integer, default=100)
    max_caption_length = Column(Integer, default=2000)

    # Hyperparameters
    lora_rank = Column(Integer, default=16)
    lora_alpha = Column(Integer, default=32)
    learning_rate = Column(Float, default=2e-4)
    num_epochs = Column(Integer, default=3)
    batch_size = Column(Integer, default=1)
    gradient_accumulation_steps = Column(Integer, default=4)
    max_seq_length = Column(Integer, default=512)
    warmup_ratio = Column(Float, default=0.1)

    # GPU selection
    target_gpu = Column(Integer, default=0)  # Which GPU to train on (0 or 1)

    # Niche tracking (for per-niche automation)
    niche = Column(String(50), nullable=True, index=True)  # motivation, fitness, cooking, travel

    # Progress tracking
    status = Column(String(30), default="pending", index=True)
    # Status values: pending, queued, preparing_data, training, evaluating, saving, completed, failed, cancelled
    progress_percent = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=True)
    current_loss = Column(Float, nullable=True)
    best_loss = Column(Float, nullable=True)

    # Data statistics
    total_samples = Column(Integer, nullable=True)
    train_samples = Column(Integer, nullable=True)
    val_samples = Column(Integer, nullable=True)

    # Results
    final_train_loss = Column(Float, nullable=True)
    final_val_loss = Column(Float, nullable=True)
    training_duration_seconds = Column(Integer, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    model = relationship("TrainedModel", back_populates="training_jobs")

    def __repr__(self):
        return f"<TrainingJob(id={self.id}, name={self.job_name}, status={self.status}, progress={self.progress_percent}%)>"


class BackgroundVideo(Base):
    """
    Background videos used as the visual layer under generated captions.
    Stored separately from the caption-source videos in `videos`.
    """
    __tablename__ = "background_videos"

    id = Column(Integer, primary_key=True, index=True)
    # video source (reddit, or a future custom importer)
    source_type = Column(String(20), default="reddit", nullable=False, index=True)
    reddit_post_id = Column(String(50), unique=True, nullable=True, index=True)
    reddit_subreddit = Column(String(100), nullable=True, index=True)
    reddit_score = Column(Integer, nullable=True)
    source_url = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    thumbnail_path = Column(Text, nullable=True)
    file_hash = Column(String(64), unique=True, index=True)

    # Video metadata
    duration_seconds = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    file_size_bytes = Column(BigInteger)

    # Source engagement metrics (Reddit score is stored as `views` for ranking)
    views = Column(Integer, default=0, index=True)
    likes = Column(Integer, default=0)

    # Content metadata
    tags = Column(JSON)  # Tags supplied by the source (subreddit name for Reddit)
    searched_tag = Column(String(100), nullable=True, index=True)  # The tag/subreddit we used to find this video
    username = Column(String(100))  # Author on the source platform
    created_at_source = Column(TIMESTAMP(timezone=True))

    # Processing status
    download_status = Column(String(20), default="pending", index=True)
    error_message = Column(Text)

    # Watermark filter status (OCR-based detection)
    # pending: Awaiting OCR check, approved: No watermark, rejected: Has watermark, error: Check failed
    filter_status = Column(String(20), default="pending", index=True)
    filter_text = Column(Text, nullable=True)  # Detected text (for debugging)
    filter_checked_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # ML tagging status (VLM-based content analysis)
    # pending: Awaiting ML tagging, processing: Being tagged, completed: Tags ready, error: Failed, skipped: Filter rejected
    # ml_tags_version tracks the tag schema (subjects/activities/setting/mood/camera/text_on_screen, see ml_tagging.py)
    ml_tags = Column(JSON, nullable=True)
    ml_tags_version = Column(Integer, default=1, nullable=True)
    ml_tagging_status = Column(String(20), default="pending", index=True)
    ml_tagging_checked_at = Column(TIMESTAMP(timezone=True), nullable=True)
    ml_tagging_error = Column(Text, nullable=True)

    # Composition tracking (for cooldown/reuse prevention)
    last_used_at = Column(TIMESTAMP(timezone=True), nullable=True, index=True)  # When used in composition

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        source_id = self.reddit_post_id or "?"
        return f"<BackgroundVideo(id={self.id}, source={self.source_type}, source_id={source_id}, filter={self.filter_status})>"


class RedditBackgroundSubreddit(Base):
    """Reddit subreddits scraped for background videos (not captions)."""
    __tablename__ = "reddit_background_subreddits"

    id = Column(Integer, primary_key=True, index=True)
    subreddit = Column(String(100), unique=True, nullable=False, index=True)
    enabled = Column(Boolean, default=True)

    # Scraping configuration
    min_score = Column(Integer, default=100)
    min_duration = Column(Integer, default=10)  # seconds
    max_duration = Column(Integer, default=60)  # seconds
    batch_size = Column(Integer, default=10)

    # Progressive depth (same stages as caption scraping)
    scrape_stage = Column(String(20), default="top_all", index=True)
    last_pagination_url = Column(Text, nullable=True)
    posts_scraped = Column(Integer, default=0)

    # Progress tracking
    videos_downloaded = Column(Integer, default=0)
    videos_failed = Column(Integer, default=0)
    last_scrape_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<RedditBackgroundSubreddit(subreddit={self.subreddit}, videos={self.videos_downloaded}, stage={self.scrape_stage})>"


class VideoCompositionJob(Base):
    """
    Batch video composition job tracking.
    Each job composes multiple videos from captions + background videos.
    """
    __tablename__ = "video_composition_jobs"

    id = Column(Integer, primary_key=True, index=True)

    # Job identification
    job_name = Column(String(200), nullable=False)
    celery_task_id = Column(String(100), nullable=True, index=True)

    # Source configuration
    caption_source = Column(String(50), default="generated")  # 'generated' or 'scraped'
    caption_status_filter = Column(String(30), nullable=True)  # e.g., 'approved' for generated
    min_upvotes = Column(Integer, default=0)  # For scraped captions
    background_tag_filter = Column(String(100), nullable=True)  # Filter backgrounds by source tag (subreddit)

    # Composition settings
    target_count = Column(Integer, default=10)  # How many videos to compose

    # Niche tracking (for per-niche automation)
    niche = Column(String(50), nullable=True, index=True)  # motivation, fitness, cooking, travel

    # Progress tracking
    status = Column(String(30), default="pending", index=True)
    # Status values: pending, queued, running, completed, failed, cancelled
    progress_percent = Column(Float, default=0.0)
    videos_composed = Column(Integer, default=0)
    videos_failed = Column(Integer, default=0)
    videos_skipped = Column(Integer, default=0)  # Caption too long for background

    # Error handling
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    composed_videos = relationship("ComposedVideo", back_populates="composition_job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<VideoCompositionJob(id={self.id}, name={self.job_name}, status={self.status}, progress={self.videos_composed}/{self.target_count})>"


class ComposedVideo(Base):
    """
    Final composed videos with caption overlay on background.
    Links a caption (generated or scraped) with a background video.
    """
    __tablename__ = "composed_videos"

    id = Column(Integer, primary_key=True, index=True)

    # Source references
    composition_job_id = Column(Integer, ForeignKey("video_composition_jobs.id"), nullable=True)
    generated_caption_id = Column(Integer, ForeignKey("generated_captions.id"), nullable=True)
    scraped_caption_id = Column(Integer, ForeignKey("scraped_captions.id"), nullable=True)
    background_video_id = Column(Integer, ForeignKey("background_videos.id"), nullable=False)

    # Output file
    storage_path = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    resolution = Column(String(50), nullable=True)  # e.g., "1920x1080"

    # Composition metadata
    caption_chunks = Column(Integer, nullable=True)  # Number of text chunks displayed
    status = Column(String(30), default="completed", index=True)  # completed, failed, deleted

    # Quality/review
    is_favorite = Column(Boolean, default=False)
    is_published = Column(Boolean, default=False)

    # Approval workflow (for Postpone scheduling)
    approval_status = Column(String(30), default="pending")  # pending, approved, scheduled, rejected
    approved_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Edited caption text (for recomposition with custom text)
    edited_caption_text = Column(Text, nullable=True)  # If set, use this instead of original caption

    # Niche tracking (for per-niche automation)
    niche = Column(String(50), nullable=True, index=True)  # motivation, fitness, cooking, travel

    # Public URL of this video on the media host (self-hosted by default, see publishers/media_host.py)
    hosted_url = Column(Text, nullable=True)

    # Reddit posting tracking
    reddit_post_url = Column(Text, nullable=True)
    reddit_posted_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Stage 4 visual review by Claude Code (interactive session, not API).
    # Status flow: NULL -> "pending" (queued by export script) -> "reviewed"
    # or "skipped". The reviewer is whoever runs scripts/claude_review_videos.py
    # in a Claude Code session and follows CLAUDE_REVIEW_INSTRUCTIONS.md.
    claude_review_status = Column(String(20), nullable=True, index=True)
    claude_review_scores = Column(JSON, nullable=True)   # {"caption_quality": 8, "caption_video_match": 7, "would_post": 8}
    claude_review_issues = Column(JSON, nullable=True)   # ["caption mentions a kitchen, video shows a trail", ...]
    claude_review_verdict = Column(String(20), nullable=True)  # pass | fail | maybe
    claude_review_notes = Column(Text, nullable=True)
    claude_review_model = Column(String(50), nullable=True)
    claude_review_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    composition_job = relationship("VideoCompositionJob", back_populates="composed_videos")
    generated_caption = relationship("GeneratedCaption")
    scraped_caption = relationship("ScrapedCaption")
    background_video = relationship("BackgroundVideo")
    publish_jobs = relationship("VideoPublishJob", back_populates="composed_video", cascade="all, delete-orphan")
    postpone_jobs = relationship("PostponeScheduleJob", back_populates="composed_video", cascade="all, delete-orphan")

    def __repr__(self):
        caption_type = "generated" if self.generated_caption_id else "scraped"
        return f"<ComposedVideo(id={self.id}, {caption_type}_caption, bg={self.background_video_id})>"


class VideoPublishJob(Base):
    """
    Publish job for composed videos.

    Workflow:
    1. Publish video to the media host (self-hosted URL by default)
    2. Post to Reddit profile with the hosted link
    3. Wait 30 minutes
    4. Crosspost to all subreddits for the niche
    """
    __tablename__ = "video_publish_jobs"

    id = Column(Integer, primary_key=True, index=True)
    composed_video_id = Column(Integer, ForeignKey("composed_videos.id"), nullable=False)

    # Content info
    title = Column(String(300), nullable=False)
    niche = Column(String(50), nullable=True, index=True)  # motivation, fitness, cooking, travel
    tags = Column(Text, nullable=True)  # Comma-separated tags for the Reddit post

    # Status tracking
    # pending, hosting_media, posting_profile, waiting_crosspost, crossposting, completed, failed, cancelled
    status = Column(String(30), default="pending", index=True)

    # Media host result (see publishers/media_host.py)
    hosted_media_id = Column(String(100), nullable=True)
    hosted_url = Column(Text, nullable=True)
    hosted_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Reddit profile post — populated per-niche at job creation from the
    # active reddit_accounts row
    profile_subreddit = Column(String(100), nullable=True)
    profile_post_id = Column(String(50), nullable=True)
    profile_post_url = Column(Text, nullable=True)
    profile_posted_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Crosspost scheduling
    crosspost_delay_minutes = Column(Integer, default=30)
    crosspost_scheduled_at = Column(TIMESTAMP(timezone=True), nullable=True)
    crosspost_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    crosspost_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Celery task tracking
    celery_task_id = Column(String(100), nullable=True, index=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_stage = Column(String(50), nullable=True)  # media_host, profile, crosspost

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    composed_video = relationship("ComposedVideo", back_populates="publish_jobs")
    crossposts = relationship("RedditCrosspost", back_populates="publish_job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<VideoPublishJob(id={self.id}, video={self.composed_video_id}, status={self.status})>"


class RedditCrosspost(Base):
    """Track individual crossposts to subreddits."""
    __tablename__ = "reddit_crossposts"

    id = Column(Integer, primary_key=True, index=True)
    publish_job_id = Column(Integer, ForeignKey("video_publish_jobs.id"), nullable=False)

    # Target subreddit
    subreddit = Column(String(100), nullable=False, index=True)

    # Status: pending, posting, posted, failed, skipped
    status = Column(String(30), default="pending", index=True)

    # Post info
    post_id = Column(String(50), nullable=True)
    post_url = Column(Text, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    posted_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    publish_job = relationship("VideoPublishJob", back_populates="crossposts")

    def __repr__(self):
        return f"<RedditCrosspost(id={self.id}, sub={self.subreddit}, status={self.status})>"


class PatreonCredential(Base):
    """
    Patreon credentials per niche.

    Each niche can have its own Patreon account for publishing videos.
    Uses Playwright browser automation with saved session cookies.
    """
    __tablename__ = "patreon_credentials"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(50), unique=True, nullable=False, index=True)  # motivation, fitness, cooking, travel
    email = Column(String(255), nullable=False)
    cookies_path = Column(Text, nullable=True)  # Path to saved session cookies
    is_configured = Column(Boolean, default=False)  # True after successful login
    last_login_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    publish_jobs = relationship("PatreonPublishJob", back_populates="credential", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<PatreonCredential(niche={self.niche}, email={self.email}, configured={self.is_configured})>"


class PatreonPublishJob(Base):
    """
    Publish job for posting composed videos to Patreon.

    Workflow:
    1. Upload video to Patreon
    2. Create post with title and description
    3. Set visibility (public/patrons-only)
    """
    __tablename__ = "patreon_publish_jobs"

    id = Column(Integer, primary_key=True, index=True)
    composed_video_id = Column(Integer, ForeignKey("composed_videos.id"), nullable=False)
    credential_id = Column(Integer, ForeignKey("patreon_credentials.id"), nullable=True)

    # Content info
    niche = Column(String(50), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)  # Comma-separated tags

    # Status: scheduled, pending, uploading, posted, failed, cancelled
    # - scheduled: queued for a future post time (waiting for beat dispatcher)
    # - pending: dispatcher promoted it; Celery task will pick it up imminently
    status = Column(String(30), default="scheduled", index=True)

    # Scheduling info (one post per niche per day, similar to Postpone)
    scheduled_date = Column(TIMESTAMP(timezone=True), nullable=True, index=True)
    base_post_time = Column(TIMESTAMP(timezone=True), nullable=True)

    # Patreon post info
    patreon_post_id = Column(String(100), nullable=True)
    patreon_post_url = Column(Text, nullable=True)
    posted_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Celery task tracking
    celery_task_id = Column(String(100), nullable=True, index=True)

    # Error handling
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    composed_video = relationship("ComposedVideo")
    credential = relationship("PatreonCredential", back_populates="publish_jobs")

    def __repr__(self):
        return f"<PatreonPublishJob(id={self.id}, niche={self.niche}, status={self.status})>"


class TelegramBot(Base):
    """
    Telegram bot credentials per niche.
    Each niche has its own bot for posting to its channel.
    """
    __tablename__ = "telegram_bots"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(50), unique=True, nullable=False, index=True)
    bot_username = Column(String(100), nullable=False)  # @bot_username
    bot_token = Column(String(200), nullable=False)  # API token from BotFather
    bot_name = Column(String(100), nullable=True)  # Display name

    is_enabled = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    channels = relationship("TelegramChannel", back_populates="bot")
    publish_jobs = relationship("TelegramPublishJob", back_populates="bot")

    def __repr__(self):
        return f"<TelegramBot(id={self.id}, niche={self.niche}, username={self.bot_username})>"


class TelegramChannel(Base):
    """
    Telegram channel configuration per niche.
    Linked to a bot that posts to this channel.
    """
    __tablename__ = "telegram_channels"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(50), unique=True, nullable=False, index=True)
    channel_id = Column(String(50), nullable=False)  # Numeric ID like -1001234567890
    channel_username = Column(String(100), nullable=True)  # @ChannelUsername (if public)
    channel_name = Column(String(200), nullable=True)  # Display name
    discussion_group_id = Column(String(50), nullable=True)  # Linked discussion group for comments

    bot_id = Column(Integer, ForeignKey("telegram_bots.id"), nullable=True)

    is_enabled = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    bot = relationship("TelegramBot", back_populates="channels")
    publish_jobs = relationship("TelegramPublishJob", back_populates="channel")

    def __repr__(self):
        return f"<TelegramChannel(id={self.id}, niche={self.niche}, channel={self.channel_username or self.channel_id})>"


class TelegramScrapeChannel(Base):
    """
    Telegram channels to scrape for training videos.
    Uses Telethon (user account) to access private channels.
    Separate from TelegramChannel which is for publishing.
    """
    __tablename__ = "telegram_scrape_channels"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String(100), unique=True, nullable=False, index=True)  # e.g., "-1001234567890"
    channel_username = Column(String(100), nullable=True)  # e.g., "@channelname" (if public)
    channel_name = Column(String(200), nullable=True)  # Display name
    is_enabled = Column(Boolean, default=True, index=True)
    batch_size = Column(Integer, default=25)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    last_scrape_at = Column(TIMESTAMP(timezone=True), nullable=True)

    def __repr__(self):
        return f"<TelegramScrapeChannel(id={self.id}, channel={self.channel_username or self.channel_id}, enabled={self.is_enabled})>"


class TelegramPublishJob(Base):
    """
    Publish job for posting composed videos to Telegram.
    """
    __tablename__ = "telegram_publish_jobs"

    id = Column(Integer, primary_key=True, index=True)
    composed_video_id = Column(Integer, ForeignKey("composed_videos.id"), nullable=False)
    niche = Column(String(50), nullable=False, index=True)
    channel_id = Column(Integer, ForeignKey("telegram_channels.id"), nullable=True)
    bot_id = Column(Integer, ForeignKey("telegram_bots.id"), nullable=True)

    # Status: scheduled, pending, uploading, completed, failed, cancelled
    # - scheduled: queued for a future post time (waiting for beat dispatcher)
    # - pending: dispatcher promoted it; Celery task will pick it up imminently
    status = Column(String(30), default="scheduled", index=True)

    # Scheduling info (one post per niche per day, similar to Postpone)
    scheduled_date = Column(TIMESTAMP(timezone=True), nullable=True, index=True)
    base_post_time = Column(TIMESTAMP(timezone=True), nullable=True)

    # Telegram post info
    telegram_message_id = Column(Integer, nullable=True)
    telegram_post_url = Column(String(500), nullable=True)
    caption_text = Column(Text, nullable=True)

    # Celery task tracking
    celery_task_id = Column(String(100), nullable=True, index=True)

    # Error handling
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    composed_video = relationship("ComposedVideo")
    channel = relationship("TelegramChannel", back_populates="publish_jobs")
    bot = relationship("TelegramBot", back_populates="publish_jobs")

    def __repr__(self):
        return f"<TelegramPublishJob(id={self.id}, niche={self.niche}, status={self.status})>"


class RedditAccount(Base):
    """
    Reddit account credentials per niche.
    Each niche can have its own Reddit account for posting.
    Uses Playwright browser automation (no API keys needed).
    """
    __tablename__ = "reddit_accounts"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(50), unique=True, nullable=False, index=True)

    # Account info
    username = Column(String(100), nullable=False)

    # Associated subreddits (comma-separated)
    subreddits = Column(Text, nullable=True)  # e.g., "motivationcaptions,motivationcaptionpalace"

    # Playwright session storage
    cookies_path = Column(String(500), nullable=True)  # Path to saved cookies
    is_logged_in = Column(Boolean, default=False)
    last_login_at = Column(TIMESTAMP(timezone=True), nullable=True)

    is_enabled = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<RedditAccount(id={self.id}, niche={self.niche}, username={self.username})>"

    def get_subreddits_list(self) -> list:
        """Return subreddits as a list."""
        if not self.subreddits:
            return []
        return [s.strip() for s in self.subreddits.split(",") if s.strip()]


class Workflow(Base):
    """
    Saved browser automation workflows.

    Workflows are recorded sequences of browser actions (clicks, typing, navigation)
    that can be replayed with variable substitution. Used for automating Reddit posting
    and other browser-based tasks that require avoiding API restrictions.
    """
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    target_site = Column(String(100), nullable=True, index=True)  # reddit, patreon, generic

    # Recorded actions as JSON array
    # Each action: {type, selector, coordinates, text, timestamp, ...}
    actions = Column(JSON, nullable=False, default=list)

    # Variables for parameterized execution
    # Array of {name: "{{title}}", default_value: "", description: "Post title"}
    variables = Column(JSON, default=list)

    # Execution tracking
    last_run_at = Column(TIMESTAMP(timezone=True), nullable=True)
    run_count = Column(Integer, default=0)
    last_run_status = Column(String(50), nullable=True)  # success, failed, cancelled
    last_run_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Workflow(id={self.id}, name={self.name}, actions={len(self.actions or [])})>"


class PostponeScheduleJob(Base):
    """
    Track Postpone-scheduled Reddit posts.

    Workflow:
    1. User approves composed videos
    2. Batch scheduling assigns each video to a consecutive day
    3. Within 24h of post time, a Celery task publishes the video to the media host
    4. Celery task calls Postpone API with the hosted link for Reddit scheduling
    """
    __tablename__ = "postpone_schedule_jobs"

    id = Column(Integer, primary_key=True, index=True)
    composed_video_id = Column(Integer, ForeignKey("composed_videos.id"), nullable=False)

    # Content info
    title = Column(String(300), nullable=False)
    niche = Column(String(50), nullable=True, index=True)
    reddit_username = Column(String(100), nullable=False)

    # Scheduling
    scheduled_date = Column(TIMESTAMP(timezone=True), nullable=False)  # Date for posting
    base_post_time = Column(TIMESTAMP(timezone=True), nullable=False)  # First subreddit post time
    stagger_minutes = Column(Integer, default=10)

    # Target subreddits (JSON array)
    target_subreddits = Column(JSON, nullable=False)

    # Postpone API tracking
    postpone_post_id = Column(String(100), nullable=True)
    postpone_response = Column(JSON, nullable=True)

    # Status: pending, scheduling, scheduled, failed, cancelled
    status = Column(String(30), default="pending", index=True)

    # Media host result — the link Postpone posts to Reddit
    hosted_media_id = Column(String(100), nullable=True)
    hosted_url = Column(Text, nullable=True)

    # Celery task tracking
    celery_task_id = Column(String(100), nullable=True, index=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    scheduled_at = Column(TIMESTAMP(timezone=True), nullable=True)  # When Postpone API was called
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    composed_video = relationship("ComposedVideo", back_populates="postpone_jobs")

    def __repr__(self):
        return f"<PostponeScheduleJob(id={self.id}, video={self.composed_video_id}, date={self.scheduled_date}, status={self.status})>"


class PatreonSubscriber(Base):
    """
    Source of truth for Patreon→Telegram membership sync.
    One row per (niche, patreon_user_id). Carries pledge status from the Patreon API
    plus the Telegram username the patron claimed in their Patreon post comment.
    """
    __tablename__ = "patreon_subscribers"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(50), nullable=False, index=True)

    patreon_user_id = Column(String(50), nullable=False)
    patreon_member_id = Column(String(50), nullable=True)
    full_name = Column(String(200), nullable=True)
    email = Column(String(200), nullable=True)

    patron_status = Column(String(50), nullable=True, index=True)
    pledge_relationship_start = Column(TIMESTAMP(timezone=True), nullable=True)
    last_charge_status = Column(String(50), nullable=True)
    last_charge_date = Column(TIMESTAMP(timezone=True), nullable=True)

    claimed_telegram_username = Column(String(100), nullable=True)
    claimed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    comment_id = Column(String(100), nullable=True)
    comment_last_modified = Column(TIMESTAMP(timezone=True), nullable=True)

    telegram_user_id = Column(BigInteger, nullable=True)
    telegram_state = Column(String(30), nullable=False, default="none")  # none/pending_request/in_channel/kicked

    last_synced_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("niche", "patreon_user_id", name="uq_patreon_subscribers_niche_user"),
    )

    def __repr__(self):
        return f"<PatreonSubscriber(id={self.id}, niche={self.niche}, user={self.patreon_user_id}, status={self.patron_status}, tg={self.telegram_state})>"


class TelegramJoinRequest(Base):
    """
    Persisted state for incoming chat_join_request updates from the Telegram bot listener.
    Pending requests are re-evaluated each hourly sync until matched, expired, or manually resolved.
    """
    __tablename__ = "telegram_join_requests"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(50), nullable=False, index=True)
    chat_id = Column(String(50), nullable=False)
    telegram_user_id = Column(BigInteger, nullable=False)
    telegram_username = Column(String(100), nullable=True)
    first_name = Column(String(200), nullable=True)
    last_name = Column(String(200), nullable=True)

    status = Column(String(30), nullable=False, default="pending")  # pending/approved/declined/expired
    received_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    resolved_at = Column(TIMESTAMP(timezone=True), nullable=True)
    matched_patreon_user_id = Column(String(50), nullable=True)
    decline_reason = Column(String(200), nullable=True)

    __table_args__ = (
        UniqueConstraint("chat_id", "telegram_user_id", name="uq_join_requests_chat_user"),
    )

    def __repr__(self):
        return f"<TelegramJoinRequest(id={self.id}, user={self.telegram_user_id}, username={self.telegram_username}, status={self.status})>"


class SyncAlert(Base):
    """
    Operational alerts surfaced in the Membership tab "Issues" view.
    Catch-all for sync failures, parse errors, expired requests, etc.
    """
    __tablename__ = "sync_alerts"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(50), nullable=True, index=True)
    severity = Column(String(20), nullable=False, default="warning")  # info/warning/error
    category = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    context = Column(JSONB, nullable=True)
    status = Column(String(20), nullable=False, default="open")  # open/resolved
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    resolved_at = Column(TIMESTAMP(timezone=True), nullable=True)

    def __repr__(self):
        return f"<SyncAlert(id={self.id}, severity={self.severity}, category={self.category}, status={self.status})>"


# ============================================================================
# Reddit analytics
# Backs the /analytics tab. Data scraped from Reddit via the existing
# RedditJsonScraper proxy pool. See tasks/reddit_analytics.py for the
# scrape jobs and api/analytics.py for the read endpoints.
# ============================================================================

class RedditAccountSnapshot(Base):
    """Per-Reddit-account karma + suspension state at fetch time."""
    __tablename__ = "reddit_account_snapshots"

    id = Column(BigInteger, primary_key=True, index=True)
    username = Column(String(100), nullable=False, index=True)
    fetched_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    total_karma = Column(Integer)
    link_karma = Column(Integer)
    comment_karma = Column(Integer)
    awardee_karma = Column(Integer)
    awarder_karma = Column(Integer)
    is_suspended = Column(Boolean, default=False, nullable=False)
    account_created_utc = Column(TIMESTAMP(timezone=True))
    verified_email = Column(Boolean)
    # Followers scraped from the rendered profile page via Playwright —
    # Reddit's JSON API returns 0 for all users. Nullable so a snapshot
    # row is still written when the Playwright probe fails.
    subscribers = Column(Integer)
    raw_about = Column(JSONB)


class RedditPost(Base):
    """One row per Reddit post we have ever observed for a tracked account.

    composed_video_id is filled in when the post's link URL matches a
    composed_videos.hosted_url. Lets the analytics tab show the caption
    text + BG context that produced this Reddit post.
    """
    __tablename__ = "reddit_posts"

    id = Column(BigInteger, primary_key=True, index=True)
    reddit_post_id = Column(String(40), nullable=False, unique=True, index=True)
    username = Column(String(100), nullable=False, index=True)
    subreddit = Column(String(100), nullable=False, index=True)
    title = Column(Text, nullable=False)
    link_url = Column(Text)
    permalink = Column(Text, nullable=False)
    created_utc = Column(TIMESTAMP(timezone=True))
    is_video = Column(Boolean, default=False)
    composed_video_id = Column(Integer, ForeignKey("composed_videos.id", ondelete="SET NULL"), nullable=True, index=True)
    first_seen_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    removed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    removed_reason = Column(String(80), nullable=True)

    def __repr__(self):
        return f"<RedditPost(reddit_id={self.reddit_post_id}, user={self.username}, sub={self.subreddit})>"


class RedditPostStat(Base):
    """One row per fetch per Reddit post — upvote/comment trajectory."""
    __tablename__ = "reddit_post_stats"

    id = Column(BigInteger, primary_key=True, index=True)
    reddit_post_id = Column(String(40), ForeignKey("reddit_posts.reddit_post_id", ondelete="CASCADE"), nullable=False, index=True)
    fetched_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    score = Column(Integer)
    num_comments = Column(Integer)
    upvote_ratio = Column(Float)
