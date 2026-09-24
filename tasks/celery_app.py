"""
Celery application configuration
"""
from celery import Celery
from celery.schedules import crontab
from config.settings import settings

# Initialize Celery app
celery_app = Celery(
    'captions_service',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'tasks.scraping_tasks',
        'tasks.incremental_scraping',
        'tasks.upvote_updater',
        'tasks.maintenance_tasks',
        'tasks.training_tasks',
        'tasks.scraper_dispatcher',
        'tasks.watchdog',
        'tasks.extraction_dispatcher',
        'tasks.watermark_filter',
        'tasks.video_composition_tasks',
        'tasks.pipeline_orchestrator',
        'tasks.publishing_tasks',
        'tasks.patreon_tasks',
        'tasks.telegram_tasks',
        'tasks.telegram_scraping',
        'tasks.reddit_background_scraping',
        'tasks.postpone_tasks',
        'tasks.media_host_tasks',
        'tasks.ml_tagging',
        'tasks.caption_judge',
        'tasks.bg_first_generation',
        'tasks.patreon_telegram_sync',
        'tasks.reddit_analytics',
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task execution
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minute soft limit
    worker_prefetch_multiplier=1,  # Fetch 1 task at a time

    # Result backend
    result_expires=86400,  # Results expire after 24 hours

    # Broker transport options - CRITICAL for long-running tasks
    # Default Redis visibility_timeout is 1 hour - if task takes longer, it gets redelivered!
    # Training tasks can take 12+ hours, so set visibility_timeout to match
    broker_transport_options={
        'visibility_timeout': 43200,  # 12 hours - matches training task time_limit
    },

    # Task routing
    # Queues:
    #   - scraping: Downloads, I/O-bound tasks (can run on CPU-only workers)
    #   - gpu: GPU-bound extraction tasks (requires CUDA worker)
    #   - maintenance: Lightweight admin tasks (schedulers, health checks)
    task_routes={
        'tasks.scraping_tasks.download_video_task': {'queue': 'scraping'},
        'tasks.scraping_tasks.scrape_subreddit': {'queue': 'scraping'},
        'tasks.scraping_tasks.scrape_all_subreddits': {'queue': 'scraping'},
        'tasks.scraping_tasks.get_scraping_stats': {'queue': 'maintenance'},
        'tasks.scraping_tasks.extract_caption_task': {'queue': 'gpu'},  # GPU extraction
        'tasks.incremental_scraping.*': {'queue': 'scraping'},
        'tasks.upvote_updater.*': {'queue': 'scraping'},
        'tasks.maintenance_tasks.*': {'queue': 'maintenance'},
        # GPU-specific maintenance tasks (need to run on GPU workers to access models)
        'tasks.maintenance_tasks.cleanup_idle_gpu_models': {'queue': 'gpu'},
        'tasks.maintenance_tasks.force_unload_gpu_models': {'queue': 'gpu'},
        # GPU status caching (runs on specific GPU workers to get accurate status)
        'tasks.maintenance_tasks.cache_gpu_0_status': {'queue': 'gpu_status_0'},
        'tasks.maintenance_tasks.cache_gpu_1_status': {'queue': 'gpu_status_1'},
        # LLM refinement runs on GPU 0 ONLY (gpu_llm queue) since GPU 1 has LLM disabled
        'tasks.scraping_tasks.llm_refine_batch_task': {'queue': 'gpu_llm'},
        # Training tasks - default to gpu but can be routed to gpu_training_0 or gpu_training_1
        # via apply_async(queue='gpu_training_0') for GPU-specific training
        'tasks.training_tasks.run_training_job': {'queue': 'gpu'},  # Default, can be overridden
        'tasks.training_tasks.get_gpu_0_status_task': {'queue': 'gpu_status_0'},  # GPU 0 status only
        'tasks.training_tasks.get_gpu_1_status_task': {'queue': 'gpu_status_1'},  # GPU 1 status only
        'tasks.training_tasks.*': {'queue': 'gpu'},  # Other training tasks use default GPU
        'tasks.watchdog.*': {'queue': 'maintenance'},  # Health monitoring
        'tasks.scraper_dispatcher.*': {'queue': 'maintenance'},  # Lightweight dispatchers
        'tasks.extraction_dispatcher.dispatch_extraction_batch': {'queue': 'maintenance'},  # Dispatch is lightweight - just queues GPU tasks
        'tasks.extraction_dispatcher.process_orphaned_videos': {'queue': 'maintenance'},
        'tasks.extraction_dispatcher.requeue_stale_extractions': {'queue': 'maintenance'},
        'tasks.extraction_dispatcher.get_extraction_queue_status': {'queue': 'maintenance'},
        # Pipeline orchestrator (lightweight state machine, runs on maintenance)
        'tasks.pipeline_orchestrator.pipeline_orchestrator': {'queue': 'maintenance'},
        # Publishing tasks (media host + Reddit posting)
        'tasks.publishing_tasks.*': {'queue': 'publishing'},
        # Patreon publishing tasks (browser-based, can share publishing queue)
        'tasks.patreon_tasks.*': {'queue': 'publishing'},
        # The schedule dispatcher is lightweight DB scanning; keep on maintenance
        'tasks.patreon_tasks.process_pending_patreon_schedules': {'queue': 'maintenance'},
        'tasks.telegram_tasks.process_pending_telegram_schedules': {'queue': 'maintenance'},
        # Telegram scraping tasks (uses scraping queue for downloads)
        'tasks.telegram_scraping.dispatch_telegram_scrapers': {'queue': 'maintenance'},
        'tasks.telegram_scraping.scrape_telegram_channel': {'queue': 'scraping'},
        'tasks.telegram_scraping.scrape_telegram_channel_manual': {'queue': 'scraping'},
        # Reddit background video scraping (downloads from Reddit for background videos)
        'tasks.reddit_background_scraping.scrape_reddit_background_subreddit': {'queue': 'scraping'},
        'tasks.reddit_background_scraping.dispatch_reddit_background_scraper': {'queue': 'maintenance'},
        # Postpone scheduling tasks
        'tasks.postpone_tasks.schedule_to_postpone': {'queue': 'publishing'},
        'tasks.postpone_tasks.process_pending_schedules': {'queue': 'maintenance'},
        # Media host tasks (for Postpone jobs missing a hosted_url)
        'tasks.media_host_tasks.publish_to_media_host': {'queue': 'publishing'},
        'tasks.media_host_tasks.dispatch_media_host_uploads': {'queue': 'maintenance'},
        # ML tagging tasks (VLM-based action classification for background videos)
        # run_tagging_batch runs on dedicated dual-GPU worker via ml_tagging queue
        'tasks.ml_tagging.dispatch_ml_tagging_batch': {'queue': 'maintenance'},
        'tasks.ml_tagging.retag_empty_actions': {'queue': 'maintenance'},
        'tasks.ml_tagging.run_tagging_batch': {'queue': 'ml_tagging'},
        # Stage 3 LLM judge: backlog scorer runs on gpu_llm where Mistral lives
        'tasks.caption_judge.judge_backlog_batch': {'queue': 'gpu_llm'},
        # Reddit analytics: per-account fetches go to scraping (proxy
        # pool, block detection), dispatchers + relinker are lightweight
        # and stay on maintenance.
        'tasks.reddit_analytics.refresh_account_snapshot': {'queue': 'scraping'},
        'tasks.reddit_analytics.refresh_account_posts': {'queue': 'scraping'},
        'tasks.reddit_analytics.refresh_post_stats_aged': {'queue': 'scraping'},
        'tasks.reddit_analytics.detect_removed_posts': {'queue': 'scraping'},
        'tasks.reddit_analytics.dispatch_account_snapshots': {'queue': 'maintenance'},
        'tasks.reddit_analytics.dispatch_post_stats_fresh': {'queue': 'maintenance'},
        'tasks.reddit_analytics.dispatch_post_stats_aged': {'queue': 'maintenance'},
        'tasks.reddit_analytics.dispatch_detect_removed': {'queue': 'maintenance'},
        'tasks.reddit_analytics.relink_posts_to_composed_videos': {'queue': 'maintenance'},
    },

    # Retry policy
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Beat schedule (periodic tasks)
    # NOTE: Scraper tasks controlled via Redis flag - use API to enable/disable
    # GET /scraper/control/status - check current state
    # POST /scraper/control/enable - enable all
    # POST /scraper/control/disable - disable all
    beat_schedule={
        # Dispatcher checks Redis and dispatches scraper if enabled (every 10 min)
        'dispatch-scraper-tasks': {
            'task': 'tasks.scraper_dispatcher.dispatch_scraper_tasks',
            'schedule': crontab(minute='*/10'),
        },

        # Dispatcher checks Redis and dispatches upvote updater if enabled (hourly)
        'dispatch-upvote-tasks': {
            'task': 'tasks.scraper_dispatcher.dispatch_upvote_tasks',
            'schedule': crontab(minute=0),
        },

        # Reddit health check hourly (checks assigned proxies, not direct IP)
        'reddit-health-check': {
            'task': 'tasks.scraper_dispatcher.check_reddit_health',
            'schedule': crontab(minute=30),  # Run at :30 of each hour
        },

        # Database maintenance daily (2 AM) - always runs
        'database-maintenance': {
            'task': 'tasks.maintenance_tasks.database_maintenance',
            'schedule': crontab(hour=2, minute=0),
        },

        # Worker health check every 5 minutes (runs on beat, monitors worker)
        'worker-health-check': {
            'task': 'tasks.watchdog.check_worker_health',
            'schedule': crontab(minute='*/5'),
        },

        # === Extraction Queue (Event-driven architecture) ===
        # GPU extraction dispatcher - processes videos from extraction queue
        # Runs every minute, queues tasks for both GPUs to process in parallel
        'dispatch-extraction-batch': {
            'task': 'tasks.extraction_dispatcher.dispatch_extraction_batch',
            'schedule': crontab(minute='*'),
            'kwargs': {'batch_size': 20},  # Queue 20 tasks, both GPUs pick them up
        },

        # Fallback: catch orphaned videos that missed the queue (every 30 min)
        'process-orphaned-videos': {
            'task': 'tasks.extraction_dispatcher.process_orphaned_videos',
            'schedule': crontab(minute='*/30'),
            'kwargs': {'max_videos': 50},
        },

        # Requeue stale extractions that hung (every 15 min)
        'requeue-stale-extractions': {
            'task': 'tasks.extraction_dispatcher.requeue_stale_extractions',
            'schedule': crontab(minute='*/15'),
            'kwargs': {'max_age_minutes': 30},
        },

        # === Watermark Filtering (Decoupled from Scraping) ===
        # Dispatches pending videos for OCR-based watermark detection
        # Enable via: POST /background-videos/filter/control/enable
        # GPU selection: POST /background-videos/filter/control/set-gpu?gpu=0|1|both
        'dispatch-filter-batch': {
            'task': 'tasks.watermark_filter.dispatch_filter_batch',
            'schedule': crontab(minute='*/5'),  # Every 5 minutes
            'kwargs': {'batch_size': 10},
        },

        # === ML Action Tagging (Background Videos) ===
        # VLM-based scene tagging via InternVL3-14B GGUF on llama-server.
        # No standalone beat — the pipeline orchestrator dispatches a
        # tagging batch from its idle path, so tagging only runs while no
        # other GPU work is queued. The transient-error reset still runs
        # so retried failures can be picked up the next time the
        # orchestrator is idle.
        'reset-transient-tagging-errors': {
            'task': 'tasks.ml_tagging.reset_transient_tagging_errors',
            'schedule': crontab(minute='15,45'),  # Twice an hour
            'kwargs': {'max_per_run': 500},
        },

        # === LLM Refinement Backlog ===
        # Process captions that were extracted without LLM refinement
        # GPU 1 worker only does OCR (8GB VRAM), LLM runs on GPU 0 (11GB VRAM)
        # This task catches up on LLM refinement every 10 minutes
        'llm-refine-backlog': {
            'task': 'tasks.scraping_tasks.llm_refine_batch_task',
            'schedule': crontab(minute='*/10'),
            'kwargs': {'batch_size': 50},
        },

        # === Caption Judge Backlog (Stage 3) ===
        # Score un-judged generated_captions with Mistral-Small judge prompt.
        # Skips itself if llama-server is not running, so it's safe even when
        # the orchestrator has the VLM profile loaded for tagging.
        'judge-caption-backlog': {
            'task': 'tasks.caption_judge.judge_backlog_batch',
            'schedule': crontab(minute='*/15'),
            'kwargs': {'batch_size': 25},
        },

        # === GPU Memory Cleanup ===
        # Unload idle models to free GPU memory when not in use
        # Runs every 2 minutes, unloads models idle for 2+ minutes
        'cleanup-idle-gpu-models': {
            'task': 'tasks.maintenance_tasks.cleanup_idle_gpu_models',
            'schedule': crontab(minute='*/2'),
            'kwargs': {'idle_timeout_minutes': 2},
        },

        # === Patreon→Telegram Membership Sync (DISABLED) ===
        # Was: hourly at :25, pulled Patreon active-member list, scraped the
        # designated post's comments for @username claims, re-evaluated
        # pending join requests, and kicked lapsed patrons. Disabled
        # 2026-05-07 — being handled manually for now.
        # 'sync-patreon-telegram-motivation': {
        #     'task': 'tasks.patreon_telegram_sync.sync_patreon_telegram',
        #     'schedule': crontab(minute=25),
        #     'args': ('motivation',),
        #     'options': {'queue': 'maintenance'},
        # },

        # === GPU Status Caching ===
        # Cache GPU status periodically so API can read from cache
        # instead of querying workers (which times out when busy)
        'cache-gpu-0-status': {
            'task': 'tasks.maintenance_tasks.cache_gpu_0_status',
            'schedule': 30.0,  # Every 30 seconds
        },
        'cache-gpu-1-status': {
            'task': 'tasks.maintenance_tasks.cache_gpu_1_status',
            'schedule': 30.0,  # Every 30 seconds
        },

        # === Pipeline Orchestrator (Automation) ===
        # Manages per-niche training, generation, and composition
        # Enable via: POST /pipeline/enable (or set pipeline:enabled=true in Redis)
        # Disabled by default - enable when ready to run automated pipeline
        'pipeline-orchestrator': {
            'task': 'tasks.pipeline_orchestrator.pipeline_orchestrator',
            'schedule': crontab(minute='*/5'),  # Every 5 minutes
        },

        # === Publishing Tasks ===
        # Check for pending crossposts that are due (fallback if ETA task missed)
        'check-pending-crossposts': {
            'task': 'tasks.publishing_tasks.check_pending_crossposts',
            'schedule': crontab(minute='*/5'),  # Every 5 minutes
        },

        # === Telegram Scraping ===
        # Dispatcher checks Redis flag and scrapes enabled channels (every 10 min)
        # Enable via: POST /telegram/scrape/control/enable
        'dispatch-telegram-scrapers': {
            'task': 'tasks.telegram_scraping.dispatch_telegram_scrapers',
            'schedule': crontab(minute='*/10'),
        },

        # === Reddit Background Video Scraping ===
        # Scrapes background-video subreddits (separate from caption scraping)
        # Enable via: POST /background-videos/reddit/control/enable
        'dispatch-reddit-background-scraper': {
            'task': 'tasks.reddit_background_scraping.dispatch_reddit_background_scraper',
            'schedule': crontab(minute='*/10'),
        },

        # === Media Host (for Postpone jobs) ===
        # Publish composed videos to the media host for jobs within 24h of post time
        'dispatch-media-host-uploads': {
            'task': 'tasks.media_host_tasks.dispatch_media_host_uploads',
            'schedule': crontab(minute='*/10'),
        },

        # === Postpone Scheduling ===
        # Process pending Postpone schedule jobs within 24h of post time
        'process-postpone-schedules': {
            'task': 'tasks.postpone_tasks.process_pending_schedules',
            'schedule': crontab(minute='*/10'),
        },

        # === Patreon Scheduling ===
        # Promote scheduled Patreon jobs to pending and dispatch the publish task
        'process-patreon-schedules': {
            'task': 'tasks.patreon_tasks.process_pending_patreon_schedules',
            'schedule': crontab(minute='*/10'),
        },

        # === Telegram Scheduling ===
        # Promote scheduled Telegram jobs to pending and dispatch the publish task
        'process-telegram-schedules': {
            'task': 'tasks.telegram_tasks.process_pending_telegram_schedules',
            'schedule': crontab(minute='*/10'),
        },

        # === Reddit Analytics ===
        # Snapshot karma for tracked Reddit accounts. Staggered at :02 so
        # we don't collide with the :00/:10/:15/:30 scrape dispatchers.
        # Suspended accounts get auto-skipped except at the 02:30 daily
        # window — see dispatch_account_snapshots() logic.
        'dispatch-reddit-account-snapshots': {
            'task': 'tasks.reddit_analytics.dispatch_account_snapshots',
            'schedule': crontab(minute='2,17,32,47'),
        },
        # Refresh post stats for fresh (<48h) posts. Same 15-min cadence,
        # offset by 5 minutes from snapshot dispatch so the proxy pool
        # has a chance to breathe between the two passes.
        'dispatch-reddit-post-stats-fresh': {
            'task': 'tasks.reddit_analytics.dispatch_post_stats_fresh',
            'schedule': crontab(minute='7,22,37,52'),
        },
        # Daily refresh for aged posts (48h-30d old).
        'dispatch-reddit-post-stats-aged': {
            'task': 'tasks.reddit_analytics.dispatch_post_stats_aged',
            'schedule': crontab(hour=2, minute=30),
        },
        # Hourly: mark posts as removed if Reddit stops returning them.
        'detect-removed-reddit-posts': {
            'task': 'tasks.reddit_analytics.dispatch_detect_removed',
            'schedule': crontab(minute=12),
        },
    }
)

if __name__ == '__main__':
    celery_app.start()
