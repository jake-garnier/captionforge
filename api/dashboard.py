"""
Dashboard API endpoints for metrics and monitoring
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from database.db import get_db
from database.models import Video, ScrapedCaption, ScrapingProgress
from tasks.celery_app import celery_app
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview")
async def get_dashboard_overview(db: Session = Depends(get_db)):
    """
    Get high-level overview metrics for dashboard
    """
    try:
        # Total videos and storage
        total_videos = db.query(Video).count()
        total_storage_bytes = db.query(func.sum(Video.file_size_bytes)).scalar() or 0
        total_storage_gb = round(total_storage_bytes / (1024**3), 2)

        # Videos by status
        status_counts = db.query(
            Video.processing_status,
            func.count(Video.id)
        ).group_by(Video.processing_status).all()

        by_status = {status: count for status, count in status_counts}

        # Videos by subreddit
        subreddit_counts = db.query(
            Video.source_subreddit,
            func.count(Video.id)
        ).group_by(Video.source_subreddit).all()

        by_subreddit = {sub: count for sub, count in subreddit_counts}

        # Videos with captions
        videos_with_captions = db.query(ScrapedCaption).distinct(ScrapedCaption.video_id).count()
        caption_extraction_rate = round((videos_with_captions / total_videos * 100), 1) if total_videos > 0 else 0

        # Active scrapers
        active_scrapers = db.query(ScrapingProgress).filter(
            ScrapingProgress.scraping_active == True
        ).count()

        # Recent activity (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        videos_last_24h = db.query(Video).filter(
            Video.download_date >= yesterday
        ).count()

        # Average upvotes
        avg_upvotes = db.query(func.avg(Video.upvotes)).scalar() or 0

        return {
            "total_videos": total_videos,
            "total_storage_gb": total_storage_gb,
            "by_status": by_status,
            "by_subreddit": by_subreddit,
            "caption_extraction_rate": caption_extraction_rate,
            "active_scrapers": active_scrapers,
            "videos_last_24h": videos_last_24h,
            "avg_upvotes": round(avg_upvotes, 0),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/progress")
async def get_all_scraping_progress(db: Session = Depends(get_db)):
    """
    Get scraping progress for all subreddits
    """
    try:
        progress_records = db.query(ScrapingProgress).all()

        subreddits_data = []
        for record in progress_records:
            # Calculate scraping velocity (posts per day)
            if record.created_at:
                # Ensure both datetimes are timezone-aware for comparison
                now_utc = datetime.now(record.created_at.tzinfo) if record.created_at.tzinfo else datetime.utcnow()
                days_active = (now_utc - record.created_at).days or 1
                posts_per_day = round(record.posts_scraped / days_active, 1)
            else:
                posts_per_day = 0

            # Time since last scrape
            hours_since_scrape = None
            if record.last_scrape_at:
                # Ensure both datetimes are timezone-aware for comparison
                now_utc = datetime.now(record.last_scrape_at.tzinfo) if record.last_scrape_at.tzinfo else datetime.utcnow()
                hours_since_scrape = round((now_utc - record.last_scrape_at).total_seconds() / 3600, 1)

            subreddits_data.append({
                "subreddit": record.subreddit,
                "posts_scraped": record.posts_scraped,
                "videos_downloaded": record.videos_downloaded,
                "videos_failed": record.videos_failed,
                "success_rate": round((record.videos_downloaded / record.posts_scraped * 100), 1) if record.posts_scraped > 0 else 0,
                "last_post_id": record.last_post_id,
                "last_post_score": record.last_post_score,
                "last_pagination_url": record.last_pagination_url,
                "target_min_score": record.target_min_score,
                "scraping_active": record.scraping_active,
                "last_scrape_at": record.last_scrape_at.isoformat() if record.last_scrape_at else None,
                "hours_since_scrape": hours_since_scrape,
                "posts_per_day": posts_per_day,
                "created_at": record.created_at.isoformat() if record.created_at else None
            })

        return {
            "total_subreddits": len(subreddits_data),
            "subreddits": subreddits_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scraping progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/schedule")
async def get_scraping_schedule(db: Session = Depends(get_db)):
    """
    Get scraper schedule information including next run times
    """
    try:
        from tasks.celery_app import celery_app
        from celery.schedules import crontab

        # Get current time
        now = datetime.utcnow()

        # Get schedule from celery beat config
        beat_schedule = celery_app.conf.beat_schedule

        schedules = []
        for task_name, task_config in beat_schedule.items():
            schedule = task_config.get('schedule')
            task_path = task_config.get('task')
            args = task_config.get('args', ())

            # Calculate next run time for crontab schedules
            next_run = None
            schedule_str = None

            if isinstance(schedule, crontab):
                # Format schedule string
                schedule_str = f"crontab(minute={schedule._orig_minute}, hour={schedule._orig_hour})"
                if schedule._orig_day_of_week:
                    schedule_str = f"crontab(day_of_week={schedule._orig_day_of_week}, hour={schedule._orig_hour}, minute={schedule._orig_minute})"

                # Calculate next run time based on pattern
                # Handle minute intervals (e.g., */30)
                if schedule._orig_minute and '*/' in str(schedule._orig_minute):
                    interval = int(str(schedule._orig_minute).split('/')[-1])
                    current_minute = now.minute
                    next_minute = ((current_minute // interval) + 1) * interval
                    if next_minute >= 60:
                        next_minute = 0
                        next_run = now.replace(minute=next_minute, second=0, microsecond=0) + timedelta(hours=1)
                    else:
                        next_run = now.replace(minute=next_minute, second=0, microsecond=0)
                        if next_run <= now:
                            next_run += timedelta(minutes=interval)

                # Handle hour intervals (e.g., */12)
                elif schedule._orig_hour and '*/' in str(schedule._orig_hour):
                    interval = int(str(schedule._orig_hour).split('/')[-1])
                    current_hour = now.hour
                    minute = int(schedule._orig_minute) if schedule._orig_minute and str(schedule._orig_minute).isdigit() else 0
                    next_hour = ((current_hour // interval) + 1) * interval
                    if next_hour >= 24:
                        next_hour = 0
                        next_run = now.replace(hour=next_hour, minute=minute, second=0, microsecond=0) + timedelta(days=1)
                    else:
                        next_run = now.replace(hour=next_hour, minute=minute, second=0, microsecond=0)
                        if next_run <= now:
                            next_run += timedelta(hours=interval)

                # Handle weekly schedules (e.g., day_of_week=1)
                elif schedule._orig_day_of_week and str(schedule._orig_day_of_week).isdigit():
                    target_day = int(schedule._orig_day_of_week)
                    hour = int(schedule._orig_hour) if schedule._orig_hour and str(schedule._orig_hour).isdigit() else 0
                    minute = int(schedule._orig_minute) if schedule._orig_minute and str(schedule._orig_minute).isdigit() else 0

                    # Calculate days until next occurrence
                    current_day = now.weekday()  # Monday=0, Sunday=6
                    days_ahead = target_day - current_day
                    if days_ahead <= 0:  # Target day already happened this week
                        days_ahead += 7

                    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days_ahead)
                    # If target day is today but time hasn't passed yet
                    if days_ahead == 7 and next_run > now:
                        next_run -= timedelta(days=7)

                # Handle hourly schedules (minute=0 or minute=X, hour=*)
                # This covers crontab(minute=0) which runs every hour at :00
                elif (schedule._orig_hour == '*' or schedule._orig_hour is None) and \
                     schedule._orig_minute is not None and str(schedule._orig_minute).isdigit():
                    target_minute = int(schedule._orig_minute)
                    next_run = now.replace(minute=target_minute, second=0, microsecond=0)
                    if next_run <= now:
                        next_run += timedelta(hours=1)

                # Handle fixed daily times (e.g., hour=4, minute=0)
                elif schedule._orig_hour and str(schedule._orig_hour).isdigit():
                    hour = int(schedule._orig_hour)
                    minute = int(schedule._orig_minute) if schedule._orig_minute and str(schedule._orig_minute).isdigit() else 0
                    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if next_run <= now:
                        next_run += timedelta(days=1)

            # Get related subreddit progress if available
            subreddit_progress = None
            if 'scrape' in task_name.lower() and args:
                subreddit_name = args[0] if isinstance(args, tuple) and len(args) > 0 else None
                if subreddit_name:
                    progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()
                    if progress:
                        subreddit_progress = {
                            "subreddit": progress.subreddit,
                            "posts_scraped": progress.posts_scraped,
                            "videos_downloaded": progress.videos_downloaded,
                            "videos_failed": progress.videos_failed,
                            "success_rate": round((progress.videos_downloaded / progress.posts_scraped * 100), 1) if progress.posts_scraped > 0 else 0,
                            "last_scrape_at": progress.last_scrape_at.isoformat() if progress.last_scrape_at else None,
                            "scraping_active": progress.scraping_active
                        }

            schedules.append({
                "task_name": task_name,
                "task": task_path,
                "schedule": schedule_str,
                "args": args,
                "next_run": next_run.isoformat() if next_run else None,
                "minutes_until_next_run": round((next_run - now).total_seconds() / 60) if next_run else None,
                "subreddit_progress": subreddit_progress
            })

        return {
            "current_time": now.isoformat(),
            "schedules": schedules,
            "timestamp": now.isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scraping schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scraping/trigger/{task_name}")
async def trigger_scheduled_task(task_name: str):
    """
    Manually trigger a scheduled task
    """
    try:
        from tasks.celery_app import celery_app

        # Get task configuration from beat schedule
        beat_schedule = celery_app.conf.beat_schedule

        if task_name not in beat_schedule:
            raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found in schedule")

        task_config = beat_schedule[task_name]
        task_path = task_config.get('task')
        args = task_config.get('args', ())
        kwargs = task_config.get('kwargs', {})

        # Trigger the task
        result = celery_app.send_task(task_path, args=args, kwargs=kwargs)

        return {
            "status": "success",
            "task_name": task_name,
            "task_id": result.id,
            "task_path": task_path,
            "args": args,
            "message": f"Task '{task_name}' triggered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/timeline")
async def get_scraping_timeline(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get videos downloaded over time (daily breakdown)
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)

        # Query videos grouped by date
        timeline_data = db.query(
            func.date(Video.download_date).label('date'),
            func.count(Video.id).label('count'),
            Video.source_subreddit
        ).filter(
            Video.download_date >= start_date
        ).group_by(
            func.date(Video.download_date),
            Video.source_subreddit
        ).order_by(
            func.date(Video.download_date)
        ).all()

        # Organize by date
        timeline_by_date = {}
        for date, count, subreddit in timeline_data:
            date_str = date.isoformat() if date else "unknown"
            if date_str not in timeline_by_date:
                timeline_by_date[date_str] = {
                    "total": 0,
                    "by_subreddit": {}
                }
            timeline_by_date[date_str]["total"] += count
            timeline_by_date[date_str]["by_subreddit"][subreddit] = count

        # Convert to list sorted by date
        timeline = [
            {
                "date": date,
                "total_videos": data["total"],
                "by_subreddit": data["by_subreddit"]
            }
            for date, data in sorted(timeline_by_date.items())
        ]

        return {
            "days": days,
            "timeline": timeline,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scraping timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upvotes/trending")
async def get_trending_videos(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get videos with highest upvote counts
    """
    try:
        trending = db.query(Video).order_by(
            desc(Video.upvotes)
        ).limit(limit).all()

        videos = []
        for video in trending:
            videos.append({
                "post_id": video.source_post_id,
                "subreddit": video.source_subreddit,
                "upvotes": video.upvotes,
                "duration": video.duration_seconds,
                "last_check": video.last_upvote_check.isoformat() if video.last_upvote_check else None,
                "download_date": video.download_date.isoformat() if video.download_date else None,
                "storage_path": video.storage_path
            })

        return {
            "trending_videos": videos,
            "count": len(videos),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting trending videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upvotes/stale")
async def get_stale_upvote_stats(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """
    Get statistics on videos with stale upvote data
    """
    try:
        stale_threshold = datetime.utcnow() - timedelta(hours=hours)

        # Count stale videos
        stale_count = db.query(Video).filter(
            and_(
                Video.last_upvote_check < stale_threshold,
                Video.last_upvote_check.isnot(None)
            )
        ).count()

        # Count never checked
        never_checked = db.query(Video).filter(
            Video.last_upvote_check.is_(None)
        ).count()

        # Count recently checked
        recent_count = db.query(Video).filter(
            Video.last_upvote_check >= stale_threshold
        ).count()

        return {
            "stale_hours": hours,
            "stale_count": stale_count,
            "never_checked_count": never_checked,
            "recently_checked_count": recent_count,
            "total_videos": stale_count + never_checked + recent_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stale upvote stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/pipeline")
async def get_processing_pipeline_stats(db: Session = Depends(get_db)):
    """
    Get video processing pipeline statistics
    """
    try:
        # Count by processing status
        pipeline_stats = db.query(
            Video.processing_status,
            func.count(Video.id)
        ).group_by(Video.processing_status).all()

        status_counts = {status: count for status, count in pipeline_stats}

        # Calculate success rate
        total_videos = sum(status_counts.values())
        completed = status_counts.get('caption_extracted', 0) + status_counts.get('completed', 0)
        success_rate = round((completed / total_videos * 100), 1) if total_videos > 0 else 0

        # Find stuck videos (downloaded but not processing)
        stuck_threshold = datetime.utcnow() - timedelta(hours=6)
        stuck_count = db.query(Video).filter(
            and_(
                Video.processing_status == 'downloaded',
                Video.download_date < stuck_threshold
            )
        ).count()

        return {
            "by_status": status_counts,
            "total_videos": total_videos,
            "success_rate": success_rate,
            "stuck_videos": stuck_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting processing pipeline stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/breakdown")
async def get_storage_breakdown(db: Session = Depends(get_db)):
    """
    Get storage usage breakdown by subreddit and status
    """
    try:
        # Storage by subreddit
        subreddit_storage = db.query(
            Video.source_subreddit,
            func.count(Video.id).label('video_count'),
            func.sum(Video.file_size_bytes).label('total_bytes')
        ).group_by(Video.source_subreddit).all()

        by_subreddit = []
        for subreddit, count, total_bytes in subreddit_storage:
            total_bytes = total_bytes or 0
            by_subreddit.append({
                "subreddit": subreddit,
                "video_count": count,
                "size_gb": round(total_bytes / (1024**3), 2),
                "avg_size_mb": round(total_bytes / count / (1024**2), 1) if count > 0 else 0
            })

        # Overall stats
        total_videos = db.query(Video).count()
        total_bytes = db.query(func.sum(Video.file_size_bytes)).scalar() or 0

        # Average video duration
        avg_duration = db.query(func.avg(Video.duration_seconds)).scalar() or 0

        return {
            "total_storage_gb": round(total_bytes / (1024**3), 2),
            "total_videos": total_videos,
            "avg_video_size_mb": round(total_bytes / total_videos / (1024**2), 1) if total_videos > 0 else 0,
            "avg_duration_seconds": round(avg_duration, 1),
            "by_subreddit": sorted(by_subreddit, key=lambda x: x['size_gb'], reverse=True),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting storage breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/celery/tasks")
async def get_celery_task_stats():
    """
    Get Celery task statistics from broker
    """
    try:
        # Get active tasks
        inspect = celery_app.control.inspect()

        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        reserved_tasks = inspect.reserved()

        # Count tasks by type
        active_count = sum(len(tasks) for tasks in (active_tasks or {}).values())
        scheduled_count = sum(len(tasks) for tasks in (scheduled_tasks or {}).values())
        reserved_count = sum(len(tasks) for tasks in (reserved_tasks or {}).values())

        # Get registered tasks
        registered = inspect.registered()
        registered_tasks = list(registered.values())[0] if registered else []

        return {
            "active_tasks": active_count,
            "scheduled_tasks": scheduled_count,
            "reserved_tasks": reserved_count,
            "registered_tasks_count": len(registered_tasks),
            "workers": list(active_tasks.keys()) if active_tasks else [],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting Celery task stats: {e}")
        return {
            "error": str(e),
            "active_tasks": 0,
            "scheduled_tasks": 0,
            "reserved_tasks": 0,
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/captions/stats")
async def get_caption_stats(db: Session = Depends(get_db)):
    """
    Get caption extraction and refinement statistics
    """
    try:
        total_captions = db.query(ScrapedCaption).count()

        # Count captions with each refinement level
        with_raw_ocr = db.query(ScrapedCaption).filter(
            ScrapedCaption.raw_ocr_text.isnot(None)
        ).count()

        with_rule_based = db.query(ScrapedCaption).filter(
            ScrapedCaption.rule_based_text.isnot(None)
        ).count()

        with_llm_refined = db.query(ScrapedCaption).filter(
            ScrapedCaption.llm_refined_text.isnot(None)
        ).count()

        # Average caption length
        avg_raw_length = db.query(func.avg(func.length(ScrapedCaption.raw_ocr_text))).scalar() or 0
        avg_rule_length = db.query(func.avg(func.length(ScrapedCaption.rule_based_text))).scalar() or 0
        avg_llm_length = db.query(func.avg(func.length(ScrapedCaption.llm_refined_text))).scalar() or 0

        return {
            "total_captions": total_captions,
            "with_raw_ocr": with_raw_ocr,
            "with_rule_based": with_rule_based,
            "with_llm_refined": with_llm_refined,
            "llm_refinement_rate": round((with_llm_refined / total_captions * 100), 1) if total_captions > 0 else 0,
            "avg_caption_length": {
                "raw_ocr": round(avg_raw_length, 0),
                "rule_based": round(avg_rule_length, 0),
                "llm_refined": round(avg_llm_length, 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting caption stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_class=HTMLResponse)
async def get_tasks_control_panel():
    """
    Simple control panel for triggering tasks
    """
    from tasks.celery_app import celery_app

    beat_schedule = celery_app.conf.beat_schedule

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Task Control Panel</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1a1a2e;
                color: #eee;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                color: #00ff41;
                border-bottom: 2px solid #00ff41;
                padding-bottom: 10px;
            }
            .task-card {
                background: #16213e;
                border: 1px solid #0f3460;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
            }
            .task-name {
                font-size: 1.3em;
                color: #00d9ff;
                margin-bottom: 10px;
            }
            .task-info {
                color: #aaa;
                font-size: 0.9em;
                margin: 5px 0;
            }
            .button-group {
                margin-top: 15px;
                display: flex;
                gap: 10px;
            }
            button {
                padding: 10px 20px;
                font-size: 1em;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s;
            }
            .trigger-btn {
                background: #00ff41;
                color: #000;
            }
            .trigger-btn:hover {
                background: #00cc34;
                transform: scale(1.05);
            }
            .logs-btn {
                background: #0f3460;
                color: #fff;
            }
            .logs-btn:hover {
                background: #1a4d7a;
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 0.85em;
                margin-left: 10px;
            }
            .status.success {
                background: #00ff4133;
                color: #00ff41;
            }
            .status.error {
                background: #ff414133;
                color: #ff4141;
            }
            .result {
                margin-top: 10px;
                padding: 10px;
                border-radius: 5px;
                display: none;
            }
        </style>
    </head>
    <body>
        <h1>🎛️ Task Control Panel</h1>
        <p>Manually trigger scheduled tasks and view their logs</p>
"""

    for task_name, task_config in beat_schedule.items():
        task_path = task_config.get('task', 'Unknown')
        args = task_config.get('args', ())
        schedule = task_config.get('schedule')

        schedule_str = str(schedule) if schedule else "No schedule"

        html_content += f"""
        <div class="task-card">
            <div class="task-name">{task_name}</div>
            <div class="task-info">Task: {task_path}</div>
            <div class="task-info">Args: {args}</div>
            <div class="task-info">Schedule: {schedule_str}</div>
            <div class="button-group">
                <button class="trigger-btn" onclick="triggerTask('{task_name}', this)">
                    ▶️ Trigger Now
                </button>
                <button class="logs-btn" onclick="window.open('/dashboard/logs/{task_name}', '_blank')">
                    📋 View Logs
                </button>
            </div>
            <div class="result" id="result-{task_name}"></div>
        </div>
"""

    html_content += """
        <script>
            async function triggerTask(taskName, button) {
                const resultDiv = document.getElementById('result-' + taskName);
                resultDiv.style.display = 'block';
                resultDiv.className = 'result';
                resultDiv.innerHTML = '⏳ Triggering task...';

                button.disabled = true;
                button.textContent = '⏳ Triggering...';

                try {
                    const response = await fetch(`/dashboard/scraping/trigger/${taskName}`, {
                        method: 'POST'
                    });

                    const data = await response.json();

                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <span class="status success">✅ Success</span><br>
                            Task ID: ${data.task_id}<br>
                            ${data.message}
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <span class="status error">❌ Error</span><br>
                            ${data.detail || 'Unknown error'}
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <span class="status error">❌ Error</span><br>
                        ${error.message}
                    `;
                }

                button.disabled = false;
                button.textContent = '▶️ Trigger Now';
            }
        </script>
    </body>
    </html>
    """

    return html_content



# NOTE: Task log streaming endpoints removed (SSE was unreliable)
# Use docker-compose logs -f celery-worker for real-time logs
# Or access Flower dashboard at :5555 for task monitoring

@router.get("/worker/status")
async def get_worker_status():
    """
    Get current worker health status from heartbeat system.

    Returns:
        Worker health status, current task info, and hang detection.
        worker_state can be: idle, processing, potentially_hung, idle_stale_heartbeat, error
    """
    try:
        from utils.worker_heartbeat import get_worker_status as get_status

        status = get_status()
        worker_state = status.get('worker_state', 'unknown')
        is_healthy = status.get('is_healthy', False)
        hang_detected = status.get('hang_detected', False)

        # Determine recommendation based on worker state
        recommendation = None
        if hang_detected or worker_state == 'potentially_hung':
            recommendation = "restart_worker"

        return {
            "is_healthy": is_healthy,
            "worker_state": worker_state,
            "hang_detected": hang_detected,
            "heartbeat": status.get('heartbeat'),
            "current_task": status.get('current_task'),
            "hang_duration_seconds": status.get('hang_duration_seconds'),
            "recommendation": recommendation
        }
    except ImportError:
        return {
            "error": "Heartbeat system not available",
            "is_healthy": None,
            "worker_state": "unknown",
            "hang_detected": None
        }
    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        return {
            "error": str(e),
            "is_healthy": False,
            "worker_state": "error",
            "hang_detected": None
        }


@router.post("/worker/restart")
async def restart_worker():
    """
    Trigger worker restart via watchdog task.

    Note: Requires WATCHDOG_AUTO_RECOVERY=true environment variable.
    """
    try:
        from tasks.watchdog import trigger_worker_restart

        success = trigger_worker_restart()

        if success:
            return {
                "status": "restart_triggered",
                "message": "Worker restart command sent successfully"
            }
        else:
            return {
                "status": "restart_failed",
                "message": "Failed to trigger worker restart. Check if docker is accessible."
            }
    except Exception as e:
        logger.error(f"Error restarting worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hang-incidents")
async def list_hang_incidents(
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """
    List recent hang incidents for debugging and pattern analysis.

    Returns:
        List of hang incidents with key diagnostic info
    """
    try:
        from database.models import HangIncident

        incidents = db.query(HangIncident).order_by(
            desc(HangIncident.detected_at)
        ).offset(offset).limit(limit).all()

        total = db.query(HangIncident).count()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "incidents": [
                {
                    "id": inc.id,
                    "detected_at": inc.detected_at.isoformat() if inc.detected_at else None,
                    "task_name": inc.task_name,
                    "video_id": inc.video_id,
                    "video_subreddit": inc.video_subreddit,
                    "video_file_size_mb": inc.video_file_size_mb,
                    "video_duration_seconds": inc.video_duration_seconds,
                    "task_duration_seconds": inc.task_duration_seconds,
                    "last_heartbeat_stage": inc.last_heartbeat_stage,
                    "heartbeat_age_seconds": inc.heartbeat_age_seconds,
                    "gpu_memory_used_mb": inc.gpu_memory_used_mb,
                    "gpu_utilization_percent": inc.gpu_utilization_percent,
                    "recovery_action": inc.recovery_action,
                    "recovery_successful": inc.recovery_successful
                }
                for inc in incidents
            ]
        }
    except Exception as e:
        logger.error(f"Error listing hang incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hang-incidents/{incident_id}")
async def get_hang_incident(incident_id: int, db: Session = Depends(get_db)):
    """
    Get full details of a specific hang incident.

    Returns:
        Complete incident data including full diagnostics JSON
    """
    try:
        from database.models import HangIncident

        incident = db.query(HangIncident).filter_by(id=incident_id).first()

        if not incident:
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

        return {
            "id": incident.id,
            "detected_at": incident.detected_at.isoformat() if incident.detected_at else None,

            # Task info
            "task_name": incident.task_name,
            "task_id": incident.task_id,
            "task_duration_seconds": incident.task_duration_seconds,

            # Heartbeat
            "last_heartbeat_stage": incident.last_heartbeat_stage,
            "last_heartbeat_at": incident.last_heartbeat_at.isoformat() if incident.last_heartbeat_at else None,
            "heartbeat_age_seconds": incident.heartbeat_age_seconds,

            # Video info
            "video_id": incident.video_id,
            "video_subreddit": incident.video_subreddit,
            "video_file_size_mb": incident.video_file_size_mb,
            "video_duration_seconds": incident.video_duration_seconds,
            "video_resolution": incident.video_resolution,

            # GPU state
            "gpu_memory_used_mb": incident.gpu_memory_used_mb,
            "gpu_memory_total_mb": incident.gpu_memory_total_mb,
            "gpu_utilization_percent": incident.gpu_utilization_percent,
            "gpu_processes": incident.gpu_processes,

            # System state
            "system_memory_used_mb": incident.system_memory_used_mb,
            "system_memory_percent": incident.system_memory_percent,
            "cpu_percent": incident.cpu_percent,

            # Worker process
            "worker_pid": incident.worker_pid,
            "worker_memory_mb": incident.worker_memory_mb,
            "worker_threads": incident.worker_threads,
            "worker_open_files": incident.worker_open_files,

            # Recovery
            "recovery_action": incident.recovery_action,
            "recovery_successful": incident.recovery_successful,

            # Full diagnostics
            "full_diagnostics": incident.full_diagnostics,

            # Notes
            "notes": incident.notes
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hang incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hang-incidents/analysis/patterns")
async def analyze_hang_patterns(db: Session = Depends(get_db)):
    """
    Analyze hang incidents to identify patterns.

    Returns:
        Statistics about hangs by subreddit, video size, heartbeat stage, etc.
    """
    try:
        from database.models import HangIncident

        total_incidents = db.query(HangIncident).count()

        if total_incidents == 0:
            return {
                "total_incidents": 0,
                "message": "No hang incidents recorded yet"
            }

        # Count by subreddit
        by_subreddit = db.query(
            HangIncident.video_subreddit,
            func.count(HangIncident.id).label('count')
        ).group_by(HangIncident.video_subreddit).all()

        # Count by heartbeat stage
        by_stage = db.query(
            HangIncident.last_heartbeat_stage,
            func.count(HangIncident.id).label('count')
        ).group_by(HangIncident.last_heartbeat_stage).all()

        # Average video size that causes hangs
        avg_video_size = db.query(func.avg(HangIncident.video_file_size_mb)).scalar()
        avg_video_duration = db.query(func.avg(HangIncident.video_duration_seconds)).scalar()
        avg_task_duration = db.query(func.avg(HangIncident.task_duration_seconds)).scalar()

        # GPU stats at hang time
        avg_gpu_memory = db.query(func.avg(HangIncident.gpu_memory_used_mb)).scalar()
        avg_gpu_util = db.query(func.avg(HangIncident.gpu_utilization_percent)).scalar()

        # Recovery stats
        recovery_success_count = db.query(HangIncident).filter(
            HangIncident.recovery_successful == True
        ).count()
        recovery_fail_count = db.query(HangIncident).filter(
            HangIncident.recovery_successful == False
        ).count()

        # Recent incidents (last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_count = db.query(HangIncident).filter(
            HangIncident.detected_at >= week_ago
        ).count()

        return {
            "total_incidents": total_incidents,
            "recent_incidents_7d": recent_count,

            "by_subreddit": {
                sub or "unknown": count for sub, count in by_subreddit
            },

            "by_heartbeat_stage": {
                stage or "unknown": count for stage, count in by_stage
            },

            "averages": {
                "video_file_size_mb": round(avg_video_size, 2) if avg_video_size else None,
                "video_duration_seconds": round(avg_video_duration, 0) if avg_video_duration else None,
                "task_duration_seconds": round(avg_task_duration, 0) if avg_task_duration else None,
                "gpu_memory_used_mb": round(avg_gpu_memory, 0) if avg_gpu_memory else None,
                "gpu_utilization_percent": round(avg_gpu_util, 0) if avg_gpu_util else None
            },

            "recovery_stats": {
                "auto_recovery_success": recovery_success_count,
                "auto_recovery_failed": recovery_fail_count,
                "manual_or_none": total_incidents - recovery_success_count - recovery_fail_count
            },

            "insights": generate_hang_insights(
                by_subreddit,
                by_stage,
                avg_video_size,
                avg_video_duration,
                avg_gpu_memory
            )
        }
    except Exception as e:
        logger.error(f"Error analyzing hang patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subreddit/{subreddit_name}/metrics")
async def get_subreddit_metrics(
    subreddit_name: str,
    days: int = 14,
    db: Session = Depends(get_db)
):
    """
    Get metrics for a specific subreddit including:
    - Videos downloaded per day
    - Caption extraction success
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)

        # Videos downloaded per day for this subreddit
        videos_by_day = db.query(
            func.date(Video.download_date).label('date'),
            func.count(Video.id).label('count')
        ).filter(
            Video.source_subreddit == subreddit_name,
            Video.download_date >= start_date
        ).group_by(
            func.date(Video.download_date)
        ).order_by(
            func.date(Video.download_date)
        ).all()

        # Create date map for all days (fill in zeros)
        date_map = {}
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days-1-i)).date()
            date_map[date.isoformat()] = 0

        for date, count in videos_by_day:
            if date:
                date_map[date.isoformat()] = count

        videos_timeline = [
            {"date": date, "count": count}
            for date, count in sorted(date_map.items())
        ]

        # Captions extracted per day
        captions_by_day = db.query(
            func.date(ScrapedCaption.scraped_at).label('date'),
            func.count(ScrapedCaption.id).label('count')
        ).filter(
            ScrapedCaption.source_subreddit == subreddit_name,
            ScrapedCaption.scraped_at >= start_date
        ).group_by(
            func.date(ScrapedCaption.scraped_at)
        ).order_by(
            func.date(ScrapedCaption.scraped_at)
        ).all()

        captions_map = {}
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days-1-i)).date()
            captions_map[date.isoformat()] = 0

        for date, count in captions_by_day:
            if date:
                captions_map[date.isoformat()] = count

        captions_timeline = [
            {"date": date, "count": count}
            for date, count in sorted(captions_map.items())
        ]

        # Aggregate stats
        total_videos = db.query(Video).filter(
            Video.source_subreddit == subreddit_name
        ).count()

        total_captions = db.query(ScrapedCaption).filter(
            ScrapedCaption.source_subreddit == subreddit_name
        ).count()

        videos_period = sum(d["count"] for d in videos_timeline)
        captions_period = sum(d["count"] for d in captions_timeline)

        # Get execution stats (successful = videos with captions, failed = videos without)
        successful_extractions = db.query(Video).filter(
            Video.source_subreddit == subreddit_name,
            Video.download_date >= start_date,
            Video.processing_status == 'caption_extracted'
        ).count()

        failed_extractions = db.query(Video).filter(
            Video.source_subreddit == subreddit_name,
            Video.download_date >= start_date,
            Video.processing_status == 'failed'
        ).count()

        # Get recent video processing results as "executions"
        recent_videos = db.query(Video).filter(
            Video.source_subreddit == subreddit_name,
            Video.download_date >= start_date
        ).order_by(Video.download_date.desc()).limit(20).all()

        executions = [
            {
                "video_id": v.id,
                "post_id": v.source_post_id,
                "status": v.processing_status,
                "success": v.processing_status == 'caption_extracted',
                "timestamp": v.download_date.isoformat() if v.download_date else None
            }
            for v in recent_videos
        ]

        return {
            "subreddit": subreddit_name,
            "period_days": days,
            "videos_timeline": videos_timeline,
            "captions_timeline": captions_timeline,
            "executions": executions,
            "summary": {
                "total_videos": total_videos,
                "total_captions": total_captions,
                "caption_rate": round((total_captions / total_videos * 100), 1) if total_videos > 0 else 0,
                "videos_in_period": videos_period,
                "captions_in_period": captions_period,
                "successful_executions": successful_extractions,
                "failed_executions": failed_extractions
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting subreddit metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_hang_insights(by_subreddit, by_stage, avg_size, avg_duration, avg_gpu_mem):
    """Generate human-readable insights from hang pattern data."""
    insights = []

    # Check for problematic subreddits
    if by_subreddit:
        max_sub = max(by_subreddit, key=lambda x: x[1])
        if max_sub[1] > 2:
            insights.append(
                f"r/{max_sub[0]} has the most hangs ({max_sub[1]}). "
                "Consider checking videos from this subreddit for problematic patterns."
            )

    # Check for problematic stages
    if by_stage:
        max_stage = max(by_stage, key=lambda x: x[1])
        if max_stage[0] and 'ocr_frame' in str(max_stage[0]):
            insights.append(
                f"Most hangs occur during OCR at stage '{max_stage[0]}'. "
                "This suggests videos may have complex frames or OCR is getting stuck."
            )

    # Check video size patterns
    if avg_size and avg_size > 50:
        insights.append(
            f"Average video size at hang is {avg_size:.1f}MB. "
            "Consider adding size limits to skip very large videos."
        )

    # Check GPU memory
    if avg_gpu_mem and avg_gpu_mem > 10000:
        insights.append(
            f"GPU memory usage at hang time averages {avg_gpu_mem:.0f}MB. "
            "Consider reducing batch sizes or implementing memory cleanup between videos."
        )

    if not insights:
        insights.append("No clear patterns identified yet. More incident data needed.")

    return insights


# === Extraction Queue Endpoints (Event-driven architecture) ===

@router.get("/extraction/queue/status")
async def get_extraction_queue_status():
    """
    Get current extraction queue status.

    Returns queue length, processing count, and stats.
    """
    try:
        from utils.extraction_queue import get_queue_stats
        return get_queue_stats()
    except Exception as e:
        logger.error(f"Error getting extraction queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extraction/queue/detailed")
async def get_extraction_queue_detailed(db: Session = Depends(get_db)):
    """
    Get detailed extraction queue status with database context.

    Includes videos pending extraction and processing metrics.
    """
    try:
        from utils.extraction_queue import get_queue_stats, get_stale_processing

        # Get queue stats
        queue_stats = get_queue_stats()

        # Get videos awaiting extraction (downloaded but no caption)
        from sqlalchemy import and_, exists

        videos_awaiting = db.query(Video).filter(
            and_(
                Video.processing_status == 'downloaded',
                ~exists().where(ScrapedCaption.video_id == Video.id)
            )
        ).count()

        # Get stale processing (potentially hung)
        stale_ids = get_stale_processing(max_age_minutes=30)

        # Recent extractions (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(hours=24)
        recent_captions = db.query(ScrapedCaption).filter(
            ScrapedCaption.scraped_at >= yesterday
        ).count()

        return {
            "queue": queue_stats,
            "database": {
                "videos_awaiting_extraction": videos_awaiting,
                "stale_processing_count": len(stale_ids),
                "stale_video_ids": stale_ids,
                "recent_extractions_24h": recent_captions
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting detailed extraction queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extraction/queue/requeue/{video_id}")
async def requeue_video_for_extraction(video_id: int, db: Session = Depends(get_db)):
    """
    Manually requeue a specific video for extraction.
    """
    try:
        from utils.extraction_queue import requeue_video

        # Verify video exists
        video = db.query(Video).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        success = requeue_video(video_id, {
            'source': 'manual_requeue',
            'media_type': video.media_type,
            'subreddit': video.source_subreddit
        })

        if success:
            return {"status": "success", "message": f"Video {video_id} requeued for extraction"}
        else:
            raise HTTPException(status_code=500, detail="Failed to requeue video")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requeuing video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extraction/queue/clear")
async def clear_extraction_queue():
    """
    Clear the extraction queue (admin operation).

    WARNING: This will discard all pending extractions.
    """
    try:
        from utils.extraction_queue import clear_queue
        cleared = clear_queue()
        return {
            "status": "success",
            "message": f"Cleared {cleared} items from extraction queue",
            "cleared_count": cleared
        }
    except Exception as e:
        logger.error(f"Error clearing extraction queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extraction/dispatch/trigger")
async def trigger_extraction_dispatch(batch_size: int = 10):
    """
    Manually trigger an extraction batch dispatch.

    Args:
        batch_size: Number of videos to process
    """
    try:
        from tasks.extraction_dispatcher import dispatch_extraction_batch
        task = dispatch_extraction_batch.delay(batch_size)
        return {
            "status": "triggered",
            "task_id": task.id,
            "batch_size": batch_size,
            "message": f"Dispatched extraction batch task"
        }
    except Exception as e:
        logger.error(f"Error triggering extraction dispatch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extraction/orphans/process")
async def process_orphaned_videos_manual(max_videos: int = 50):
    """
    Manually trigger orphan video processing.

    Finds videos that were downloaded but never queued for extraction.
    """
    try:
        from tasks.extraction_dispatcher import process_orphaned_videos
        task = process_orphaned_videos.delay(max_videos)
        return {
            "status": "triggered",
            "task_id": task.id,
            "max_videos": max_videos,
            "message": f"Triggered orphan video processing"
        }
    except Exception as e:
        logger.error(f"Error triggering orphan processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Reddit IP Block Detection Endpoints ===

@router.get("/reddit/block-status")
async def get_reddit_block_status():
    """
    Get current Reddit IP block status.

    Returns the most recent block check result.
    """
    try:
        from utils.reddit_block_detector import get_current_status
        return get_current_status()
    except Exception as e:
        logger.error(f"Error getting block status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reddit/block-stats")
async def get_reddit_block_stats():
    """
    Get Reddit block statistics.

    Returns block rate, incident counts, and breakdown by type.
    """
    try:
        from utils.reddit_block_detector import get_block_stats
        return get_block_stats()
    except Exception as e:
        logger.error(f"Error getting block stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reddit/block-history")
async def get_reddit_block_history(limit: int = 20):
    """
    Get recent Reddit block check history.

    Args:
        limit: Number of events to return (max 100)
    """
    try:
        from utils.reddit_block_detector import get_block_history
        return {"history": get_block_history(min(limit, 100))}
    except Exception as e:
        logger.error(f"Error getting block history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reddit/check-access")
async def check_reddit_access_now():
    """
    Perform a live Reddit access check through assigned proxies.

    Tests if Reddit is accessible via each proxy currently assigned
    to scrapers. Returns per-proxy results.
    """
    try:
        from utils.reddit_block_detector import check_reddit_access, record_block_event
        from utils.proxy_pool import get_proxy_pool

        # Get proxy pool and find assigned proxies
        pool = get_proxy_pool()
        assignments = pool._get_assignments()

        results = []

        if not assignments:
            # No proxies assigned - do a single check without proxy
            status = check_reddit_access()
            record_block_event(status)
            results.append({
                "proxy": "direct",
                "result": status.to_dict(),
                "recommendation": _get_block_recommendation(status)
            })
        else:
            # Check each assigned proxy
            checked_proxies = set()
            for subreddit, proxy_url in assignments.items():
                # Extract host:port for deduplication
                if '@' in proxy_url:
                    proxy_host = proxy_url.split('@')[-1]
                else:
                    proxy_host = proxy_url

                # Skip if we already checked this proxy
                if proxy_host in checked_proxies:
                    continue
                checked_proxies.add(proxy_host)

                # Format full proxy URL
                full_proxy_url = f"http://{proxy_url}" if not proxy_url.startswith('http') else proxy_url

                status = check_reddit_access(proxy_url=full_proxy_url)
                record_block_event(status)

                results.append({
                    "proxy": proxy_host,
                    "subreddit": subreddit,
                    "result": status.to_dict(),
                    "recommendation": _get_block_recommendation(status)
                })

        # Summary
        blocked_count = sum(1 for r in results if r["result"].get("blocked") == "true")
        ok_count = len(results) - blocked_count

        return {
            "check_performed": True,
            "proxies_checked": len(results),
            "proxies_blocked": blocked_count,
            "proxies_ok": ok_count,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error checking Reddit access: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reddit/clear-block-status")
async def clear_reddit_block_status():
    """
    Clear block status (admin operation).

    Use after changing IP or resolving block.
    """
    try:
        from utils.reddit_block_detector import clear_block_status
        clear_block_status()
        return {"status": "success", "message": "Block status cleared"}
    except Exception as e:
        logger.error(f"Error clearing block status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Proxy Configuration ===

@router.get("/proxy/status")
async def get_proxy_status():
    """
    Get current proxy configuration status.

    Returns whether proxy is enabled, the host (not credentials),
    and tests connectivity if enabled.
    """
    from config.settings import settings

    result = {
        "enabled": settings.proxy_enabled,
        "configured": settings.proxy_url is not None,
        "proxy_host": None,
        "proxy_type": settings.proxy_type,
        "test_result": None
    }

    if settings.proxy_url:
        # Extract host only (hide credentials)
        if '@' in settings.proxy_url:
            result["proxy_host"] = settings.proxy_url.split('@')[-1]
        else:
            # Remove protocol prefix
            host = settings.proxy_url
            for prefix in ['http://', 'https://', 'socks5://', 'socks4://']:
                host = host.replace(prefix, '')
            result["proxy_host"] = host

    return result


@router.post("/proxy/test")
async def test_proxy_connection():
    """
    Test proxy connectivity by making a request through it.

    Tests both the proxy connection and Reddit accessibility.
    """
    import requests
    from config.settings import settings

    if not settings.proxy_enabled or not settings.proxy_url:
        return {
            "status": "skipped",
            "message": "Proxy not enabled or configured"
        }

    try:
        # Test 1: Basic connectivity (httpbin returns your IP)
        proxies = {
            'http': settings.proxy_url,
            'https': settings.proxy_url,
        }

        # Get IP through proxy
        ip_response = requests.get(
            'https://httpbin.org/ip',
            proxies=proxies,
            timeout=15
        )
        proxy_ip = ip_response.json().get('origin', 'unknown')

        # Get IP without proxy for comparison
        direct_response = requests.get('https://httpbin.org/ip', timeout=10)
        direct_ip = direct_response.json().get('origin', 'unknown')

        # Test 2: Reddit accessibility through proxy
        reddit_response = requests.get(
            'https://www.reddit.com/r/test.json?limit=1',
            proxies=proxies,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=15
        )

        reddit_accessible = reddit_response.status_code == 200
        reddit_data = None
        if reddit_accessible:
            try:
                reddit_data = reddit_response.json()
                reddit_accessible = 'data' in reddit_data
            except:
                reddit_accessible = False

        return {
            "status": "success",
            "proxy_working": True,
            "proxy_ip": proxy_ip,
            "direct_ip": direct_ip,
            "ip_changed": proxy_ip != direct_ip,
            "reddit_accessible": reddit_accessible,
            "reddit_status_code": reddit_response.status_code
        }

    except requests.exceptions.ProxyError as e:
        return {
            "status": "error",
            "proxy_working": False,
            "error": f"Proxy connection failed: {str(e)}"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "proxy_working": False,
            "error": "Proxy connection timed out"
        }
    except Exception as e:
        return {
            "status": "error",
            "proxy_working": False,
            "error": str(e)
        }


# === Proxy Pool Management ===

@router.get("/proxy/pool/status")
async def get_proxy_pool_status():
    """
    Get status of the proxy pool.

    Returns total proxies, available/assigned counts, and assignments.
    """
    try:
        from utils.proxy_pool import get_proxy_pool
        pool = get_proxy_pool()
        return pool.get_pool_status()
    except Exception as e:
        logger.error(f"Error getting proxy pool status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proxy/pool/assign/{subreddit}")
async def assign_proxy_to_subreddit(subreddit: str, proxy_index: Optional[int] = None):
    """
    Assign a proxy from the pool to a subreddit.

    Args:
        subreddit: Subreddit name
        proxy_index: Specific proxy index (0-based), auto-selects if not provided
    """
    try:
        from utils.proxy_pool import get_proxy_pool
        pool = get_proxy_pool()
        proxy_url = pool.assign_proxy_to_subreddit(subreddit, proxy_index)

        if proxy_url:
            proxy_host = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
            return {
                "status": "success",
                "subreddit": subreddit,
                "proxy_host": proxy_host,
                "message": f"Assigned proxy to r/{subreddit}"
            }
        else:
            return {
                "status": "error",
                "message": "No proxies available in pool"
            }
    except Exception as e:
        logger.error(f"Error assigning proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/proxy/pool/assign/{subreddit}")
async def unassign_proxy_from_subreddit(subreddit: str):
    """Remove proxy assignment from a subreddit."""
    try:
        from utils.proxy_pool import get_proxy_pool
        pool = get_proxy_pool()
        result = pool.unassign_proxy(subreddit)

        return {
            "status": "success" if result else "not_found",
            "subreddit": subreddit,
            "message": f"Unassigned proxy from r/{subreddit}" if result else "No assignment found"
        }
    except Exception as e:
        logger.error(f"Error unassigning proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proxy/pool/clear")
async def clear_all_proxy_assignments():
    """Clear all proxy assignments (admin operation)."""
    try:
        from utils.proxy_pool import get_proxy_pool
        pool = get_proxy_pool()
        pool.clear_all_assignments()

        return {
            "status": "success",
            "message": "All proxy assignments cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing assignments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proxy/pool/test/{proxy_index}")
async def test_specific_proxy(proxy_index: int):
    """
    Test a specific proxy from the pool.

    Args:
        proxy_index: Index of proxy in pool (0-based)
    """
    import requests
    from utils.proxy_pool import get_proxy_pool

    try:
        pool = get_proxy_pool()

        if proxy_index < 0 or proxy_index >= pool.size:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid proxy index. Pool has {pool.size} proxies (0-{pool.size-1})"
            )

        proxy = pool.proxies[proxy_index]
        proxy_url = proxy.full_url

        proxies = {
            'http': proxy_url,
            'https': proxy_url,
        }

        # Test connectivity
        ip_response = requests.get(
            'https://httpbin.org/ip',
            proxies=proxies,
            timeout=15
        )
        proxy_ip = ip_response.json().get('origin', 'unknown')

        # Test Reddit access
        reddit_response = requests.get(
            'https://www.reddit.com/r/test.json?limit=1',
            proxies=proxies,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=15
        )

        return {
            "status": "success",
            "proxy_index": proxy_index,
            "proxy_host": proxy.display_url,
            "assigned_to": proxy.assigned_to,
            "proxy_ip": proxy_ip,
            "reddit_accessible": reddit_response.status_code == 200,
            "reddit_status_code": reddit_response.status_code
        }

    except requests.exceptions.ProxyError as e:
        return {
            "status": "error",
            "proxy_index": proxy_index,
            "error": f"Proxy connection failed: {str(e)}"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "proxy_index": proxy_index,
            "error": "Proxy connection timed out"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proxy/pool/test-all")
async def test_all_proxies():
    """
    Test all proxies in the pool.

    Returns status of each proxy (working/failed).
    WARNING: This can take a while with many proxies.
    """
    import requests
    from utils.proxy_pool import get_proxy_pool

    try:
        pool = get_proxy_pool()
        results = []

        for i, proxy in enumerate(pool.proxies):
            proxy_url = proxy.full_url
            proxies = {
                'http': proxy_url,
                'https': proxy_url,
            }

            try:
                response = requests.get(
                    'https://httpbin.org/ip',
                    proxies=proxies,
                    timeout=10
                )
                proxy_ip = response.json().get('origin', 'unknown')
                results.append({
                    "index": i,
                    "host": proxy.display_url,
                    "assigned_to": proxy.assigned_to,
                    "working": True,
                    "ip": proxy_ip
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "host": proxy.display_url,
                    "assigned_to": proxy.assigned_to,
                    "working": False,
                    "error": str(e)[:100]
                })

        working = sum(1 for r in results if r['working'])
        return {
            "total": len(results),
            "working": working,
            "failed": len(results) - working,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error testing proxies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Caption Re-extraction Progress Tracking ===

@router.get("/extraction/progress")
async def get_extraction_progress(db: Session = Depends(get_db)):
    """
    Get detailed progress of caption extraction/re-extraction.

    Returns completion percentage, rate, and estimated time remaining.
    Use this to monitor bulk re-extraction operations.
    """
    try:
        from sqlalchemy import and_, or_
        import redis

        # Total videos in database
        total_videos = db.query(Video).count()

        # Videos with captions (completed)
        videos_with_captions = db.query(ScrapedCaption).distinct(ScrapedCaption.video_id).count()

        # Check actual Celery queue length (tasks waiting to be processed)
        celery_queue_length = 0
        try:
            redis_client = redis.Redis(host='redis', port=6379, db=0)
            celery_queue_length = redis_client.llen('gpu') or 0
        except Exception as e:
            logger.debug(f"Could not check Celery queue: {e}")

        # Videos with LLM refined captions (fully processed)
        with_llm_refined = db.query(ScrapedCaption).filter(
            ScrapedCaption.llm_refined_text.isnot(None),
            ScrapedCaption.llm_refined_text != ""
        ).count()

        # Videos pending extraction
        pending = db.query(Video).filter(
            Video.processing_status == 'downloaded'
        ).count()

        # Videos currently extracting
        extracting = db.query(Video).filter(
            Video.processing_status == 'extracting'
        ).count()

        # Calculate completion percentage
        completed = videos_with_captions
        completion_percent = round((completed / total_videos * 100), 1) if total_videos > 0 else 0
        llm_completion_percent = round((with_llm_refined / total_videos * 100), 1) if total_videos > 0 else 0

        # Get extraction rate (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        captions_last_hour = db.query(ScrapedCaption).filter(
            ScrapedCaption.scraped_at >= one_hour_ago
        ).count()

        # Estimate time remaining (initial estimate without queue - will be recalculated)
        eta_hours = None
        eta_display = "Unable to estimate (no recent progress)"

        # Videos without captions (failed or skipped)
        without_captions = total_videos - completed

        # Determine extraction status - include Celery queue in the check
        is_running = pending > 0 or extracting > 0 or celery_queue_length > 0
        all_processed = pending == 0 and extracting == 0 and celery_queue_length == 0
        all_succeeded = completed == total_videos

        # Recalculate ETA based on queue length if we have tasks queued
        total_remaining = pending + extracting + celery_queue_length
        if captions_last_hour > 0 and total_remaining > 0:
            hours_remaining = total_remaining / captions_last_hour
            eta_hours = round(hours_remaining, 1)
            eta_display = f"{eta_hours:.1f} hours" if eta_hours >= 1 else f"{int(eta_hours * 60)} minutes"

        return {
            "total_videos": total_videos,
            "completed": completed,
            "without_captions": without_captions,
            "with_llm_refined": with_llm_refined,
            "pending": pending,
            "extracting": extracting,
            "celery_queue_length": celery_queue_length,
            "total_remaining": total_remaining,
            "completion_percent": completion_percent,
            "llm_completion_percent": llm_completion_percent,
            "rate_per_hour": captions_last_hour,
            "estimated_hours_remaining": eta_hours,
            "eta_display": eta_display,
            "is_running": is_running,
            "all_processed": all_processed,
            "all_succeeded": all_succeeded,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting extraction progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_block_recommendation(status) -> str:
    """Generate recommendation based on block status."""
    if not status.blocked:
        return "All clear - Reddit is accessible"

    recommendations = {
        "rate_limited": f"Wait {status.retry_after or 60} seconds before retrying. Consider increasing delays between requests.",
        "forbidden": "IP appears to be blocked. Consider: 1) Wait 24 hours, 2) Use a VPN/proxy, 3) Check if issue persists.",
        "service_unavailable": "Reddit may be experiencing issues or blocking. Wait and retry in a few minutes.",
        "shadow_blocked": "Possible shadow block - requests succeed but return no data. Try different IP.",
        "captcha": "CAPTCHA required - automated access detected. Need to solve CAPTCHA or change IP.",
        "login_required": "Login required - session may have expired. This is unusual for JSON API.",
        "connection_error": "Network connectivity issue. Check your internet connection.",
    }

    return recommendations.get(status.block_type.value, "Unknown block type - investigate manually")
