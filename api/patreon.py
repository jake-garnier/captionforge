"""
Patreon Publishing API Endpoints

Handles Patreon credentials management and video publishing.
Each niche has its own Patreon account.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, date, timedelta, timezone
import os
import logging

logger = logging.getLogger(__name__)

from database.db import get_db
from database.models import (
    PatreonCredential, PatreonPublishJob, ComposedVideo, GeneratedCaption
)
from config.automation_config import get_automation_config
from config.settings import settings

router = APIRouter(prefix="/patreon", tags=["patreon"])


# === Request/Response Models ===

class PatreonCredentialCreate(BaseModel):
    """Create Patreon credentials for a niche."""
    niche: str
    email: str


class PatreonCredentialResponse(BaseModel):
    """Patreon credential response."""
    id: int
    niche: str
    email: str
    is_configured: bool
    last_login_at: Optional[str]
    created_at: str


class PatreonPublishRequest(BaseModel):
    """Request to publish a video to Patreon.

    By default this *schedules* the job for the next open day for the niche
    (one post per niche per day). Pass publish_immediately=True to bypass
    the schedule and dispatch the publish task right away — this is what the
    VNC session flow uses.
    """
    composed_video_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[str] = None
    public: bool = True
    publish_immediately: bool = False  # If True, post now instead of scheduling


class PatreonScheduleRequest(BaseModel):
    """Schedule a Patreon publish job for the next open slot for the niche."""
    composed_video_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[str] = None
    scheduled_date: Optional[str] = None  # YYYY-MM-DD; defaults to next free day
    posting_hour_utc: Optional[int] = None
    posting_minute: Optional[int] = None


class PatreonPublishResponse(BaseModel):
    """Response from publish request."""
    job_id: int
    status: str
    message: str


class PatreonJobStatusResponse(BaseModel):
    """Patreon publish job status response."""
    id: int
    composed_video_id: int
    niche: str
    title: str
    status: str
    patreon_post_url: Optional[str]
    error_message: Optional[str]
    created_at: str


# === Credentials Management Endpoints ===

@router.get("/credentials", response_model=List[PatreonCredentialResponse])
def list_patreon_credentials(db: Session = Depends(get_db)):
    """List all Patreon credentials by niche."""
    credentials = db.query(PatreonCredential).all()

    # Also show available niches that don't have credentials yet
    config = get_automation_config()
    configured_niches = {c.niche for c in credentials}

    return [_format_credential(c) for c in credentials]


@router.post("/credentials", response_model=PatreonCredentialResponse)
def create_patreon_credential(request: PatreonCredentialCreate, db: Session = Depends(get_db)):
    """Create or update Patreon credentials for a niche."""
    # Validate niche
    config = get_automation_config()
    valid_niches = [f.name for f in config.get_enabled_niches()]

    if request.niche not in valid_niches:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid niche '{request.niche}'. Valid options: {valid_niches}"
        )

    # Check if credential already exists
    existing = db.query(PatreonCredential).filter_by(niche=request.niche).first()

    if existing:
        # Update existing
        existing.email = request.email
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return _format_credential(existing)

    # Create new
    credential = PatreonCredential(
        niche=request.niche,
        email=request.email,
        cookies_path=f"/data/patreon_cookies/{request.niche}_cookies.json"
    )
    db.add(credential)
    db.commit()
    db.refresh(credential)

    return _format_credential(credential)


@router.get("/credentials/{niche}", response_model=PatreonCredentialResponse)
def get_patreon_credential(niche: str, db: Session = Depends(get_db)):
    """Get Patreon credentials for a specific niche."""
    credential = db.query(PatreonCredential).filter_by(niche=niche).first()
    if not credential:
        raise HTTPException(status_code=404, detail=f"No credentials found for niche '{niche}'")

    return _format_credential(credential)


@router.delete("/credentials/{niche}")
def delete_patreon_credential(niche: str, db: Session = Depends(get_db)):
    """Delete Patreon credentials for a niche."""
    credential = db.query(PatreonCredential).filter_by(niche=niche).first()
    if not credential:
        raise HTTPException(status_code=404, detail=f"No credentials found for niche '{niche}'")

    # Delete cookies file if exists
    if credential.cookies_path and os.path.exists(credential.cookies_path):
        try:
            os.remove(credential.cookies_path)
        except Exception:
            pass

    db.delete(credential)
    db.commit()

    return {"status": "deleted", "niche": niche}


@router.post("/credentials/{niche}/test-connection")
def test_patreon_connection(niche: str, db: Session = Depends(get_db)):
    """Test if Patreon connection is working for a niche.

    Note: This only validates that cookies exist and are valid JSON.
    Actual connection is tested during publish (browser automation is slow).
    """
    import json

    credential = db.query(PatreonCredential).filter_by(niche=niche).first()
    if not credential:
        raise HTTPException(status_code=404, detail=f"No credentials found for niche '{niche}'")

    # Check if cookies file exists
    cookies_exist = credential.cookies_path and os.path.exists(credential.cookies_path)

    if not cookies_exist:
        return {
            "status": "not_configured",
            "message": "Session cookies not found. Run interactive_login() to set up.",
            "niche": niche,
            "email": credential.email
        }

    # Validate cookies file is valid JSON with session data
    try:
        with open(credential.cookies_path, 'r') as f:
            cookies = json.load(f)

        if not isinstance(cookies, list) or len(cookies) == 0:
            return {
                "status": "invalid",
                "message": "Cookies file is empty or invalid. Run interactive_login() again.",
                "niche": niche,
                "email": credential.email
            }

        # Check for key Patreon session cookies
        cookie_names = {c.get('name', '') for c in cookies}
        has_session = 'session_id' in cookie_names or any('patreon' in name.lower() for name in cookie_names)

        if not has_session:
            return {
                "status": "invalid",
                "message": "Cookies don't contain Patreon session. Run interactive_login() again.",
                "niche": niche,
                "email": credential.email
            }

        # Update credential status
        credential.is_configured = True
        credential.last_login_at = datetime.utcnow()
        db.commit()

        return {
            "status": "configured",
            "message": f"Session cookies found ({len(cookies)} cookies). Ready to publish.",
            "niche": niche,
            "email": credential.email,
            "cookie_count": len(cookies)
        }

    except json.JSONDecodeError:
        return {
            "status": "invalid",
            "message": "Cookies file is corrupted. Run interactive_login() again.",
            "niche": niche,
            "email": credential.email
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading cookies: {str(e)}",
            "niche": niche
        }


# === Publishing Endpoints ===

@router.post("/publish", response_model=PatreonPublishResponse)
def publish_to_patreon(request: PatreonPublishRequest, db: Session = Depends(get_db)):
    """
    Create a Patreon publish job for a composed video.

    Default behavior (publish_immediately=False): schedule for the next open
    day for the niche (one post per niche per day, building a backlog).

    With publish_immediately=True: bypass the schedule and dispatch the
    publish task right away. This is what the VNC session flow uses.
    """
    niche, credential = _resolve_niche_and_credential(request.composed_video_id, db)
    title = _resolve_title(request.composed_video_id, request.title, db)
    _ensure_no_active_patreon_job(request.composed_video_id, db)

    if request.publish_immediately:
        job = PatreonPublishJob(
            composed_video_id=request.composed_video_id,
            credential_id=credential.id,
            niche=niche,
            title=title,
            description=request.description,
            tags=request.tags,
            status="pending",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        from tasks.patreon_tasks import publish_to_patreon_task
        task = publish_to_patreon_task.delay(job.id)
        job.celery_task_id = task.id
        db.commit()

        return PatreonPublishResponse(
            job_id=job.id,
            status="started",
            message=f"Publishing to Patreon ({niche}) started.",
        )

    # Schedule for the next open day for this niche
    target_date = _next_open_patreon_date(niche, db)
    post_time = _post_time_for(target_date)

    job = PatreonPublishJob(
        composed_video_id=request.composed_video_id,
        credential_id=credential.id,
        niche=niche,
        title=title,
        description=request.description,
        tags=request.tags,
        status="scheduled",
        scheduled_date=post_time,
        base_post_time=post_time,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return PatreonPublishResponse(
        job_id=job.id,
        status="scheduled",
        message=f"Patreon ({niche}) scheduled for {target_date.isoformat()} (one/day backlog).",
    )


@router.post("/schedule", response_model=PatreonPublishResponse)
def schedule_patreon(request: PatreonScheduleRequest, db: Session = Depends(get_db)):
    """
    Schedule a Patreon publish job for a specific date or the next open slot.

    One post per niche per day. If `scheduled_date` is omitted the job lands
    on the first day with no pending/scheduled/uploading job for the niche.
    """
    niche, credential = _resolve_niche_and_credential(request.composed_video_id, db)
    title = _resolve_title(request.composed_video_id, request.title, db)
    _ensure_no_active_patreon_job(request.composed_video_id, db)

    if request.scheduled_date:
        try:
            target_date = date.fromisoformat(request.scheduled_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid scheduled_date format. Use YYYY-MM-DD")
    else:
        target_date = _next_open_patreon_date(niche, db)

    post_time = _post_time_for(
        target_date,
        hour=request.posting_hour_utc,
        minute=request.posting_minute,
    )

    job = PatreonPublishJob(
        composed_video_id=request.composed_video_id,
        credential_id=credential.id,
        niche=niche,
        title=title,
        description=request.description,
        tags=request.tags,
        status="scheduled",
        scheduled_date=post_time,
        base_post_time=post_time,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return PatreonPublishResponse(
        job_id=job.id,
        status="scheduled",
        message=f"Patreon ({niche}) scheduled for {target_date.isoformat()}.",
    )


@router.post("/jobs/{job_id}/publish-now")
def publish_patreon_now(job_id: int, db: Session = Depends(get_db)):
    """Skip the schedule and dispatch a Patreon job immediately."""
    job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("scheduled", "pending", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot publish job with status '{job.status}'",
        )

    if job.status == "failed":
        job.error_message = None
    job.status = "pending"
    db.commit()

    from tasks.patreon_tasks import publish_to_patreon_task
    task = publish_to_patreon_task.delay(job.id)
    job.celery_task_id = task.id
    db.commit()

    return {"status": "dispatched", "job_id": job.id, "task_id": task.id}


@router.get("/jobs", response_model=List[PatreonJobStatusResponse])
def list_patreon_jobs(
    status: Optional[str] = None,
    niche: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List Patreon publish jobs."""
    query = db.query(PatreonPublishJob).order_by(PatreonPublishJob.created_at.desc())

    if status:
        query = query.filter(PatreonPublishJob.status == status)
    if niche:
        query = query.filter(PatreonPublishJob.niche == niche)

    jobs = query.limit(limit).all()
    return [_format_job(job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=PatreonJobStatusResponse)
def get_patreon_job(job_id: int, db: Session = Depends(get_db)):
    """Get details of a specific Patreon publish job."""
    job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return _format_job(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_patreon_job(job_id: int, db: Session = Depends(get_db)):
    """Cancel a pending Patreon publish job."""
    job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ["posted", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status}'"
        )

    # Revoke Celery task if running
    if job.celery_task_id:
        from tasks.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)

    job.status = "cancelled"
    job.completed_at = datetime.utcnow()
    db.commit()

    return {"status": "cancelled", "job_id": job_id}


@router.post("/jobs/{job_id}/retry")
def retry_patreon_job(job_id: int, db: Session = Depends(get_db)):
    """Retry a pending or failed Patreon publish job."""
    job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ["pending", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry job with status '{job.status}'"
        )

    # Reset job status to pending
    job.status = "pending"
    job.error_message = None
    job.started_at = None
    job.completed_at = None

    # Dispatch the publish task
    from tasks.patreon_tasks import publish_to_patreon_task
    task = publish_to_patreon_task.delay(job.id)
    job.celery_task_id = task.id
    db.commit()

    return {"status": "retrying", "job_id": job_id, "task_id": task.id}


@router.delete("/jobs/{job_id}")
def delete_patreon_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a Patreon publish job."""
    job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ["pending", "uploading"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete job with status '{job.status}'. Cancel it first."
        )

    db.delete(job)
    db.commit()

    return {"status": "deleted", "job_id": job_id}


@router.get("/stats")
def get_patreon_stats(db: Session = Depends(get_db)):
    """Get Patreon publishing statistics."""
    from sqlalchemy import func

    total_jobs = db.query(PatreonPublishJob).count()
    by_status = db.query(
        PatreonPublishJob.status,
        func.count(PatreonPublishJob.id)
    ).group_by(PatreonPublishJob.status).all()

    by_niche = db.query(
        PatreonPublishJob.niche,
        func.count(PatreonPublishJob.id)
    ).filter(PatreonPublishJob.status == "posted").group_by(PatreonPublishJob.niche).all()

    # Credentials status
    credentials = db.query(PatreonCredential).all()
    cred_status = {
        c.niche: {
            "email": c.email,
            "configured": c.is_configured,
            "last_login": c.last_login_at.isoformat() if c.last_login_at else None
        }
        for c in credentials
    }

    return {
        "jobs": {
            "total": total_jobs,
            "by_status": {status: count for status, count in by_status}
        },
        "posted_by_niche": {niche: count for niche, count in by_niche},
        "credentials": cred_status
    }


# === Helper Functions ===

def _format_credential(cred: PatreonCredential) -> dict:
    """Format a credential for API response."""
    return {
        "id": cred.id,
        "niche": cred.niche,
        "email": cred.email,
        "is_configured": cred.is_configured,
        "last_login_at": cred.last_login_at.isoformat() if cred.last_login_at else None,
        "created_at": cred.created_at.isoformat() if cred.created_at else None
    }


def _format_job(job: PatreonPublishJob) -> dict:
    """Format a publish job for API response."""
    return {
        "id": job.id,
        "composed_video_id": job.composed_video_id,
        "niche": job.niche,
        "title": job.title,
        "status": job.status,
        "patreon_post_url": job.patreon_post_url,
        "error_message": job.error_message,
        "scheduled_date": job.scheduled_date.isoformat() if job.scheduled_date else None,
        "base_post_time": job.base_post_time.isoformat() if job.base_post_time else None,
        "created_at": job.created_at.isoformat() if job.created_at else None
    }


# === Scheduling Helpers ===

def _resolve_niche_and_credential(composed_video_id: int, db: Session):
    """Resolve the niche + Patreon credential for a composed video.

    Raises HTTPException if anything is missing/misconfigured.
    """
    video = db.query(ComposedVideo).filter_by(id=composed_video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Composed video not found")

    niche = video.niche
    if not niche and video.generated_caption_id:
        caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
        if caption:
            niche = caption.niche

    if not niche:
        raise HTTPException(
            status_code=400,
            detail="Cannot determine niche for this video. Please set the niche field.",
        )

    credential = db.query(PatreonCredential).filter_by(niche=niche).first()
    if not credential:
        raise HTTPException(
            status_code=400,
            detail=f"No Patreon credentials configured for niche '{niche}'. Add credentials first.",
        )
    if not credential.is_configured:
        raise HTTPException(
            status_code=400,
            detail=f"Patreon credentials for '{niche}' are not configured. Run interactive_login() first.",
        )
    return niche, credential


def _resolve_title(composed_video_id: int, requested_title: Optional[str], db: Session) -> str:
    if requested_title:
        return requested_title

    video = db.query(ComposedVideo).filter_by(id=composed_video_id).first()
    title = None
    if video and video.generated_caption_id:
        caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
        if caption and caption.generated_title:
            title = caption.generated_title

    if not title and video:
        caption_text = video.edited_caption_text or ""
        if not caption_text and video.generated_caption_id:
            caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
            if caption:
                caption_text = caption.caption_text
        if caption_text:
            title = caption_text[:100] + "..." if len(caption_text) > 100 else caption_text

    if not title:
        raise HTTPException(status_code=400, detail="No title provided and no caption available")
    return title


def _ensure_no_active_patreon_job(composed_video_id: int, db: Session):
    """Block stacking multiple in-flight or queued Patreon jobs on one video."""
    existing = db.query(PatreonPublishJob).filter(
        PatreonPublishJob.composed_video_id == composed_video_id,
        PatreonPublishJob.status.in_(["scheduled", "pending", "uploading"]),
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Video already has an active Patreon publish job (id={existing.id}, status={existing.status})",
        )


def _next_open_patreon_date(niche: str, db: Session) -> date:
    """Find the next day (>= today) with no scheduled/pending/uploading/posted
    Patreon job for this niche.

    Today counts as available if we haven't already posted today and no
    in-flight job is reserving today's slot — that way the first post of the
    day lands today instead of getting pushed to tomorrow.
    """
    taken: set = set()
    rows = db.query(PatreonPublishJob).filter(
        PatreonPublishJob.niche == niche,
        PatreonPublishJob.status.in_(["scheduled", "pending", "uploading", "posted"]),
    ).all()
    for row in rows:
        # For posted jobs, prefer the actual post timestamp;
        # for in-flight jobs, the scheduled date is what reserves the slot.
        d = row.posted_at or row.scheduled_date or row.base_post_time
        if d:
            taken.add(d.date() if hasattr(d, "date") else d)

    target = date.today()
    while target in taken:
        target += timedelta(days=1)
    return target


def _post_time_for(target_date: date, hour: Optional[int] = None, minute: Optional[int] = None) -> datetime:
    """Build a UTC datetime at the configured posting hour for the given date."""
    cfg = get_automation_config().patreon_schedule
    h = hour if hour is not None else cfg.default_posting_hour_utc
    m = minute if minute is not None else cfg.default_posting_minute
    return datetime(target_date.year, target_date.month, target_date.day, h, m, tzinfo=timezone.utc)


# === Browser Session Management ===
# Provides a persistent browser window for interactive Patreon publishing
# All session methods are async to work properly with playwright's async API

@router.get("/session/status")
def get_session_status():
    """Get the status of the persistent browser session."""
    from publishers.patreon_session import patreon_session_manager
    return patreon_session_manager.get_status()


@router.post("/session/start")
async def start_session(request: Request, niche: str):
    """
    Start a persistent browser session for Patreon with VNC display.

    Returns when the browser is ready, including noVNC URL to view/interact with the browser.
    Connect to the noVNC URL to see the browser window, complete CAPTCHAs, and login.
    """
    from publishers.patreon_session import patreon_session_manager

    # Check if already running
    status = patreon_session_manager.get_status()
    if status["active"]:
        if status["niche"] == niche:
            # Return existing session with the public noVNC URL
            return {
                "status": "already_running",
                "niche": niche,
                "novnc_url": settings.NOVNC_PUBLIC_URL,
                "novnc_port": 6080
            }
        # Different niche - will restart

    # Await the async start_session method
    result = await patreon_session_manager.start_session(niche)

    # Add the public noVNC URL
    if result.get("status") == "started" or result.get("novnc_port"):
        result["novnc_url"] = settings.NOVNC_PUBLIC_URL

    return result


@router.post("/session/stop")
async def stop_session():
    """Stop the persistent browser session."""
    from publishers.patreon_session import patreon_session_manager
    return await patreon_session_manager.stop_session()


@router.post("/session/login")
async def navigate_to_login():
    """Navigate the browser to the Patreon login page."""
    from publishers.patreon_session import patreon_session_manager
    return await patreon_session_manager.navigate_to_login()


@router.get("/session/login-status")
async def check_login_status():
    """Check if currently logged in to Patreon."""
    from publishers.patreon_session import patreon_session_manager
    return await patreon_session_manager.check_login_status()


@router.post("/session/solve-cloudflare")
async def solve_cloudflare():
    """
    Attempt to solve a Cloudflare CAPTCHA challenge.

    If the browser is blocked by Cloudflare's "Verify you are human" page,
    this will try to click the checkbox automatically. If that fails,
    use the VNC viewer to solve it manually.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        return {
            "status": "error",
            "error": "Browser session not started. Call POST /patreon/session/start first."
        }

    return await patreon_session_manager.try_solve_cloudflare()


@router.post("/session/publish")
async def publish_via_session(
    video_path: str,
    title: str,
    description: Optional[str] = None,
    tags: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Publish a video using the open browser session.

    The video will be uploaded and posted while you watch in the browser window.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    tags_list = tags.split(",") if tags else None
    result = await patreon_session_manager.publish_video(video_path, title, description, tags_list)

    return {
        "success": result.success,
        "post_id": result.post_id,
        "post_url": result.post_url,
        "error": result.error
    }


@router.get("/session/screenshot")
def get_session_screenshot():
    """
    Get the latest screenshot from the browser session.

    Returns base64-encoded PNG image data.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    return patreon_session_manager.get_latest_screenshot()


@router.post("/session/screenshot")
async def take_session_screenshot():
    """
    Take a new screenshot of the current browser state.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    # Take screenshot (async method)
    path = await patreon_session_manager._take_screenshot("manual_screenshot")

    if path:
        return patreon_session_manager.get_latest_screenshot()
    else:
        raise HTTPException(status_code=500, detail="Failed to take screenshot")


@router.post("/session/save-cookies")
async def save_session_cookies(db: Session = Depends(get_db)):
    """
    Save cookies from the current browser session to disk.

    After logging in via the VNC browser, call this to save the session
    for headless publishing to use.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Start a session and login first."
        )

    niche = status.get("niche")
    if not niche:
        raise HTTPException(status_code=400, detail="No niche set for current session")

    # Save cookies from browser session
    cookie_count = await patreon_session_manager.save_cookies()

    if cookie_count == 0:
        raise HTTPException(status_code=400, detail="No cookies found in browser session")

    # Update credential in database
    credential = db.query(PatreonCredential).filter_by(niche=niche).first()
    if credential:
        credential.is_configured = True
        credential.last_login_at = datetime.utcnow()
    else:
        # Create credential entry
        credential = PatreonCredential(
            niche=niche,
            email=f"{niche}@patreon.local",
            cookies_path=f"/data/patreon_cookies/{niche}_cookies.json",
            is_configured=True,
            last_login_at=datetime.utcnow()
        )
        db.add(credential)

    db.commit()

    return {
        "status": "saved",
        "niche": niche,
        "cookie_count": cookie_count,
        "message": "Cookies saved. You can now use headless publishing."
    }


class CookieUploadRequest(BaseModel):
    """Request to upload cookies for a niche."""
    niche: str
    cookies: List[dict]


@router.post("/session/upload-cookies")
def upload_cookies(request: CookieUploadRequest, db: Session = Depends(get_db)):
    """
    Upload Patreon cookies exported from your browser.

    Use a browser extension like "Cookie-Editor" to export cookies from Patreon,
    then upload them here. This allows headless publishing without interactive login.
    """
    import json

    if not request.cookies:
        raise HTTPException(status_code=400, detail="No cookies provided")

    # Save cookies to file
    cookies_dir = "/data/patreon_cookies"
    os.makedirs(cookies_dir, exist_ok=True)

    cookies_path = f"{cookies_dir}/{request.niche}_cookies.json"
    with open(cookies_path, 'w') as f:
        json.dump(request.cookies, f, indent=2)

    # Update or create credential
    credential = db.query(PatreonCredential).filter_by(niche=request.niche).first()
    if credential:
        credential.cookies_path = cookies_path
        credential.is_configured = True
        credential.last_login_at = datetime.utcnow()
    else:
        credential = PatreonCredential(
            niche=request.niche,
            email=f"{request.niche}@patreon.local",  # Placeholder
            cookies_path=cookies_path,
            is_configured=True,
            last_login_at=datetime.utcnow()
        )
        db.add(credential)

    db.commit()

    return {
        "status": "uploaded",
        "niche": request.niche,
        "cookie_count": len(request.cookies),
        "path": cookies_path
    }


@router.post("/session/publish-job/{job_id}")
async def publish_job_via_session(job_id: int, db: Session = Depends(get_db)):
    """
    Publish a Patreon job using the open browser session.

    Use this instead of the Celery task for interactive publishing.
    """
    from publishers.patreon_session import patreon_session_manager

    # Check session is active
    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    # Get the job
    job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ["pending", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot publish job with status '{job.status}'"
        )

    # Get video info
    video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Update job status
    job.status = "uploading"
    job.started_at = datetime.utcnow()
    job.error_message = None
    db.commit()

    # Publish via session (async)
    tags_list = job.tags.split(",") if job.tags else None
    result = await patreon_session_manager.publish_video(
        video.storage_path,
        job.title,
        job.description,
        tags_list
    )

    # Update job with result
    if result.success:
        job.status = "posted"
        job.patreon_post_id = result.post_id
        job.patreon_post_url = result.post_url
        job.posted_at = datetime.utcnow()
    else:
        job.status = "failed"
        job.error_message = result.error

    job.completed_at = datetime.utcnow()
    db.commit()

    return {
        "job_id": job_id,
        "success": result.success,
        "post_url": result.post_url,
        "error": result.error
    }


class SessionPublishRequest(BaseModel):
    """Request to publish a video via VNC session."""
    composed_video_id: int
    title: str
    description: Optional[str] = None
    tags: Optional[str] = None


@router.post("/session/publish-video")
async def publish_video_via_session(request: SessionPublishRequest, db: Session = Depends(get_db)):
    """
    Publish a composed video directly via the active VNC browser session.

    This bypasses the credential check - if you're logged in via VNC, it will publish.
    Use this when you have a VNC session open and want to publish without pre-saving cookies.
    """
    from publishers.patreon_session import patreon_session_manager

    # Check session is active
    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    # Get the video
    video = db.query(ComposedVideo).filter_by(id=request.composed_video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Composed video not found")

    if not video.storage_path:
        raise HTTPException(status_code=400, detail="Video has no storage path")

    # Check file exists
    if not os.path.exists(video.storage_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {video.storage_path}")

    # Publish via the active session
    tags_list = request.tags.split(",") if request.tags else None
    result = await patreon_session_manager.publish_video(
        video.storage_path,
        request.title,
        request.description,
        tags_list
    )

    return {
        "success": result.success,
        "post_id": result.post_id,
        "post_url": result.post_url,
        "error": result.error
    }


class EmbedPublishRequest(BaseModel):
    """Request to publish a post with embedded video link."""
    embed_url: str  # Public video URL (e.g. the self-hosted stream URL)
    title: str
    description: Optional[str] = None
    tags: Optional[str] = None  # Comma-separated tags


@router.post("/session/publish-embed")
async def publish_embed_via_session(request: EmbedPublishRequest):
    """
    Publish a Patreon post with an embedded video link.

    Keeps the video on your own media host instead of uploading it to
    Patreon; the post body contains the link and Patreon renders a
    preview/embed for it where supported.

    Args:
        embed_url: Public URL of the video (e.g. the self-hosted stream URL)
        title: Post title
        description: Optional post description
        tags: Comma-separated tags (optional)
    """
    from publishers.patreon_session import patreon_session_manager

    # Check session is active
    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    # Validate embed URL
    if not request.embed_url:
        raise HTTPException(status_code=400, detail="embed_url is required")

    from urllib.parse import urlparse
    parsed = urlparse(request.embed_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="embed_url must be an absolute http(s) URL")

    # Publish via the active session
    tags_list = request.tags.split(",") if request.tags else None
    result = await patreon_session_manager.publish_with_embed(
        request.embed_url,
        request.title,
        request.description,
        tags_list
    )

    return {
        "success": result.success,
        "post_id": result.post_id,
        "post_url": result.post_url,
        "error": result.error
    }


# === Action Recording Endpoints ===
# Record and replay user actions for automated Patreon workflows

class StartRecordingRequest(BaseModel):
    """Request to start recording actions."""
    name: str


class ReplayRecordingRequest(BaseModel):
    """Request to replay a recording."""
    variables: Optional[dict] = None
    speed_multiplier: float = 1.0


@router.post("/session/recording/start")
async def start_recording(request: StartRecordingRequest):
    """
    Start recording user actions in the VNC browser session.

    After starting, perform actions in the VNC browser window.
    All clicks, typing, and navigation will be captured.
    Call /session/recording/stop to save the recording.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    return await patreon_session_manager.start_recording(request.name)


@router.post("/session/recording/stop")
async def stop_recording():
    """
    Stop recording and save the captured actions.

    Returns the number of actions captured and save location.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started."
        )

    return await patreon_session_manager.stop_recording()


@router.get("/session/recording/status")
async def get_recording_status():
    """
    Get current recording status including action count.
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        return {
            "session_active": False,
            "recording_active": False,
            "action_count": 0
        }

    recording_status = await patreon_session_manager.get_recording_status()
    return {
        "session_active": True,
        **recording_status
    }


@router.get("/recordings")
def list_recordings():
    """
    List all saved action recordings.

    Returns recordings with name, niche, recorded_at, and action_count.
    """
    from publishers.patreon_session import patreon_session_manager
    return patreon_session_manager.list_recordings()


@router.get("/recordings/{name}")
def get_recording(name: str):
    """
    Get details of a specific recording including all actions.
    """
    from publishers.patreon_session import patreon_session_manager

    recording = patreon_session_manager.get_recording(name)
    if not recording:
        raise HTTPException(status_code=404, detail=f"Recording not found: {name}")

    return recording


@router.delete("/recordings/{name}")
def delete_recording(name: str):
    """
    Delete a saved recording.
    """
    from publishers.patreon_session import patreon_session_manager

    if patreon_session_manager.delete_recording(name):
        return {"status": "deleted", "name": name}
    else:
        raise HTTPException(status_code=404, detail=f"Recording not found: {name}")


@router.post("/recordings/{name}/replay")
async def replay_recording(name: str, request: ReplayRecordingRequest):
    """
    Replay a saved recording in the active browser session.

    Placeholders in the recording can be replaced with actual values:
    - {{title}} - Video title
    - {{description}} - Video description
    - {{video_path}} - Path to video file

    Args:
        name: Name of the recording to replay
        variables: Dict of placeholder replacements
        speed_multiplier: Speed up (>1) or slow down (<1) replay
    """
    from publishers.patreon_session import patreon_session_manager

    status = patreon_session_manager.get_status()
    if not status["active"]:
        raise HTTPException(
            status_code=400,
            detail="Browser session not started. Call POST /patreon/session/start first."
        )

    return await patreon_session_manager.replay_recording(
        name=name,
        variables=request.variables,
        speed_multiplier=request.speed_multiplier
    )
