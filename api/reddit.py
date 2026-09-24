"""
Reddit Account Management and Posting API (Playwright-based).

Uses browser automation with VNC for interactive login.
No Reddit API credentials needed - just log in via browser.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import asyncio
import io

from database.db import get_db
from database.models import RedditAccount, ComposedVideo, GeneratedCaption
from config.settings import settings

router = APIRouter(prefix="/reddit", tags=["reddit"])


# === Request/Response Models ===

class RedditAccountCreate(BaseModel):
    """Create a new Reddit account."""
    niche: str
    username: str
    subreddits: Optional[str] = None  # Comma-separated


class RedditAccountUpdate(BaseModel):
    """Update an existing Reddit account."""
    username: Optional[str] = None
    subreddits: Optional[str] = None
    is_enabled: Optional[bool] = None


class RedditAccountResponse(BaseModel):
    """Reddit account response."""
    id: int
    niche: str
    username: str
    subreddits: Optional[str]
    is_logged_in: bool
    last_login_at: Optional[str]
    is_enabled: bool
    created_at: str

    class Config:
        from_attributes = True


class PostToRedditRequest(BaseModel):
    """Request to post a composed video to Reddit."""
    composed_video_id: int
    title: Optional[str] = None  # Uses generated_title if not provided


class SessionStatusResponse(BaseModel):
    """Reddit session status."""
    active: bool
    niche: Optional[str]
    logged_in: bool
    novnc_url: Optional[str]


# === Reddit Account CRUD Endpoints ===

@router.get("/accounts", response_model=List[RedditAccountResponse])
def list_reddit_accounts(db: Session = Depends(get_db)):
    """List all Reddit accounts."""
    accounts = db.query(RedditAccount).order_by(RedditAccount.niche).all()
    return [_format_account(a) for a in accounts]


@router.get("/accounts/{niche}", response_model=RedditAccountResponse)
def get_reddit_account(niche: str, db: Session = Depends(get_db)):
    """Get Reddit account for a specific niche."""
    account = db.query(RedditAccount).filter_by(niche=niche).first()
    if not account:
        raise HTTPException(status_code=404, detail=f"No Reddit account for niche '{niche}'")
    return _format_account(account)


@router.post("/accounts", response_model=RedditAccountResponse)
def create_reddit_account(request: RedditAccountCreate, db: Session = Depends(get_db)):
    """Create a new Reddit account for a niche."""
    existing = db.query(RedditAccount).filter_by(niche=request.niche).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Reddit account already exists for niche '{request.niche}'"
        )

    account = RedditAccount(
        niche=request.niche,
        username=request.username,
        subreddits=request.subreddits,
        cookies_path=f"/data/reddit_cookies/{request.niche}_cookies.json"
    )
    db.add(account)
    db.commit()
    db.refresh(account)

    return _format_account(account)


@router.put("/accounts/{niche}", response_model=RedditAccountResponse)
def update_reddit_account(niche: str, request: RedditAccountUpdate, db: Session = Depends(get_db)):
    """Update a Reddit account."""
    account = db.query(RedditAccount).filter_by(niche=niche).first()
    if not account:
        raise HTTPException(status_code=404, detail=f"No Reddit account for niche '{niche}'")

    if request.username is not None:
        account.username = request.username
    if request.subreddits is not None:
        account.subreddits = request.subreddits
    if request.is_enabled is not None:
        account.is_enabled = request.is_enabled

    account.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(account)

    return _format_account(account)


@router.delete("/accounts/{niche}")
def delete_reddit_account(niche: str, db: Session = Depends(get_db)):
    """Delete a Reddit account."""
    account = db.query(RedditAccount).filter_by(niche=niche).first()
    if not account:
        raise HTTPException(status_code=404, detail=f"No Reddit account for niche '{niche}'")

    db.delete(account)
    db.commit()

    return {"status": "deleted", "niche": niche}


# === VNC Browser Session Endpoints ===

@router.get("/session/status")
async def get_session_status():
    """Get current Reddit browser session status."""
    import publishers.reddit_poster_playwright as rpm

    # Check if there's an active session
    if hasattr(rpm, '_active_sessions') and rpm._active_sessions:
        # Return first active session info
        for niche, session in rpm._active_sessions.items():
            return {
                "active": True,
                "niche": niche,
                "logged_in": False,  # Can't know until cookies are checked
                "novnc_url": settings.NOVNC_PUBLIC_URL
            }

    return {
        "active": False,
        "niche": None,
        "logged_in": False,
        "novnc_url": None
    }


@router.post("/session/start")
async def start_reddit_session(niche: str, db: Session = Depends(get_db)):
    """
    Start a VNC browser session for Reddit login.

    Connect to the noVNC URL to see and interact with the browser.
    Log in manually, then call /session/save-cookies to persist the session.
    """
    account = db.query(RedditAccount).filter_by(niche=niche).first()
    if not account:
        raise HTTPException(status_code=404, detail=f"No Reddit account for niche '{niche}'")

    try:
        from utils.vnc_manager import VNCDisplayManager
        from publishers.reddit_poster_playwright import RedditPosterPlaywright

        # Start VNC display
        vnc_manager = VNCDisplayManager()
        display_info = vnc_manager.start_display()

        if not display_info:
            raise HTTPException(status_code=500, detail="Failed to start VNC display")

        # Store session info globally for later use
        import publishers.reddit_poster_playwright as rpm
        if not hasattr(rpm, '_active_sessions'):
            rpm._active_sessions = {}

        # Create poster with VNC display
        poster = RedditPosterPlaywright(headless=False)
        poster.connect()

        # Navigate to Reddit
        if poster.page:
            poster.page.goto('https://www.reddit.com/login', wait_until='domcontentloaded', timeout=30000)

        rpm._active_sessions[niche] = {
            'poster': poster,
            'vnc_manager': vnc_manager,
            'started_at': datetime.utcnow()
        }

        return {
            "status": "started",
            "niche": niche,
            "novnc_url": settings.NOVNC_PUBLIC_URL,
            "message": "Connect to the noVNC URL, log in to Reddit, then call /session/save-cookies"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/save-cookies")
async def save_reddit_cookies(db: Session = Depends(get_db)):
    """Save the current browser session cookies after logging in via VNC."""
    import publishers.reddit_poster_playwright as rpm

    if not hasattr(rpm, '_active_sessions') or not rpm._active_sessions:
        raise HTTPException(status_code=400, detail="No active session")

    # Get the first (and usually only) active session
    niche = next(iter(rpm._active_sessions.keys()))
    session = rpm._active_sessions[niche]
    poster = session['poster']

    try:
        # Save cookies
        if poster.context:
            import json
            from pathlib import Path

            cookies = poster.context.cookies()
            reddit_cookies = [c for c in cookies if 'reddit.com' in c.get('domain', '')]

            cookies_path = f"/data/reddit_cookies/{niche}_cookies.json"
            Path(cookies_path).parent.mkdir(parents=True, exist_ok=True)

            with open(cookies_path, 'w') as f:
                json.dump(reddit_cookies, f)

            # Update database
            account = db.query(RedditAccount).filter_by(niche=niche).first()
            if account:
                account.cookies_path = cookies_path
                account.is_logged_in = True
                account.last_login_at = datetime.utcnow()
                db.commit()

            return {
                "status": "saved",
                "cookies_count": len(reddit_cookies),
                "niche": niche
            }
        else:
            raise HTTPException(status_code=500, detail="No browser context available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/stop")
async def stop_reddit_session():
    """Stop the Reddit browser session."""
    import publishers.reddit_poster_playwright as rpm

    if not hasattr(rpm, '_active_sessions') or not rpm._active_sessions:
        return {"status": "no_session", "niche": None}

    # Stop all active sessions
    stopped_niches = []
    for niche in list(rpm._active_sessions.keys()):
        session = rpm._active_sessions[niche]
        try:
            if session.get('poster'):
                session['poster'].close()
            if session.get('vnc_manager'):
                session['vnc_manager'].stop_display()
        except:
            pass
        del rpm._active_sessions[niche]
        stopped_niches.append(niche)

    return {"status": "stopped", "niches": stopped_niches}


@router.get("/session/screenshot")
async def get_reddit_screenshot():
    """Get a screenshot of the current Reddit browser session."""
    import publishers.reddit_poster_playwright as rpm

    if not hasattr(rpm, '_active_sessions') or not rpm._active_sessions:
        raise HTTPException(status_code=400, detail="No active session")

    # Get first active session
    niche = next(iter(rpm._active_sessions.keys()))
    session = rpm._active_sessions[niche]
    poster = session['poster']

    try:
        if poster.page:
            screenshot = poster.page.screenshot()
            return StreamingResponse(io.BytesIO(screenshot), media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="No page available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Reddit Posting Endpoints ===

@router.post("/post")
async def post_to_reddit(request: PostToRedditRequest, db: Session = Depends(get_db)):
    """
    Post a composed video to Reddit using saved session.

    1. Posts to user's profile first
    2. Crossposts to all configured subreddits
    """
    from publishers.reddit_poster_playwright import RedditPosterPlaywright

    # Get the composed video
    video = db.query(ComposedVideo).filter_by(id=request.composed_video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Composed video not found")

    if video.reddit_post_url:
        raise HTTPException(status_code=400, detail=f"Already posted: {video.reddit_post_url}")

    # Get niche
    niche = video.niche
    if not niche and video.generated_caption_id:
        caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
        if caption:
            niche = caption.niche

    if not niche:
        raise HTTPException(status_code=400, detail="Cannot determine niche for this video")

    # Get Reddit account
    account = db.query(RedditAccount).filter_by(niche=niche, is_enabled=True).first()
    if not account:
        raise HTTPException(status_code=400, detail=f"No enabled Reddit account for niche '{niche}'")

    if not account.is_logged_in:
        raise HTTPException(
            status_code=400,
            detail="Reddit account not logged in. Start a session via /reddit/session/start and log in via VNC."
        )

    # Get title
    title = request.title
    if not title and video.generated_caption_id:
        caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
        if caption and caption.generated_title:
            title = caption.generated_title

    if not title:
        raise HTTPException(status_code=400, detail="No title provided")

    # Get subreddits
    subreddits = account.get_subreddits_list()

    # Make sure the video has a public URL on the media host
    if not video.hosted_url:
        from publishers.media_host import get_media_host, MediaHostError
        try:
            hosted = get_media_host().publish(
                video_path=video.storage_path,
                title=title,
                composed_video_id=video.id,
            )
        except MediaHostError as e:
            raise HTTPException(status_code=400, detail=f"Could not host video: {e}")
        video.hosted_url = hosted.url
        db.commit()

    try:
        # Use Playwright to post
        poster = RedditPosterPlaywright(headless=True)

        # Load saved cookies
        if account.cookies_path:
            poster.COOKIES_FILE = account.cookies_path

        if not poster.connect():
            raise HTTPException(status_code=500, detail="Failed to connect to Reddit")

        # Post to profile first
        profile_result = poster.post_to_profile(
            title=title,
            url=video.hosted_url
        )

        if not profile_result.success:
            poster.close()
            raise HTTPException(status_code=500, detail=f"Failed to post: {profile_result.error}")

        # Crosspost to subreddits
        crosspost_results = []
        if subreddits and profile_result.post_id:
            crosspost_batch = poster.crosspost_to_subreddits(
                source_post_id=profile_result.post_id,
                subreddits=subreddits,
                title=title
            )
            crosspost_results = crosspost_batch.results

        poster.close()

        # Update video record
        video.reddit_post_url = profile_result.post_url
        video.reddit_posted_at = datetime.utcnow()
        video.is_published = True
        db.commit()

        return {
            "success": True,
            "profile_post_url": profile_result.post_url,
            "crosspost_results": crosspost_results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/videos/{video_id}/hosted-url")
def set_hosted_url(video_id: int, hosted_url: str, db: Session = Depends(get_db)):
    """Manually set the public media-host URL for a composed video."""
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Composed video not found")

    video.hosted_url = hosted_url
    db.commit()

    return {"status": "updated", "video_id": video_id, "hosted_url": hosted_url}


def _format_account(account: RedditAccount) -> dict:
    """Format account for API response."""
    return {
        "id": account.id,
        "niche": account.niche,
        "username": account.username,
        "subreddits": account.subreddits,
        "is_logged_in": account.is_logged_in or False,
        "last_login_at": account.last_login_at.isoformat() if account.last_login_at else None,
        "is_enabled": account.is_enabled,
        "created_at": account.created_at.isoformat() if account.created_at else None
    }
