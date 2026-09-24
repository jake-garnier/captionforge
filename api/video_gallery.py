"""
Video Gallery API Router
Endpoints for browsing and streaming videos
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.db import get_db
from database.models import Video, ScrapedCaption
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gallery", tags=["gallery"])


@router.get("/", response_class=HTMLResponse)
async def video_gallery_page():
    """Render video gallery HTML page with dashboard"""
    # The HTML embeds inline JS that's iterated frequently. Browsers were
    # serving stale cached HTML across deploys, so disable caching for this
    # route — the page is small enough (~700KB once, gzipped) that a fresh
    # fetch on each load is fine.
    no_cache_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    html_path = os.path.join(os.path.dirname(__file__), "gallery_with_dashboard.html")
    try:
        with open(html_path, 'r') as f:
            return HTMLResponse(content=f.read(), headers=no_cache_headers)
    except FileNotFoundError:
        # Fallback to embedded HTML if file not found
        logger.warning("gallery_with_dashboard.html not found, using fallback")
        html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Gallery - Captions Service</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: #0f0f0f;
                color: #fff;
                padding: 20px;
            }

            .header {
                max-width: 1400px;
                margin: 0 auto 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #333;
            }

            h1 {
                font-size: 32px;
                margin-bottom: 10px;
            }

            .stats {
                display: flex;
                gap: 20px;
                color: #aaa;
                font-size: 14px;
            }

            .filters {
                max-width: 1400px;
                margin: 0 auto 30px;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }

            .filter-group {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .filter-group label {
                font-size: 14px;
                color: #aaa;
            }

            select, input[type="number"] {
                background: #222;
                color: #fff;
                border: 1px solid #444;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
            }

            button {
                background: #3ea6ff;
                color: #fff;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
            }

            button:hover {
                background: #2d8fd8;
            }

            .delete-btn {
                background: transparent;
                border: none;
                color: #aaa;
                font-size: 13px;
                cursor: pointer;
                padding: 0;
                margin: 0;
                transition: color 0.2s ease;
            }

            .delete-btn:hover {
                color: #ff4444;
            }

            .gallery {
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 20px;
            }

            .video-card {
                background: #1a1a1a;
                border-radius: 8px;
                overflow: hidden;
                transition: transform 0.2s;
            }

            .video-card:hover {
                transform: translateY(-4px);
            }

            .video-container {
                position: relative;
                width: 100%;
                padding-top: 56.25%; /* 16:9 aspect ratio */
                background: #000;
            }

            video {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
                filter: blur(20px);
                transition: filter 0.3s ease;
            }

            video.playing {
                filter: blur(0);
            }

            .video-info {
                padding: 15px;
            }

            .video-title {
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 8px;
                line-height: 1.4;
            }

            .video-meta {
                display: flex;
                gap: 12px;
                font-size: 13px;
                color: #aaa;
                flex-wrap: wrap;
            }

            .video-meta span {
                display: flex;
                align-items: center;
                gap: 4px;
            }

            .badge {
                background: #333;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 12px;
            }

            .caption-text {
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid #333;
                font-size: 14px;
                line-height: 1.6;
                color: #ddd;
                max-height: 300px;
                overflow-y: auto;
            }

            .caption-text::-webkit-scrollbar {
                width: 6px;
            }

            .caption-text::-webkit-scrollbar-track {
                background: #1a1a1a;
            }

            .caption-text::-webkit-scrollbar-thumb {
                background: #444;
                border-radius: 3px;
            }

            .caption-label {
                font-size: 12px;
                color: #888;
                margin-bottom: 6px;
                font-weight: 500;
            }

            .caption-version {
                margin-bottom: 12px;
                padding: 10px;
                border-radius: 4px;
                background: rgba(255, 255, 255, 0.03);
            }

            .caption-version:last-child {
                margin-bottom: 0;
            }

            .version-label {
                font-size: 11px;
                font-weight: 600;
                margin-bottom: 6px;
                padding: 3px 8px;
                border-radius: 3px;
                display: inline-block;
            }

            .version-raw {
                background: rgba(255, 107, 107, 0.2);
                color: #ff6b6b;
            }

            .version-rule {
                background: rgba(255, 184, 108, 0.2);
                color: #ffb86c;
            }

            .version-llm {
                background: rgba(80, 250, 123, 0.2);
                color: #50fa7b;
            }

            .version-text {
                font-size: 13px;
                line-height: 1.5;
                color: #ddd;
            }

            .no-caption {
                color: #666;
                font-style: italic;
                font-size: 13px;
            }

            .loading {
                text-align: center;
                padding: 40px;
                color: #aaa;
            }

            .error {
                text-align: center;
                padding: 40px;
                color: #ff4444;
            }

            .pagination {
                max-width: 1400px;
                margin: 30px auto;
                display: flex;
                justify-content: center;
                gap: 10px;
            }

            .pagination button {
                background: #222;
            }

            .pagination button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎬 Video Gallery</h1>
            <div class="stats" id="stats">
                <span id="total-videos">Loading...</span>
            </div>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label>Subreddit:</label>
                <select id="subreddit-filter">
                    <option value="">All Subreddits</option>
                </select>
            </div>

            <div class="filter-group">
                <label>Status:</label>
                <select id="status-filter">
                    <option value="">All Statuses</option>
                    <option value="downloaded">Downloaded</option>
                    <option value="processing">Processing</option>
                    <option value="completed">Completed</option>
                    <option value="failed">Failed</option>
                </select>
            </div>

            <div class="filter-group">
                <label>Per Page:</label>
                <input type="number" id="limit" value="12" min="6" max="50" step="6">
            </div>

            <button onclick="loadVideos()">Apply Filters</button>
        </div>

        <div class="gallery" id="gallery">
            <div class="loading">Loading videos...</div>
        </div>

        <div class="pagination" id="pagination"></div>

        <script>
            let currentPage = 0;

            async function loadVideos() {
                const gallery = document.getElementById('gallery');
                const subreddit = document.getElementById('subreddit-filter').value;
                const status = document.getElementById('status-filter').value;
                const limit = parseInt(document.getElementById('limit').value);
                const skip = currentPage * limit;

                gallery.innerHTML = '<div class="loading">Loading videos...</div>';

                try {
                    const params = new URLSearchParams({
                        skip: skip,
                        limit: limit
                    });

                    if (subreddit) params.append('subreddit', subreddit);
                    if (status) params.append('status', status);

                    const response = await fetch(`/videos?${params}`);
                    const data = await response.json();

                    document.getElementById('total-videos').textContent =
                        `${data.total} videos total`;

                    if (data.videos.length === 0) {
                        gallery.innerHTML = '<div class="loading">No videos found</div>';
                        return;
                    }

                    gallery.innerHTML = data.videos.map(video => `
                        <div class="video-card">
                            <div class="video-container">
                                <video controls preload="metadata" onplay="this.classList.add('playing')">
                                    <source src="/gallery/stream/${video.post_id}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                            <div class="video-info">
                                <div class="video-title">
                                    ${video.post_id}
                                </div>
                                <div class="video-meta">
                                    <span class="badge">r/${video.subreddit}</span>
                                    ${video.upvotes ? `<span>⬆️ ${video.upvotes.toLocaleString()} upvotes</span>` : ''}
                                    ${video.duration ? `<span>⏱️ ${Math.floor(video.duration)}s</span>` : ''}
                                    ${video.resolution ? `<span>📐 ${video.resolution}</span>` : ''}
                                    ${video.file_size_mb ? `<span>💾 ${video.file_size_mb.toFixed(1)} MB</span>` : ''}
                                    <button class="delete-btn" onclick="deleteVideo(${video.id}, '${video.post_id}')">🗑️</button>
                                </div>
                                ${video.llm_refined ? `
                                    <div class="caption-text">
                                        ${video.llm_refined}
                                    </div>
                                ` : `
                                    <div class="caption-text">
                                        <div class="no-caption">No caption extracted yet</div>
                                    </div>
                                `}
                            </div>
                        </div>
                    `).join('');

                    // Update pagination
                    updatePagination(data.total, skip, limit);

                } catch (error) {
                    console.error('Error loading videos:', error);
                    gallery.innerHTML = '<div class="error">Failed to load videos</div>';
                }
            }

            function updatePagination(total, skip, limit) {
                const pagination = document.getElementById('pagination');
                const totalPages = Math.ceil(total / limit);
                const currentPageNum = Math.floor(skip / limit);

                if (totalPages <= 1) {
                    pagination.innerHTML = '';
                    return;
                }

                pagination.innerHTML = `
                    <button onclick="goToPage(0)" ${currentPageNum === 0 ? 'disabled' : ''}>First</button>
                    <button onclick="goToPage(${currentPageNum - 1})" ${currentPageNum === 0 ? 'disabled' : ''}>Previous</button>
                    <span style="color: #aaa; padding: 8px 12px;">Page ${currentPageNum + 1} of ${totalPages}</span>
                    <button onclick="goToPage(${currentPageNum + 1})" ${currentPageNum >= totalPages - 1 ? 'disabled' : ''}>Next</button>
                    <button onclick="goToPage(${totalPages - 1})" ${currentPageNum >= totalPages - 1 ? 'disabled' : ''}>Last</button>
                `;
            }

            function goToPage(page) {
                currentPage = page;
                loadVideos();
            }

            async function loadSubreddits() {
                try {
                    const response = await fetch('/videos?limit=1000');
                    const data = await response.json();

                    const subreddits = [...new Set(data.videos.map(v => v.subreddit))];
                    const select = document.getElementById('subreddit-filter');

                    subreddits.forEach(sub => {
                        const option = document.createElement('option');
                        option.value = sub;
                        option.textContent = `r/${sub}`;
                        select.appendChild(option);
                    });
                } catch (error) {
                    console.error('Error loading subreddits:', error);
                }
            }

            async function deleteVideo(videoId, postId) {
                if (!confirm(`Are you sure you want to delete video ${postId}? This cannot be undone.`)) {
                    return;
                }

                try {
                    const response = await fetch(`/videos/${videoId}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        // Reload videos to update the gallery
                        loadVideos();
                    } else {
                        const error = await response.json();
                        alert(`Failed to delete video: ${error.detail || 'Unknown error'}`);
                    }
                } catch (error) {
                    console.error('Error deleting video:', error);
                    alert('Failed to delete video. Please try again.');
                }
            }

            // Load on page load
            window.addEventListener('DOMContentLoaded', () => {
                loadVideos();
                loadSubreddits();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/stream/{post_id}")
async def stream_video(post_id: str, db: Session = Depends(get_db)):
    """
    Stream a media file by post ID with Range request support for seeking

    Args:
        post_id: Reddit post ID
        db: Database session

    Returns:
        Media file stream with appropriate MIME type
    """
    from fastapi import Request
    from fastapi.responses import Response

    # Find video in database
    video = db.query(Video).filter(Video.source_post_id == post_id).first()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found in database")

    if not video.storage_path:
        raise HTTPException(status_code=404, detail="Video file path not set")

    # Check if file exists
    if not os.path.exists(video.storage_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    # Get file size
    file_size = os.path.getsize(video.storage_path)

    # Determine MIME type based on file extension and media_type
    file_ext = os.path.splitext(video.storage_path)[1].lower()
    mime_types = {
        '.gif': 'image/gif',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mov': 'video/quicktime',
    }
    content_type = mime_types.get(file_ext, 'video/mp4')

    # Also check media_type field as fallback
    if video.media_type == 'gif' and content_type == 'video/mp4':
        content_type = 'image/gif'
    elif video.media_type in ['image', 'gallery'] and content_type == 'video/mp4':
        content_type = 'image/jpeg'  # Default to jpeg for images

    # Stream the entire file with proper headers
    def iterfile():
        with open(video.storage_path, mode="rb") as file:
            # Stream in 64KB chunks
            chunk_size = 64 * 1024
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # Determine appropriate filename extension
    ext_map = {
        'image/gif': '.gif',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/webp': '.webp',
        'video/mp4': '.mp4',
        'video/webm': '.webm',
    }
    file_extension = ext_map.get(content_type, '.mp4')

    # Return streaming response with proper headers
    return StreamingResponse(
        iterfile(),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'inline; filename="{post_id}{file_extension}"'
        }
    )
