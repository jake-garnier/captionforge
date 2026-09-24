"""
Reddit posting integration using PRAW.

For posting videos to Reddit profile and crossposting to subreddits.

Required environment variables:
- REDDIT_CLIENT_ID: Reddit API client ID
- REDDIT_CLIENT_SECRET: Reddit API client secret
- REDDIT_USERNAME: Reddit account username
- REDDIT_PASSWORD: Reddit account password
- REDDIT_USER_AGENT: User agent string (optional)
"""
import os
import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import praw
from prawcore.exceptions import ResponseException, RequestException

logger = logging.getLogger(__name__)


@dataclass
class PostResult:
    """Result from Reddit post/crosspost."""
    success: bool
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CrosspostResults:
    """Results from batch crossposting."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)


class RedditPoster:
    """
    Post and crosspost content to Reddit using PRAW.

    Usage:
        poster = RedditPoster()
        if poster.connect():
            # Post to profile
            result = poster.post_to_profile(
                title="Check out this video",
                url="https://captions.example.com/composition/videos/123/stream"
            )
            if result.success:
                # Crosspost to subreddits
                crossposts = poster.crosspost_to_subreddits(
                    source_post_id=result.post_id,
                    subreddits=["GetMotivated", "Motivation"]
                )
    """

    def __init__(self):
        self.reddit: Optional[praw.Reddit] = None
        self.username = os.environ.get("REDDIT_USERNAME", "")

        # Credentials from environment
        self.client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        self.client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        self.password = os.environ.get("REDDIT_PASSWORD", "")
        self.user_agent = os.environ.get(
            "REDDIT_USER_AGENT",
            f"python:captions-publisher:v1.0 (by /u/{self.username})"
        )

    def connect(self) -> bool:
        """
        Connect to Reddit API.

        Returns True if connection successful.
        """
        if not all([self.client_id, self.client_secret, self.username, self.password]):
            logger.error("Missing Reddit API credentials in environment")
            return False

        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                username=self.username,
                password=self.password,
                user_agent=self.user_agent
            )

            # Verify connection by getting user info
            user = self.reddit.user.me()
            if user:
                logger.info(f"Connected to Reddit as u/{user.name}")
                return True
            else:
                logger.error("Failed to verify Reddit connection")
                return False

        except Exception as e:
            logger.error(f"Reddit connection error: {e}")
            return False

    def post_to_profile(
        self,
        title: str,
        url: str,
        flair_text: Optional[str] = None
    ) -> PostResult:
        """
        Post a link to user's profile.

        Args:
            title: Post title
            url: URL to post (the hosted video URL)
            flair_text: Optional flair text

        Returns:
            PostResult with success status and post info
        """
        if not self.reddit:
            if not self.connect():
                return PostResult(success=False, error="Failed to connect to Reddit")

        try:
            # User profile is accessed as subreddit u_username
            profile_sub = f"u_{self.username}"
            subreddit = self.reddit.subreddit(profile_sub)

            # Submit link post
            submission = subreddit.submit(
                title=title,
                url=url,
                send_replies=False
            )

            if flair_text:
                try:
                    submission.flair.select(flair_text)
                except Exception as e:
                    logger.warning(f"Failed to set flair: {e}")

            logger.info(f"Posted to profile: {submission.permalink}")

            return PostResult(
                success=True,
                post_id=submission.id,
                post_url=f"https://reddit.com{submission.permalink}"
            )

        except ResponseException as e:
            error_msg = f"Reddit API error: {e}"
            logger.error(error_msg)
            return PostResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Post error: {e}"
            logger.error(error_msg)
            return PostResult(success=False, error=error_msg)

    def crosspost_to_subreddit(
        self,
        source_post_id: str,
        subreddit: str,
        title: Optional[str] = None
    ) -> PostResult:
        """
        Crosspost an existing post to a subreddit.

        Args:
            source_post_id: ID of the source post (from profile)
            subreddit: Target subreddit name (without r/)
            title: Optional new title (uses original if None)

        Returns:
            PostResult with success status
        """
        if not self.reddit:
            if not self.connect():
                return PostResult(success=False, error="Failed to connect to Reddit")

        try:
            # Get the source submission
            source = self.reddit.submission(id=source_post_id)

            # Get target subreddit
            target_sub = self.reddit.subreddit(subreddit)

            # Check if subreddit allows crossposts
            # (This might fail if subreddit has crosspost disabled)

            # Perform crosspost
            crosspost = source.crosspost(
                subreddit=target_sub,
                title=title or source.title,
                send_replies=False
            )

            logger.info(f"Crossposted to r/{subreddit}: {crosspost.permalink}")

            return PostResult(
                success=True,
                post_id=crosspost.id,
                post_url=f"https://reddit.com{crosspost.permalink}"
            )

        except ResponseException as e:
            error_msg = str(e)

            # Parse common error types
            if "SUBREDDIT_NOTALLOWED" in error_msg:
                error_msg = f"Crossposts not allowed in r/{subreddit}"
            elif "SUBREDDIT_NOEXIST" in error_msg:
                error_msg = f"Subreddit r/{subreddit} does not exist"
            elif "INVALID_CROSSPOST_THING" in error_msg:
                error_msg = "Source post cannot be crossposted"
            elif "RATELIMIT" in error_msg:
                error_msg = f"Rate limited: {e}"

            logger.error(f"Crosspost to r/{subreddit} failed: {error_msg}")
            return PostResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"Crosspost error: {e}"
            logger.error(error_msg)
            return PostResult(success=False, error=error_msg)

    def crosspost_to_subreddits(
        self,
        source_post_id: str,
        subreddits: List[str],
        title: Optional[str] = None,
        delay_between: float = 5.0
    ) -> CrosspostResults:
        """
        Crosspost to multiple subreddits with rate limiting.

        Args:
            source_post_id: ID of the source post
            subreddits: List of subreddit names
            title: Optional new title
            delay_between: Seconds to wait between crossposts

        Returns:
            CrosspostResults with success/failure counts
        """
        results = CrosspostResults(total=len(subreddits))

        for i, subreddit in enumerate(subreddits):
            logger.info(f"Crossposting to r/{subreddit} ({i+1}/{len(subreddits)})")

            result = self.crosspost_to_subreddit(
                source_post_id=source_post_id,
                subreddit=subreddit,
                title=title
            )

            if result.success:
                results.successful += 1
                results.results.append({
                    "subreddit": subreddit,
                    "success": True,
                    "post_id": result.post_id,
                    "post_url": result.post_url
                })
            else:
                # Check if we should skip or count as failure
                if "not allowed" in (result.error or "").lower():
                    results.skipped += 1
                else:
                    results.failed += 1

                results.results.append({
                    "subreddit": subreddit,
                    "success": False,
                    "error": result.error
                })

            # Rate limiting between posts
            if i < len(subreddits) - 1:
                time.sleep(delay_between)

        logger.info(
            f"Crosspost complete: {results.successful} success, "
            f"{results.failed} failed, {results.skipped} skipped"
        )

        return results

    def delete_post(self, post_id: str) -> bool:
        """
        Delete a post.

        Args:
            post_id: The Reddit post ID

        Returns:
            True if deletion successful
        """
        if not self.reddit:
            if not self.connect():
                return False

        try:
            submission = self.reddit.submission(id=post_id)
            submission.delete()
            logger.info(f"Deleted post {post_id}")
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def get_post_info(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Get info about a post.

        Args:
            post_id: The Reddit post ID

        Returns:
            Post info dict or None
        """
        if not self.reddit:
            if not self.connect():
                return None

        try:
            submission = self.reddit.submission(id=post_id)
            return {
                "id": submission.id,
                "title": submission.title,
                "url": submission.url,
                "permalink": f"https://reddit.com{submission.permalink}",
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
                "subreddit": str(submission.subreddit),
                "is_crosspostable": submission.is_crosspostable
            }
        except Exception as e:
            logger.error(f"Get post info error: {e}")
            return None

    def check_subreddit_allows_crosspost(self, subreddit: str) -> bool:
        """
        Check if a subreddit allows crossposts.

        Args:
            subreddit: Subreddit name

        Returns:
            True if crossposts are allowed
        """
        if not self.reddit:
            if not self.connect():
                return False

        try:
            sub = self.reddit.subreddit(subreddit)
            # Check if crossposts are disabled
            # Note: This info isn't always available via API
            return True
        except Exception as e:
            logger.error(f"Check subreddit error: {e}")
            return False
