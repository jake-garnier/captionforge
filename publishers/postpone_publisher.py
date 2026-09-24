"""
Postpone API client for scheduling Reddit link posts.

Flow:
1. The media host task publishes the composed video and stores its public
   URL (hosted_url) on the PostponeScheduleJob
2. Automation calls schedule_reddit_post() with that URL
3. Postpone creates a Reddit link post pointing to the hosted video

We deliberately create *link* posts (the `link` field in scheduleRedditPost)
rather than uploading media to Reddit: many subreddits restrict native video
posts, and a link post works everywhere while keeping the video on our host.

IMPORTANT: Do NOT use `mediaUrl` — that creates native video posts, which are
rejected by subreddits that only allow links.

GraphQL API at https://api.postpone.app/gql
Auth: Bearer token from POSTPONE_API_KEY
Rate limit: 300 requests per 10 minutes (paid plan)
"""
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

POSTPONE_GQL_URL = "https://api.postpone.app/gql"


@dataclass
class PostponeScheduleResult:
    """Result from scheduling a post via Postpone."""
    success: bool
    postpone_post_id: Optional[str] = None
    submissions: Optional[List[Dict[str, Any]]] = None
    errors: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = field(default=None, repr=False)


class PostponePublisher:
    """
    Schedule Reddit link posts via Postpone GraphQL API.

    Creates link posts pointing at the public URL of a composed video on the
    media host (self-hosted by default, see publishers/media_host.py).

    Usage:
        publisher = PostponePublisher(api_key="YOUR_KEY")
        result = publisher.schedule_reddit_post(
            username="motivation_clips",
            title="Check this out",
            media_url="https://captions.example.com/composition/videos/123/stream",
            subreddits=["GetMotivated", "Motivation"],
            base_post_time=datetime(2026, 2, 20, 18, 0),
            stagger_minutes=10,
        )
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _execute_query(self, query: str, variables: Optional[Dict] = None, operation_name: Optional[str] = None) -> Dict:
        """Execute a GraphQL query/mutation against Postpone API."""
        payload: Dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        if operation_name:
            payload["operationName"] = operation_name
        response = self.session.post(POSTPONE_GQL_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise Exception(f"GraphQL errors: {data['errors']}")
        return data.get("data", {})

    def health_check(self) -> bool:
        """Verify API key is valid by querying user profile."""
        try:
            query = """
            query profile {
                profile {
                    id
                    username
                    email
                }
            }
            """
            result = self._execute_query(query, operation_name="profile")
            profile = result.get("profile", {})
            if profile:
                logger.info(f"Postpone health check passed: user={profile.get('username')}")
                return True
            return False
        except Exception as e:
            logger.error(f"Postpone health check failed: {e}")
            return False

    def schedule_reddit_post(
        self,
        username: str,
        title: str,
        media_url: str,
        subreddits: List[str],
        base_post_time: datetime,
        stagger_minutes: int = 10,
        subreddit_flairs: Optional[Dict[str, str]] = None,
    ) -> PostponeScheduleResult:
        """
        Schedule a Reddit link post to multiple subreddits via Postpone.

        Creates a link post pointing at the hosted video URL.

        Args:
            username: Connected Reddit account username (exact case, e.g. "motivation_clips")
            title: Post title, max 300 chars
            media_url: Public URL of the composed video on the media host
            subreddits: List of subreddit names (no r/ prefix)
            base_post_time: First subreddit posting time (must be in future)
            stagger_minutes: Minutes between subreddit posts (min 10)
            subreddit_flairs: Optional dict of subreddit name -> flair text

        Returns:
            PostponeScheduleResult with success/error info
        """
        stagger_minutes = max(10, stagger_minutes)

        # If base_post_time is in the past, bump to 5 minutes from now
        now = datetime.utcnow()
        # Handle timezone-aware base_post_time by comparing naive UTC
        compare_time = base_post_time.replace(tzinfo=None) if base_post_time.tzinfo else base_post_time
        if compare_time < now:
            new_time = now + timedelta(minutes=5)
            logger.warning(
                f"base_post_time {base_post_time.isoformat()} is in the past, "
                f"bumping to {new_time.isoformat()}"
            )
            base_post_time = new_time

        subreddit_flairs = subreddit_flairs or {}

        submissions = []
        for i, subreddit in enumerate(subreddits):
            post_at = base_post_time + timedelta(minutes=i * stagger_minutes)
            submission = {
                "validationId": f"sub_{i}_{subreddit}",
                "subreddit": subreddit,
                "postAt": post_at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "profileCrosspost": False,
            }
            flair = subreddit_flairs.get(subreddit)
            if flair:
                submission["flairText"] = flair
            submissions.append(submission)

        mutation = """
        mutation ScheduleRedditPost($input: ScheduleRedditPostInput!) {
            scheduleRedditPost(input: $input) {
                success
                errors {
                    field
                    message
                }
                post {
                    id
                    title
                }
            }
        }
        """

        variables = {
            "input": {
                "username": username,
                "title": title[:300],
                "link": media_url,
                "submissions": submissions,
                "skipPostRequirementsValidation": True,
                "publishingStatus": "READY_TO_PUBLISH",
            }
        }

        try:
            logger.info(
                f"Scheduling Reddit link post via Postpone: title='{title[:50]}...', "
                f"subreddits={subreddits}, base_time={base_post_time.isoformat()}, "
                f"media_url={media_url}"
            )
            data = self._execute_query(mutation, variables, operation_name="ScheduleRedditPost")
            result = data.get("scheduleRedditPost", {})

            if result.get("success"):
                post = result.get("post", {})
                post_id = post.get("id")

                logger.info(
                    f"Postpone schedule success: post_id={post_id}, "
                    f"link={media_url}, subreddits={subreddits}"
                )

                return PostponeScheduleResult(
                    success=True,
                    postpone_post_id=post_id,
                    raw_response=data,
                )
            else:
                errors = result.get("errors", [])
                error_msg = "; ".join(f"{e.get('field', '?')}: {e.get('message', '?')}" for e in errors)
                logger.error(f"Postpone schedule failed with validation errors: {error_msg}")
                return PostponeScheduleResult(
                    success=False,
                    errors=errors,
                    error=error_msg,
                    raw_response=data,
                )

        except requests.exceptions.HTTPError as e:
            logger.error(f"Postpone HTTP error: {e.response.status_code} - {e.response.text[:500]}")
            return PostponeScheduleResult(success=False, error=f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.error(f"Postpone schedule error: {e}")
            return PostponeScheduleResult(success=False, error=str(e))

    def delete_post(self, post_id: str) -> bool:
        """Delete a scheduled post from Postpone."""
        mutation = """
        mutation DeletePlatformPost($platform: SocialPlatform!, $postId: ID!, $hardDelete: Boolean) {
            deletePlatformPost(platform: $platform, postId: $postId, hardDelete: $hardDelete) {
                success
            }
        }
        """
        try:
            data = self._execute_query(
                mutation,
                {"platform": "REDDIT", "postId": post_id, "hardDelete": True},
                operation_name="DeletePlatformPost",
            )
            success = data.get("deletePlatformPost", {}).get("success", False)
            if success:
                logger.info(f"Postpone post {post_id} deleted")
            return success
        except Exception as e:
            logger.error(f"Failed to delete Postpone post {post_id}: {e}")
            return False
