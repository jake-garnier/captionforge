"""Patreon OAuth v2 API client (per-niche creator credentials).

Used by the Patreon→Telegram membership sync. Reads:
  - PATREON_{NICHE}_CLIENT_ID
  - PATREON_{NICHE}_CLIENT_SECRET
  - PATREON_{NICHE}_ACCESS_TOKEN
  - PATREON_{NICHE}_REFRESH_TOKEN
  - PATREON_{NICHE}_CAMPAIGN_ID  (optional — looked up once if missing)

Tokens auto-refresh on 401. The latest pair is persisted to
/data/patreon_tokens/{niche}.json so refreshes survive container restarts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://www.patreon.com/api/oauth2/v2"
TOKEN_URL = "https://www.patreon.com/api/oauth2/token"
TOKEN_DIR = Path("/data/patreon_tokens")


@dataclass
class Member:
    """Active or recent Patreon member, decoded from JSON:API resource."""
    member_id: str
    user_id: str
    full_name: Optional[str]
    email: Optional[str]
    patron_status: Optional[str]
    pledge_relationship_start: Optional[str]
    last_charge_status: Optional[str]
    last_charge_date: Optional[str]


class PatreonAuthError(Exception):
    pass


class PatreonAPI:
    """Per-niche Patreon v2 client. Construct with PatreonAPI("motivation")."""

    def __init__(self, niche: str, *, http_timeout: float = 30.0):
        self.niche = niche
        prefix = f"PATREON_{niche.upper()}_"
        self.client_id = os.environ.get(f"{prefix}CLIENT_ID")
        self.client_secret = os.environ.get(f"{prefix}CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise PatreonAuthError(
                f"Missing {prefix}CLIENT_ID/CLIENT_SECRET in environment"
            )

        self._campaign_id_env = os.environ.get(f"{prefix}CAMPAIGN_ID") or None
        self._campaign_id: Optional[str] = self._campaign_id_env

        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        self._token_path = TOKEN_DIR / f"{niche}.json"

        access, refresh = self._load_tokens(prefix)
        if not access or not refresh:
            raise PatreonAuthError(
                f"No Patreon tokens for {niche}: set {prefix}ACCESS_TOKEN/"
                f"{prefix}REFRESH_TOKEN in .env or persist {self._token_path}"
            )
        self._access_token = access
        self._refresh_token = refresh

        self._http = httpx.Client(timeout=http_timeout)

    def _load_tokens(self, env_prefix: str) -> tuple[Optional[str], Optional[str]]:
        # Disk cache wins (it's the most recent post-refresh value).
        if self._token_path.exists():
            try:
                data = json.loads(self._token_path.read_text())
                return data.get("access_token"), data.get("refresh_token")
            except Exception as e:
                logger.warning("Failed reading %s: %s — falling back to env", self._token_path, e)
        return (
            os.environ.get(f"{env_prefix}ACCESS_TOKEN"),
            os.environ.get(f"{env_prefix}REFRESH_TOKEN"),
        )

    def _persist_tokens(self):
        payload = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "saved_at": int(time.time()),
        }
        tmp = self._token_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self._token_path)

    def _refresh_access_token(self):
        logger.info("Refreshing Patreon access token for %s", self.niche)
        resp = self._http.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        if resp.status_code >= 400:
            raise PatreonAuthError(
                f"Token refresh failed ({resp.status_code}): {resp.text[:300]}"
            )
        data = resp.json()
        self._access_token = data["access_token"]
        # Patreon rotates refresh tokens on each refresh.
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        self._persist_tokens()

    def _request(self, method: str, path: str, *, params: dict | None = None) -> dict:
        url = f"{API_BASE}{path}"
        for attempt in (0, 1):  # one retry after refresh
            resp = self._http.request(
                method,
                url,
                params=params,
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
            if resp.status_code == 401 and attempt == 0:
                self._refresh_access_token()
                continue
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                logger.warning("Patreon 429 — sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            if resp.status_code >= 400:
                raise PatreonAuthError(
                    f"Patreon {method} {path} failed ({resp.status_code}): {resp.text[:300]}"
                )
            return resp.json()
        raise PatreonAuthError(f"Patreon {method} {path} exhausted retries")

    def get_campaign_id(self) -> str:
        """Fetch and cache the creator's campaign id."""
        if self._campaign_id:
            return self._campaign_id
        data = self._request("GET", "/identity", params={"include": "campaign"})
        rels = (data.get("data") or {}).get("relationships") or {}
        campaign = rels.get("campaign") or {}
        cid = (campaign.get("data") or {}).get("id")
        if not cid:
            raise PatreonAuthError(
                "Could not derive campaign id from /identity response. "
                "Confirm the creator token belongs to a creator account with a campaign."
            )
        self._campaign_id = cid
        return cid

    def list_active_members(self) -> Iterator[Member]:
        """Yield every member of the campaign (active, declined, former)."""
        cid = self.get_campaign_id()
        params = {
            "include": "user",
            "fields[member]": (
                "patron_status,full_name,email,last_charge_status,"
                "last_charge_date,pledge_relationship_start"
            ),
            "fields[user]": "full_name",
            "page[count]": "100",
        }
        cursor: Optional[str] = None
        while True:
            if cursor:
                params["page[cursor]"] = cursor
            data = self._request("GET", f"/campaigns/{cid}/members", params=params)
            for resource in data.get("data") or []:
                member_id = resource.get("id")
                attrs = resource.get("attributes") or {}
                user_rel = (resource.get("relationships") or {}).get("user") or {}
                user_id = (user_rel.get("data") or {}).get("id")
                if not user_id or not member_id:
                    continue
                yield Member(
                    member_id=member_id,
                    user_id=user_id,
                    full_name=attrs.get("full_name"),
                    email=attrs.get("email"),
                    patron_status=attrs.get("patron_status"),
                    pledge_relationship_start=attrs.get("pledge_relationship_start"),
                    last_charge_status=attrs.get("last_charge_status"),
                    last_charge_date=attrs.get("last_charge_date"),
                )
            next_link = (data.get("links") or {}).get("next")
            cursor = (data.get("meta") or {}).get("pagination", {}).get("cursors", {}).get("next")
            if not next_link and not cursor:
                break

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
