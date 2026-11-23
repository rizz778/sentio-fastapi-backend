import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional
import httpx
from core.config import settings

logger = logging.getLogger(__name__)


class NodeBackendClient:
    """
    HTTP client for asynchronously interacting with the Node.js backend
    for profile fetching and encrypted MongoDB chat persistence.
    """

    def __init__(self, base_url: Optional[str] = None):
        url = (base_url or settings.NODE_BACKEND_URL).rstrip("/")
        # Only rewrite localhost to 127.0.0.1 if it's explicitly a local URL
        if "localhost" in url and not url.startswith("https://") and not ".onrender.com" in url:
            self.base_url = url.replace("localhost", "127.0.0.1")
        else:
            self.base_url = url

    def _build_headers(self, auth_token: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if user_id:
            headers["x-user-id"] = user_id
        return headers

    def _get_candidate_urls(self, path: str) -> List[str]:
        """Returns primary IPv4 URL with fallback URL for Windows loopback reliability."""
        clean_path = path.lstrip("/")
        primary = f"{self.base_url}/{clean_path}"
        if "127.0.0.1" in self.base_url:
            fallback = f"{self.base_url.replace('127.0.0.1', 'localhost')}/{clean_path}"
        elif "localhost" in self.base_url:
            fallback = f"{self.base_url.replace('localhost', '127.0.0.1')}/{clean_path}"
        else:
            return [primary]
        return [primary, fallback]

    async def fetch_user_profile(
        self, user_id: str, auth_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetches user profile details from Node backend."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        for url in self._get_candidate_urls(f"api/profile/{user_id}"):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    res = await client.get(url, headers=headers)
                    if res.status_code == 200:
                        return res.json()
                    logger.warning(
                        f"[NodeClient] Profile fetch returned {res.status_code} for user {user_id}"
                    )
            except Exception as e:
                logger.warning(f"[NodeClient] Failed to fetch profile from {url}: {repr(e)}")
        return None

    async def fetch_session_messages(
        self, session_id: str, auth_token: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetches prior messages for a session (used for rehydration on failover)."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        for url in self._get_candidate_urls(f"api/chat/sessions/{session_id}"):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    res = await client.get(url, headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        return data.get("messages", [])
                    logger.warning(
                        f"[NodeClient] Session fetch returned {res.status_code} for session {session_id}"
                    )
            except Exception as e:
                logger.warning(f"[NodeClient] Failed to fetch messages from {url}: {repr(e)}")
        return []

    async def _send_message_task(
        self,
        session_id: str,
        sender: str,
        text: str,
        emotion: Optional[str] = None,
        distress_score: Optional[float] = None,
        auth_token: Optional[str] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Internal worker that posts the message to the Node backend with retry."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        payload = {
            "sender": sender,
            "text": text,
            "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if emotion:
            payload["emotion"] = emotion
        if distress_score is not None:
            payload["distressScore"] = distress_score

        candidate_urls = self._get_candidate_urls(f"api/chat/sessions/{session_id}/messages")
        last_error = None

        for url in candidate_urls:
            for attempt in range(2):
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        res = await client.post(url, json=payload, headers=headers)
                        if res.status_code in (200, 201):
                            logger.info(
                                f"[NodeClient] Message persisted to Node backend for session {session_id} ({sender}, {len(text)} chars)"
                            )
                            return
                        else:
                            logger.warning(
                                f"[NodeClient] Persist message returned {res.status_code} from {url} for session {session_id}: {res.text}"
                            )
                            if 400 <= res.status_code < 500:
                                return
                except Exception as e:
                    last_error = e
                    await asyncio.sleep(0.15)

        logger.error(
            f"[NodeClient] Non-blocking persist to Node failed for session {session_id}: {repr(last_error)}"
        )

    def fire_and_forget_persist(
        self,
        session_id: str,
        sender: str,
        text: str,
        emotion: Optional[str] = None,
        distress_score: Optional[float] = None,
        auth_token: Optional[str] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Schedules non-blocking async persist task so live voice is never blocked."""
        asyncio.create_task(
            self._send_message_task(
                session_id=session_id,
                sender=sender,
                text=text,
                emotion=emotion,
                distress_score=distress_score,
                auth_token=auth_token,
                user_id=user_id,
                timestamp=timestamp,
            )
        )


node_client = NodeBackendClient()
