import asyncio
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
        self.base_url = (base_url or settings.NODE_BACKEND_URL).rstrip("/")

    def _build_headers(self, auth_token: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if user_id:
            headers["x-user-id"] = user_id
        return headers

    async def fetch_user_profile(
        self, user_id: str, auth_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetches user profile details from Node backend."""
        url = f"{self.base_url}/api/profile/{user_id}"
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code == 200:
                    return res.json()
                logger.warning(
                    f"[NodeClient] Profile fetch returned {res.status_code} for user {user_id}"
                )
        except Exception as e:
            logger.warning(f"[NodeClient] Failed to fetch profile for user {user_id}: {e}")
        return None

    async def fetch_session_messages(
        self, session_id: str, auth_token: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetches prior messages for a session (used for rehydration on failover)."""
        url = f"{self.base_url}/api/chat/sessions/{session_id}"
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
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
            logger.warning(f"[NodeClient] Failed to fetch messages for session {session_id}: {e}")
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
    ) -> None:
        """Internal worker that posts the message to the Node backend."""
        url = f"{self.base_url}/api/chat/sessions/{session_id}/messages"
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        payload = {
            "sender": sender,
            "text": text,
            "timestamp": httpx._utils.get_environment_proxies,  # default placeholder replaced below
        }
        import datetime
        payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if emotion:
            payload["emotion"] = emotion
        if distress_score is not None:
            payload["distressScore"] = distress_score

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code in (200, 201):
                    logger.debug(
                        f"[NodeClient] Message persisted to Node backend for session {session_id}"
                    )
                else:
                    logger.warning(
                        f"[NodeClient] Persist message returned {res.status_code} for session {session_id}: {res.text}"
                    )
        except Exception as e:
            logger.warning(f"[NodeClient] Non-blocking persist to Node failed: {e}")

    def fire_and_forget_persist(
        self,
        session_id: str,
        sender: str,
        text: str,
        emotion: Optional[str] = None,
        distress_score: Optional[float] = None,
        auth_token: Optional[str] = None,
        user_id: Optional[str] = None,
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
            )
        )


node_client = NodeBackendClient()
