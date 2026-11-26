"""
Session Node Client
===================
Handles session lifecycle, profile fetching, conversation messages,
and rolling summary persistence with MongoDB via Node backend.
"""

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional
from clients.node.base import BaseNodeClient

logger = logging.getLogger(__name__)


class SessionClient(BaseNodeClient):
    """
    Client for interacting with Node.js /api/profile and /api/chat endpoints.
    """

    async def fetch_user_profile(
        self, user_id: str, auth_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetches user profile details from Node backend."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        res = await self._request("GET", f"api/profile/{user_id}", headers=headers, timeout=3.5)
        if res and res.status_code == 200:
            return res.json()
        if res:
            logger.warning(f"[SessionClient] Profile fetch returned {res.status_code} for user {user_id}")
        return None

    async def fetch_session_data(
        self, session_id: str, auth_token: Optional[str] = None, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetches full session data (messages, conversationSummary) from Node backend."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        res = await self._request("GET", f"api/chat/sessions/{session_id}", headers=headers, timeout=3.5)
        if res and res.status_code == 200:
            return res.json()
        if res:
            logger.warning(
                f"[SessionClient] Session data fetch returned {res.status_code} for session {session_id}"
            )
        return None

    async def fetch_session_messages(
        self, session_id: str, auth_token: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetches prior messages for a session (used for rehydration on failover)."""
        data = await self.fetch_session_data(session_id=session_id, auth_token=auth_token, user_id=user_id)
        if data and isinstance(data, dict):
            return data.get("messages", [])
        return []

    async def patch_session_summary(
        self,
        session_id: str,
        summary: str,
        auth_token: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Updates the conversation summary in MongoDB via Node backend (cache write-back)."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        payload = {"summary": summary}
        res = await self._request(
            "PATCH", f"api/chat/sessions/{session_id}/summary", headers=headers, json=payload, timeout=5.0
        )
        if res and res.status_code == 200:
            logger.info(
                f"[SessionClient] Summary persisted to MongoDB for session {session_id} ({len(summary)} chars)"
            )
            return True
        elif res and 400 <= res.status_code < 500:
            logger.warning(
                f"[SessionClient] Patch summary rejected ({res.status_code}) for session {session_id}: {res.text}"
            )
        return False

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
        """Internal background worker that posts the message to the Node backend."""
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

        res = await self._request(
            "POST", f"api/chat/sessions/{session_id}/messages", headers=headers, json=payload, timeout=5.0
        )
        if res and res.status_code in (200, 201):
            logger.info(
                f"[SessionClient] Message persisted to Node backend for session {session_id} ({sender}, {len(text)} chars)"
            )
        elif res:
            logger.warning(
                f"[SessionClient] Persist message returned {res.status_code} for session {session_id}: {res.text}"
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
        """Schedules non-blocking async persist task so live voice audio is never blocked."""
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


session_client = SessionClient()
