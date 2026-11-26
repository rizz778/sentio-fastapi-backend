"""
Memory Node Client
==================
Dedicated client for interacting with the Node.js Episodic Memory API (/api/episodes).
Extends BaseNodeClient for consistent URL candidate handling, auth headers, and retries.
"""

import logging
from typing import Any, Dict, List, Optional
from clients.node.base import BaseNodeClient

logger = logging.getLogger(__name__)


class MemoryClient(BaseNodeClient):
    """
    Client for interacting with Node.js /api/episodes endpoints.
    """

    async def create_episode(
        self,
        user_id: str,
        session_id: Optional[str],
        episode_data: Dict[str, Any],
        auth_token: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Stores a new structured episodic memory with embedding in MongoDB via Node backend."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        payload = {
            "userId": user_id,
            "sessionId": session_id,
            "eventType": episode_data.get("eventType", "general"),
            "eventSummary": episode_data.get("eventSummary", ""),
            "dominantEmotions": episode_data.get("dominantEmotions", []),
            "peakDistressScore": episode_data.get("peakDistressScore", 0),
            "peopleMentioned": episode_data.get("peopleMentioned", []),
            "copingAttempted": episode_data.get("copingAttempted", ""),
            "resolutionStatus": episode_data.get("resolutionStatus", "unresolved"),
            "embedding": episode_data.get("embedding", []),
            "timestamp": episode_data.get("timestamp"),
        }

        res = await self._request("POST", "api/episodes", headers=headers, json=payload, timeout=5.0)
        if res and res.status_code in (200, 201):
            data = res.json()
            logger.info(f"[MemoryClient] Episode created for user {user_id}: {episode_data.get('eventType')}")
            return data.get("episode")
        elif res:
            logger.warning(f"[MemoryClient] Create episode returned {res.status_code}: {res.text}")
        return None

    async def fetch_user_episodes(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 5,
        auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetches recent episodic memories (e.g. status='unresolved' for session start)."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        path = f"api/episodes?limit={limit}"
        if status and status != "all":
            path += f"&status={status}"

        res = await self._request("GET", path, headers=headers, timeout=4.0)
        if res and res.status_code == 200:
            data = res.json()
            return data.get("episodes", [])
        elif res:
            logger.warning(f"[MemoryClient] Fetch episodes returned {res.status_code}: {res.text}")
        return []

    async def search_episodes(
        self,
        user_id: str,
        query_vector: List[float],
        limit: int = 3,
        auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Searches episodic memories via Atlas $vectorSearch (with cosine fallback)."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        payload = {
            "userId": user_id,
            "queryVector": query_vector,
            "limit": limit,
        }

        res = await self._request("POST", "api/episodes/search", headers=headers, json=payload, timeout=5.0)
        if res and res.status_code == 200:
            data = res.json()
            return data.get("matches", [])
        elif res:
            logger.warning(f"[MemoryClient] Search episodes returned {res.status_code}: {res.text}")
        return []

    async def patch_episode(
        self,
        user_id: str,
        episode_id: str,
        resolution_status: Optional[str] = None,
        coping_attempted: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates an episode's therapeutic status (resolutionStatus, copingAttempted)."""
        headers = self._build_headers(auth_token=auth_token, user_id=user_id)
        payload = {}
        if resolution_status:
            payload["resolutionStatus"] = resolution_status
        if coping_attempted is not None:
            payload["copingAttempted"] = coping_attempted

        if not payload:
            return None

        res = await self._request(
            "PATCH", f"api/episodes/{episode_id}", headers=headers, json=payload, timeout=4.0
        )
        if res and res.status_code == 200:
            data = res.json()
            logger.info(f"[MemoryClient] Updated episode {episode_id} for user {user_id}")
            return data.get("episode")
        elif res:
            logger.warning(f"[MemoryClient] Patch episode returned {res.status_code}: {res.text}")
        return None


memory_client = MemoryClient()
