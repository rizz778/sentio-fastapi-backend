"""
Base Node Client
================
Common base class for HTTP communication with the Node.js backend.
Handles base URL normalization, candidate loopback routing for Windows,
auth header formatting, and shared request helpers.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import httpx
from core.config import settings

logger = logging.getLogger(__name__)


class BaseNodeClient:
    """
    Common base class for all Node.js backend API clients.
    """

    def __init__(self, base_url: Optional[str] = None):
        url = (base_url or settings.NODE_BACKEND_URL).rstrip("/")
        # Rewrite localhost to 127.0.0.1 for local URLs to prevent Windows IPv6 resolution latency
        if "localhost" in url and not url.startswith("https://") and ".onrender.com" not in url:
            self.base_url = url.replace("localhost", "127.0.0.1")
        else:
            self.base_url = url

    def _build_headers(
        self, auth_token: Optional[str] = None, user_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Constructs common request headers with downstream authentication."""
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if user_id:
            headers["x-user-id"] = user_id
        return headers

    def _get_candidate_urls(self, path: str) -> List[str]:
        """
        Returns primary URL and loopback fallback URL for reliable local dev connectivity.
        """
        clean_path = path.lstrip("/")
        primary = f"{self.base_url}/{clean_path}"
        if "127.0.0.1" in self.base_url:
            fallback = f"{self.base_url.replace('127.0.0.1', 'localhost')}/{clean_path}"
        elif "localhost" in self.base_url:
            fallback = f"{self.base_url.replace('localhost', '127.0.0.1')}/{clean_path}"
        else:
            return [primary]
        return [primary, fallback]

    async def _request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
        max_retries: int = 2,
    ) -> Optional[httpx.Response]:
        """
        Executes a resilient HTTP request across candidate URLs with retry.
        """
        candidate_urls = self._get_candidate_urls(path)
        last_error = None

        for url in candidate_urls:
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.request(
                            method=method,
                            url=url,
                            headers=headers,
                            json=json,
                            params=params,
                        )
                        return response
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.15)

        logger.warning(
            f"[{self.__class__.__name__}] {method} {path} failed across candidates: {repr(last_error)}"
        )
        return None
