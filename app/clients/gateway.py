from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings
from app.core.logging import api_logger


def _build_webhook_url() -> str:
    """Resolve the gateway webhook URL from settings.

    Priority:
    1) settings.gateway_webhook_url
    2) settings.gateway_base_url + settings.gateway_webhook_path
    """
    if settings.gateway_webhook_url:
        return settings.gateway_webhook_url.rstrip("/")
    if settings.gateway_base_url:
        base = settings.gateway_base_url.rstrip("/")
        path = (settings.gateway_webhook_path or "/webhooks/chat").lstrip("/")
        return f"{base}/{path}"
    raise ValueError(
        "Gateway URL is not configured. Set GATEWAY_WEBHOOK_URL or GATEWAY_BASE_URL."
    )


class GatewayClient:
    """Client for posting chat updates to an external gateway webhook."""

    def __init__(
        self,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        self.webhook_url = webhook_url or _build_webhook_url()
        self.timeout = timeout or float(settings.gateway_timeout or 10)
        self.api_key = api_key or settings.gateway_api_key

    async def send_chat_webhook(
        self,
        *,
        intent: str,
        reply: str,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST chat intent + reply to the configured gateway webhook.

        Returns parsed JSON if available; otherwise a dict with status info.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "intent": intent,
            "reply": reply,
        }
        if chat_id is not None:
            payload["chat_id"] = chat_id
        if user_id is not None:
            payload["user_id"] = user_id
        if metadata:
            payload["metadata"] = metadata

        api_logger.info(
            "gateway.send: url=%s chat_id=%s intent=%s reply_len=%d",
            self.webhook_url,
            chat_id,
            intent,
            len(reply or ""),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.webhook_url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                api_logger.exception("gateway.send: HTTP error %s", e)
                return {
                    "ok": False,
                    "status_code": resp.status_code,
                    "error": str(e),
                    "body": resp.text,
                }

            try:
                data = resp.json()
            except Exception:
                data = {"ok": True, "status_code": resp.status_code, "body": resp.text}
            return data

    async def update_lead_score(self, lead_id: str, score: str) -> dict:
        """Call  webhook /leads/score."""
        url = f"{self.base_url}/leads/score"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"lead_id": lead_id, "score": score}

        api_logger.info("Sending lead score update: %s -> %s", lead_id, score)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                api_logger.exception("Error calling webhook: %s", e)
                return {
                    "ok": False,
                    "status": resp.status_code,
                    "error": str(e),
                    "body": resp.text,
                }

            try:
                return resp.json()
            except Exception:
                return {"ok": True, "status": resp.status_code, "body": resp.text}
