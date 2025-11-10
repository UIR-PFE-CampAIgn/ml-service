"""
Score webhook for sending lead scores to external endpoints.
"""

import os
import asyncio
import httpx
from typing import Dict, Any
from app.core.logging import ml_logger


class ScoreWebhook:
    """Score webhook for sending lead scores."""

    def __init__(self, path: str = "/leads/score"):
        """Initialize score webhook."""
        base_url = os.getenv("WEBHOOK_BASE_URL")
        self.webhook_url = f"{base_url.rstrip('/')}{path}"

    async def send_lead_score(self, lead_id: str, score: str) -> Dict[str, Any]:
        """Send lead score to webhook endpoint."""
        payload = {"lead_id": lead_id, "score": score}

        try:
            ml_logger.info(f"Sending lead score to {self.webhook_url}: {payload}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

                result = {
                    "success": True,
                    "status_code": response.status_code,
                    "response": response.json() if response.content else {},
                    "payload": payload,
                }
                ml_logger.info(f"Successfully sent lead score for {lead_id}: {score}")
                return result

        except httpx.TimeoutException:
            error_msg = (
                f"Timeout sending lead score for {lead_id} to {self.webhook_url}"
            )
            ml_logger.error(error_msg)
            return {
                "success": False,
                "error": "timeout",
                "message": error_msg,
                "payload": payload,
            }

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} sending lead score for {lead_id}: {e.response.text}"
            ml_logger.error(error_msg)
            return {
                "success": False,
                "error": "http_error",
                "status_code": e.response.status_code,
                "message": error_msg,
                "payload": payload,
            }

        except Exception as e:
            error_msg = f"Error sending lead score for {lead_id}: {str(e)}"
            ml_logger.error(error_msg)
            return {
                "success": False,
                "error": "unexpected_error",
                "message": error_msg,
                "payload": payload,
            }


async def send_lead_score(
    lead_id: str, score: str, path: str = "/leads/score"
) -> Dict[str, Any]:
    """Convenience function to send a lead score."""
    webhook = ScoreWebhook(path)
    return await webhook.send_lead_score(lead_id, score)
