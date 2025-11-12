from typing import Any, Dict, List
from datetime import datetime, timedelta
import json
import requests
from pydantic import BaseModel, Field


class CampaignRequest(BaseModel):
    prompt: str
    timezone: str = "UTC"


class MessageTemplate(BaseModel):
    message: str
    target_segment: str
    approach: str
    personalization_tips: str


class SendSchedule(BaseModel):
    segment: str
    send_datetime: str
    reasoning: str
    priority: str


class CampaignStrategy(BaseModel):
    target_segments: List[str]
    reasoning: str
    campaign_type: str
    key_message: str
    expected_response_rates: Dict[str, str]


class CampaignResponse(BaseModel):
    strategy: CampaignStrategy
    templates: List[MessageTemplate]
    schedule: List[SendSchedule]
    insights: Dict[str, List[str]]


class WhatsAppCampaignGenerator:
    """WhatsApp Campaign Generator using Groq API (sync version, no aiohttp)"""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = "llama-3.1-8b-instant"

    def __init__(self, groq_api_key: str, logger=None):
        self.groq_api_key = groq_api_key
        self.logger = logger

    def _log(self, level: str, message: str):
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")

    def _build_ai_prompt(self, user_prompt: str, timezone: str) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")

        return f"""You are an expert WhatsApp marketing campaign strategist. Analyze this campaign request and create a complete strategy.
        IMPORTANT:
- Choose optimal send_datetime based on campaign analysis
- In your reasoning, refer to dates and times, NOT day names (you may calculate incorrectly)
- Example: "Send at 10:00 AM on Nov 19" NOT "Send on Thursday"

USER'S CAMPAIGN REQUEST:
"{user_prompt}"

CONTEXT:
- Current date: {current_date}
- Timezone: {timezone}

LEAD SEGMENTS:
- HOT: High engagement, ready to buy
- WARM: Considering, need nurturing
- COLD: Awareness stage

Return ONLY valid JSON (no markdown):
{{
  "strategy": {{
    "target_segments": ["hot"],
    "reasoning": "Explain reasoning",
    "campaign_type": "promotional",
    "key_message": "Core message",
    "expected_response_rates": {{"hot": "40-50%"}}
  }},
  "templates": [
    {{
      "message": "Hi {{{{name}}}}! 🎯 Check this out...",
      "target_segment": "hot",
      "approach": "direct",
      "personalization_tips": "Mention previous interest"
    }}
  ],
 "schedule": [
    {{
      "segment": "hot",
      "send_datetime": "YYYY-MM-DDTHH:MM:SS format (choose optimal time)",
      "reasoning": "Explain why this time works best for this segment",
      "priority": "high"
    }}
  ],
  "insights": {{
    "success_factors": ["Timing", "Personalization"],
    "warnings": ["Avoid spam"],
    "optimization_tips": ["Test multiple messages"]
  }}
}}"""

    def _call_groq_api(self, prompt: str) -> Dict[str, Any]:
        """Call Groq API synchronously"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key}"
        }

        payload = {
            "model": self.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a JSON-only assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2500,
        }

        response = requests.post(self.GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Groq API error {response.status_code}: {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Clean JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)
    def generate_campaign(self, request: CampaignRequest) -> CampaignResponse:
        """Main entry to generate campaign"""
        try:
            prompt = self._build_ai_prompt(request.prompt, request.timezone)
            campaign_data = self._call_groq_api(prompt)
        except Exception as e:
            self._log("error", f"Groq failed: {e}")
            raise  # <-- Stop here instead of using fallback


        strategy = CampaignStrategy(**campaign_data["strategy"])
        templates = [MessageTemplate(**t) for t in campaign_data["templates"]]
        schedule = [SendSchedule(**s) for s in campaign_data["schedule"]]

        return CampaignResponse(
            strategy=strategy,
            templates=templates,
            schedule=schedule,
            insights=campaign_data["insights"]
        )


# Quick helper function
def generate_campaign(prompt: str, groq_api_key: str, timezone: str = "UTC") -> Dict[str, Any]:
    generator = WhatsAppCampaignGenerator(groq_api_key)
    request = CampaignRequest(prompt=prompt, timezone=timezone)
    response = generator.generate_campaign(request)
    return {
        "strategy": response.strategy.dict(),
        "templates": [t.dict() for t in response.templates],
        "schedule": [s.dict() for s in response.schedule],
        "insights": response.insights
    } 