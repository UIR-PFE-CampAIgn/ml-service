from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import os

from app.ml.campaign_generator import (
    WhatsAppCampaignGenerator,
    CampaignRequest,
)

router = APIRouter()


class GenerateCampaignRequest(BaseModel):
    """API request model for campaign generation"""
    prompt: str = Field(
        ...,
        description="Natural language description of the campaign",
        min_length=10,
        example="Launch our new premium CRM feature with 20% early bird discount for the next 48 hours"
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for scheduling",
        example="UTC"
    )


@router.post("/generate_campaign")
async def generate_campaign(request: GenerateCampaignRequest) -> Dict[str, Any]:
    """
    Generate a complete WhatsApp campaign using Groq AI (Llama 3.1).
    
    The AI will automatically determine:
    - Which lead segments to target (hot/warm/cold)
    - Message templates
    - Optimal send times
    
    **FREE API** - Get your Groq API key: https://console.groq.com/keys
    """
    try:
        # Get Groq API key from environment variable
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY not configured. Get free key at https://console.groq.com"
            )
        
        # Initialize generator
        generator = WhatsAppCampaignGenerator(groq_api_key=groq_api_key)
        
        # Create campaign request
        campaign_request = CampaignRequest(
            prompt=request.prompt,
            timezone=request.timezone
        )
        
        # Generate campaign
        campaign =  generator.generate_campaign(campaign_request)
        
        # Return as dict
        return {
            "strategy": {
                "target_segments": campaign.strategy.target_segments,
                "reasoning": campaign.strategy.reasoning,
                "campaign_type": campaign.strategy.campaign_type,
                "key_message": campaign.strategy.key_message,
                "expected_response_rates": campaign.strategy.expected_response_rates
            },
            "templates": [
                {
                    "message": t.message,
                    "target_segment": t.target_segment,
                    "approach": t.approach,
                    "personalization_tips": t.personalization_tips
                }
                for t in campaign.templates
            ],
            "schedule": [
                {
                    "segment": s.segment,
                    "send_datetime": s.send_datetime,
                    "reasoning": s.reasoning,
                    "priority": s.priority
                }
                for s in campaign.schedule
            ],
            "insights": campaign.insights
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Campaign generation failed: {str(e)}"
        )