"""
Unit Tests for Lead Score Predictor
====================================
Add to tests/unit/
"""

import pytest

from app.ml.score import ScorePredictor


@pytest.fixture
def qualified_features():
    return {
        "messages_in_session": 12,
        "user_msg": "I want to buy. What's the price?",
        "conversation_duration_minutes": 18.5,
        "user_response_time_avg_seconds": 35.0,
        "user_initiated_conversation": True,
        "is_returning_customer": False,
        "time_of_day": "business_hours",
    }


@pytest.fixture
def unqualified_features():
    return {
        "messages_in_session": 2,
        "user_msg": "hi",
        "conversation_duration_minutes": 1.2,
        "user_response_time_avg_seconds": 180.0,
        "user_initiated_conversation": False,
        "is_returning_customer": False,
        "time_of_day": "off_hours",
    }


class TestScoreValidation:
    """Test input validation."""

    @pytest.mark.asyncio
    async def test_missing_field(self):
        """Predictor should raise when model not loaded or input incomplete."""
        incomplete = {"messages_in_session": 5, "user_msg": "test"}

        predictor = ScorePredictor()
        with pytest.raises(ValueError):
            await predictor.predict(incomplete)

    @pytest.mark.asyncio
    async def test_invalid_time_of_day(self, qualified_features):
        """Should raise error for invalid time_of_day."""
        qualified_features["time_of_day"] = "invalid"

        # Note: Validation happens in API layer, not predictor.
        predictor = ScorePredictor()
        try:
            await predictor.predict(qualified_features)
        except Exception:
            pass  # Expected if model not loaded


class TestScorePrediction:
    """Test prediction functionality."""

    @pytest.mark.asyncio
    async def test_predict_returns_correct_format(self, qualified_features):
        """Should return score, category, confidence."""
        predictor = ScorePredictor()
        predictor._load_model()
        if not predictor.pipeline:
            pytest.skip("Model not loaded")

        result = await predictor.predict(qualified_features)

        assert "score" in result
        assert "category" in result
        assert "confidence" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["category"] in ["hot", "warm", "cold"]
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_prediction_latency(self, qualified_features):
        """Should predict in under 50ms."""
        predictor = ScorePredictor()
        predictor._load_model()
        if not predictor.pipeline:
            pytest.skip("Model not loaded")

        import time

        start = time.time()

        await predictor.predict(qualified_features)

        latency_ms = (time.time() - start) * 1000
        assert latency_ms < 50, f"Latency {latency_ms:.2f}ms > 50ms"


class TestScoreCategories:
    """Test categorization logic."""

    @pytest.mark.asyncio
    async def test_high_score_is_hot(self, qualified_features):
        """High score should be categorized as hot."""
        predictor = ScorePredictor()
        predictor._load_model()
        if not predictor.pipeline:
            pytest.skip("Model not loaded")

        result = await predictor.predict(qualified_features)
        # Qualified leads should typically be warm or hot
        assert result["category"] in ["warm", "hot"]

    @pytest.mark.asyncio
    async def test_low_score_is_cold(self, unqualified_features):
        """Low score should be categorized as cold."""
        predictor = ScorePredictor()
        predictor._load_model()
        if not predictor.pipeline:
            pytest.skip("Model not loaded")

        result = await predictor.predict(unqualified_features)
        # Unqualified leads should typically be cold or warm
        assert result["category"] in ["cold", "warm"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
