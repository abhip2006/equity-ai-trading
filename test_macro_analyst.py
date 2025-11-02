"""
Tests for MacroAnalyst.

Tests LLM integration, prompt building, and signal generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from active_trader_llm.analysts.macro_analyst import MacroAnalyst, MacroSignal
from active_trader_llm.feature_engineering.models import MacroSnapshot


class TestMacroSignal:
    """Test suite for MacroSignal model"""

    def test_valid_signal_creation(self):
        """Test creating a valid macro signal"""
        signal = MacroSignal(
            analyst="Macro",
            market_environment="risk_on",
            tailwinds=["Low volatility", "Stable yields"],
            headwinds=["Rising dollar"],
            key_observations=["VIX at 15", "10Y yield at 4%"],
            confidence=0.75
        )

        assert signal.analyst == "Macro"
        assert signal.market_environment == "risk_on"
        assert len(signal.tailwinds) == 2
        assert len(signal.headwinds) == 1
        assert signal.confidence == 0.75

    def test_confidence_validation(self):
        """Test confidence must be between 0 and 1"""
        # Valid confidence
        signal = MacroSignal(
            analyst="Macro",
            market_environment="neutral",
            tailwinds=[],
            headwinds=[],
            key_observations=[],
            confidence=0.5
        )
        assert signal.confidence == 0.5

        # Invalid confidence
        with pytest.raises(ValueError):
            MacroSignal(
                analyst="Macro",
                market_environment="neutral",
                tailwinds=[],
                headwinds=[],
                key_observations=[],
                confidence=1.5
            )

    def test_environment_validation(self):
        """Test market_environment must be valid literal"""
        # Valid environments
        for env in ["risk_on", "risk_off", "neutral", "transitioning"]:
            signal = MacroSignal(
                analyst="Macro",
                market_environment=env,
                tailwinds=[],
                headwinds=[],
                key_observations=[],
                confidence=0.5
            )
            assert signal.market_environment == env

        # Invalid environment
        with pytest.raises(ValueError):
            MacroSignal(
                analyst="Macro",
                market_environment="invalid_env",
                tailwinds=[],
                headwinds=[],
                key_observations=[],
                confidence=0.5
            )


class TestMacroAnalyst:
    """Test suite for MacroAnalyst"""

    def test_initialization(self):
        """Test analyst initializes correctly"""
        analyst = MacroAnalyst(api_key="test_key", model="gpt-3.5-turbo")

        assert analyst.model == "gpt-3.5-turbo"
        assert analyst.client is not None

    def test_build_analysis_prompt_full_data(self):
        """Test prompt building with full data"""
        analyst = MacroAnalyst()

        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=17.5,
            vxn=21.2,
            move_index=105.3,
            treasury_10y=4.10,
            treasury_2y=3.72,
            treasury_30y=4.67,
            yield_curve_spread=0.38,
            gold_price=3982.20,
            oil_price=60.98,
            dollar_index=99.80
        )

        prompt = analyst._build_analysis_prompt(snapshot)

        # Check all data points are in prompt
        assert "VIX (S&P 500 Volatility): 17.50" in prompt
        assert "VXN (Nasdaq Volatility): 21.20" in prompt
        assert "MOVE Index (Treasury Volatility): 105.30" in prompt
        assert "10-Year Treasury Yield: 4.10%" in prompt
        assert "2-Year Treasury Yield: 3.72%" in prompt
        assert "Yield Curve Spread (10Y-2Y): +0.38%" in prompt
        assert "Gold: $3982.20/oz" in prompt
        assert "Crude Oil: $60.98/barrel" in prompt
        assert "US Dollar Index (DXY): 99.80" in prompt

        # Check structure
        assert "VOLATILITY ENVIRONMENT:" in prompt
        assert "INTEREST RATE ENVIRONMENT:" in prompt
        assert "COMMODITIES:" in prompt
        assert "CURRENCY:" in prompt

    def test_build_analysis_prompt_partial_data(self):
        """Test prompt building with partial data"""
        analyst = MacroAnalyst()

        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=17.5,
            vxn=None,
            move_index=None,
            treasury_10y=4.10,
            treasury_2y=None,
            treasury_30y=None,
            yield_curve_spread=None,
            gold_price=None,
            oil_price=60.98,
            dollar_index=99.80
        )

        prompt = analyst._build_analysis_prompt(snapshot)

        # Available data should be in prompt
        assert "VIX (S&P 500 Volatility): 17.50" in prompt
        assert "10-Year Treasury Yield: 4.10%" in prompt
        assert "Crude Oil: $60.98/barrel" in prompt
        assert "US Dollar Index (DXY): 99.80" in prompt

        # Missing data should show NOT AVAILABLE
        assert "VXN: NOT AVAILABLE" in prompt
        assert "MOVE Index: NOT AVAILABLE" in prompt
        assert "2-Year Treasury: NOT AVAILABLE" in prompt
        assert "Gold: NOT AVAILABLE" in prompt

    @patch('openai.OpenAI')
    def test_analyze_success(self, mock_openai_class):
        """Test successful analysis"""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "analyst": "Macro",
            "market_environment": "risk_on",
            "tailwinds": ["Low VIX at 15", "Positive yield curve"],
            "headwinds": ["Rising dollar"],
            "key_observations": ["VIX subdued", "Yields stable", "Oil strong"],
            "confidence": 0.75
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        analyst = MacroAnalyst(api_key="test_key")
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=15.0,
            treasury_10y=4.0,
            oil_price=65.0,
            dollar_index=100.0
        )

        signal = analyst.analyze(snapshot)

        assert signal is not None
        assert signal.market_environment == "risk_on"
        assert signal.confidence == 0.75
        assert len(signal.tailwinds) == 2
        assert len(signal.headwinds) == 1

    @patch('openai.OpenAI')
    def test_analyze_with_markdown_response(self, mock_openai_class):
        """Test handling of markdown-wrapped JSON response"""
        # Mock OpenAI response with markdown
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """```json
{
    "analyst": "Macro",
    "market_environment": "neutral",
    "tailwinds": ["Stable markets"],
    "headwinds": ["Mixed signals"],
    "key_observations": ["VIX moderate"],
    "confidence": 0.5
}
```"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        analyst = MacroAnalyst(api_key="test_key")
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=18.0
        )

        signal = analyst.analyze(snapshot)

        assert signal is not None
        assert signal.market_environment == "neutral"
        assert signal.confidence == 0.5

    @patch('openai.OpenAI')
    def test_analyze_invalid_json(self, mock_openai_class):
        """Test handling of invalid JSON response"""
        # Mock invalid response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        analyst = MacroAnalyst(api_key="test_key")
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=18.0
        )

        signal = analyst.analyze(snapshot)

        # Should return None on JSON parse error
        assert signal is None

    @patch('openai.OpenAI')
    def test_analyze_llm_exception(self, mock_openai_class):
        """Test handling of LLM API exception"""
        # Mock exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        analyst = MacroAnalyst(api_key="test_key")
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=18.0
        )

        signal = analyst.analyze(snapshot)

        # Should return None on exception
        assert signal is None

    def test_analyze_no_data(self):
        """Test analysis with no macro data"""
        analyst = MacroAnalyst()

        # All fields None
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat()
        )

        signal = analyst.analyze(snapshot)

        # Should return None when no data available
        assert signal is None

    @patch('openai.OpenAI')
    def test_analyze_model_dump(self, mock_openai_class):
        """Test signal can be converted to dict"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "analyst": "Macro",
            "market_environment": "risk_off",
            "tailwinds": [],
            "headwinds": ["High VIX", "Inverted curve"],
            "key_observations": ["Risk-off signals"],
            "confidence": 0.8
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        analyst = MacroAnalyst(api_key="test_key")
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=30.0,
            yield_curve_spread=-0.2
        )

        signal = analyst.analyze(snapshot)
        signal_dict = signal.model_dump()

        assert isinstance(signal_dict, dict)
        assert signal_dict['analyst'] == "Macro"
        assert signal_dict['market_environment'] == "risk_off"
        assert signal_dict['confidence'] == 0.8

    @patch('openai.OpenAI')
    def test_system_prompt_structure(self, mock_openai_class):
        """Test system prompt is used correctly"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "analyst": "Macro",
            "market_environment": "neutral",
            "tailwinds": [],
            "headwinds": [],
            "key_observations": [],
            "confidence": 0.5
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        analyst = MacroAnalyst(api_key="test_key")
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            vix=18.0
        )

        analyst.analyze(snapshot)

        # Verify system prompt was passed
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert 'MacroAnalyst' in call_args[1]['messages'][0]['content']
        assert 'RAW numerical values' in call_args[1]['messages'][0]['content']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
