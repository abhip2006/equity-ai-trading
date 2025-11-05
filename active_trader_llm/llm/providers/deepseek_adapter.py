"""
DeepSeek Adapter

Supports: DeepSeek models
API: OpenAI-compatible HTTP API
Endpoint: https://api.deepseek.com/v1
"""

import json
import logging
import requests
from typing import Any, Dict, List, Optional

from active_trader_llm.llm.llm_client import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    LLMClientFactory
)

logger = logging.getLogger(__name__)


class DeepSeekAdapter(LLMClient):
    """DeepSeek API adapter - OpenAI-compatible"""

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.timeout = kwargs.get('timeout', 60.0)
        self.max_retries = kwargs.get('max_retries', 3)

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert LLMMessage to DeepSeek format (same as OpenAI)"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to DeepSeek API with retries"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    raise RateLimitError(
                        f"Rate limit exceeded: {response.text}",
                        self.provider_name,
                        self.model
                    )
                elif response.status_code == 401:
                    raise AuthenticationError(
                        f"Authentication failed: {response.text}",
                        self.provider_name,
                        self.model
                    )
                elif response.status_code >= 400:
                    raise InvalidRequestError(
                        f"HTTP {response.status_code}: {response.text}",
                        self.provider_name,
                        self.model
                    )
                else:
                    raise LLMError(
                        f"HTTP {response.status_code}: {response.text}",
                        self.provider_name,
                        self.model
                    )

            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"Request timeout after {self.max_retries} attempts",
                        self.provider_name,
                        self.model
                    )
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"Request failed: {str(e)}",
                        self.provider_name,
                        self.model
                    )

        raise LLMError("Max retries exceeded", self.provider_name, self.model)

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using DeepSeek API"""
        temperature = self._normalize_temperature(temperature)

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        response_data = self._make_request(payload)

        usage = response_data.get("usage", {})
        usage_dict = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        } if usage else None

        content = response_data["choices"][0]["message"]["content"].strip()

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            usage=usage_dict,
            raw_response=response_data
        )

    def generate_structured(
        self,
        messages: List[LLMMessage],
        response_schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate structured JSON response using DeepSeek API"""
        temperature = self._normalize_temperature(temperature)

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        response_data = self._make_request(payload)

        usage = response_data.get("usage", {})
        usage_dict = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        } if usage else None

        content = response_data["choices"][0]["message"]["content"].strip()

        # Validate JSON
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise InvalidRequestError(
                f"Invalid JSON in response: {str(e)}",
                self.provider_name,
                self.model
            )

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            usage=usage_dict,
            raw_response=response_data
        )


# Register provider
LLMClientFactory.register_provider("deepseek", DeepSeekAdapter)
