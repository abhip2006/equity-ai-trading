"""
OpenRouter Adapter

Supports: Multiple models including Qwen, LLaMA, Mistral, etc.
API: OpenAI-compatible HTTP API
Endpoint: https://openrouter.ai/api/v1
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


class OpenRouterAdapter(LLMClient):
    """OpenRouter API adapter - supports multiple models via unified API"""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.timeout = kwargs.get('timeout', 60.0)
        self.max_retries = kwargs.get('max_retries', 3)
        self.app_name = kwargs.get('app_name', 'ActiveTraderLLM')

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert LLMMessage to OpenRouter format (same as OpenAI)"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenRouter API with retries"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/equity-ai-trading",  # Optional
            "X-Title": self.app_name  # Optional
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
        """Generate completion using OpenRouter API"""
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
        """Generate structured JSON response using OpenRouter API"""
        temperature = self._normalize_temperature(temperature)

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature
        }

        # OpenRouter supports response_format for some models
        if kwargs.get('supports_json_mode', False):
            payload["response_format"] = {"type": "json_object"}

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Remove custom kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'supports_json_mode'}
        payload.update(clean_kwargs)

        response_data = self._make_request(payload)

        usage = response_data.get("usage", {})
        usage_dict = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        } if usage else None

        content = response_data["choices"][0]["message"]["content"].strip()

        # Remove markdown if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
            if content.startswith("json"):
                content = content[4:].strip()

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
LLMClientFactory.register_provider("openrouter", OpenRouterAdapter)
