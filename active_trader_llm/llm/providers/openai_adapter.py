"""
OpenAI Adapter

Supports: GPT-4, GPT-4-Turbo, GPT-3.5-Turbo
API: https://platform.openai.com/docs/api-reference
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError, RateLimitError as OpenAIRateLimit, AuthenticationError as OpenAIAuthError

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


class OpenAIAdapter(LLMClient):
    """OpenAI API adapter (GPT-4, GPT-3.5-Turbo)"""

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.client = OpenAI(
            api_key=api_key,
            timeout=kwargs.get('timeout', 60.0),
            max_retries=kwargs.get('max_retries', 3)
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert LLMMessage to OpenAI format"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI API"""
        temperature = self._normalize_temperature(temperature)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None

            return LLMResponse(
                content=response.choices[0].message.content.strip(),
                model=self.model,
                provider=self.provider_name,
                usage=usage,
                raw_response=response
            )

        except OpenAIRateLimit as e:
            raise RateLimitError(str(e), self.provider_name, self.model)
        except OpenAIAuthError as e:
            raise AuthenticationError(str(e), self.provider_name, self.model)
        except OpenAIError as e:
            raise LLMError(str(e), self.provider_name, self.model)
        except Exception as e:
            raise LLMError(f"Unexpected error: {str(e)}", self.provider_name, self.model)

    def generate_structured(
        self,
        messages: List[LLMMessage],
        response_schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate structured JSON response using OpenAI API"""
        temperature = self._normalize_temperature(temperature)

        # For structured output, we'll use the response_format parameter (available in newer models)
        # Fallback: parse JSON from text response
        try:
            # Try using JSON mode (requires "JSON" in system prompt)
            api_kwargs = kwargs.copy()
            api_kwargs['response_format'] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **api_kwargs
            )

            content = response.choices[0].message.content.strip()

            # Validate JSON
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                raise InvalidRequestError(
                    f"Invalid JSON in response: {str(e)}",
                    self.provider_name,
                    self.model
                )

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None

            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.provider_name,
                usage=usage,
                raw_response=response
            )

        except OpenAIRateLimit as e:
            raise RateLimitError(str(e), self.provider_name, self.model)
        except OpenAIAuthError as e:
            raise AuthenticationError(str(e), self.provider_name, self.model)
        except OpenAIError as e:
            raise LLMError(str(e), self.provider_name, self.model)
        except Exception as e:
            raise LLMError(f"Unexpected error: {str(e)}", self.provider_name, self.model)


# Register provider
LLMClientFactory.register_provider("openai", OpenAIAdapter)
