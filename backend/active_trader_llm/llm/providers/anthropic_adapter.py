"""
Anthropic Adapter

Supports: Claude Sonnet, Claude Opus, Claude Haiku
API: https://docs.anthropic.com/claude/reference/
"""

import json
import logging
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, APIError, RateLimitError as AnthropicRateLimit, AuthenticationError as AnthropicAuthError

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


class AnthropicAdapter(LLMClient):
    """Anthropic API adapter (Claude models)"""

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.client = Anthropic(
            api_key=api_key,
            timeout=kwargs.get('timeout', 60.0),
            max_retries=kwargs.get('max_retries', 3)
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[str, List[Dict[str, str]]]:
        """
        Convert LLMMessage to Anthropic format
        Anthropic requires system message separate from conversation

        Returns:
            (system_message, conversation_messages)
        """
        system_message = ""
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return system_message, conversation

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Anthropic API"""
        temperature = self._normalize_temperature(temperature)
        system_msg, conversation = self._convert_messages(messages)

        # Default max_tokens if not provided (Anthropic requires this)
        if max_tokens is None:
            max_tokens = 2000

        try:
            api_kwargs = {"model": self.model, "messages": conversation, "temperature": temperature, "max_tokens": max_tokens}

            if system_msg:
                api_kwargs["system"] = system_msg

            api_kwargs.update(kwargs)

            response = self.client.messages.create(**api_kwargs)

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

            # Extract text from content blocks
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text

            return LLMResponse(
                content=content.strip(),
                model=self.model,
                provider=self.provider_name,
                usage=usage,
                raw_response=response
            )

        except AnthropicRateLimit as e:
            raise RateLimitError(str(e), self.provider_name, self.model)
        except AnthropicAuthError as e:
            raise AuthenticationError(str(e), self.provider_name, self.model)
        except APIError as e:
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
        """
        Generate structured JSON response using Anthropic API
        Note: Claude doesn't have native JSON mode, so we parse from text
        """
        temperature = self._normalize_temperature(temperature)
        system_msg, conversation = self._convert_messages(messages)

        # Add JSON instruction to system message
        json_instruction = "\n\nYou must respond with valid JSON only. No markdown, no explanations."
        if system_msg:
            system_msg = system_msg + json_instruction
        else:
            system_msg = json_instruction.strip()

        if max_tokens is None:
            max_tokens = 2000

        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_msg,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

            # Extract text from content blocks
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text

            content = content.strip()

            # Remove markdown if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # Remove first and last line
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
                usage=usage,
                raw_response=response
            )

        except AnthropicRateLimit as e:
            raise RateLimitError(str(e), self.provider_name, self.model)
        except AnthropicAuthError as e:
            raise AuthenticationError(str(e), self.provider_name, self.model)
        except APIError as e:
            raise LLMError(str(e), self.provider_name, self.model)
        except Exception as e:
            raise LLMError(f"Unexpected error: {str(e)}", self.provider_name, self.model)


# Register provider
LLMClientFactory.register_provider("anthropic", AnthropicAdapter)
