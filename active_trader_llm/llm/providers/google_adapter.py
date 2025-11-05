"""
Google Gemini Adapter

Supports: Gemini Pro, Gemini Flash
API: https://ai.google.dev/api/python/google/generativeai
"""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

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


class GoogleAdapter(LLMClient):
    """Google Gemini API adapter"""

    def __init__(self, model: str, api_key: str, **kwargs):
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        super().__init__(model, api_key, **kwargs)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    @property
    def provider_name(self) -> str:
        return "google"

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[str, List[Dict[str, str]]]:
        """
        Convert LLMMessage to Gemini format
        Gemini uses system instruction separately

        Returns:
            (system_instruction, conversation_parts)
        """
        system_instruction = ""
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                # Gemini uses 'user' and 'model' instead of 'assistant'
                role = "model" if msg.role == "assistant" else msg.role
                conversation.append({
                    "role": role,
                    "parts": [msg.content]
                })

        return system_instruction, conversation

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Gemini API"""
        temperature = self._normalize_temperature(temperature)
        system_instruction, conversation = self._convert_messages(messages)

        try:
            # Configure generation
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )

            # Update model with system instruction if provided
            if system_instruction:
                self.client = genai.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction
                )

            # Start chat or generate
            if len(conversation) > 1:
                # Multi-turn conversation
                chat = self.client.start_chat(history=conversation[:-1])
                response = chat.send_message(
                    conversation[-1]["parts"][0],
                    generation_config=generation_config
                )
            else:
                # Single-turn generation
                response = self.client.generate_content(
                    conversation[0]["parts"][0],
                    generation_config=generation_config
                )

            # Extract usage info
            usage_dict = None
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                usage_dict = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }

            return LLMResponse(
                content=response.text.strip(),
                model=self.model,
                provider=self.provider_name,
                usage=usage_dict,
                raw_response=response
            )

        except Exception as e:
            error_str = str(e).lower()

            # Classify errors
            if "quota" in error_str or "rate" in error_str:
                raise RateLimitError(str(e), self.provider_name, self.model)
            elif "api key" in error_str or "auth" in error_str:
                raise AuthenticationError(str(e), self.provider_name, self.model)
            elif "invalid" in error_str:
                raise InvalidRequestError(str(e), self.provider_name, self.model)
            else:
                raise LLMError(str(e), self.provider_name, self.model)

    def generate_structured(
        self,
        messages: List[LLMMessage],
        response_schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate structured JSON response using Gemini API
        Note: Gemini doesn't have native JSON mode, so we parse from text
        """
        temperature = self._normalize_temperature(temperature)
        system_instruction, conversation = self._convert_messages(messages)

        # Add JSON instruction
        json_instruction = "\n\nYou must respond with valid JSON only. No markdown, no explanations."
        if system_instruction:
            system_instruction = system_instruction + json_instruction
        else:
            system_instruction = json_instruction.strip()

        try:
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )

            # Update model with system instruction
            self.client = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction
            )

            # Generate
            if len(conversation) > 1:
                chat = self.client.start_chat(history=conversation[:-1])
                response = chat.send_message(
                    conversation[-1]["parts"][0],
                    generation_config=generation_config
                )
            else:
                response = self.client.generate_content(
                    conversation[0]["parts"][0],
                    generation_config=generation_config
                )

            # Extract usage info
            usage_dict = None
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                usage_dict = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }

            content = response.text.strip()

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
                raw_response=response
            )

        except Exception as e:
            error_str = str(e).lower()

            if "quota" in error_str or "rate" in error_str:
                raise RateLimitError(str(e), self.provider_name, self.model)
            elif "api key" in error_str or "auth" in error_str:
                raise AuthenticationError(str(e), self.provider_name, self.model)
            elif "invalid" in error_str:
                raise InvalidRequestError(str(e), self.provider_name, self.model)
            else:
                raise LLMError(str(e), self.provider_name, self.model)


# Register provider
if GOOGLE_AVAILABLE:
    LLMClientFactory.register_provider("google", GoogleAdapter)
