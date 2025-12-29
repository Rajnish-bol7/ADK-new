# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model Factory for creating LLM instances.

This module provides a factory for creating LLM instances based on model names
and tenant API keys. It supports multiple providers including Google Gemini,
OpenAI, Anthropic, and others via LiteLLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

from google.adk.models import LiteLlm
from google.adk.models.google_llm import Gemini

logger = logging.getLogger(__name__)


# Model provider mappings
MODEL_PROVIDERS: Dict[str, str] = {
    # OpenAI models
    "gpt-4": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4-turbo-preview": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-3.5-turbo": "openai",
    "gpt-3.5-turbo-16k": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    "o1-preview": "openai",
    # Anthropic models
    "claude-3-opus": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-sonnet": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "claude-3-haiku": "anthropic",
    "claude-3-haiku-20240307": "anthropic",
    "claude-3.5-sonnet": "anthropic",
    "claude-3-5-sonnet-20240620": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    # Google Gemini models (native ADK support)
    "gemini-2.5-flash": "google",
    "gemini-2.5-pro": "google",
    "gemini-2.0-flash": "google",
    "gemini-2.0-flash-exp": "google",
    "gemini-1.5-flash": "google",
    "gemini-1.5-flash-latest": "google",
    "gemini-1.5-pro": "google",
    "gemini-1.5-pro-latest": "google",
    "gemini-1.0-pro": "google",
    # Google Gemini Live/streaming models
    "gemini-2.5-flash-native-audio-latest": "google-live",
    "gemini-live-2.5-flash-preview": "google-live",
    "gemini-2.0-flash-live-preview-04-09": "google-live",
    # Mistral models
    "mistral-tiny": "mistral",
    "mistral-small": "mistral",
    "mistral-medium": "mistral",
    "mistral-large": "mistral",
    "mistral-large-latest": "mistral",
    # Cohere models
    "command": "cohere",
    "command-light": "cohere",
    "command-r": "cohere",
    "command-r-plus": "cohere",
    # Ollama models (local)
    "ollama/llama2": "ollama",
    "ollama/llama3": "ollama",
    "ollama/mistral": "ollama",
    "ollama/codellama": "ollama",
    # Grok models (xAI)
    "grok-2": "xai",
    "grok-2-mini": "xai",
    "grok-beta": "xai",
    "grok-1": "xai",
    # Groq models (fast inference)
    "groq/llama3-8b-8192": "groq",
    "groq/llama3-70b-8192": "groq",
    "groq/mixtral-8x7b-32768": "groq",
    "groq/gemma-7b-it": "groq",
    # Deepseek models
    "deepseek-chat": "deepseek",
    "deepseek-coder": "deepseek",
}

# Default model for voice calls (only Gemini Live supports native audio)
# User has NO choice for call model - Gemini Live is required
DEFAULT_CALL_MODEL = "gemini-2.0-flash-exp"


class ModelFactory:
    """
    Factory for creating LLM instances based on model name and API keys.

    This factory supports:
    - Google Gemini models (native ADK integration)
    - OpenAI models (via LiteLLM)
    - Anthropic Claude models (via LiteLLM)
    - Mistral, Cohere, and other providers (via LiteLLM)

    Example:
        factory = ModelFactory(api_keys={
            "google": "AIza...",
            "openai": "sk-...",
            "anthropic": "sk-ant-...",
        })

        model = factory.create_model("gpt-4")
    """

    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the model factory.

        Args:
            api_keys: Dict mapping provider names to API keys.
                Supported providers: google, openai, anthropic, mistral, cohere
        """
        self.api_keys = api_keys
        self._set_environment_variables()

    def _set_environment_variables(self) -> None:
        """Set environment variables for API keys if not already set."""
        env_mapping = {
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "cohere": "COHERE_API_KEY",
            "xai": "XAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }

        for provider, env_var in env_mapping.items():
            if provider in self.api_keys and self.api_keys[provider]:
                # Only set if not already in environment
                if not os.environ.get(env_var):
                    os.environ[env_var] = self.api_keys[provider]

    def create_model(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[Gemini, LiteLlm]:
        """
        Create an LLM instance for the given model name.

        Args:
            model_name: The model identifier (e.g., "gpt-4", "gemini-2.5-flash")
            temperature: Optional temperature setting for the model
            max_tokens: Optional max tokens setting

        Returns:
            An LLM instance compatible with ADK
        """
        provider = MODEL_PROVIDERS.get(model_name)

        if provider is None:
            # Check if it's a model with provider prefix (e.g., "openai/gpt-4")
            if "/" in model_name:
                provider = model_name.split("/")[0]
            else:
                # Default to LiteLLM for unknown models
                logger.warning(
                    f"Unknown model '{model_name}', using LiteLLM as fallback"
                )
                return self._create_litellm_model(model_name, temperature, max_tokens)

        if provider in ("google", "google-live"):
            return self._create_gemini_model(model_name, temperature, max_tokens)

        return self._create_litellm_model(model_name, temperature, max_tokens, provider)

    def _create_gemini_model(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Gemini:
        """Create a native Gemini model instance."""
        # Build generation config if parameters provided
        config: Dict[str, Any] = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        if config:
            from google.genai import types

            generation_config = types.GenerateContentConfig(**config)
            return Gemini(model=model_name, generation_config=generation_config)

        return Gemini(model=model_name)

    def _create_litellm_model(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> LiteLlm:
        """Create a LiteLLM model instance."""
        # Map provider to LiteLLM model prefix if needed
        litellm_model_name = model_name

        if provider == "anthropic" and not model_name.startswith("anthropic/"):
            litellm_model_name = f"anthropic/{model_name}"
        elif provider == "mistral" and not model_name.startswith("mistral/"):
            litellm_model_name = f"mistral/{model_name}"
        elif provider == "cohere" and not model_name.startswith("cohere/"):
            litellm_model_name = f"cohere/{model_name}"
        elif provider == "xai" and not model_name.startswith("xai/"):
            litellm_model_name = f"xai/{model_name}"
        elif provider == "groq" and not model_name.startswith("groq/"):
            litellm_model_name = f"groq/{model_name}"
        elif provider == "deepseek" and not model_name.startswith("deepseek/"):
            litellm_model_name = f"deepseek/{model_name}"

        # Get API key for the provider
        api_key = None
        if provider:
            api_key = self.api_keys.get(provider)

        # Build kwargs
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        return LiteLlm(model=litellm_model_name, **kwargs)

    def is_live_model(self, model_name: str) -> bool:
        """
        Check if the model supports live/streaming audio.

        Args:
            model_name: The model identifier

        Returns:
            True if the model supports live audio streaming
        """
        return MODEL_PROVIDERS.get(model_name) == "google-live"

    def get_provider(self, model_name: str) -> Optional[str]:
        """
        Get the provider for a model name.

        Args:
            model_name: The model identifier

        Returns:
            Provider name or None if unknown
        """
        return MODEL_PROVIDERS.get(model_name)

    @staticmethod
    def get_supported_models() -> Dict[str, str]:
        """
        Get a dict of all supported models and their providers.

        Returns:
            Dict mapping model names to provider names
        """
        return MODEL_PROVIDERS.copy()

    @staticmethod
    def get_models_for_provider(provider: str) -> list[str]:
        """
        Get all models for a specific provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            List of model names for that provider
        """
        return [model for model, prov in MODEL_PROVIDERS.items() if prov == provider]
