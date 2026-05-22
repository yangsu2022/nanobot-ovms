"""Semantic routing for hybrid local/cloud model selection.

This module provides intelligent model routing based on prompt classification
using a local LLM instead of rigid keyword matching.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import Config, ModelPresetConfig
    from nanobot.providers.base import LLMProvider

RouteTarget = Literal["LOCAL", "CLOUD"]

# Classification prompt embedded in the router
CLASSIFICATION_PROMPT = """You are a routing classifier. Analyze the user's request and respond with ONLY one word: LOCAL or CLOUD.

Route to LOCAL:
- Local document search, RAG retrieval
- Meeting summary, personal notes & private content
- Local image/video search
- Daily chat, simple Q&A & greetings
- Short code snippet & simple script writing
- Personal private file operation

Route to CLOUD:
- In-depth professional knowledge research
- Complex multi-step tasks & multiple tool calls
- Agent & custom skill creation
- Long-form article/report generation
- Super long context over 32K tokens
- Large codebase analysis & complex architecture design

User request: {prompt}

Classification (respond with only LOCAL or CLOUD):"""


class HybridRouter:
    """Hybrid router using local LLM semantic classification.

    This is a minimal v1 implementation that:
    - Uses fixed classification prompt
    - Invokes local LLM for semantic routing decisions
    - Skips exception handling (marked as TODO for v2)
    - Maintains simple routing priority chain
    """

    def __init__(
        self,
        config: Config,
        local_preset: ModelPresetConfig,
        cloud_preset: ModelPresetConfig,
        local_provider: LLMProvider,
    ):
        """Initialize hybrid router.

        Args:
            config: Full nanobot configuration
            local_preset: Configuration for local model (e.g., OVMS)
            cloud_preset: Configuration for cloud model (e.g., DeepSeek)
            local_provider: Pre-instantiated local provider for classification
        """
        self.config = config
        self.local_preset = local_preset
        self.cloud_preset = cloud_preset
        self._classifier_provider = local_provider
        self._stats = {"local_routes": 0, "cloud_routes": 0, "classification_errors": 0}

    async def classify_prompt(self, prompt: str) -> RouteTarget:
        """Classify prompt using local LLM.

        Sends classification prompt to local model and parses response.

        Args:
            prompt: User's input message

        Returns:
            "LOCAL" or "CLOUD" routing target

        Note:
            TODO v2: Add exception handling for classification failures
            TODO v2: Add retry logic with fallback to default route
        """
        classification_prompt = CLASSIFICATION_PROMPT.format(prompt=prompt[:1000])

        # TODO v2: Add try-except for classification request errors
        response = await self._classifier_provider.chat(
            messages=[{"role": "user", "content": classification_prompt}],
            model=self.local_preset.model,
            max_tokens=300,  # Allow enough for thinking + classification result
            temperature=0.0,
        )

        result = self._parse_classification(response.content or "")
        logger.debug(f"Classification result for prompt: {result}")
        return result

    def _parse_classification(self, response: str) -> RouteTarget:
        """Parse LLM response to extract LOCAL or CLOUD decision.

        Handles responses with <think> tags by searching after the thinking section.

        Args:
            response: Raw LLM response text

        Returns:
            Extracted route target, defaults to LOCAL if parsing fails
        """
        # Remove <think>...</think> section if present to avoid false matches
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Extract first occurrence of LOCAL or CLOUD (case-insensitive) after thinking
        match = re.search(r'\b(LOCAL|CLOUD)\b', cleaned_response.upper())
        if match:
            return match.group(1)  # type: ignore

        # Fallback: check original response if cleaning removed everything
        match = re.search(r'\b(LOCAL|CLOUD)\b', response.upper())
        if match:
            return match.group(1)  # type: ignore

        # Fallback to LOCAL if unclear
        logger.warning(f"Failed to parse classification response: {response[:100]}")
        return "LOCAL"

    async def route(
        self,
        prompt: str,
        user_command: str | None = None,
        context_tokens: int = 0,
    ) -> tuple[ModelPresetConfig, str]:
        """Determine routing target with priority chain.

        Priority order:
        1. User explicit route instruction (e.g., /local, /cloud commands)
        2. Local LLM semantic classification
        3. Cloud API key availability check
        4. Local & cloud service status check
        5. Context length limit check
        6. Default to local

        Args:
            prompt: User's input message
            user_command: Explicit routing command if present
            context_tokens: Current conversation context token count

        Returns:
            Tuple of (selected_preset, routing_reason)

        Note:
            TODO v2: Add service availability checks
            TODO v2: Add automatic fallback when selected service unavailable
        """
        # Priority 1: User explicit command
        if user_command:
            if user_command.lower() in ["local", "ovms"]:
                self._stats["local_routes"] += 1
                return self.local_preset, "user_command:local"
            elif user_command.lower() in ["cloud", "deepseek"]:
                self._stats["cloud_routes"] += 1
                return self.cloud_preset, "user_command:cloud"

        # Priority 2: Context length check (hard constraint)
        # If context exceeds local model capacity, force cloud
        if context_tokens > self.local_preset.context_window_tokens:
            logger.info(
                f"Context {context_tokens} exceeds local capacity "
                f"{self.local_preset.context_window_tokens}, routing to cloud"
            )
            self._stats["cloud_routes"] += 1
            return self.cloud_preset, "context_length_exceeded"

        # Priority 3: Cloud API key check
        # If no cloud key configured, must use local
        cloud_provider_config = self.config.get_provider(
            self.cloud_preset.model,
            preset=self.cloud_preset
        )
        if not cloud_provider_config or not cloud_provider_config.api_key:
            logger.debug("No cloud API key configured, routing to local")
            self._stats["local_routes"] += 1
            return self.local_preset, "no_cloud_key"

        # Priority 4: Semantic classification
        # TODO v2: Wrap in try-except and add fallback on classification error
        classification = await self.classify_prompt(prompt)

        if classification == "LOCAL":
            self._stats["local_routes"] += 1
            return self.local_preset, "classification:local"
        else:
            self._stats["cloud_routes"] += 1
            return self.cloud_preset, "classification:cloud"

    def get_stats(self) -> dict[str, int]:
        """Return routing statistics for monitoring."""
        return self._stats.copy()
