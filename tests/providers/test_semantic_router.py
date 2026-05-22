"""Tests for semantic router."""

import pytest

from nanobot.config.schema import Config, ModelPresetConfig, ProviderConfig
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.semantic_router import HybridRouter


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(self, response_text: str = "LOCAL"):
        self._response_text = response_text
        self.generation = type('obj', (object,), {'max_tokens': 2048, 'temperature': 0.1})()

    def get_default_model(self) -> str:
        return "mock-model"

    async def chat(self, **kwargs) -> LLMResponse:
        return LLMResponse(
            content=self._response_text,
            finish_reason="stop",
        )

    async def chat_stream(self, **kwargs) -> LLMResponse:
        return await self.chat(**kwargs)


@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    config = Config(
        workspace_path=".",
        model_presets={
            "local": ModelPresetConfig(
                model="local-model",
                provider="ovms",
                context_window_tokens=32768,
            ),
            "cloud": ModelPresetConfig(
                model="cloud-model",
                provider="deepseek",
                context_window_tokens=65536,
            ),
        },
        providers={
            "ovms": ProviderConfig(api_key="local-key"),
            "deepseek": ProviderConfig(api_key="cloud-key"),
        },
    )
    return config


@pytest.fixture
def router(mock_config):
    """Create router with mock provider."""
    local_preset = mock_config.model_presets["local"]
    cloud_preset = mock_config.model_presets["cloud"]
    local_provider = MockProvider("LOCAL")

    return HybridRouter(
        config=mock_config,
        local_preset=local_preset,
        cloud_preset=cloud_preset,
        local_provider=local_provider,
    )


@pytest.mark.asyncio
async def test_parse_classification_valid_local(router):
    """Test parsing valid LOCAL response."""
    result = router._parse_classification("LOCAL")
    assert result == "LOCAL"


@pytest.mark.asyncio
async def test_parse_classification_valid_cloud(router):
    """Test parsing valid CLOUD response."""
    result = router._parse_classification("CLOUD")
    assert result == "CLOUD"


@pytest.mark.asyncio
async def test_parse_classification_with_text(router):
    """Test parsing response with surrounding text."""
    result = router._parse_classification("The classification is: LOCAL for this request")
    assert result == "LOCAL"


@pytest.mark.asyncio
async def test_parse_classification_fallback(router):
    """Test fallback to LOCAL on parse failure."""
    result = router._parse_classification("unclear response")
    assert result == "LOCAL"


@pytest.mark.asyncio
async def test_route_user_command_local(router):
    """Test routing with explicit local command."""
    preset, reason = await router.route(
        prompt="Test prompt",
        user_command="local",
    )
    assert preset.model == "local-model"
    assert reason == "user_command:local"


@pytest.mark.asyncio
async def test_route_user_command_cloud(router):
    """Test routing with explicit cloud command."""
    preset, reason = await router.route(
        prompt="Test prompt",
        user_command="cloud",
    )
    assert preset.model == "cloud-model"
    assert reason == "user_command:cloud"


@pytest.mark.asyncio
async def test_route_context_length_exceeded(router):
    """Test routing when context exceeds local capacity."""
    preset, reason = await router.route(
        prompt="Test prompt",
        context_tokens=40000,  # Exceeds local 32K limit
    )
    assert preset.model == "cloud-model"
    assert reason == "context_length_exceeded"


@pytest.mark.asyncio
async def test_route_no_cloud_key(mock_config):
    """Test routing when cloud key unavailable."""
    # Remove cloud API key
    mock_config.providers["deepseek"].api_key = None

    local_preset = mock_config.model_presets["local"]
    cloud_preset = mock_config.model_presets["cloud"]
    local_provider = MockProvider("CLOUD")  # Classification says CLOUD

    router = HybridRouter(
        config=mock_config,
        local_preset=local_preset,
        cloud_preset=cloud_preset,
        local_provider=local_provider,
    )

    preset, reason = await router.route(prompt="Test prompt")
    assert preset.model == "local-model"
    assert reason == "no_cloud_key"


@pytest.mark.asyncio
async def test_route_classification_local(router):
    """Test routing based on LOCAL classification."""
    preset, reason = await router.route(prompt="Hello, how are you?")
    assert preset.model == "local-model"
    assert reason == "classification:local"


@pytest.mark.asyncio
async def test_route_classification_cloud(mock_config):
    """Test routing based on CLOUD classification."""
    local_preset = mock_config.model_presets["local"]
    cloud_preset = mock_config.model_presets["cloud"]
    local_provider = MockProvider("CLOUD")  # Mock returns CLOUD

    router = HybridRouter(
        config=mock_config,
        local_preset=local_preset,
        cloud_preset=cloud_preset,
        local_provider=local_provider,
    )

    preset, reason = await router.route(
        prompt="Write a comprehensive research paper on quantum computing"
    )
    assert preset.model == "cloud-model"
    assert reason == "classification:cloud"


@pytest.mark.asyncio
async def test_routing_statistics(router):
    """Test routing statistics tracking."""
    await router.route(prompt="Test 1", user_command="local")
    await router.route(prompt="Test 2", user_command="cloud")
    await router.route(prompt="Test 3")  # Classification -> LOCAL

    stats = router.get_stats()
    assert stats["local_routes"] == 2
    assert stats["cloud_routes"] == 1
