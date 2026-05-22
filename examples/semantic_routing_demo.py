"""Demo script for semantic routing functionality.

This script shows how to use the HybridRouter for intelligent
local/cloud model selection based on prompt classification.

Usage:
    python examples/semantic_routing_demo.py
"""

import asyncio
from pathlib import Path

from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.providers.factory import _make_provider_core
from nanobot.providers.semantic_router import HybridRouter


async def demo_routing():
    """Demonstrate semantic routing with sample prompts."""
    # Load config - try fallback first if main config has no presets
    config_path = Path.home() / ".nanobot" / "config.json"
    fallback_path = Path.home() / ".nanobot" / "config.fallback.json"

    # Try fallback config if it exists
    if fallback_path.exists():
        print(f"Using fallback config: {fallback_path}")
        config = resolve_config_env_vars(load_config(fallback_path))
    else:
        config = resolve_config_env_vars(load_config(config_path))

    # Get local and cloud presets from config
    if "local" not in config.model_presets:
        print("Error: 'local' preset not found in config.json")
        print("Please configure model_presets with 'local' and 'cloud' entries")
        return

    if "cloud" not in config.model_presets:
        print("Error: 'cloud' preset not found in config.json")
        print("Please configure model_presets with 'local' and 'cloud' entries")
        return

    local_preset = config.model_presets["local"]
    cloud_preset = config.model_presets["cloud"]

    # Create local provider for classification
    local_provider = _make_provider_core(config, preset=local_preset)

    # Initialize router
    router = HybridRouter(
        config=config,
        local_preset=local_preset,
        cloud_preset=cloud_preset,
        local_provider=local_provider,
    )

    # Test prompts
    test_cases = [
        # Should route to LOCAL
        ("Hello! How are you?", "Simple greeting"),
        ("Summarize my meeting notes from today", "Local document task"),
        ("Write a short Python script to sort a list", "Simple code snippet"),
        ("What files are in my current directory?", "Local file operation"),

        # Should route to CLOUD
        ("Explain the latest developments in quantum computing", "Professional knowledge"),
        ("Write a comprehensive research paper on climate change", "Long-form content"),
        ("Design a microservices architecture for a large e-commerce platform", "Complex architecture"),
        ("Create a custom agent with multiple tools for data analysis", "Agent creation"),
    ]

    print("=" * 80)
    print("SEMANTIC ROUTING DEMO")
    print("=" * 80)
    print()

    for prompt, description in test_cases:
        print(f"Test: {description}")
        print(f"Prompt: {prompt[:70]}...")
        print()

        # Route the prompt
        selected_preset, reason = await router.route(
            prompt=prompt,
            context_tokens=100,  # Simulate small context
        )

        print(f"  ✓ Route: {selected_preset.model}")
        print(f"  ✓ Provider: {selected_preset.provider}")
        print(f"  ✓ Reason: {reason}")
        print()
        print("-" * 80)
        print()

    # Show statistics
    stats = router.get_stats()
    print()
    print("ROUTING STATISTICS")
    print("=" * 80)
    print(f"Local routes:  {stats['local_routes']}")
    print(f"Cloud routes:  {stats['cloud_routes']}")
    print(f"Classification errors: {stats['classification_errors']}")
    print()


if __name__ == "__main__":
    asyncio.run(demo_routing())
