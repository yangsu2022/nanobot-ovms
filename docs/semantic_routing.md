# Semantic Routing

## Overview

Semantic routing provides intelligent model selection between local and cloud providers using local LLM classification instead of keyword matching.

**Note**: This implementation is built on top of the existing **Model Presets** feature in nanobot's main branch. It extends the preset system with semantic classification for automatic preset selection.

## Architecture

### Routing Priority Chain

```
User Command → Context Length → Cloud Key Check → Semantic Classification → Default Local
```

### Classification Criteria

**Route to LOCAL:**
- Local document search & personal notes
- Simple Q&A & greetings
- Short code snippets & file operations

**Route to CLOUD:**
- Professional knowledge research
- Complex multi-step tasks
- Long-form content generation
- Large codebase analysis
- Context over 32K tokens

## Configuration

This implementation uses nanobot's existing **Model Presets** feature (already in main branch). 

Add `local` and `cloud` presets to `~/.nanobot/config.json`:

```json
{
  "model_presets": {
    "local": {
      "model": "OpenVINO/Qwen3-4B-int4-ov",
      "provider": "ovms",
      "maxTokens": 2048,
      "contextWindowTokens": 32768,
      "temperature": 0.1
    },
    "cloud": {
      "model": "cloud-model-name",
      "provider": "cloud-provider",
      "maxTokens": 4096,
      "contextWindowTokens": 65536,
      "temperature": 0.1
    }
  }
}
```

## Usage

Run the demo script:

```bash
.venv\Scripts\python.exe examples/semantic_routing_demo.py
```

## Performance

- **Latency**: ~7 seconds per classification (with thinking mode)
- **Token usage**: ~255-310 tokens per classification
- **Accuracy**: 87.5% semantic accuracy (8 test cases)

## Limitations

This is a minimal v1 implementation. Future improvements (v2) may include:
- Exception handling for classification failures
- Service health checks and automatic fallback
- Classification result caching

## References

- **Implementation**: [nanobot/providers/semantic_router.py](../nanobot/providers/semantic_router.py)
- **Demo**: [examples/semantic_routing_demo.py](../examples/semantic_routing_demo.py)
- **Tests**: [tests/providers/test_semantic_router.py](../tests/providers/test_semantic_router.py)
- **Test Results**: [TEST_RESULTS.md](../TEST_RESULTS.md)
