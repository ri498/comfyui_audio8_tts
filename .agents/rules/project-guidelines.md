# audio8_tts — Project Guidelines

## Overview
**comfyui_audio8_tts** is a set of ComfyUI custom nodes that wrap the Audio8 TTS model for in-graph text-to-speech synthesis.

- Paper: https://arxiv.org/abs/2610.14776
- GitHub (upstream model): https://github.com/Edge0-AI/Audio8_TTS
- HuggingFace: https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b
- GitHub (this repo): https://github.com/ri498/comfyui_audio8_tts

---

## Project Structure

```
comfyui_audio8_tts/
├── __init__.py          # ComfyUI entry point — exports NODE_CLASS_MAPPINGS
├── audio8_nodes.py      # All node definitions and model loading logic
├── requirements.txt     # Python dependencies
├── README.md
├── CLAUDE.md            # Agent instructions
└── .agents/
    └── rules/
        └── project-guidelines.md  ← this file
```

---

## ComfyUI Node Conventions

- Every node class must define:
  - `INPUT_TYPES(cls)` — classmethod returning required/optional input dict
  - `RETURN_TYPES` — tuple of type strings
  - `RETURN_NAMES` — tuple of output names
  - `FUNCTION` — name of the method to call
  - `CATEGORY` — shown in the node menu (`"Audio8-TTS"`)
  - `DESCRIPTION` — short string shown in UI tooltip

- ComfyUI `AUDIO` format: `{"waveform": torch.Tensor([B, C, S], float32, -1 to 1), "sample_rate": int}`
- Default sample rate for Audio8: **44100 Hz**
- Always resample reference audio to 44100 Hz before passing to the model

---

## Model Caching

- Module-level `_MODEL_CACHE: dict` keyed by `"<model_size>:<device>"` keeps weights in VRAM
- `_load_model()` checks the cache before downloading
- Weights are stored under `ComfyUI/models/TTS/Audio8-TTS-<size>/`
- Download via `huggingface_hub.snapshot_download()`

---

## Supported Models

| Size | Repo ID | Notes |
|------|---------|-------|
| `0.6b` | `Edge0/Audio8-TTS-Realistic-0.6b` | Default, higher quality |
| `0.1b` | `Edge0/Audio8-TTS-Realistic-0.1b` | Lightweight |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch>=2.0.0` | Inference backend |
| `torchaudio>=2.0.0` | Audio resampling |
| `huggingface_hub>=0.23.0` | Model download |
| `numpy>=1.24.0` | Array utilities |

---

## Registered Nodes

| Class | Display Name | Description |
|-------|-------------|-------------|
| `Audio8TTSNode` | Audio8 Text-to-Speech | Main TTS generation node |
| `Audio8ModelLoaderNode` | Audio8 Model Loader | Explicit weight pre-loader |

---

## Branching Strategy

- `main` — stable, tagged releases
- `development` — active work, merged into main via PR

---

## Coding Standards

- Follow ComfyUI custom node conventions strictly
- Use type hints on all functions
- Model inference stub marked with `# TODO:` until upstream SDK is available
- Print progress with `[Audio8-TTS]` prefix for ComfyUI console visibility
- No secrets or API keys in code — use environment variables if needed
