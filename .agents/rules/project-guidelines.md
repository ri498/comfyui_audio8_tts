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

## Inference Pipeline

Audio8 uses HuggingFace `transformers` with `trust_remote_code=True`:

1. `AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)` — tokenises text + reference audio
2. `AutoModel.from_pretrained(...)` loaded as `bfloat16` on CUDA, `float32` on CPU
3. `model.generate(**inputs, return_dict_in_generate=True)` → `output.codes`
4. `model.decode_audio(output.codes)` → `(waveforms, waveform_lengths)`
5. Output sample rate from `model.config.codec_sample_rate`

### Voice Cloning
- Requires BOTH `reference_audio` AND `reference_text` (exact transcript)
- Reference waveform is saved to a temp WAV via `torchaudio.save()`, passed as a file path to the processor
- If only `reference_audio` is given without `reference_text`, a warning is logged and generation runs without cloning

## Model Caching

- Module-level `_MODEL_CACHE: dict` keyed by `"<model_size>:<device>"` keeps weights in VRAM
- `_load_model()` returns `{"processor": ..., "model": ..., "device": ...}`
- Weights cached under `ComfyUI/models/TTS/Audio8-TTS-<size>/` via `cache_dir`

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
| `torch>=2.5.0,<3` | Inference backend |
| `torchaudio>=2.5.0,<3` | Audio resampling + temp WAV save for reference audio |
| `transformers>=4.57.0,<5` | `AutoModel` / `AutoProcessor` for Audio8 loading |
| `huggingface_hub>=0.23.0` | Model download / caching |
| `soundfile>=0.12.1` | Audio file I/O |
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
