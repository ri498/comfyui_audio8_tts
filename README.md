# comfyui_audio8_tts

> ComfyUI custom nodes for [Audio8 TTS](https://github.com/Edge0-AI/Audio8_TTS) — a compact DualAR text-to-speech model with zero-shot voice cloning.

- 📄 Paper: [arxiv.org/abs/2610.14776](https://arxiv.org/abs/2610.14776)
- 🤗 Model: [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b)

---

## Features

- **Two model sizes**: `0.6b` (Realistic) and `0.1b` (lightweight)
- **6 languages**: English, Chinese, Japanese, Spanish, French, German
- **Zero-shot voice cloning** via optional reference audio input
- **Speed control**: 0.5× – 2.0×
- **Model caching**: weights stay in VRAM between executions
- Standard ComfyUI `AUDIO` tensor output (`{"waveform": Tensor[B,C,S], "sample_rate": int}`)

---

## Installation

### Via ComfyUI Manager (recommended)
Search for `Audio8 TTS` and click Install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ri498/comfyui_audio8_tts
cd comfyui_audio8_tts
pip install -r requirements.txt
```
Restart ComfyUI.

---

## Nodes

| Node | Description |
|------|-------------|
| **Audio8 Text-to-Speech** | Generate speech from text. Optionally accepts a reference audio for voice cloning. |
| **Audio8 Model Loader** | Pre-download and cache model weights. Connect its output to the TTS node for explicit control. |

### Audio8 Text-to-Speech — Inputs

| Input | Type | Description |
|-------|------|-------------|
| `text` | STRING | Text to synthesize |
| `model_size` | `0.6b` / `0.1b` | Model variant |
| `language` | en/zh/ja/es/fr/de | Output language |
| `speed` | FLOAT (0.5–2.0) | Speech rate |
| `reference_audio` *(optional)* | AUDIO | Reference voice for zero-shot cloning |

---

## Model Weights

Weights are downloaded automatically from HuggingFace on first use and cached under `ComfyUI/models/TTS/Audio8-TTS-<size>/`.

---

## Status

> ⚠️ The upstream Audio8 Python SDK is not yet publicly released. The inference call is stubbed — audio output is currently silent placeholder audio. This will be updated as soon as the SDK is available.

---

## License

See [Audio8 TTS repository](https://github.com/Edge0-AI/Audio8_TTS) for model license.

