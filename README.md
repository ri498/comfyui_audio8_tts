# comfyui_audio8_tts

> ComfyUI custom nodes for Audio8 TTS — a compact DualAR text-to-speech model with zero-shot voice cloning.

- 📄 Paper: [arxiv.org/abs/2610.14776](https://arxiv.org/abs/2610.14776)
- 🤗 Model: [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b)

---

## Features

- **Two model sizes**: `0.6b` (Realistic) and `0.1b` (lightweight)
- **15 languages**: English, Chinese, Japanese, Spanish, French, German, Portuguese, Italian, Korean, Arabic, Russian, Dutch, Polish, Turkish, Swedish
- **Zero-shot voice cloning** via optional reference audio + exact transcript
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