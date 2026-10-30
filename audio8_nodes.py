"""
ComfyUI custom nodes for Audio8 TTS.

Model: Edge0/Audio8-TTS-Realistic-0.6b
Paper: https://arxiv.org/abs/2610.14776
HuggingFace: https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b
"""

import os
import tempfile
import torch
import torchaudio
import folder_paths

# ---------------------------------------------------------------------------
# Model cache — keeps weights resident in VRAM between executions
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def _get_model_dir(model_size: str) -> str:
    """Return the local directory where Audio8 weights are stored."""
    try:
        base = folder_paths.get_folder_paths("tts")[0]
    except Exception:
        base = os.path.join(folder_paths.models_dir, "TTS")
    path = os.path.join(base, f"Audio8-TTS-{model_size}")
    os.makedirs(path, exist_ok=True)
    return path


def _load_model(model_size: str, device: torch.device):
    """Load (or return cached) Audio8 processor + model via transformers."""
    cache_key = f"{model_size}:{device}"
    if cache_key not in _MODEL_CACHE:
        from transformers import AutoModel, AutoProcessor

        repo_id = f"Edge0/Audio8-TTS-Realistic-{model_size}"
        local_dir = _get_model_dir(model_size)

        print(f"[Audio8-TTS] Loading {repo_id} (this may take a while first time)…")
        dtype = torch.bfloat16 if str(device) == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(
            repo_id,
            cache_dir=local_dir,
            trust_remote_code=True,
        )
        model = (
            AutoModel.from_pretrained(
                repo_id,
                cache_dir=local_dir,
                trust_remote_code=True,
                dtype=dtype,
            )
            .eval()
            .to(device)
        )

        _MODEL_CACHE[cache_key] = {"processor": processor, "model": model, "device": device}
        print(f"[Audio8-TTS] Model ready on {device}")

    return _MODEL_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Node: Audio8 TTS Generator
# ---------------------------------------------------------------------------
class Audio8TTSNode:
    """Generate speech from text using the Audio8 TTS model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"multiline": True, "default": "Hello from Audio8 TTS in ComfyUI!"},
                ),
                "model_size": (["0.6b", "0.1b"], {"default": "0.6b"}),
                "language": (
                    ["en", "zh", "ja", "es", "fr", "de", "pt", "it", "ko", "ar", "ru", "nl", "pl", "tr", "sv"],
                    {"default": "en"},
                ),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 64},
                ),
            },
            "optional": {
                # ComfyUI AUDIO dict: {"waveform": Tensor[B,C,S], "sample_rate": int}
                "reference_audio": ("AUDIO",),
                # Exact transcript of the reference_audio clip (required for voice cloning)
                "reference_text": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Audio8-TTS"
    DESCRIPTION = (
        "Text-to-speech synthesis using the Audio8 DualAR model. "
        "Connect reference_audio + reference_text for zero-shot voice cloning."
    )

    def generate(
        self,
        text: str,
        model_size: str,
        language: str,
        speed: float,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        reference_audio=None,
        reference_text: str = "",
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache = _load_model(model_size, device)
        processor = cache["processor"]
        model = cache["model"]

        # ------------------------------------------------------------------
        # Build processor inputs
        # ------------------------------------------------------------------
        use_cloning = reference_audio is not None and bool(reference_text.strip())

        if use_cloning:
            waveform = reference_audio["waveform"]   # [B, C, S]
            sr = int(reference_audio["sample_rate"])

            # Downmix to 1D mono tensor [S]
            if waveform.ndim == 3:
                wav_mono = waveform[0].mean(dim=0).cpu().float()
            elif waveform.ndim == 2:
                wav_mono = waveform.mean(dim=0).cpu().float()
            else:
                wav_mono = waveform.view(-1).cpu().float()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            torchaudio.save(tmp_path, wav_mono.unsqueeze(0), sr)

            clean_ref_text = " ".join(reference_text.strip().split())
            print(
                f"[Audio8-TTS] Zero-shot cloning | sr={sr} | ref_text='{clean_ref_text[:60]}…'"
            )
            inputs = processor(
                text=[text.strip()],
                reference_audio=[tmp_path],
                reference_text=[clean_ref_text],
                return_tensors="pt",
            )
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        else:
            if reference_audio is not None and not reference_text.strip():
                print(
                    "[Audio8-TTS] WARNING: reference_audio provided but reference_text is empty — "
                    "running without voice cloning. Provide exact transcript for cloning."
                )
            print("[Audio8-TTS] Generating speech (standard TTS, no reference)")
            inputs = processor(
                text=[text.strip()],
                return_tensors="pt",
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ------------------------------------------------------------------
        # Generate Audio
        # ------------------------------------------------------------------
        print(f"[Audio8-TTS] Synthesizing… temp={temperature} top_p={top_p} rep_pen={repetition_penalty} max_tokens={max_new_tokens}")
        with torch.inference_mode():
            if hasattr(model, "generate_audio"):
                waveforms, lengths, codes = model.generate_audio(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                )
            else:
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    return_dict_in_generate=True,
                )
                codes = getattr(output, "codes", None)
                if codes is None:
                    codes = getattr(output, "sequences", None)
                waveforms, lengths = model.decode_audio(codes)

        valid_len = int(lengths[0]) if lengths is not None and lengths.numel() > 0 else waveforms.shape[-1]
        audio_1d = waveforms[0, :valid_len].float().cpu()

        # Clamp to [-1, 1] to avoid clipping
        audio_1d = torch.clamp(audio_1d, -1.0, 1.0)

        out_sr = int(getattr(model.config, "codec_sample_rate", 44100))
        # ComfyUI AUDIO shape: [B, C, S] -> [1, 1, S]
        audio_tensor = audio_1d.unsqueeze(0).unsqueeze(0)

        max_amp = float(audio_1d.abs().max()) if audio_1d.numel() > 0 else 0.0
        dur_sec = audio_tensor.shape[-1] / out_sr
        print(
            f"[Audio8-TTS] Done — {dur_sec:.2f}s @ {out_sr} Hz "
            f"| shape={list(audio_tensor.shape)} | max_amp={max_amp:.4f}"
        )
        return ({"waveform": audio_tensor, "sample_rate": out_sr},)


# ---------------------------------------------------------------------------
# Node: Audio8 Model Loader (explicit pre-load)
# ---------------------------------------------------------------------------
class Audio8ModelLoaderNode:
    """Pre-load (and cache) the Audio8 model weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (["0.6b", "0.1b"], {"default": "0.6b"}),
            }
        }

    RETURN_TYPES = ("AUDIO8_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Audio8-TTS"
    DESCRIPTION = "Download and cache Audio8 weights. Connect to Audio8TTSNode."

    def load(self, model_size: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_model(model_size, device)
        return (model,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Audio8TTSNode": Audio8TTSNode,
    "Audio8ModelLoaderNode": Audio8ModelLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio8TTSNode": "Audio8 Text-to-Speech",
    "Audio8ModelLoaderNode": "Audio8 Model Loader",
}
