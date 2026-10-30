"""
ComfyUI custom nodes for Audio8 TTS.

Model: Edge0/Audio8-TTS-Realistic-0.6b
Paper: https://arxiv.org/abs/2610.14776
HuggingFace: https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b
"""

import os
import torch
import torchaudio
import folder_paths

# ---------------------------------------------------------------------------
# Model cache — keeps weights resident in VRAM between executions
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def _get_model_dir(model_size: str) -> str:
    """Return the local directory where Audio8 weights are stored."""
    base = (
        folder_paths.get_folder_paths("tts")[0]
        if "tts" in folder_paths.folder_names_and_paths
        else os.path.join(folder_paths.models_dir, "TTS")
    )
    path = os.path.join(base, f"Audio8-TTS-{model_size}")
    os.makedirs(path, exist_ok=True)
    return path


def _load_model(model_size: str, device: torch.device):
    """Load (or return cached) Audio8 model."""
    cache_key = f"{model_size}:{device}"
    if cache_key not in _MODEL_CACHE:
        from huggingface_hub import snapshot_download

        repo_id = f"Edge0/Audio8-TTS-Realistic-{model_size}"
        local_dir = _get_model_dir(model_size)
        print(f"[Audio8-TTS] Downloading {repo_id} → {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)

        # TODO: replace with actual Audio8 model loading once the upstream
        # Python SDK is published. The placeholder below stores the path so
        # inference code can reference it.
        _MODEL_CACHE[cache_key] = {"path": local_dir, "device": device}
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
                    ["en", "zh", "ja", "es", "fr", "de"],
                    {"default": "en"},
                ),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                # ComfyUI AUDIO dict: {"waveform": Tensor[B,C,S], "sample_rate": int}
                "reference_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Audio8-TTS"
    DESCRIPTION = "Text-to-speech synthesis using the Audio8 DualAR model."

    def generate(
        self,
        text: str,
        model_size: str,
        language: str,
        speed: float,
        reference_audio=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_model(model_size, device)

        # ------------------------------------------------------------------
        # Reference audio for zero-shot voice cloning
        # ------------------------------------------------------------------
        ref_waveform = None
        if reference_audio is not None:
            waveform = reference_audio["waveform"]  # [B, C, S]
            sr = reference_audio["sample_rate"]
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
            ref_waveform = waveform.to(device)

        # ------------------------------------------------------------------
        # Inference
        # TODO: replace the stub below with the real Audio8 inference call
        # once the upstream Python SDK is available:
        #
        #   audio_np, out_sr = model["pipeline"].synthesize(
        #       text=text,
        #       language=language,
        #       speed=speed,
        #       voice_prompt=ref_waveform,
        #   )
        #   audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        # ------------------------------------------------------------------
        print(
            f"[Audio8-TTS] Generating speech | model={model_size} lang={language} "
            f"speed={speed} ref={'yes' if ref_waveform is not None else 'no'}"
        )
        out_sr = 44100
        # Placeholder: 2-second silent tensor (remove once real SDK is wired)
        audio_tensor = torch.zeros((1, 1, out_sr * 2), dtype=torch.float32)

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
