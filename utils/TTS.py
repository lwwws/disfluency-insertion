"""TTS wrapper module

Provides a small, dependency-tolerant wrapper around multiple TTS backends:
- Bark (suno/bark)
- Tortoise TTS
- OpenAI TTS (via openai python package)

This module is for creating a backend instance and calling
``synthesize(text, ...)`` which returns (audio_np, sample_rate). You can also
pass ``save_path`` to write output to disk.

Note: TTS model selection can be provided in ``config.yaml`` under a ``tts`` mapping.
This file includes a small ``_load_tts_registry`` helper
that looks for a mapping called ``tts``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Lazy optional deps
_np = None
_soundfile = None
_openai = None

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import soundfile as _soundfile
except Exception:
    _soundfile = None

try:
    import openai as _openai
except Exception:
    _openai = None


def _load_tts_registry(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Load TTS mapping from config.yaml if present.

    Expected config structure (example):

    tts:
      bark: "bark-model-name-or-key"
      tortoise: "pretrained/tortoise-id"
      openai: "gpt-4o-mini-tts"

    Returns a dict mapping keys to model identifiers (may be empty).
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    path = config_path or (Path(__file__).resolve().parent.parent / "config.yaml")
    if not path.exists():
        return {}

    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        tts = parsed.get("tts", {}) if isinstance(parsed, dict) else {}
        return {str(k): str(v) for k, v in tts.items()} if isinstance(tts, dict) else {}
    except Exception:
        return {}


class BaseTTS(ABC):
    """Abstract base class for TTS implementations.

    Implementations must provide ``synthesize(text, **kwargs) -> (np.ndarray, sr)``.
    """

    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_key = model_key
        self.model_id = model_id
        self.device = device or ("cuda" if _np is not None else "cpu")

    @abstractmethod
    def synthesize(self, text: str, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        """Synthesize `text` into (audio_numpy_array, sample_rate).

        The audio array should be a 1D or 2D numpy float32 array in the range
        [-1.0, 1.0]. The sample rate is an integer (commonly 22050 or 24000).
        """

    def save(self, audio: "_np.ndarray", sample_rate: int, path: str) -> None:
        """Save audio numpy array to disk as WAV (or other format supported by soundfile).

        Uses `soundfile` if available; otherwise falls back to the standard
        library ``wave`` module (which expects int16 PCM).
        """
        if _np is None:
            raise RuntimeError("numpy is required to save audio. Install numpy.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if _soundfile is not None:
            # soundfile accepts float32 and writes proper WAV header
            _soundfile.write(str(out_path), audio, samplerate=sample_rate)
            return

        # fallback: convert to int16 PCM and write with wave
        import wave

        pcm = (_np.clip(audio, -1.0, 1.0) * 32767.0).astype("int16")
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())


class BarkTTS(BaseTTS):
    """Wrapper for Bark TTS (https://github.com/suno-ai/bark).

    Note: bark must be installed (pip install git+https://github.com/suno-ai/bark)
    and the repository's model files must be available.
    """

    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)

        # Optional: Preload models during initialization if bark is available
        try:
            import bark
            bark.preload_models()
        except ImportError:
            pass  # Handled during synthesize

    def synthesize(self, text: str, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if _np is None:
            raise ImportError("numpy is required for Bark TTS: pip install numpy")

        try:
            import bark
        except ImportError as exc:
            raise ImportError(
                "Bark is not installed. Install with: pip install git+https://github.com/suno-ai/bark"
            ) from exc

        if not hasattr(bark, "generate_audio"):
            raise RuntimeError(
                "Could not find `generate_audio` in the installed Bark package. Check your installation.")

        # Generate audio array and retrieve the official sample rate
        audio_array = bark.generate_audio(text, **kwargs)
        rate = bark.SAMPLE_RATE

        return audio_array, rate


class TortoiseTTS(BaseTTS):
    """Wrapper for Tortoise TTS.

    Tortoise installation and usage vary. This wrapper uses the
    convenience API `tortoise.api.TextToSpeech`.
    """

    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)

        self.tts = None
        try:
            from tortoise.api import TextToSpeech  # type: ignore
            # Initialize once so we don't reload models on every call
            self.tts = TextToSpeech()
        except ImportError:
            pass  # Handled during synthesize

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if _np is None:
            raise ImportError("numpy is required for Tortoise TTS: pip install numpy")

        if self.tts is None:
            raise ImportError(
                "Tortoise TTS package not available. See https://github.com/neonbjb/tortoise-tts for installation instructions."
            )

        # out is typically a PyTorch tensor, or a tuple where the first item is the tensor
        out = self.tts.tts(text, voice=voice, progress=False, **kwargs)

        if isinstance(out, tuple):
            audio_tensor = out[0]
        else:
            audio_tensor = out

        audio = audio_tensor.squeeze().cpu().numpy()
        return audio, 24000


class OpenAITTS(BaseTTS):
    """Wrapper for OpenAI's TTS capabilities via the `openai` package.

    Requires environment variable OPENAI_API_KEY to be set.
    """

    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)

        self.client = None
        try:
            import openai
            # Client automatically detects OPENAI_API_KEY from environment
            self.client = openai.OpenAI()
        except ImportError:
            pass  # Handled during synthesize

    def synthesize(self, text: str, model: Optional[str] = None, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if self.client is None:
            raise ImportError("openai package is required for OpenAI TTS: pip install openai")
        if _np is None:
            raise ImportError("numpy is required: pip install numpy")

        model_name = model or getattr(self, "model_id", None) or "gpt-4o-mini-tts"

        try:
            # Force response_format to wav so wave module doesn't crash on MP3s
            resp = self.client.audio.speech.create(
                model=model_name,
                input=text,
                response_format="wav",
                **kwargs
            )
            audio_bytes = resp.content
        except Exception as exc:
            raise RuntimeError(f"OpenAI TTS request failed: {exc}") from exc

        import io
        buf = io.BytesIO(audio_bytes)

        # Make sure to explicitly cast to float32
        if _soundfile is not None:
            audio, sr = _soundfile.read(buf, dtype="float32")
            return _np.asarray(audio), int(sr)

        # fallback: write to temporary file and read with wave
        import wave
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with wave.open(tmp_path, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            import struct
            import numpy as _np_local

            pcm = struct.unpack("<{}h".format(wf.getnframes() * wf.getnchannels()), frames)
            audio = _np_local.asarray(pcm, dtype=_np_local.int16).astype(_np_local.float32) / 32767.0
            return audio, sr


def get_tts(tts_key: str, config_path: Optional[str] = None, device: Optional[str] = None) -> BaseTTS:
    """Factory that returns a TTS backend instance.

    - If `tts_key` matches a name in config.yaml's `tts:` mapping, the mapping's
      value will be used as `model_id` for the backend.
    - Otherwise, `tts_key` may be a backend name ("bark", "tortoise", "openai").

    Examples:
      get_tts("bark")  -> BarkTTS
      get_tts("openai") -> OpenAITTS
      get_tts("custom", config_path=...) -> attempt to resolve from config
    """
    registry = _load_tts_registry(Path(config_path) if config_path else None)
    model_id = registry.get(tts_key) or None

    key = tts_key.lower()
    if key in registry:
        model_id = registry[key]

    if "bark" in key:
        return BarkTTS(model_key=tts_key, model_id=model_id, device=device)
    if "tortoise" in key:
        return TortoiseTTS(model_key=tts_key, model_id=model_id, device=device)
    if "openai" in key:
        return OpenAITTS(model_key=tts_key, model_id=model_id, device=device)

    # If a model id suggests a backend, try to smart-match
    if model_id:
        mid = model_id.lower()
        if "bark" in mid:
            return BarkTTS(model_key=tts_key, model_id=model_id, device=device)
        if "tortoise" in mid:
            return TortoiseTTS(model_key=tts_key, model_id=model_id, device=device)
        if "openai" in mid or "gpt" in mid or "tts" in mid:
            return OpenAITTS(model_key=tts_key, model_id=model_id, device=device)

    # Default: raise with guidance
    raise ValueError(
        "Unknown TTS backend key. Use 'bark', 'tortoise' or 'openai', or add a mapping in config.yaml under 'tts:'"
    )