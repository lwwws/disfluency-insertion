"""TTS wrapper module

Provides a small, dependency-tolerant wrapper around multiple TTS backends:
- Bark (suno/bark)
- Tortoise TTS
- OpenAI TTS (via openai python package)
- ParlerTTS
- ChatTTS
- CartesiaTTS
- ElevenLabsTTS

This module is for creating a backend instance and calling
``synthesize(text, ...)`` which returns (audio_np, sample_rate). You can also
pass ``save_path`` to write output to disk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Lazy optional deps
_np = None
_soundfile = None

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import soundfile as _soundfile
except Exception:
    _soundfile = None


def _load_tts_registry(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Load TTS mapping from config.yaml if present."""
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

        # Flatten open_source and closed_source sections
        flat_tts = {}
        for key, value in tts.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_tts[str(sub_key)] = str(sub_val)
            else:
                flat_tts[str(key)] = str(value)
        return flat_tts
    except Exception:
        return {}


class BaseTTS(ABC):
    """Abstract base class for TTS implementations.

    Implementations must provide ``synthesize(text, **kwargs) -> (np.ndarray, sr)``.
    """

    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_key = model_key
        self.model_id = model_id
        self.device = device or self._default_device()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @abstractmethod
    def synthesize(self, text: str, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        """Synthesize `text` into (audio_numpy_array, sample_rate).

        The audio array should be a 1D or 2D numpy float32 array in the range
        [-1.0, 1.0]. The sample rate is an integer (commonly 22050 or 24000).
        """

    def save(self, audio: "_np.ndarray", sample_rate: int, path: str) -> None:
        if _np is None:
            raise RuntimeError("numpy is required to save audio. Install numpy.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if _soundfile is not None:
            _soundfile.write(str(out_path), audio, samplerate=sample_rate)
            return

        import wave
        pcm = (_np.clip(audio, -1.0, 1.0) * 32767.0).astype("int16")
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    def synthesize_to_file(self, text: str, path: str, **kwargs: Any) -> str:
        """Synthesize text and save it as an audio file."""
        audio, sample_rate = self.synthesize(text, **kwargs)
        self.save(audio, sample_rate, path)
        return path

    def sample_speaker(self) -> Optional[str]:
        """Return a reusable speaker/voice token when the backend supports it."""
        return None


def prepare_text_for_tts(text: str, backend: str = "chattts") -> str:
    """Convert dataset markers into backend-friendly speech/pause markers.

    The dataset uses <sil> for silent pauses. ChatTTS has explicit pause tokens;
    Bark tends to respond better to ellipses for hesitation-like pauses.
    """
    cleaned = " ".join(str(text).split())
    backend = backend.lower()
    if "chattts" in backend:
        return cleaned.replace("<sil>", "[uv_break]")
    if "bark" in backend:
        return cleaned.replace("<sil>", "...")
    return cleaned.replace("<sil>", "...")


def sanitize_for_tts(text: str, keep_cutoffs: bool = False) -> str:
    """Strip characters some TTS backends choke on, preserving ``<sil>`` markers.

    This is the sanitizer used for the Turing-test stimuli (lifted out of
    ``turing_test/tts_for_turing_test.py`` so both pipelines share one copy).

    ``keep_cutoffs=False`` reproduces that behaviour exactly, INCLUDING the fact
    that it deletes token-final hyphens - which is Switchboard's cut-off marker,
    so ``innova-`` is read aloud as ``innova``. Pass ``keep_cutoffs=True`` to
    preserve speech-error disfluencies; only do so if you are willing to have
    stimuli that differ from the Turing test.
    """
    import re

    if not text:
        return ""
    placeholder = "__SIL__"
    txt = str(text).replace("<sil>", placeholder)
    txt = " ".join(txt.split())
    txt = txt.replace("'", "").replace("’", "")
    if not keep_cutoffs:
        txt = re.sub(r"(\w)-(?=\s|$)", r"\1", txt)
        txt = re.sub(r"(?<=\s)-(?=\w)", "", txt)
    txt = re.sub(r"[\t\r\n]+", " ", txt)
    return txt.replace(placeholder, "<sil>")


def build_tts_text(text: str, backend: str = "chattts", *, keep_cutoffs: bool = False,
                   lbreak: bool = False) -> str:
    """The exact string handed to the backend: prepared, sanitized, optionally padded.

    Split out from ``synthesize_utterance`` so callers can hash what will ACTUALLY
    be spoken. Caching on the upstream text instead silently serves stale audio
    when a flag like ``lbreak`` or ``keep_cutoffs`` changes the rendered string.
    """
    out = sanitize_for_tts(prepare_text_for_tts(text, backend), keep_cutoffs=keep_cutoffs)
    if lbreak and "chattts" in backend.lower():
        out = out.rstrip() + " [lbreak]"
    return out


def synthesize_utterance(tts: "BaseTTS", text: str, path: str, *, backend: str = "chattts",
                         speaker: Optional[str] = None, keep_cutoffs: bool = False,
                         lbreak: bool = False, preserve_tags: bool = True,
                         **synth_kwargs: Any) -> str:
    """Render one utterance to ``path``: prepare -> sanitize -> synthesize.

    The single implementation of the synthesis step, shared by
    ``turing_test/tts_for_turing_test.py`` and ``conv_tts.py`` so stimuli from
    the two studies are produced identically.

    On backend failure it retries with a coarser scrub.

    ``preserve_tags`` controls that retry. ``True`` keeps ``[...]`` control tags,
    so ``[uv_break]`` still renders as a pause. ``False`` reproduces the original
    inline fallback in ``tts_for_turing_test.py`` byte-for-byte - INCLUDING its
    bug, where stripping the brackets makes ChatTTS speak "uv break" aloud. The
    Turing-test script passes ``False`` so its already-collected stimuli stay
    exactly reproducible; new work should leave it ``True``.
    """
    import re

    tts_text = build_tts_text(text, backend, keep_cutoffs=keep_cutoffs, lbreak=lbreak)

    kwargs = dict(synth_kwargs)
    if speaker is not None:
        kwargs["speaker"] = speaker
    try:
        tts.synthesize_to_file(tts_text, str(path), **kwargs)
    except Exception as exc:
        if preserve_tags:
            fallback = " ".join(re.sub(r"[^\w\s<>\[\]\.\,\?\!]+", " ", tts_text).split())
        else:  # verbatim HEAD behaviour: brackets are stripped, no whitespace collapse
            fallback = re.sub(r"[^\w\s<>\.\,\?\!]+", " ", tts_text)
        print(f"  ! TTS failed on {text!r} ({exc}); retrying with {fallback!r}")
        tts.synthesize_to_file(fallback, str(path), **kwargs)
    return str(path)


class BarkTTS(BaseTTS):
    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)
        try:
            import bark
            bark.preload_models()
        except ImportError:
            pass

    def synthesize(self, text: str, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if _np is None:
            raise ImportError("numpy is required for Bark TTS: pip install numpy")
        try:
            import bark
        except ImportError as exc:
            raise ImportError("Bark is not installed.") from exc

        audio_array = bark.generate_audio(text, **kwargs)
        rate = bark.SAMPLE_RATE
        return audio_array, rate


class TortoiseTTS(BaseTTS):
    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)
        self.tts = None
        try:
            from tortoise.api import TextToSpeech
            self.tts = TextToSpeech()
        except ImportError:
            pass

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if _np is None or self.tts is None:
            raise ImportError("numpy and tortoise-tts are required")

        out = self.tts.tts(text, voice=voice, progress=False, **kwargs)
        audio_tensor = out[0] if isinstance(out, tuple) else out
        audio = audio_tensor.squeeze().cpu().numpy()
        return audio, 24000


class OpenAITTS(BaseTTS):
    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)
        self.client = None
        try:
            import openai
            self.client = openai.OpenAI()
        except ImportError:
            pass

    def synthesize(self, text: str, model: Optional[str] = None, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if self.client is None or _np is None:
            raise ImportError("openai and numpy are required")

        model_name = model or self.model_id or "gpt-4o-mini-tts"
        try:
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


class ParlerTTS(BaseTTS):
    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)
        self.model = None
        self.tokenizer = None
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            model_name = self.model_id or "parler-tts/parler_tts_mini_v0.1"
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ImportError:
            pass

    def synthesize(self, text: str, description: Optional[str] = None, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if self.model is None or self.tokenizer is None or _np is None:
            raise ImportError("parler_tts, transformers, and numpy are required")

        # Parler uses a text prompt for speech characteristics
        desc = description or "A speaker delivers their words clearly and expressively."
        input_ids = self.tokenizer(desc, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, **kwargs)
        audio = generation.cpu().numpy().squeeze()
        rate = self.model.config.sampling_rate
        return audio, rate


class ChatTTS(BaseTTS):
    def __init__(self, model_key: str, model_id: Optional[str] = None,
                 device: Optional[str] = None, source: str = "huggingface"):
        super().__init__(model_key, model_id, device)
        self.chat = None
        self._chat_module = None

        try:
            import ChatTTS as _ChatTTS
        except ImportError as exc:
            raise ImportError(
                "ChatTTS is not installed in this environment. Install it with:\n"
                "    pip install ChatTTS"
            ) from exc

        self._chat_module = _ChatTTS
        chat = _ChatTTS.Chat()

        # source="huggingface" resolves weights through the standard HF cache.
        # The library's own default ("local") uses a bundled downloader that
        # writes into ./asset and reports a bare 'no dns available' on some
        # networks even when huggingface.co is perfectly reachable.
        #
        # load() RETURNS False on failure rather than raising; ignoring that
        # leaves a half-built Chat whose first real call dies with a confusing
        # AttributeError, so check it here.
        if not chat.load(source=source, compile=False):
            raise RuntimeError(
                f"ChatTTS.load(source={source!r}) failed - model weights are missing "
                f"or could not be fetched.\n"
                f"The weights normally live in the HF cache under "
                f"models--2Noise--ChatTTS. Try source='local' to download into "
                f"./asset instead, or pre-fetch with:\n"
                f"    huggingface-cli download 2Noise/ChatTTS"
            )
        self.chat = chat

    def sample_speaker(self) -> Optional[str]:
        if self.chat is None:
            raise ImportError("ChatTTS is required to sample a speaker")
        return self.chat.sample_random_speaker()

    def sample_audio_speaker(self, audio: "_np.ndarray") -> str:
        """Encode a rendered waveform into a ``spk_smp`` reference token.

        Pass the result back as ``spk_smp`` (with the matching text as
        ``txt_smp``) to keep a voice stable across separate synthesis calls.
        """
        if self.chat is None:
            raise ImportError("ChatTTS is required to sample an audio speaker")
        return self.chat.sample_audio_speaker(audio)

    def synthesize(self, text: str, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if self.chat is None or _np is None:
            raise ImportError("ChatTTS and numpy are required")

        speaker = kwargs.pop("speaker", None) or kwargs.pop("spk_emb", None)

        spk_smp = kwargs.pop("spk_smp", None)
        txt_smp = kwargs.pop("txt_smp", None)
        manual_seed = kwargs.pop("manual_seed", None)
        infer_temperature = kwargs.pop("infer_temperature", None)
        if infer_temperature is None:
            infer_temperature = 0.1

        # 1. Lower the temperature of the acoustic model to reduce hallucinations
        if "params_infer_code" not in kwargs:
            kwargs["params_infer_code"] = self._chat_module.Chat.InferCodeParams(
                spk_emb=speaker,
                spk_smp=spk_smp,
                txt_smp=txt_smp,
                manual_seed=manual_seed,
                temperature=infer_temperature,
            )

        # 2. Explicitly zero out the control tags in the refinement stage
        kwargs["skip_refine_text"] = False
        if "params_refine_text" not in kwargs:
            kwargs["params_refine_text"] = self._chat_module.Chat.RefineTextParams(
                prompt="[oral_0][laugh_0][break_0]"
            )

        # 3. Disable normalization so it doesn't alter your exact script
        kwargs.setdefault("do_text_normalization", False)
        kwargs.setdefault("do_homophone_replacement", False)

        text = prepare_text_for_tts(text, "chattts")
        wavs = self.chat.infer([text], **kwargs)
        return wavs[0], 24000


class CartesiaTTS(BaseTTS):
    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)
        self.client = None
        try:
            from cartesia import Cartesia
            self.client = Cartesia() # Automatically detects CARTESIA_API_KEY
        except ImportError:
            pass

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if self.client is None or _np is None:
            raise ImportError("cartesia package and numpy are required")

        model_name = self.model_id or "sonic-3.5"
        voice_id = voice or "e07c00bc-4134-4eae-9ea4-1a55fb45746b"

        import tempfile
        try:
            response = self.client.tts.generate(
                model_id=model_name,
                output_format={"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
                transcript=text,
                voice={"mode": "id", "id": voice_id},
                **kwargs
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                response.write_to_file(tmp_path)

            if _soundfile is not None:
                audio, sr = _soundfile.read(tmp_path, dtype="float32")
                return _np.asarray(audio), int(sr)

            # Fallback
            import wave
            with wave.open(tmp_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                import struct
                import numpy as _np_local
                pcm = struct.unpack("<{}f".format(wf.getnframes() * wf.getnchannels()), frames)
                audio = _np_local.asarray(pcm, dtype=_np_local.float32)
                return audio, sr

        except Exception as exc:
            raise RuntimeError(f"Cartesia TTS failed: {exc}") from exc


class ElevenLabsTTS(BaseTTS):
    def __init__(self, model_key: str, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_key, model_id, device)
        self.client = None
        try:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs() # Automatically detects ELEVENLABS_API_KEY
        except ImportError:
            pass

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs: Any) -> Tuple["_np.ndarray", int]:
        if self.client is None or _np is None:
            raise ImportError("elevenlabs package and numpy are required")

        model_name = self.model_id or "eleven_multilingual_v2"
        voice_id = voice or "JBFqnCBsd6RMkjVDRZzb"

        try:
            audio_iterator = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_name,
                output_format=kwargs.pop("output_format", "mp3_44100_128"),
                **kwargs
            )
            audio_bytes = b"".join(audio_iterator)

            import io
            if _soundfile is not None:
                audio, sr = _soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
                return _np.asarray(audio), int(sr)

            raise ImportError("soundfile is required to decode ElevenLabs MP3 output")
        except Exception as exc:
            raise RuntimeError(f"ElevenLabs TTS failed: {exc}") from exc


def get_tts(tts_key: str, config_path: Optional[str] = None, device: Optional[str] = None) -> BaseTTS:
    """Factory that returns a TTS backend instance."""
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
    if "parler" in key:
        return ParlerTTS(model_key=tts_key, model_id=model_id, device=device)
    if "chattts" in key:
        return ChatTTS(model_key=tts_key, model_id=model_id, device=device)
    if "cartesia" in key:
        return CartesiaTTS(model_key=tts_key, model_id=model_id, device=device)
    if "elevenlabs" in key or "eleven" in key:
        return ElevenLabsTTS(model_key=tts_key, model_id=model_id, device=device)

    if model_id:
        mid = model_id.lower()
        if "bark" in mid:
            return BarkTTS(model_key=tts_key, model_id=model_id, device=device)
        if "tortoise" in mid:
            return TortoiseTTS(model_key=tts_key, model_id=model_id, device=device)
        if "openai" in mid or "gpt" in mid or "tts" in mid:
            return OpenAITTS(model_key=tts_key, model_id=model_id, device=device)
        if "parler" in mid:
            return ParlerTTS(model_key=tts_key, model_id=model_id, device=device)
        if "chattts" in mid:
            return ChatTTS(model_key=tts_key, model_id=model_id, device=device)
        if "cartesia" in mid or "sonic" in mid:
            return CartesiaTTS(model_key=tts_key, model_id=model_id, device=device)
        if "elevenlabs" in mid or "eleven" in mid:
            return ElevenLabsTTS(model_key=tts_key, model_id=model_id, device=device)

    raise ValueError("Unknown TTS backend key. Check config.yaml.")
