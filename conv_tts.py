"""Synthesize the study conversations as full dialogue clips.

Reads what ``conv_predict.py`` wrote and renders one WAV per conversation per
condition, with a distinct voice per speaker and a configurable pause between
turns.

A condition is either ``fluent`` (every turn clean) or ``MODEL:WHO`` where WHO
is ``all``, or a speaker label such as ``A``/``B`` - so ``L3.1-8B-Instruct:B``
gives disfluencies on B's turns only while A stays fluent. That is the knob for
the "should both speakers be disfluent?" question: predictions are generated for
every turn, and the condition decides which ones actually get voiced.

Per-turn audio is cached under ``turns/``, so a fluent turn is synthesized once
and reused by every condition that needs it, and reruns are cheap.

Usage:
    python conv_tts.py --conditions fluent L3.1-8B-Instruct:B L3.1-8B-Instruct:all
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # soundfile arrives with most TTS installs, but stdlib wave is enough here
    import soundfile as _sf
except Exception:
    _sf = None


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a WAV as mono float32 in [-1, 1]."""
    if _sf is not None:
        audio, sr = _sf.read(str(path), dtype="float32", always_2d=False)
    else:
        import wave

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_ch, width = wf.getnchannels(), wf.getsampwidth()
            if width != 2:
                raise RuntimeError(f"{path}: expected 16-bit PCM, got {width * 8}-bit. "
                                   f"Install soundfile (pip install soundfile).")
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype="float32"), int(sr)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _sf is not None:
        _sf.write(str(path), audio, sample_rate)
        return
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def voice_seed_for(base: int, speaker: str) -> int:
    """Per-speaker RNG seed, derived from the label rather than its position.

    Order-independent on purpose: adding a third speaker must not change the
    voices A and B already had. Uses sha1, not the builtin ``hash``, which is
    salted per process and would give different voices on every run.
    """
    h = int(hashlib.sha1(speaker.encode("utf-8")).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


def level_audio(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """Scale ``audio`` to a common RMS, backing off if that would clip.

    ChatTTS picks its own level on every call. That is invisible when each clip
    is heard alone (the Turing test) but jumps out when ten turns are joined into
    one conversation, so level them before concatenating.
    """
    rms = float(np.sqrt(np.mean(np.square(audio, dtype="float64")))) if audio.size else 0.0
    if rms < 1e-6:
        return audio
    gain = target_rms / rms
    peak = float(np.abs(audio).max()) * gain
    if peak > 0.99:
        gain *= 0.99 / peak
    return (audio * gain).astype("float32")


def parse_condition(spec: str) -> Tuple[str, Optional[str], Optional[str]]:
    """``'fluent'`` or ``'MODEL:WHO'`` -> (label, model_key, who)."""
    if spec.lower() == "fluent":
        return "fluent", None, None
    parts = spec.split(":")
    model_key = parts[0]
    who = (parts[1] if len(parts) > 1 else "all").upper()
    who = "all" if who == "ALL" else who
    return f"{model_key}__{who}".replace("/", "_"), model_key, who


def load_rows(pred_dir: Path, clean_path: Path) -> Tuple[Dict, Dict]:
    """Return (clean_by_key, preds_by_model) keyed by ``(conv, turn)``."""
    clean: Dict[Tuple[str, int], Dict] = {}
    if not clean_path.exists():
        raise SystemExit(f"Missing {clean_path}. Run conv_predict.py first.")
    for line in clean_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            clean[(r["conv"], int(r["turn"]))] = r

    preds: Dict[str, Dict[Tuple[str, int], Dict]] = {}
    for path in sorted(pred_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            preds.setdefault(r["model_key"], {})[(r["conv"], int(r["turn"]))] = r
    return clean, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="e.g. fluent L3.1-8B-Instruct:B G4-4EB-it:all")
    ap.add_argument("--input-dir", default="outputs/conversations")
    ap.add_argument("--output-dir", default="outputs/conversations/audio")
    ap.add_argument("--backend", default="chattts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--turn-gap", type=float, default=0.45,
                    help="Seconds of silence inserted between turns")
    ap.add_argument("--keep-cutoffs", action="store_true",
                    help="Preserve token-final hyphens (cut-off disfluencies). OFF by default")
    ap.add_argument("--lbreak", action=argparse.BooleanOptionalAction, default=True,
                    help="Append ChatTTS's [lbreak]. ON by default because the text "
                         "has been stripped of punctuation, leaving "
                         "ChatTTS no end-of-sentence cue, so it clips the final syllable "
                    )
    ap.add_argument("--no-stitch", action="store_true",
                    help="Only write per-turn WAVs, skip the joined conversation clip.")
    ap.add_argument("--voice-seed", type=int, default=None,
                    help="Make voice selection reproducible. The same seed always yields "
                         "the same voice for a given speaker label. Omit for an "
                         "unseeded random draw (the previous behaviour).")
    ap.add_argument("--normalize-turns", dest="normalize_turns",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Level each turn to a common loudness when joining the clip. "
                         "ChatTTS's output amplitude varies per call, which is inaudible "
                         "for standalone clips but obvious once turns are concatenated. "
                         "Applied at stitch time only, so toggling costs no synthesis.")
    ap.add_argument("--resample-voices", action="store_true",
                    help="Ignore _speakers.json and draw fresh voices. Combine with "
                         "--voice-seed to hunt for a pair you like.")
    ap.add_argument("--force", action="store_true", help="Re-synthesize cached turn audio")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    clean, preds = load_rows(in_dir / "predictions", in_dir / "clean_conversations.jsonl")

    conditions = [parse_condition(c) for c in args.conditions]
    for label, model_key, _ in conditions:
        if model_key and model_key not in preds:
            raise SystemExit(f"No predictions for {model_key}. Have: {sorted(preds)}")

    out_root = Path(args.output_dir) / args.backend
    turns_dir = out_root / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    from utils.TTS import build_tts_text, get_tts, synthesize_utterance

    tts = get_tts(args.backend, device=args.device)

    # One voice per speaker label, persisted so every condition, model and rerun
    # uses the same two voices - otherwise voice becomes a confound.
    speakers_path = out_root / "_speakers.json"
    voices: Dict[str, str] = {}
    prev_seed = None
    if speakers_path.exists() and not args.resample_voices:
        stored = json.loads(speakers_path.read_text(encoding="utf-8"))
        # {"voice_seed": N, "voices": {...}}; tolerate the older flat {label: emb}
        if isinstance(stored.get("voices"), dict):
            voices, prev_seed = stored["voices"], stored.get("voice_seed")
        else:
            voices = stored

    drawn: List[str] = []
    for speaker in sorted({r["speaker"] for r in clean.values()}):
        if speaker in voices:
            continue
        if args.voice_seed is not None:
            import torch  # ChatTTS samples speakers via the global torch RNG
            torch.manual_seed(voice_seed_for(args.voice_seed, speaker))
        voices[speaker] = tts.sample_speaker()
        drawn.append(speaker)

    # NOTE: do NOT try to stabilise the voice with spk_smp/txt_smp. ChatTTS decodes
    # spk_smp as an AUDIO PROMPT and continues from it, and only ever uses it where
    # the reference is the first chunk of the same utterance. Feeding a fixed
    # unrelated sentence makes every turn resume from that clip's acoustic state,
    # which comes out raspy and wrong. spk_emb alone is the supported way.
    seed_used = args.voice_seed if args.voice_seed is not None else prev_seed
    speakers_path.write_text(json.dumps(
        {"voice_seed": seed_used, "voices": voices}, indent=2), encoding="utf-8")
    reused = sorted(set(voices) - set(drawn))
    print(f"Voices -> {speakers_path}")
    if drawn:
        print(f"  drawn : {', '.join(sorted(drawn))}"
              + (f" (seed {args.voice_seed})" if args.voice_seed is not None else " (unseeded)"))
    if reused:
        print(f"  reused: {', '.join(reused)}")

    def synth_turn(conv: str, turn: int, variant: str, speaker: str, text: str) -> Path:
        """Synthesize one turn (cached). ``variant`` is 'fluent' or the model key.

        The filename carries a hash of the exact text AND the speaker embedding,
        so hand-editing a prediction or changing a voice produces a new cache key
        and re-synthesizes, while untouched turns are still reused.
        """
        # Hash the string that will actually be SPOKEN (so --lbreak / --keep-cutoffs
        # invalidate too), plus the voice.
        spoken = build_tts_text(text, args.backend, keep_cutoffs=args.keep_cutoffs,
                                lbreak=args.lbreak)
        key = f"{spoken}\x00{voices.get(speaker, '')}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        wav = turns_dir / conv / f"t{turn:02d}_{variant.replace('/', '_')}_{digest}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        if wav.exists() and not args.force:
            return wav
        synthesize_utterance(tts, text, wav, backend=args.backend,
                             speaker=voices.get(speaker),
                             keep_cutoffs=args.keep_cutoffs, lbreak=args.lbreak)
        return wav

    convs = sorted({conv for conv, _ in clean})
    manifest: List[Dict] = []

    for label, model_key, who in conditions:
        clip_dir = out_root / "clips" / label
        clip_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- condition {label} ---")

        for conv in convs:
            turn_ids = sorted(t for c, t in clean if c == conv)
            segments: List[np.ndarray] = []
            sample_rate: Optional[int] = None

            for turn in turn_ids:
                base = clean[(conv, turn)]
                speaker = base["speaker"]
                disfluent = model_key is not None and (who == "all" or who == speaker)

                if disfluent:
                    row = preds[model_key].get((conv, turn))
                    if row is None:
                        raise SystemExit(f"No {model_key} prediction for {conv} turn {turn}")
                    text, variant = row["prediction"], model_key
                else:
                    text, variant = base["fluent"], "fluent"

                wav = synth_turn(conv, turn, variant, speaker, text)
                audio, sr = read_wav(wav)
                sample_rate = sample_rate or sr
                if sr != sample_rate:
                    raise SystemExit(f"Sample-rate mismatch in {wav}: {sr} vs {sample_rate}")

                if segments:
                    segments.append(np.zeros(int(args.turn_gap * sample_rate), dtype="float32"))
                segments.append(level_audio(audio) if args.normalize_turns else audio)

                manifest.append({
                    "condition": label, "conv": conv, "turn": turn, "speaker": speaker,
                    "disfluent": disfluent, "variant": variant,
                    "model_key": model_key, "text": text,
                    "turn_wav": str(wav), "clip_wav": str(clip_dir / f"{conv}.wav"),
                    "tts_backend": args.backend,
                })

            if args.no_stitch:
                print(f"  {conv}: {len(turn_ids)} turn WAVs (not stitched)")
                continue
            clip = np.concatenate(segments)
            clip_path = clip_dir / f"{conv}.wav"
            write_wav(clip_path, clip, sample_rate)
            print(f"  {conv}: {len(turn_ids)} turns, "
                  f"{len(clip) / sample_rate:.1f}s -> {clip_path}")

    # Merge with any existing manifest, keyed by (condition, conv, turn), so a run
    # covering only some conditions refreshes those and leaves the rest intact -
    # their clips are still on disk, and this CSV is what the questionnaire is
    # built from. Rows from this run win.
    manifest_path = out_root / "manifest.csv"
    df = pd.DataFrame(manifest)
    if manifest_path.exists() and not df.empty:
        try:
            df = (pd.concat([pd.read_csv(manifest_path), df], ignore_index=True)
                    .drop_duplicates(subset=["condition", "conv", "turn"], keep="last"))
        except Exception as exc:
            print(f"  ! could not merge existing manifest ({exc}); writing this run only")
    df = df.sort_values(["condition", "conv", "turn"])
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(manifest)} rows this run; {len(df)} total -> {manifest_path}")
    print(f"conditions in manifest: {', '.join(sorted(df['condition'].unique()))}")


if __name__ == "__main__":
    main()
