"""Synthesize model outputs (and reference disfluent text) for Turing test listening.

Select N random examples from the canonical test split (outputs/llm_runs_enriched/splits/test.jsonl)
and for each sample synthesize:
 - the reference disfluent text (as the "original" when raw audio is not available)
 - each requested model's prediction (reads predictions JSONL under outputs/* and matches model_key)

Outputs are written to outputs/tts_for_turing_test/<sample_idx>/ with a small manifest.csv.

Usage example:
  python tts_for_turing_test.py --models L3.1-8B-Instruct Llama-2-13B --num-samples 10

If you DO have original disfluent audio files available locally, provide their root using
--original-audio-root and ensure test records include metadata to map into that dir.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Avoid importing utils.TTS at module import time to prevent heavy HF/torch model loads.
# The factory (get_tts) will be imported lazily inside main().

# Utility: sanitize text for TTS backends (remove problematic characters while preserving <sil>)
import re

def sanitize_for_tts(text: str) -> str:
    """Sanitize text before sending to TTS backends.
    Preserves <sil> tokens but removes stray apostrophes and internal hyphens that some backends reject.
    """
    if not text:
        return ""
    # preserve <sil> markers by temporarily replacing them
    placeholder = "__SIL__"
    txt = text.replace("<sil>", placeholder)
    # normalize whitespace
    txt = " ".join(txt.split())
    # remove straight and curly apostrophes
    txt = txt.replace("'", "").replace("’", "")
    # remove hyphens that occur at token edges (e.g., st- or -member)
    txt = re.sub(r"(\w)-(?=\s|$)", r"\1", txt)
    txt = re.sub(r"(?<=\s)-(?=\w)", "", txt)
    # collapse repeated punctuation that some backends dislike
    txt = re.sub(r"[\t\r\n]+", " ", txt)
    # restore <sil>
    txt = txt.replace(placeholder, "<sil>")
    return txt

# Avoid importing evaluate.py to prevent heavy HF/torch model loads and downloads.
# Load the canonical test split JSONL directly when needed.

# Support for decoding/temperature variants: maps human label -> filename suffix
DECODINGS_MAP = {"greedy": "", "T=0.5": "_sampled_t05", "T=1.0": "_sampled", "T=1.5": "_sampled_t15"}


def find_prediction_files(roots: List[Path], suffixes: List[str]) -> List[Path]:
    """Find prediction files for the given suffixes (e.g., greedy or sampled variants).

    suffixes: list of filename suffix strings like '', '_sampled', '_sampled_t05'
    """
    out = []
    for root in roots:
        if not root.exists():
            continue
        for suf in suffixes:
            pattern = f"test_predictions{suf}.jsonl"
            for p in root.rglob(pattern):
                out.append(p)
    return out


def parse_model_spec(spec: str) -> Dict:
    """Parse a model spec string into a dict.

    Syntax (tokens separated by ':'):
      MODEL_KEY
      MODEL_KEY:with_context
      MODEL_KEY:no_context
      MODEL_KEY:any          # accept either context mode
      MODEL_KEY:prompt       # select prompt-engineered baseline
      MODEL_KEY:shot=NAME    # when prompt is used, filter by shot name
      MODEL_KEY:shot_k=3     # few-shot k
      MODEL_KEY:T=0.5        # decoding/temperature label (also 'T=1.0', 'T=1.5')
      MODEL_KEY:decoding=T=0.5
      MODEL_KEY:temp=0.5

    Examples:
      "L3.1-8B-Instruct:with_context"
      "L2-7B-Chat:prompt:shot=few:context=no_context"
      "L3.1-8B-Instruct:T=0.5:with_context"
    """
    parts = [p.strip() for p in spec.split(":") if p.strip()]
    out = {"model_key": parts[0], "prompt": False, "shot": None, "shot_k": None, "context_mode": None, "decoding": None}
    for tok in parts[1:]:
        if tok == "prompt":
            out["prompt"] = True
        elif tok in ("with_context", "no_context", "any"):
            out["context_mode"] = tok
        elif tok.startswith("context="):
            out["context_mode"] = tok.split("=", 1)[1]
        elif tok.startswith("shot="):
            out["shot"] = tok.split("=", 1)[1]
        elif tok.startswith("shot_k="):
            try:
                out["shot_k"] = int(tok.split("=", 1)[1])
            except Exception:
                out["shot_k"] = None
        elif tok.startswith("decoding="):
            out["decoding"] = tok.split("=", 1)[1]
        elif tok.startswith("temp="):
            # normalize to T=... form
            out["decoding"] = "T=" + tok.split("=", 1)[1]
        elif tok in DECODINGS_MAP or tok.startswith("T="):
            out["decoding"] = tok
        else:
            # allow shorthand like 'few' to set shot
            out["shot"] = tok
    return out


def _make_label(spec: Dict) -> str:
    parts = [spec["model_key"].replace("/", "_")]
    if spec.get("prompt"):
        parts.append("prompt")
    if spec.get("shot"):
        parts.append(f"shot={spec['shot']}")
    if spec.get("shot_k") is not None:
        parts.append(f"shot_k={spec['shot_k']}")
    if spec.get("context_mode"):
        parts.append(f"context={spec['context_mode']}")
    # include decoding label (temperature) if present so variants are distinct
    if spec.get("decoding"):
        parts.append(f"decoding={spec['decoding']}")
    return "__".join(parts).replace(" ", "_")


def load_predictions_map(pred_paths: List[Path], specs: List[Dict]) -> Dict[str, Dict]:
    """Return mapping: label -> {by_index: {idx: row}, by_id: {id: row}, spec: spec}
    Only rows that match any spec are recorded. Later files override earlier ones.
    """
    mapping: Dict[str, Dict] = {}
    # prepare labels
    labels = {_make_label(s): s for s in specs}
    for label, spec in labels.items():
        mapping[label] = {"by_index": {}, "by_id": {}, "spec": spec}

    for p in pred_paths:
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    mk = r.get("model_key")
                    cm = r.get("context_mode")
                    # try id and source_index
                    sid = r.get("id")
                    sidx = r.get("source_index")
                    for label, entry in mapping.items():
                        spec = entry["spec"]
                        if mk != spec["model_key"]:
                            continue
                        # context mode check
                        if spec.get("context_mode") and spec["context_mode"] != "any":
                            if cm != spec["context_mode"]:
                                continue
                        # prompt vs fine-tuned
                        has_shot = r.get("shot") is not None or r.get("prompt_engineering") is not None
                        if spec.get("prompt") and not has_shot:
                            continue
                        if not spec.get("prompt") and has_shot:
                            # user asked for fine-tuned adapter but this is a prompt row
                            continue
                        # shot filter
                        if spec.get("shot") and r.get("shot") != spec.get("shot"):
                            continue
                        # passed all filters -> record
                        if sidx is not None:
                            try:
                                entry["by_index"][int(sidx)] = r
                            except Exception:
                                pass
                        if sid is not None:
                            entry["by_id"][str(sid)] = r
        except Exception:
            continue
    return mapping


def select_examples(test_records: List[Dict], n: int, seed: int = 42):
    candidates = [r for r in test_records if r.get("disfluent") and r.get("n_tokens_fluent", 0) >= 8 and r.get("n_tokens_fluent", 0) <= 25]
    if not candidates:
        raise RuntimeError("No suitable test records found in the split. Run the dataset builder / ensure splits exist.")
    random.Random(seed).shuffle(candidates)
    return candidates[:min(n, len(candidates))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="Legacy: model keys to include (e.g. L3.1-8B-Instruct). If --model-specs is provided it takes precedence.")
    ap.add_argument("--model-specs", nargs="+",
                    help=("Model specifications with optional qualifiers. Syntax: MODEL[:prompt][:shot=NAME][:shot_k=K][:with_context|no_context|any]. "
                          "Examples: 'L3.1-8B-Instruct:with_context', 'L2-7B-Chat:prompt:shot=few:no_context'"))
    ap.add_argument("--decodings", nargs="+", default=["greedy"],
                    help=("Decoding variants to include. Use labels: 'greedy' (default), 'T=0.5', 'T=1.0', 'T=1.5'. "
                          "Prompt-engineered runs typically have only greedy outputs."))
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--output-dir", default="outputs/tts_for_turing_test")
    ap.add_argument("--results-roots", nargs="+", default=["outputs/llm_runs_enriched", "outputs/prompt_eng"],
                    help="Root dirs to search for test_predictions.jsonl (and their sampled variants)")
    ap.add_argument("--context-mode", choices=["with_context", "no_context", "any"], default="with_context",
                    help="Default context mode for legacy --models list (ignored when --model-specs is used).")
    ap.add_argument("--tts-backend", default="chattts", help="TTS backend key from config.yaml")
    ap.add_argument("--speaker-mode", choices=("same_all", "same_pair", "backend_default"), default="same_all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--original-audio-root", default=None,
                    help="Optional: root dir containing original disfluent audio files. If provided, script will try to locate originals; otherwise it will synthesize the reference text.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing audio files and update manifest entries")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical test split (from llm_runs_enriched by default) without importing evaluate
    splits_dir = Path("outputs/llm_runs_enriched") / "splits"
    test_path = splits_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}. Run llm_training.py to create it.")
    test_records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))
    # Assign source_index and id (predictions may refer to either)
    for idx, r in enumerate(test_records):
        if "source_index" not in r:
            r["source_index"] = idx
        if "id" not in r:
            r["id"] = idx

    picked = select_examples(test_records, args.num_samples, seed=args.seed)

    # Build model specs (prefer --model-specs, fallback to legacy --models)
    specs_raw: List[str] = args.model_specs if args.model_specs else (args.models or [])
    if not specs_raw:
        raise SystemExit("No models specified. Use --model-specs or --models.")
    specs = [parse_model_spec(s) if ":" in s or "prompt" in s or "shot" in s else {"model_key": s, "prompt": False, "shot": None, "shot_k": None, "context_mode": args.context_mode} for s in specs_raw]

    # Find prediction files for requested decoding variants and expand specs per decoding
    roots = [Path(r) for r in args.results_roots]
    # Map decoding labels to filename suffixes
    requested_decodings = args.decodings
    suffixes = []
    for d in requested_decodings:
        if d in DECODINGS_MAP:
            suffixes.append(DECODINGS_MAP[d])
        else:
            # allow users to pass raw suffix strings too
            suffixes.append(d)
    # Expand specs per decoding so each decoding variant becomes a distinct label.
    # If a spec already contains a decoding (e.g., '...:T=0.5'), honor that and do not expand it.
    expanded_specs: List[Dict] = []
    for s in specs:
        if s.get("decoding"):
            # spec already names its decoding; include as-is
            expanded_specs.append(dict(s))
        else:
            for d in requested_decodings:
                s2 = dict(s)
                s2["decoding"] = d
                expanded_specs.append(s2)

    pred_files = find_prediction_files(roots, suffixes)
    preds_map = load_predictions_map(pred_files, expanded_specs)

    # TTS setup (lazy import to avoid pulling heavy ML deps when not needed)
    from utils.TTS import get_tts, prepare_text_for_tts
    tts = get_tts(args.tts_backend, device="cpu")

    # Speaker selection and persistence
    saved_global_speaker = out_dir / "_saved_speaker.json"
    shared_speaker = None
    if args.speaker_mode == "same_all":
        # Reuse a global speaker across runs; persist to outputs dir
        if saved_global_speaker.exists():
            try:
                js = json.loads(saved_global_speaker.read_text(encoding="utf-8"))
                shared_speaker = js.get("speaker")
            except Exception:
                shared_speaker = None
        if shared_speaker is None:
            shared_speaker = tts.sample_speaker()
            try:
                saved_global_speaker.write_text(json.dumps({"speaker": shared_speaker}), encoding="utf-8")
            except Exception:
                pass

    manifest = []
    # Load existing manifest if present so re-running adds new models only
    existing_manifest_path = out_dir / "manifest.csv"
    existing_manifest = None
    if existing_manifest_path.exists():
        try:
            existing_manifest = pd.read_csv(existing_manifest_path)
        except Exception:
            existing_manifest = None
    for i, rec in enumerate(picked, start=1):
        sample_dir = out_dir / f"sample_{i:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Reference: either original audio (if provided and found) or synthesized reference text
        ref_wav = None
        if args.original_audio_root:
            orig_root = Path(args.original_audio_root)
            candidates = []
            conv = rec.get("conv")
            speaker = rec.get("speaker")
            if conv and speaker:
                candidates.extend([
                    orig_root / f"{conv}_{speaker}.wav",
                    orig_root / f"{conv}.{speaker}.wav",
                ])
            if conv:
                for p in orig_root.rglob(f"*{conv}*.wav"):
                    candidates.append(p)
            existing = [c for c in candidates if c.exists()]
            if existing:
                ref_wav = existing[0]

        # sample-level speaker persistence (for same_pair)
        pair_speaker = None
        sample_speaker_path = sample_dir / "_speaker.json"
        if args.speaker_mode == "same_pair":
            if sample_speaker_path.exists():
                try:
                    js = json.loads(sample_speaker_path.read_text(encoding="utf-8"))
                    pair_speaker = js.get("speaker")
                except Exception:
                    pair_speaker = None
            if pair_speaker is None:
                pair_speaker = tts.sample_speaker()
                try:
                    sample_speaker_path.write_text(json.dumps({"speaker": pair_speaker}), encoding="utf-8")
                except Exception:
                    pass

        if ref_wav is None:
            # Synthesize the reference disfluent text
            text = prepare_text_for_tts(rec.get("disfluent", ""), args.tts_backend)
            text = sanitize_for_tts(text)
            ref_wav = sample_dir / "reference_disfluent.wav"
            if args.speaker_mode == "same_pair":
                synth_kwargs = {"speaker": pair_speaker} if pair_speaker is not None else {}
            else:
                synth_kwargs = {"speaker": shared_speaker} if args.speaker_mode == "same_all" and shared_speaker is not None else {}
            try:
                if not ref_wav.exists():
                    tts.synthesize_to_file(text, str(ref_wav), **synth_kwargs)
            except Exception as exc:
                # fallback: try again with more aggressive sanitization
                text2 = re.sub(r"[^\w\s<>\.\,\?\!]+", " ", text)
                tts.synthesize_to_file(text2, str(ref_wav), **synth_kwargs)

        # For each requested spec, synthesize its prediction (if available)
        for label, entry in preds_map.items():
            spec = entry["spec"]
            row = None
            # try source_index first
            sidx = rec.get("source_index")
            if sidx is not None and int(sidx) in entry["by_index"]:
                row = entry["by_index"].get(int(sidx))
            else:
                # try by id
                rid = rec.get("id")
                if rid is not None and str(rid) in entry["by_id"]:
                    row = entry["by_id"].get(str(rid))

            if row is None:
                # skip if not found
                print(f"Warning: no prediction found for {label} on sample {i}; skipping")
                continue

            pred_text = row.get("prediction") or (row.get("predictions") and row.get("predictions")[0]) or ""
            tts_text = prepare_text_for_tts(pred_text, args.tts_backend)
            tts_text = sanitize_for_tts(tts_text)
            safe_label = label.replace("/", "_").replace(" ", "_")
            model_wav = sample_dir / f"model_{safe_label}.wav"

            # skip existing model wavs if present in existing manifest (so reruns add only new models)
            already_done = False
            if existing_manifest is not None:
                try:
                    # match by sample number and label
                    row_sel = existing_manifest[(existing_manifest['sample'] == i) & (existing_manifest['label'] == label)]
                    if not row_sel.empty and Path(row_sel.iloc[0]['model_wav']).exists():
                        already_done = True
                except Exception:
                    already_done = False
            if already_done and not args.force:
                print(f"Skipping existing model {label} for sample {i}")
                continue
            if already_done and args.force:
                # remove existing file so we overwrite
                try:
                    Path(existing_manifest.loc[(existing_manifest['sample'] == i) & (existing_manifest['label'] == label), 'model_wav'].values[0]).unlink()
                except Exception:
                    pass

            if args.speaker_mode == "same_pair":
                synth_kwargs = {"speaker": pair_speaker} if pair_speaker is not None else {}
            else:
                synth_kwargs = {"speaker": shared_speaker} if args.speaker_mode == "same_all" and shared_speaker is not None else {}

            try:
                tts.synthesize_to_file(tts_text, str(model_wav), **synth_kwargs)
            except Exception as exc:
                # try a more aggressive sanitize fallback
                tts_text2 = re.sub(r"[^\w\s<>\.\,\?\!]+", " ", tts_text)
                tts.synthesize_to_file(tts_text2, str(model_wav), **synth_kwargs)

            # record row in manifest (after synth, regardless of fallback)
            manifest.append({
                "sample": i,
                "conv": rec.get("conv"),
                "id": rec.get("id"),
                "model_key": spec.get("model_key"),
                "label": label,
                "prompt": spec.get("prompt"),
                "shot": spec.get("shot"),
                "shot_k": spec.get("shot_k"),
                "context_mode": spec.get("context_mode"),
                "decoding": spec.get("decoding"),
                "model_text": pred_text,
                "model_wav": str(model_wav),
                "reference_wav": str(ref_wav),
                "fluent": rec.get("fluent"),
                "reference_text": rec.get("disfluent"),
            })

    df_new = pd.DataFrame(manifest)
    if existing_manifest is not None and not args.force:
        try:
            df_combined = pd.concat([existing_manifest, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["sample", "label"], keep="first")
            df_combined.to_csv(out_dir / "manifest.csv", index=False)
            print(f"Appended {len(df_combined) - len(existing_manifest)} new rows -> {out_dir / 'manifest.csv'}")
        except Exception:
            df_new.to_csv(out_dir / "manifest.csv", index=False)
            print(f"Wrote {len(df_new)} rows -> {out_dir / 'manifest.csv'}")
    else:
        # overwrite or no existing manifest
        df_new.to_csv(out_dir / "manifest.csv", index=False)
        print(f"Wrote {len(df_new)} rows -> {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
