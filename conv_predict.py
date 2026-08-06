"""Insert disfluencies into the study conversations under ``convos/``.

Walks each conversation turn by turn, builds the SAME input the adapters were
fine-tuned on (``{context}\\n\\nCurrent: {fluent}``, where ``context`` is the
previous k=3 turns of CLEAN text with speaker prefixes), samples a disfluent
rewrite, and writes one JSONL per model plus a combined CSV for eyeballing.

Two conventions are load-bearing and match ``llm_training.py``:
  * context is built from the FLUENT turns, never from the model's own disfluent
    output - that is what ``add_context(source="fluent")`` did at training time;
  * conversation text is normalized to Switchboard style first (see
    ``utils.text_norm``), so the model is not fed capitals and full stops it
    never saw.

Usage:
    python conv_predict.py --models L3.1-8B-Instruct G4-4EB-it --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.swbd_disfluency_loader import SwbdDisfluencyDataset
from utils.text_norm import find_digits, is_unchanged, parse_conversation

# render_context is the SAME function that built the `context` field during
# training, so this pipeline cannot drift from it.
render_context = SwbdDisfluencyDataset.render_context

CONTEXT_K = 3  # must match llm_training.py's context_k


def clean_prediction(raw: str) -> str:
    """Trim a raw generation down to the rewritten utterance.

    Guards against the model echoing the prompt scaffolding or rambling onto a
    second line, which sampling at T=1.0 occasionally does.
    """
    out = str(raw).strip()
    if "\n" in out:
        out = out.split("\n", 1)[0].strip()
    out = re.sub(r"^(current|disfluent|output)\s*:\s*", "", out, flags=re.IGNORECASE)
    return " ".join(out.split())


def load_conversations(paths: List[Path]) -> Dict[str, List[Dict[str, str]]]:
    """``{conv_name: [{"speaker", "text"}, ...]}`` from a list of transcript files."""
    convs: Dict[str, List[Dict[str, str]]] = {}
    for path in paths:
        turns = parse_conversation(path.read_text(encoding="utf-8"), normalize=True)
        if not turns:
            continue
        convs[path.stem] = turns
        for t in turns:
            digits = find_digits(t["text"])
            if digits:
                print(f"  ! {path.name}: digits {digits} in {t['text']!r} - Switchboard "
                      f"spells numbers out; edit the source conversation.")
    if not convs:
        raise SystemExit("No conversations parsed.")

    # Switchboard only ever has speakers A and B, so that is all the adapters saw
    # in a context string. Any other label is an unseen token in the model input.
    labels = {t["speaker"] for turns in convs.values() for t in turns}
    unexpected = sorted(labels - {"A", "B"})
    if unexpected:
        print(f"  ! speaker labels {unexpected} are not A/B. The models were fine-tuned "
              f"on Switchboard, whose context strings only ever contain 'A:' and 'B:', "
              f"so these go into the prompt as unseen tokens. Relabel the transcripts "
              f"to A/B unless you specifically want to test this.")
    return convs


def merge_jsonl(path: Path, rows: List[Dict], overwrite: bool = False) -> int:
    """Write ``rows`` to ``path``, keyed by ``(conv, turn)``.

    Merges with whatever is already there unless ``overwrite``, so re-running on
    one conversation refreshes only that conversation instead of clobbering the
    rest of the set. Returns the total row count written.
    """
    merged: Dict[tuple, Dict] = {}
    if path.exists() and not overwrite:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                merged[(r["conv"], int(r["turn"]))] = r
    for r in rows:
        merged[(r["conv"], int(r["turn"]))] = r

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(merged):
            f.write(json.dumps(merged[key], ensure_ascii=False) + "\n")
    return len(merged)


def resolve_inputs(inputs: List[str]) -> List[Path]:
    """Accept a mix of directories and individual .txt files."""
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.txt")))
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit(f"No such file or directory: {item}")
    if not paths:
        raise SystemExit(f"No .txt files found in {inputs}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["L3.1-8B-Instruct", "G4-4EB-it"])
    ap.add_argument("--context-mode", default="with_context",
                    choices=["with_context", "no_context"])
    ap.add_argument("--inputs", nargs="+", default=["convos"],
                    help="Directories and/or individual .txt transcripts")
    ap.add_argument("--adapter-dir", default="outputs/llm_runs_enriched",
                    help="Root holding <context_mode>/<model>_<context_mode>/final_model")
    ap.add_argument("--output-dir", default="outputs/conversations")
    ap.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true", help="Greedy decoding instead of sampling")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-retries", type=int, default=1,
                    help="Attempts per turn. >1 resamples with a fresh seed when the model "
                         "returns the input unchanged. Default 1 (OFF)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Discard existing rows for conversations not in this run. By "
                         "default results are MERGED, so running on a single file only "
                         "refreshes that conversation and leaves the others intact.")
    args = ap.parse_args()

    paths = resolve_inputs(args.inputs)
    convs = load_conversations(paths)
    print(f"Loaded {len(convs)} conversations "
          f"({sum(len(t) for t in convs.values())} turns) from {args.inputs}")

    out_dir = Path(args.output_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    # Freeze the normalized fluent text so conv_tts.py voices exactly this.
    clean_path = out_dir / "clean_conversations.jsonl"
    clean_rows = [{"conv": conv, "turn": i, "speaker": t["speaker"], "fluent": t["text"]}
                  for conv, turns in convs.items() for i, t in enumerate(turns)]
    n_clean = merge_jsonl(clean_path, clean_rows, overwrite=args.overwrite)
    print(f"Wrote normalized fluent turns -> {clean_path} ({n_clean} total)")

    from predict import FinetunedPredictor  # heavy import, only once we know we need it

    all_rows: List[Dict] = []
    for model_key in args.models:
        print(f"\n=== {model_key} ({args.context_mode}) ===")
        predictor = FinetunedPredictor(
            model_key, context_mode=args.context_mode,
            output_dir=args.adapter_dir, quantization=args.quantization,
        )

        rows: List[Dict] = []
        for conv_i, (conv, turns) in enumerate(convs.items()):
            for turn_i, turn in enumerate(turns):
                speaker, fluent = turn["speaker"], turn["text"]
                prior = turns[max(0, turn_i - CONTEXT_K):turn_i]
                context = render_context(prior) if prior else ""

                # Deterministic per-turn seed so re-runs reproduce and adding a
                # model does not shift any other turn's sample.
                base_seed = args.seed + conv_i * 1000 + turn_i * 10
                prediction, used_seed, attempts = "", base_seed, 0
                for attempt in range(max(1, args.max_retries)):
                    used_seed = base_seed + attempt
                    attempts = attempt + 1
                    raw = predictor.predict(
                        fluent, context=context or None,
                        do_sample=not args.greedy,
                        temperature=args.temperature, top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens, seed=used_seed,
                    )
                    prediction = clean_prediction(raw if isinstance(raw, str) else raw[0])
                    if not is_unchanged(prediction, fluent) or args.greedy:
                        break

                unchanged = is_unchanged(prediction, fluent)
                if unchanged:
                    print(f"  ! {conv} turn {turn_i} ({speaker}): no disfluency inserted "
                          f"after {attempts} attempt(s)")

                rows.append({
                    "conv": conv, "turn": turn_i, "speaker": speaker,
                    "fluent": fluent, "context": context, "prediction": prediction,
                    "unchanged": unchanged,
                    "model_key": model_key, "context_mode": args.context_mode,
                    "do_sample": not args.greedy,
                    "temperature": args.temperature if not args.greedy else None,
                    "top_p": args.top_p if not args.greedy else None,
                    "seed": used_seed, "attempts": attempts,
                })
            print(f"  {conv}: {len(turns)} turns done")

        pred_path = out_dir / "predictions" / f"{model_key}_{args.context_mode}.jsonl"
        total = merge_jsonl(pred_path, rows, overwrite=args.overwrite)
        print(f"Wrote {len(rows)} rows -> {pred_path} ({total} total)")

        del predictor  # release GPU memory before loading the next base model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Rebuild the combined view from EVERY prediction file on disk, not just the
    # models in this run, so the CSV stays a complete picture.
    for path in sorted((out_dir / "predictions").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    csv_path = out_dir / "predictions_all.csv"
    pd.DataFrame(all_rows).sort_values(["model_key", "conv", "turn"]).to_csv(
        csv_path, index=False, encoding="utf-8")
    print(f"\nCombined {len(all_rows)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
