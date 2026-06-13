"""Generate a few fluent/disfluent audio pairs for TTS backend comparison.

Usage:
    python tts_smoke_test.py --backend chattts --num-examples 3

The script expects `swbd_disfluency.jsonl` to exist locally. Create it from the
loader notebook first, or pass `--dataset path/to/file.jsonl`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.TTS import get_tts, prepare_text_for_tts
from utils.swbd_disfluency_loader import SwbdDisfluencyDataset


def load_examples(dataset_path: Path, num_examples: int) -> pd.DataFrame:
    ds = SwbdDisfluencyDataset.from_jsonl(str(dataset_path))
    df = ds.to_pandas()
    examples = df[
        (df["n_disfl_tokens"] > 0)
        & (df["n_tokens_fluent"] >= 8)
        & (df["n_tokens_fluent"] <= 25)
    ].copy()
    return examples.sample(n=min(num_examples, len(examples)), random_state=42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="chattts", help="TTS backend key from config.yaml")
    parser.add_argument("--dataset", default="swbd_disfluency.jsonl")
    parser.add_argument("--output-dir", default="outputs/tts_smoke_test")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument(
        "--speaker-mode",
        choices=("same_all", "same_pair", "backend_default"),
        default="same_all",
        help=(
            "same_all reuses one voice for every clip, same_pair reuses one voice "
            "within each fluent/disfluent pair, backend_default lets the backend choose."
        ),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} was not found. Run the dataset loader notebook and save_jsonl first."
        )

    out_dir = Path(args.output_dir) / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = get_tts(args.backend)
    examples = load_examples(dataset_path, args.num_examples)
    shared_speaker = tts.sample_speaker() if args.speaker_mode == "same_all" else None

    manifest_rows = []
    for idx, (_, row) in enumerate(examples.iterrows(), start=1):
        pair_speaker = tts.sample_speaker() if args.speaker_mode == "same_pair" else shared_speaker
        for version in ("fluent", "disfluent"):
            text = prepare_text_for_tts(row[version], args.backend)
            wav_path = out_dir / f"example_{idx:02d}_{version}.wav"
            synth_kwargs = {"speaker": pair_speaker} if pair_speaker is not None else {}
            tts.synthesize_to_file(text, str(wav_path), **synth_kwargs)
            manifest_rows.append(
                {
                    "example_id": idx,
                    "version": version,
                    "path": str(wav_path),
                    "text": text,
                    "n_disfl_tokens": row["n_disfl_tokens"],
                    "speaker_mode": args.speaker_mode,
                    "speaker_id": pair_speaker,
                }
            )
            print(f"saved {wav_path}")

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"wrote {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
