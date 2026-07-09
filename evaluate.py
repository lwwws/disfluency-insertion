"""
Generate disfluency predictions on the held-out test split for each fine-tuned model.

This script:
1. Loads the canonical test split (``<output_dir>/splits/test.jsonl``) written by
   ``llm_training.py`` — every model is scored on byte-identical rows.
2. Loads each trained LoRA adapter onto its (4-bit, bf16) base model.
3. Generates a disfluent rewrite for every test row.
4. Writes one predictions JSONL per model x context, next to that run's ``final_model/``.

No metrics are computed here. Each output row carries the clean input, the gold
disfluent reference, and the model's disfluent output, so BLEU / BERTScore /
token-placement analysis can all be computed downstream from this single file.

Usage:
    python evaluate.py [--output-dir outputs/llm_runs] [--limit N] [--sample]
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from utils.models import BaseDisfluencyModel, _load_model_registry, _infer_task_type
from llm_training import build_input_text, _cleanup_cuda_memory

load_dotenv()

try:
    from huggingface_hub import login as hf_login  # type: ignore
except Exception:  # pragma: no cover
    hf_login = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mirror the context labels used by llm_training.py for run folders.
CONTEXT_MODES = {False: "no_context", True: "with_context"}


def _hf_login() -> None:
    """Authenticate with the HF hub so gated base models can be downloaded."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token and hf_login is not None:
        try:
            hf_login(token=token)
        except Exception as exc:  # pragma: no cover
            logger.warning("HF login failed (continuing; cached models may still work): %s", exc)


def load_test_split(splits_dir: Path) -> List[Dict[str, Any]]:
    """Load the canonical test records dumped by llm_training.save_splits."""
    test_path = splits_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found at {test_path}. Run llm_training.py first "
            "(it writes splits/test.jsonl)."
        )
    records = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} test records from {test_path}")
    return records


def predict_for_model(
    model_key: str,
    model_id: str,
    adapter_dir: Path,
    test_records: List[Dict[str, Any]],
    use_context: bool,
    out_path: Path,
    *,
    quantization: str = "4bit",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    num_samples: int = 1,
    seed: int = 42,
    limit: Optional[int] = None,
    batch_size: int = 8,
    prompt_style: str = "plain",
    added_tokens: Optional[List[str]] = None,
) -> int:
    """Load one adapter, generate on the test set, write a predictions JSONL.

    When do_sample=True, draws `num_samples` samples per row (stored in
    'predictions'); 'prediction' keeps the first draw for back-compat.
    Greedy (do_sample=False) always collapses to a single draw.

    Rows are generated `batch_size` at a time. If a batch fails outright
    (e.g. one pathological row), we retry it row-by-row so a single bad
    example doesn't null out the whole chunk.
    """
    task_type = _infer_task_type(model_key, model_id)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info(f"Loading base {model_key} ({model_id}) + adapter {adapter_dir}...")
    model = BaseDisfluencyModel.from_config(
        model_key, model_id=model_id, task_type=task_type, device_map="auto",
        load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
        torch_dtype=compute_dtype,
        prompt_style=prompt_style, added_tokens=added_tokens,
    )
    model.load()
    # load_adapter reloads the adapter's tokenizer and auto-resizes embeddings, so
    # any train-time "<sil>" token is reproduced even if added_tokens wasn't passed.
    model.load_adapter(str(adapter_dir))

    # Greedy is deterministic -> one draw is enough; sampling needs a seed for reproducibility.
    k = num_samples if do_sample else 1
    torch.manual_seed(seed)

    context_label = CONTEXT_MODES[use_context]
    records = test_records[:limit] if limit else test_records

    def generate_batch(texts: List[str], draw: int) -> List[Optional[str]]:
        """One batched generate() call, falling back to per-row calls on failure."""
        try:
            return model.generate(
                texts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
            )
        except Exception as exc:
            logger.warning(
                "Batch generation failed (draw %s, %s rows): %s; retrying row-by-row",
                draw, len(texts), exc,
            )
            preds: List[Optional[str]] = []
            for text in texts:
                try:
                    preds.append(model.generate(
                        [text],
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                    )[0])
                except Exception as row_exc:
                    logger.warning("Generation failed on row (draw %s): %s", draw, row_exc)
                    preds.append(None)
            return preds

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f, tqdm(
        total=len(records), desc=f"{model_key} [{context_label}]", unit="row"
    ) as pbar:
        for batch_start in range(0, len(records), batch_size):
            batch = list(enumerate(records[batch_start:batch_start + batch_size], start=batch_start))

            valid = []
            for idx, r in batch:
                fluent = r.get("fluent", "").strip()
                reference = r.get("disfluent", "").strip()
                if not fluent or not reference:
                    continue
                input_text = build_input_text(r, use_context=use_context)
                valid.append((idx, r, fluent, reference, input_text))

            if not valid:
                pbar.update(len(batch))
                continue

            input_texts = [v[4] for v in valid]
            samples_per_row: List[List[Optional[str]]] = [[] for _ in valid]
            for s in range(k):
                preds = generate_batch(input_texts, s)
                for row_samples, pred in zip(samples_per_row, preds):
                    row_samples.append(pred)

            for (idx, r, fluent, reference, input_text), samples in zip(valid, samples_per_row):
                row = {
                    "id": r.get("id", r.get("uid", idx)),
                    "conv": r.get("conv"),
                    "split": r.get("split"),
                    "model_key": model_key,
                    "model_id": model_id,
                    "context_mode": context_label,
                    "context": r.get("context") if use_context else None,
                    "input_text": input_text,
                    "fluent": fluent,
                    "reference": reference,
                    "prediction": samples[0],      # back-compat: first draw
                    "predictions": samples,        # all k draws
                    "decode": {                    # provenance so you can't mix protocols by accident
                        "do_sample": do_sample, "temperature": temperature,
                        "top_p": top_p, "num_samples": k, "seed": seed,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            pbar.update(len(batch))

    logger.info(f"✓ Wrote {written} predictions to {out_path}")

    # Release VRAM before the next model.
    del model
    _cleanup_cuda_memory()
    return written


def main(
    output_dir: str = "outputs/llm_runs",
    quantization: str = "4bit",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    num_samples: int = 1,
    seed: int = 42,
    pred_filename: str = "test_predictions.jsonl",
    limit: Optional[int] = None,
    skip_models: Optional[List[str]] = None,
    overwrite: bool = False,
    batch_size: int = 8,
    prompt_style_override: Optional[str] = None,
) -> None:
    skip_models = skip_models or []
    _hf_login()

    out_root = Path(output_dir)
    test_records = load_test_split(out_root / "splits")
    registry = _load_model_registry()

    summary = []
    # Evaluate whatever context modes were actually trained (dir present on disk).
    for use_context, context_label in CONTEXT_MODES.items():
        ctx_dir = out_root / context_label
        if not ctx_dir.is_dir():
            continue

        for model_key, model_id in registry.items():
            if model_key in skip_models:
                continue
            run_key = f"{model_key}_{context_label}"
            adapter_dir = ctx_dir / run_key / "final_model"
            if not adapter_dir.is_dir():
                logger.info(f"Skipping {run_key}: no adapter at {adapter_dir}")
                continue

            out_path = ctx_dir / run_key / pred_filename
            if out_path.exists() and not overwrite:
                logger.info(f"Skipping {run_key}: predictions already exist ({out_path}); pass --overwrite to regenerate")
                continue

            # Reconstruct the training recipe so inference matches it (prompt style,
            # extra vocab). CLI --prompt-style overrides the recorded value.
            run_cfg: Dict[str, Any] = {}
            run_cfg_path = adapter_dir / "run_config.json"
            if run_cfg_path.exists():
                try:
                    with open(run_cfg_path) as rc:
                        run_cfg = json.load(rc)
                except Exception as exc:
                    logger.warning("Could not read %s: %s", run_cfg_path, exc)
            prompt_style = prompt_style_override or run_cfg.get("prompt_style", "plain")
            added_tokens = run_cfg.get("added_tokens") or None
            logger.info("Recipe for %s: prompt_style=%s, added_tokens=%s", run_key, prompt_style, added_tokens)

            try:
                n = predict_for_model(
                    model_key=model_key, model_id=model_id, adapter_dir=adapter_dir,
                    test_records=test_records, use_context=use_context, out_path=out_path,
                    quantization=quantization, max_new_tokens=max_new_tokens,
                    do_sample=do_sample, temperature=temperature, top_p=top_p,
                    num_samples=num_samples, seed=seed, limit=limit,
                    batch_size=batch_size,
                    prompt_style=prompt_style, added_tokens=added_tokens,
                )
                summary.append((run_key, n, str(out_path)))
            except Exception as exc:
                logger.exception("Failed to evaluate %s: %s", run_key, exc)
                summary.append((run_key, -1, str(exc)))

    logger.info("\n%s\nEvaluation Summary\n%s", "=" * 80, "=" * 80)
    for run_key, n, where in summary:
        if n < 0:
            logger.error(f"✗ {run_key}: {where}")
        else:
            logger.info(f"✓ {run_key}: {n} predictions -> {where}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test-set disfluency predictions for fine-tuned models"
    )
    parser.add_argument("--output-dir", default="outputs/llm_runs", help="Root dir used by llm_training.py")
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"], help="Base-model quantization")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate per example")
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding (greedy is default for reproducible eval)")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N test rows (for quick checks)")
    parser.add_argument("--skip-models", nargs="+", default=[], help="Model keys to skip")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate predictions even if test_predictions.jsonl already exists")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Draws per row when --sample is set (ignored for greedy)")
    parser.add_argument("--pred-filename", default="test_predictions.jsonl",
                        help="Output filename; change it to avoid overwriting greedy predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8, help="Rows generated per batched model.generate() call")
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default=None,
                        help="Override the prompt style; by default it's read from each run's "
                             "run_config.json (falls back to 'plain' for older runs)")

    args = parser.parse_args()

    main(
        output_dir=args.output_dir, quantization=args.quantization,
        max_new_tokens=args.max_new_tokens, do_sample=args.sample,
        temperature=args.temperature, top_p=args.top_p, num_samples=args.num_samples,
        seed=args.seed, pred_filename=args.pred_filename,
        limit=args.limit, skip_models=args.skip_models, overwrite=args.overwrite,
        batch_size=args.batch_size, prompt_style_override=args.prompt_style,
    )
