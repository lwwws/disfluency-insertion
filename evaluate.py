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
    limit: Optional[int] = None,
) -> int:
    """Load one adapter, generate on the test set, and write a predictions JSONL.

    Returns the number of rows written.
    """
    task_type = _infer_task_type(model_key, model_id)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info(f"Loading base {model_key} ({model_id}) + adapter {adapter_dir}...")
    model = BaseDisfluencyModel.from_config(
        model_key,
        model_id=model_id,
        task_type=task_type,
        device_map="auto",
        load_in_4bit=quantization == "4bit",
        load_in_8bit=quantization == "8bit",
        torch_dtype=compute_dtype,
    )
    model.load()
    model.load_adapter(str(adapter_dir))

    context_label = CONTEXT_MODES[use_context]
    records = test_records[:limit] if limit else test_records

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for idx, r in enumerate(records):
            fluent = r.get("fluent", "").strip()
            reference = r.get("disfluent", "").strip()
            if not fluent or not reference:
                continue  # same rows training would have skipped

            input_text = build_input_text(r, use_context=use_context)
            try:
                prediction = model.generate(
                    input_text,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
            except Exception as exc:  # keep going; record the failure
                logger.warning("Generation failed on row %s: %s", idx, exc)
                prediction = None

            row = {
                "id": r.get("id", r.get("uid", idx)),
                "conv": r.get("conv"),
                "split": r.get("split"),
                "model_key": model_key,
                "model_id": model_id,
                "context_mode": context_label,
                "context": r.get("context") if use_context else None,
                "input_text": input_text,   # exact text fed to the model
                "fluent": fluent,           # clean input
                "reference": reference,     # gold disfluent target
                "prediction": prediction,   # model's disfluent output
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if written % 50 == 0:
                logger.info(f"  {model_key} [{context_label}]: {written} rows...")

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
    limit: Optional[int] = None,
    skip_models: Optional[List[str]] = None,
    overwrite: bool = False,
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

            out_path = ctx_dir / run_key / "test_predictions.jsonl"
            if out_path.exists() and not overwrite:
                logger.info(f"Skipping {run_key}: predictions already exist ({out_path}); pass --overwrite to regenerate")
                continue
            try:
                n = predict_for_model(
                    model_key=model_key,
                    model_id=model_id,
                    adapter_dir=adapter_dir,
                    test_records=test_records,
                    use_context=use_context,
                    out_path=out_path,
                    quantization=quantization,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    limit=limit,
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
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        limit=args.limit,
        skip_models=args.skip_models,
        overwrite=args.overwrite,
    )
