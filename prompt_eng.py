"""Prompt-engineering baseline: disfluency predictions from RAW base models.

The no-fine-tuning counterpart to ``evaluate.py``. For each base instruct model
it sweeps zero / one / few-shot x no-context / with-context, varying ONLY the
prompt (native chat template + a convention-describing instruction + a curated
few-shot suite), and writes one predictions JSONL per condition in the SAME
schema and layout as the fine-tuned runs, so ``silver.ipynb`` scores them with
the identical pipeline.

Key differences from evaluate.py:
  * loads the base model with ``.load()`` only -- NO ``load_adapter``;
  * prompt_style defaults to "chat" (the model's own template);
  * passes the ``prompting`` instruction + selected examples into generate();
  * loads each base model ONCE and reuses it across all 6 conditions.

Output layout (mirrors outputs/llm_runs_*):
    outputs/prompt_eng/
      {no_context,with_context}/{model}_{shot}_{context}/
          test_predictions.jsonl
          run_config.json
      manifest.json

Usage:
    python prompt_eng.py --limit 8                    # smoke test
    python prompt_eng.py                              # full greedy sweep
    python prompt_eng.py --sample --temperature 0.5 --num-samples 3 \
        --pred-filename test_predictions_t05.jsonl    # later sweep
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from utils.models import BaseDisfluencyModel, _load_model_registry, _infer_task_type
from utils import prompting
from llm_training import _cleanup_cuda_memory
from evaluate import load_test_split, _hf_login, CONTEXT_MODES

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reverse of CONTEXT_MODES ({False:"no_context", True:"with_context"}): CLI label -> bool.
CONTEXT_FLAG = {"no": False, "with": True}


def _generate_batch(
    model: BaseDisfluencyModel,
    texts: List[str],
    *,
    examples: Optional[Sequence[Dict[str, str]]],
    instruction: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    draw: int,
) -> List[Optional[str]]:
    """One batched generate() call, falling back to per-row on failure.

    Mirrors evaluate.py's retry logic so a single pathological row can't null
    out a whole chunk. ``examples``/``instruction`` are what make this the
    prompt-engineering path rather than the adapter path.
    """
    gen_kwargs = dict(
        instruction=instruction, examples=examples,
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
    )
    try:
        return model.generate(texts, **gen_kwargs)
    except Exception as exc:
        logger.warning(
            "Batch generation failed (draw %s, %s rows): %s; retrying row-by-row",
            draw, len(texts), exc,
        )
        preds: List[Optional[str]] = []
        for text in texts:
            try:
                preds.append(model.generate([text], **gen_kwargs)[0])
            except Exception as row_exc:
                logger.warning("Generation failed on row (draw %s): %s", draw, row_exc)
                preds.append(None)
        return preds


def run_condition(
    model: BaseDisfluencyModel,
    model_key: str,
    model_id: str,
    pool: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    *,
    shot: str,
    use_context: bool,
    out_path: Path,
    few_k: Optional[int],
    example_seed: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_samples: int,
    seed: int,
    max_new_tokens: int,
    batch_size: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    """Generate + write one (shot x context) condition on an already-loaded model."""
    context_label = CONTEXT_MODES[use_context]
    examples = prompting.select(pool, shot, use_context, few_k=few_k, seed=example_seed)
    examples_arg = examples or None  # zero-shot -> None
    ids = prompting.example_ids(examples)

    k = num_samples if do_sample else 1
    torch.manual_seed(seed)
    records = test_records[:limit] if limit else test_records

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Crash-safe: write to a .partial file and atomically rename only once the
    # whole condition finishes. A crash mid-generation leaves the .partial behind
    # (ignored on resume) and NO final file, so the run is redone rather than
    # skipped as if complete.
    tmp_path = out_path.with_name(out_path.name + ".partial")
    with open(tmp_path, "w") as f, tqdm(
        total=len(records), desc=f"{model_key} [{shot}/{context_label}]", unit="row"
    ) as pbar:
        for start in range(0, len(records), batch_size):
            batch = list(enumerate(records[start:start + batch_size], start=start))
            valid = []
            for idx, r in batch:
                fluent = r.get("fluent", "").strip()
                reference = r.get("disfluent", "").strip()
                if not fluent or not reference:
                    continue
                # Labelled query ("Fluent:" / "Context:...Fluent:") matching the
                # few-shot example inputs, so the demonstrations and the live query
                # read identically.
                input_text = prompting.format_input(fluent, r.get("context"), use_context)
                valid.append((idx, r, fluent, reference, input_text))

            if not valid:
                pbar.update(len(batch))
                continue

            input_texts = [v[4] for v in valid]
            samples_per_row: List[List[Optional[str]]] = [[] for _ in valid]
            for s in range(k):
                preds = _generate_batch(
                    model, input_texts, examples=examples_arg,
                    instruction=prompting.INSTRUCTION, do_sample=do_sample,
                    temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens, draw=s,
                )
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
                    "shot": shot,
                    "context": r.get("context") if use_context else None,
                    "input_text": input_text,
                    "fluent": fluent,
                    "reference": reference,
                    "prediction": samples[0],
                    "predictions": samples,
                    "decode": {
                        "do_sample": do_sample, "temperature": temperature,
                        "top_p": top_p, "num_samples": k, "seed": seed,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            pbar.update(len(batch))

    # All rows written successfully -> publish atomically.
    os.replace(tmp_path, out_path)

    run_cfg = {
        "model_key": model_key, "model_id": model_id,
        "shot": shot, "context_mode": context_label,
        "prompt_style": "chat", "instruction": prompting.INSTRUCTION,
        "example_ids": ids, "few_k": few_k, "example_seed": example_seed,
        "decode": {
            "do_sample": do_sample, "temperature": temperature,
            "top_p": top_p, "num_samples": k, "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
        "n_written": written,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_path.parent / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    logger.info("✓ Wrote %s predictions to %s (examples=%s)", written, out_path, ids)
    return {"run_key": out_path.parent.name, "n_written": written,
            "path": str(out_path), "example_ids": ids}


def main(
    output_dir: str = "outputs/prompt_eng",
    test_split: str = "outputs/llm_runs_enriched/splits/test.jsonl",
    pool_path: str = "prompt_eng/examples.yaml",
    models: Optional[List[str]] = None,
    shots: Optional[List[str]] = None,
    contexts: Optional[List[str]] = None,
    few_k: Optional[int] = None,
    example_seed: int = 0,
    quantization: str = "4bit",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    num_samples: int = 1,
    seed: int = 42,
    pred_filename: str = "test_predictions.jsonl",
    limit: Optional[int] = None,
    overwrite: bool = False,
    batch_size: int = 8,
) -> None:
    _hf_login()
    shots = shots or list(prompting.SHOTS)
    contexts = contexts or list(CONTEXT_FLAG.keys())

    out_root = Path(output_dir)
    test_records = load_test_split(Path(test_split).parent)
    pool = prompting.load_pool(pool_path)
    registry = _load_model_registry()
    model_keys = models or list(registry.keys())

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    manifest: List[Dict[str, Any]] = []

    def flush_manifest() -> None:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for model_key in model_keys:
        model_id = registry.get(model_key, model_key)

        # Decide up front whether every condition for this model is already done,
        # so we can skip loading the (heavy) base model entirely when resuming.
        planned = []
        for ctx in contexts:
            use_context = CONTEXT_FLAG[ctx]
            for shot in shots:
                run_key = f"{model_key}_{shot}_{CONTEXT_MODES[use_context]}"
                out_path = out_root / CONTEXT_MODES[use_context] / run_key / pred_filename
                if out_path.exists() and not overwrite:
                    logger.info("Skipping %s: exists (%s); pass --overwrite", run_key, out_path)
                    manifest.append({"run_key": run_key, "status": "skipped_exists",
                                     "path": str(out_path)})
                    continue
                planned.append((shot, use_context, out_path, run_key))
        if not planned:
            flush_manifest()
            continue

        logger.info("Loading base %s (%s) [chat prompt, %s]...", model_key, model_id, quantization)
        task_type = _infer_task_type(model_key, model_id)
        model = BaseDisfluencyModel.from_config(
            model_key, model_id=model_id, task_type=task_type, device_map="auto",
            load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
            torch_dtype=compute_dtype, prompt_style="chat",
        )
        try:
            model.load()
            for shot, use_context, out_path, run_key in planned:
                try:
                    info = run_condition(
                        model, model_key, model_id, pool, test_records,
                        shot=shot, use_context=use_context, out_path=out_path,
                        few_k=few_k, example_seed=example_seed,
                        do_sample=do_sample, temperature=temperature, top_p=top_p,
                        num_samples=num_samples, seed=seed, max_new_tokens=max_new_tokens,
                        batch_size=batch_size, limit=limit,
                    )
                    manifest.append({"run_key": run_key, "status": "ok", **info})
                except Exception as exc:
                    logger.exception("Failed %s: %s", run_key, exc)
                    manifest.append({"run_key": run_key, "status": "error", "error": str(exc)})
                flush_manifest()
        finally:
            del model
            _cleanup_cuda_memory()

    flush_manifest()
    logger.info("\n%s\nPrompt-Engineering Summary\n%s", "=" * 80, "=" * 80)
    for entry in manifest:
        logger.info("%-40s %s", entry["run_key"], entry.get("status"))
    logger.info("Manifest -> %s", out_root / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt-engineering disfluency baseline on base models")
    parser.add_argument("--output-dir", default="outputs/prompt_eng")
    parser.add_argument("--test-split", default="outputs/llm_runs_enriched/splits/test.jsonl",
                        help="Canonical test split (reused byte-identical from the trained runs)")
    parser.add_argument("--pool-path", default="prompt_eng/examples.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys (default: all in config.yaml)")
    parser.add_argument("--shots", nargs="+", default=None, choices=list(prompting.SHOTS))
    parser.add_argument("--contexts", nargs="+", default=None, choices=list(CONTEXT_FLAG.keys()))
    parser.add_argument("--few-k", type=int, default=None,
                        help="Cap few-shot to the first N suite entries (default: use all)")
    parser.add_argument("--example-seed", type=int, default=0, help="Seed for few-shot type-coverage tie-breaks")
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy (default greedy)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1, help="Draws per row when --sample is set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred-filename", default="test_predictions.jsonl",
                        help="Output filename; change it for temperature-sweep files")
    parser.add_argument("--limit", type=int, default=None, help="Only first N test rows (quick checks)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    main(
        output_dir=args.output_dir, test_split=args.test_split, pool_path=args.pool_path,
        models=args.models, shots=args.shots, contexts=args.contexts,
        few_k=args.few_k, example_seed=args.example_seed,
        quantization=args.quantization, max_new_tokens=args.max_new_tokens,
        do_sample=args.sample, temperature=args.temperature, top_p=args.top_p,
        num_samples=args.num_samples, seed=args.seed, pred_filename=args.pred_filename,
        limit=args.limit, overwrite=args.overwrite, batch_size=args.batch_size,
    )
