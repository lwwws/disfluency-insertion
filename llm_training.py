"""
LLM fine-tuning with LoRA on disfluency insertion task.

This script:
1. Loads the Switchboard disfluency dataset from the dataloader
2. Splits data into train/eval/test sets
3. Fine-tunes all configured LLMs with LoRA adapters
4. Saves checkpoints during training for resumption
5. Logs training metrics and model configurations

Usage:
    python llm_training.py [--seed 42] [--output-dir outputs/llm_runs]

LoRA Configuration (from paper):
    - Rank (r): 32
    - Scaling factor (alpha): 64
    - Dropout: 0.1
    - Target modules: ['q_proj', 'v_proj'] (model-specific)

Training Configuration:
    - Batch size: 1 (default, CLI-overridable)
    - Gradient accumulation steps: 8 (default, CLI-overridable)
    - Learning rate: 2e-4
    - Gradient checkpointing: enabled by default
    - Save strategy: steps (every 500 steps)
"""

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
from utils.models import BaseDisfluencyModel, _load_model_registry
from utils.swbd_disfluency_loader import SwbdDisfluencyDataset, Config as DataConfig
import os

from dotenv import load_dotenv
load_dotenv()

try:
    # prefer the modern name; huggingface_hub is a light dependency and safe to require for auth
    from huggingface_hub import login as hf_login  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    hf_login = None  # type: ignore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================================================================================
# LoRA Configuration per Model Family
# ================================================================================
LORA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "causal": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj"],  # works for Llama, Qwen, etc.
    },
    "seq2seq": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "SEQ_2_SEQ_LM",
        "target_modules": ["q", "v"],  # T5 uses 'q' and 'v' (not 'q_proj', 'v_proj')
    },
}


# Wider adapter surface (opt-in via --lora-profile wide): attention out + MLP
# projections on top of q/v.
WIDE_CAUSAL_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def resolve_lora_config(
    task_type: str,
    model_id: str,
    *,
    lora_profile: str = "paper",
    modules_to_save: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a LoRA config for this model, applying per-family overrides.

    Gemma-4 E4B is multimodal: its text decoder uses plain ``nn.Linear``
    q_proj/v_proj (which LoRA supports), but its vision and audio towers wrap
    the same-named projections in a custom ``Gemma4ClippableLinear`` module
    that PEFT cannot adapt. A bare ``["q_proj", "v_proj"]`` suffix match hits
    both, so PEFT errors on the tower wrappers. We exclude the towers instead:
    LoRA then attaches only to the text decoder's query/value projections,
    identical to the Llama setup, and text-only batches actually drive them
    (adapting the towers gives them no gradient -> grad-scaler crash).

    lora_profile     : "paper" (q/v only, the default/reproduction) or "wide"
                       (adds k/o + MLP projections; causal models only).
    modules_to_save  : full (non-LoRA) modules to train and save, e.g.
                       ["lm_head"] when a new "<sil>" token needs an output row
                       the frozen base can't provide.
    """
    cfg = dict(LORA_CONFIGS[task_type])  # copy so we never mutate the shared dict

    if task_type == "causal" and lora_profile == "wide":
        cfg["target_modules"] = list(WIDE_CAUSAL_MODULES)

    if task_type == "causal" and "gemma" in model_id.lower():
        # Keep the plain-Linear text q_proj/v_proj; drop the vision/audio towers.
        cfg["exclude_modules"] = r".*(vision_tower|audio_tower).*"

    if modules_to_save:
        cfg["modules_to_save"] = list(modules_to_save)

    return cfg


# ================================================================================
# Training Configuration
# ================================================================================
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 8,  # eval is forward-only; larger batch is safe + much faster
    "gradient_accumulation_steps": 8,  # keeps effective batch size ~8 with lower VRAM
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 50,
    "gradient_checkpointing": True,
    "save_steps": 500,
    "save_total_limit": 2,
    "save_strategy": "steps",
    "eval_strategy": "steps",
    "eval_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "seed": 42,
}


def _cleanup_cuda_memory() -> None:
    """Free GPU allocator cache between model runs to reduce fragmentation/OOMs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ================================================================================
# Data Loading & Preparation
# ================================================================================
def load_disfluency_dataset(
    seed: int = 42,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_len: int = 10,
    add_context: bool = True,
    context_k: int = 3,
    full_data_path: str = "data/cache/swbd_disfluency_full.jsonl",
    force_rebuild: bool = False,
    enriched_only: bool = False,
    min_disfl_tokens: int = 0,
) -> tuple:
    """
    Load the Switchboard disfluency dataset and return train/eval/test splits.

    Args:
        seed: Random seed for reproducibility
        train_ratio, eval_ratio, test_ratio: Split ratios (must sum to ~1.0)
        min_len: Minimum number of tokens for a record to be kept
        add_context: If True, add previous context turns to each record
        context_k: Number of context turns to include
        full_data_path: Path to save/load the full built dataset (JSONL)
        force_rebuild: If True, ignore any existing cache and rebuild the dataset
        enriched_only: If True, keep only annotation-complete rows (NXT alignment
            succeeded, so typical <sil>/filled pauses are present). Non-enriched
            rows are annotation-INCOMPLETE for typical disfluencies, so as targets
            they may teach omission.
        min_disfl_tokens: Drop rows with fewer than this many disfluency tokens
            (0 = keep all; 1 = drop the ~30% pure-copy rows where target == input).

    Returns:
        (train_records, eval_records, test_records) where each is a list of dicts
        Each dict has keys: 'fluent', 'disfluent', 'context' (if add_context=True), etc.
    """
    logger.info("Loading Switchboard disfluency dataset...")

    cfg = DataConfig()

    if not force_rebuild and Path(full_data_path).exists():
        logger.info(f"Loading full built dataset from {full_data_path}...")
        dataset = SwbdDisfluencyDataset.from_jsonl(full_data_path, config=cfg)
    else:
        logger.info("Building dataset from raw Zayats/NXT sources...")
        dataset = SwbdDisfluencyDataset(cfg).build()
        Path(full_data_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_jsonl(full_data_path)
        logger.info(f"Saved full built dataset to {full_data_path}")

    if add_context:
        logger.info(f"Adding context (k={context_k} turns)...")
        dataset.add_context(k=context_k, source="fluent", include_other_speaker=True)

    # Filter records. Runs AFTER add_context, so non-enriched / low-signal rows
    # still served as context for their neighbours before being dropped as targets.
    custom_fn = None
    if enriched_only or min_disfl_tokens > 0:
        def custom_fn(r):
            if enriched_only and not r.get("enriched", False):
                return False
            if r.get("n_disfl_tokens", 0) < min_disfl_tokens:
                return False
            return True
    logger.info(
        "Filtering: min_len=%s, enriched_only=%s, min_disfl_tokens=%s ...",
        min_len, enriched_only, min_disfl_tokens,
    )
    dataset.filter_records(min_tokens_fluent=min_len, custom_fn=custom_fn)

    # Add splits
    logger.info(f"Splitting data: train={train_ratio}, eval={eval_ratio}, test={test_ratio}...")
    dataset.add_splits(
        ratios=(train_ratio, eval_ratio, test_ratio),
        names=("train", "eval", "test"),  # must match the split names filtered below
        seed=seed,
        by="conv",  # split by conversation to avoid speaker leakage
    )

    # Partition by split
    train_records = [r for r in dataset.records if r.get("split") == "train"]
    eval_records = [r for r in dataset.records if r.get("split") == "eval"]
    test_records = [r for r in dataset.records if r.get("split") == "test"]

    logger.info(f"Dataset splits: train={len(train_records)}, eval={len(eval_records)}, test={len(test_records)}")

    return train_records, eval_records, test_records


def save_splits(splits_dir: str, train_records, eval_records, test_records) -> None:
    """Persist the exact train/eval/test records (post context+filter+split) as JSONL.

    These are the canonical splits for this seed. Evaluation loads ``test.jsonl``
    from here so every model is scored on byte-identical rows — no re-splitting.
    """
    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("train", train_records), ("eval", eval_records), ("test", test_records)):
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(records)} {name} records to {path}")


def prepare_training_data(
    records: List[Dict[str, Any]],
    use_context: bool = True,
) -> List[tuple]:
    """
    Convert dataset records into (input, output) pairs for fine-tuning.

    Each pair uses:
        input:  context (if available) + fluent utterance
        output: disfluent utterance

    Args:
        records: List of dicts from the dataset with 'fluent', 'disfluent', and optionally 'context'
        use_context: If True, prepend context to input

    Returns:
        List of (input_text, output_text) tuples
    """
    pairs = []
    for r in records:
        fluent = r.get("fluent", "").strip()
        disfluent = r.get("disfluent", "").strip()

        if not fluent or not disfluent:
            continue

        pairs.append((build_input_text(r, use_context=use_context), disfluent))

    logger.info(f"Prepared {len(pairs)} training pairs")
    return pairs


def build_input_text(record: Dict[str, Any], use_context: bool = True) -> str:
    """Build the model input string for a single record (context-aware).

    Single source of truth shared by training-data prep and evaluation, so the
    text fed to the model at eval time matches training byte-for-byte.
    """
    fluent = record.get("fluent", "").strip()
    if use_context and "context" in record:
        context = record.get("context", "").strip()
        if context:
            return f"{context}\n\nCurrent: {fluent}"
    return fluent


# ================================================================================
# Model Training
# ================================================================================
def train_model(
    model_key: str,
    model_id: str,
    train_data: List[tuple],
    eval_data: Optional[List[tuple]],
    output_dir: str,
    task_type: str,
    seed: int = 42,
    run_name: Optional[str] = None,
    quantization: str = "4bit",
    prompt_style: str = "plain",
    add_sil_token: bool = False,
    sil_train_embeddings: bool = False,
    lora_profile: str = "paper",
    **training_kwargs: Any,
) -> Dict[str, Any]:
    """
    Fine-tune a single model with LoRA.

    Args:
        model_key: Human-readable model name (e.g., "L2-7B-Chat")
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
        train_data: List of (input, output) tuples
        eval_data: Optional list of (input, output) tuples for evaluation
        output_dir: Directory to save model and checkpoints
        task_type: "causal" or "seq2seq"
        seed: Random seed
        prompt_style: "plain" (hand-rolled scaffold) or "chat" (native template)
        add_sil_token: register "<sil>" as an atomic token + resize embeddings
        sil_train_embeddings: with add_sil_token, also train input embeddings
            (heavier VRAM); default trains only lm_head so the model can emit it
        lora_profile: "paper" (q/v only) or "wide" (adds k/o + MLP projections)
        **training_kwargs: Additional training arguments (learning_rate, num_epochs, etc.)

    Returns:
        Dict with training results including model name, best metrics, etc.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_key} ({model_id})")
    logger.info(f"{'='*80}")

    # Use run_name (which may include context label) for folder naming, but keep model_key for lookup
    folder_name = run_name or model_key
    output_path = Path(output_dir) / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    final_model_dir = output_path / "final_model"

    try:
        # Load model (use the config key to resolve registry; pass explicit model_id to be safe)
        logger.info(f"Loading model {model_key} (id={model_id})...")

        quantization = (quantization or "none").lower().strip()
        load_in_4bit = quantization == "4bit"
        load_in_8bit = quantization == "8bit"

        # Match the mixed-precision dtype used for training (bf16 where supported)
        # so quantized compute and non-quantized weights don't fight fp16 overflow.
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        added_tokens = ["<sil>"] if add_sil_token else None
        model = BaseDisfluencyModel.from_config(
            model_key,
            model_id=model_id,
            task_type=task_type,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=compute_dtype,
            prompt_style=prompt_style,
            added_tokens=added_tokens,
        )
        model.load()
        logger.info(
            "Model loaded successfully. Task type: %s, quantization: %s",
            model.task_type,
            quantization,
        )

        # Enable LoRA. A new "<sil>" token needs a trainable output row (and
        # optionally input embedding) since the frozen base has none — otherwise
        # the model can't learn to emit it. Default to lm_head-only to keep VRAM
        # sane on a single 24GB card.
        modules_to_save = None
        if add_sil_token:
            modules_to_save = ["embed_tokens", "lm_head"] if sil_train_embeddings else ["lm_head"]

        logger.info("Enabling LoRA adapters...")
        lora_cfg = resolve_lora_config(
            task_type, model_id, lora_profile=lora_profile, modules_to_save=modules_to_save,
        )
        logger.info(
            "LoRA config: r=%s, alpha=%s, dropout=%s, profile=%s, target_modules=%s, modules_to_save=%s",
            lora_cfg["r"], lora_cfg["lora_alpha"], lora_cfg["lora_dropout"], lora_profile,
            lora_cfg["target_modules"], lora_cfg.get("modules_to_save"),
        )
        model.enable_lora(lora_cfg)
        logger.info("LoRA enabled successfully")

        # Get trainable parameters
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")

        # Prepare training config
        train_config = TRAINING_CONFIG.copy()
        train_config.update(training_kwargs)
        train_config["output_dir"] = str(checkpoint_dir)
        train_config["seed"] = seed

        # Quantized runs benefit from bitsandbytes optimizer when available.
        if (load_in_4bit or load_in_8bit) and "optim" not in train_config:
            train_config["optim"] = "paged_adamw_8bit"

        # If eval is disabled for OOM mitigation, avoid loading best model at end.
        if eval_data is None:
            train_config["eval_strategy"] = "no"
            train_config["load_best_model_at_end"] = False
            train_config.pop("metric_for_best_model", None)
            train_config.pop("greater_is_better", None)
            train_config.pop("eval_steps", None)

        # Set device-specific flags
        if torch.cuda.is_available():
            if "fp16" not in train_config and "bf16" not in train_config:
                # Prefer bf16 when the GPU supports it: fp16's narrow range makes
                # Gemma overflow to inf/nan grads. bf16 has fp32's exponent range.
                if torch.cuda.is_bf16_supported():
                    train_config["bf16"] = True
                else:
                    train_config["fp16"] = True
            logger.info("Using CUDA training")
        else:
            logger.warning("CUDA not available; using CPU training (slower)")

        # Enable gradient checkpointing when requested to reduce activation memory.
        if train_config.get("gradient_checkpointing", False):
            if hasattr(model.model, "gradient_checkpointing_enable"):
                model.model.gradient_checkpointing_enable()
            if hasattr(model.model, "config") and hasattr(model.model.config, "use_cache"):
                model.model.config.use_cache = False

        # Fine-tune
        logger.info(f"Starting training with {len(train_data)} training examples...")
        if eval_data:
            logger.info(f"Using {len(eval_data)} eval examples")

        trainer = model.fine_tune(
            train_data=train_data,
            eval_data=eval_data,
            **train_config,
        )

        # Save final model
        logger.info(f"Saving final model to {final_model_dir}...")
        model.save(str(final_model_dir))

        # Record the decoding-relevant recipe next to the adapter so evaluate.py
        # scores it with the SAME prompt style / vocab it was trained on (can't
        # silently mix a chat-trained adapter with plain-prompt inference).
        run_config = {
            "model_key": model_key,
            "model_id": model_id,
            "prompt_style": prompt_style,
            "added_tokens": added_tokens or [],
            "lora_profile": lora_profile,
            "sil_train_embeddings": bool(add_sil_token and sil_train_embeddings),
        }
        try:
            with open(final_model_dir / "run_config.json", "w") as f:
                json.dump(run_config, f, indent=2)
        except Exception as exc:
            logger.warning("Could not write run_config.json: %s", exc)

        # Gather metrics
        metrics = {
            "model_key": model_key,
            "model_id": model_id,
            "task_type": task_type,
            "output_dir": str(final_model_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "n_train": len(train_data),
            "n_eval": len(eval_data) if eval_data else 0,
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "lora_config": lora_cfg,
            "training_config": train_config,
        }

        # Try to extract best metrics from trainer
        if hasattr(trainer, "state") and hasattr(trainer.state, "best_metric"):
            metrics["best_eval_loss"] = trainer.state.best_metric
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
            metrics["log_history"] = trainer.state.log_history[-5:]  # last 5 log entries

        logger.info(f"✓ Training complete for {model_key}")
        return metrics

    except Exception as exc:
        logger.exception(f"Error training {model_key}: {exc}")
        return {
            "model_key": model_key,
            "model_id": model_id,
            "error": str(exc),
        }
    finally:
        # Always release references between runs to avoid cumulative VRAM growth.
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        _cleanup_cuda_memory()


# ================================================================================
# Main Orchestration
# ================================================================================
def main(
        output_dir: str = "outputs/llm_runs",
        seed: int = 42,
        skip_models: Optional[List[str]] = None,
        max_models: Optional[int] = None,
        run_eval: bool = True,
        run_test_eval: bool = True,
        training_overrides: Optional[Dict[str, Any]] = None,
        quantization: str = "4bit",
        full_data_path: str = "data/cache/swbd_disfluency_full.jsonl",
        force_rebuild: bool = False,
        prompt_style: str = "plain",
        add_sil_token: bool = False,
        sil_train_embeddings: bool = False,
        lora_profile: str = "paper",
        enriched_only: bool = False,
        min_disfl_tokens: int = 0,
) -> None:
    """
    Main training orchestration.
    """
    skip_models = skip_models or []
    training_overrides = training_overrides or {}
    logger.info(f"Starting LLM fine-tuning with LoRA")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seed: {seed}")

    # Ensure Hugging Face token is available in environment
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error(
            "A Hugging Face token is required. Please set the environment variable HUGGINGFACE_HUB_TOKEN or HF_HUB_TOKEN."
        )
        raise SystemExit(1)

    if hf_login is None:
        logger.warning(
            "huggingface_hub is not available; cannot programmatically log in. "
            "If models require authentication the script may fail. Install huggingface_hub to enable automatic login."
        )
    else:
        try:
            hf_login(token=hf_token)
            logger.info("Logged in to Hugging Face hub using environment token")
        except Exception as exc:
            logger.exception("Failed to login to Hugging Face hub: %s", exc)
            raise

    # Load dataset (load with context globally so we have the option to use it)
    train_records, eval_records, test_records = load_disfluency_dataset(
        seed=seed,
        train_ratio=0.8,
        eval_ratio=0.1,
        test_ratio=0.1,
        add_context=True,
        context_k=3,
        full_data_path=full_data_path,
        force_rebuild=force_rebuild,
        enriched_only=enriched_only,
        min_disfl_tokens=min_disfl_tokens,
    )

    # Persist the canonical splits for this seed so evaluation scores every model
    # on the identical test rows (evaluate.py reads splits/test.jsonl).
    save_splits(str(Path(output_dir) / "splits"), train_records, eval_records, test_records)

    # # TEMP: tiny debug run
    # train_records = train_records[:30]
    # eval_records = eval_records[:5]
    # test_records = test_records[:5]

    # Load model registry
    registry = _load_model_registry()

    # Filter models
    models_to_train = [
        (key, model_id) for key, model_id in registry.items()
        if key not in skip_models
    ]
    if max_models:
        models_to_train = models_to_train[:max_models]

    results = []

    # Iterate over both context states
    for use_context in [False, True]:
        context_label = "with_context" if use_context else "no_context"
        logger.info(f"\n\n{'*' * 80}\nSTARTING RUN: {context_label.upper()}\n{'*' * 80}")

        # Prepare data pairs for the current context state
        train_data = prepare_training_data(train_records, use_context=use_context)
        eval_data = (
            prepare_training_data(eval_records, use_context=use_context)
            if run_eval and eval_records
            else None
        )

        # Create a specific output directory for this context state
        current_output_dir = str(Path(output_dir) / context_label)

        # Train each model
        for model_key, model_id in models_to_train:
            # We append the context label to the model key so we can distinguish them in logs/folders
            run_key = f"{model_key}_{context_label}"

            # Skip runs that already finished (final adapter saved), so re-launching
            # only trains what's missing instead of redoing completed models.
            final_model_dir = Path(current_output_dir) / run_key / "final_model"
            if (final_model_dir / "adapter_model.safetensors").exists():
                logger.info(f"⏭  Skipping {run_key}: already trained ({final_model_dir} exists)")
                continue

            from utils.models import _infer_task_type
            task_type = _infer_task_type(model_key, model_id)

            result = train_model(
                model_key=model_key,  # original config key for lookup
                model_id=model_id,
                train_data=train_data,
                eval_data=eval_data,
                output_dir=current_output_dir,
                task_type=task_type,
                seed=seed,
                run_name=run_key,
                quantization=quantization,
                prompt_style=prompt_style,
                add_sil_token=add_sil_token,
                sil_train_embeddings=sil_train_embeddings,
                lora_profile=lora_profile,
                **training_overrides,
            )
            # Add the context state to the results dictionary for tracking
            result["used_context"] = use_context
            results.append(result)

            # Generate + save test-set predictions with the just-trained model.
            # train_model has already freed its in-memory copy, so predict_for_model
            # reloads base+adapter fresh (avoids holding two models at once).
            if run_test_eval and "error" not in result and test_records:
                from evaluate import predict_for_model  # deferred: evaluate imports from this module
                adapter_dir = Path(current_output_dir) / run_key / "final_model"
                preds_path = Path(current_output_dir) / run_key / "test_predictions.jsonl"
                try:
                    predict_for_model(
                        model_key=model_key,
                        model_id=model_id,
                        adapter_dir=adapter_dir,
                        test_records=test_records,
                        use_context=use_context,
                        out_path=preds_path,
                        quantization=quantization,
                        prompt_style=prompt_style,
                        added_tokens=(["<sil>"] if add_sil_token else None),
                    )
                    result["test_predictions"] = str(preds_path)
                except Exception as exc:
                    logger.exception("Test prediction generation failed for %s: %s", run_key, exc)

    # Save results summary globally
    results_file = Path(output_dir) / "training_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    clean_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                clean[k] = v
            elif isinstance(v, dict):
                clean[k] = {kk: (str(vv) if not isinstance(vv, (str, int, float, bool, type(None))) else vv)
                            for kk, vv in v.items()}
            elif isinstance(v, list):
                clean[k] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item
                            for item in v]
            else:
                clean[k] = str(v)
        clean_results.append(clean)

    with open(results_file, "w") as f:
        json.dump(clean_results, f, indent=2)

    logger.info(f"\n{'=' * 80}")
    logger.info("Training Summary")
    logger.info(f"{'=' * 80}")
    for r in results:
        if "error" in r:
            logger.error(f"✗ {r['model_key']}: {r['error']}")
        else:
            logger.info(f"✓ {r['model_key']}: saved to {r['output_dir']}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs with LoRA on disfluency insertion task"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llm_runs",
        help="Root output directory for all models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-models",
        nargs="+",
        default=[],
        help="Model keys to skip (e.g., 'L2-7B-Chat' 'Flan-T5-L')"
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Maximum number of models to train (useful for testing)"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size (lower this first if CUDA OOM occurs)"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size (eval is forward-only, so a larger batch than "
             "train is safe and much faster; lower it if eval OOMs)"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training to reduce VRAM usage"
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--no-test-eval",
        action="store_true",
        help="Skip generating test-set predictions (test_predictions.jsonl) after each model trains"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 mixed-precision training"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bf16 mixed-precision training (supported GPUs only)"
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Model loading quantization mode"
    )
    parser.add_argument(
        "--full-data-path",
        default="data/cache/swbd_disfluency_full.jsonl",
        help="Path to the full built-dataset JSONL (loaded if present, else built and saved here)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore any existing cache and rebuild the dataset from raw sources"
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Restrict training to specific GPUs by setting CUDA_VISIBLE_DEVICES, "
             "e.g. '0' for one GPU or '0,2' for two. Controls both which and how many "
             "GPUs device_map='auto' can shard across. Omit to use all visible GPUs."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=384,
        help="Maximum source/prompt token length used in tokenization"
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Maximum target token length used in tokenization"
    )
    parser.add_argument(
        "--prompt-style",
        choices=["plain", "chat"],
        default="plain",
        help="Prompt format: 'plain' (hand-rolled scaffold, default/current behavior) "
             "or 'chat' (the model's native chat template; causal instruct models only)"
    )
    parser.add_argument(
        "--add-sil-token",
        action="store_true",
        help="Register '<sil>' as a single atomic token + resize embeddings, and train "
             "lm_head so the model can emit it (VRAM: adds a full trainable output head)"
    )
    parser.add_argument(
        "--sil-train-embeddings",
        action="store_true",
        help="With --add-sil-token, also train input embeddings (heavier VRAM; default "
             "trains only lm_head)"
    )
    parser.add_argument(
        "--lora-profile",
        choices=["paper", "wide"],
        default="paper",
        help="'paper' = q/v only (default/reproduction); 'wide' = adds k/o + MLP "
             "projections for more capacity (causal models only)"
    )
    parser.add_argument(
        "--enriched-only",
        action="store_true",
        help="Train only on annotation-complete (NXT-aligned) rows; drops rows "
             "missing typical <sil>/filled-pause annotation. Non-enriched rows are "
             "still used as context. Default off = original full dataset."
    )
    parser.add_argument(
        "--min-disfl-tokens",
        type=int,
        default=0,
        help="Drop training rows with fewer than N disfluency tokens (0 = keep all, "
             "the original behavior; 1 = drop the ~30%% pure-copy rows)"
    )

    args = parser.parse_args()

    # Must run before any CUDA initialization (i.e. before main() touches torch.cuda
    # or loads a model) so device_map='auto' only sees the requested GPUs.
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Restricting to GPUs via CUDA_VISIBLE_DEVICES={args.gpus}")

    training_overrides = {
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "max_length": args.max_length,
        "max_target_length": args.max_target_length,
    }
    if args.fp16 and args.bf16:
        raise SystemExit("Please choose only one of --fp16 or --bf16")
    if args.fp16:
        training_overrides["fp16"] = True
        training_overrides["bf16"] = False
    elif args.bf16:
        training_overrides["bf16"] = True
        training_overrides["fp16"] = False

    main(
        output_dir=args.output_dir,
        seed=args.seed,
        skip_models=args.skip_models,
        max_models=args.max_models,
        run_eval=not args.no_eval,
        run_test_eval=not args.no_test_eval,
        training_overrides=training_overrides,
        quantization=args.quantization,
        full_data_path=args.full_data_path,
        force_rebuild=args.force_rebuild,
        prompt_style=args.prompt_style,
        add_sil_token=args.add_sil_token,
        sil_train_embeddings=args.sil_train_embeddings,
        lora_profile=args.lora_profile,
        enriched_only=args.enriched_only,
        min_disfl_tokens=args.min_disfl_tokens,
    )
