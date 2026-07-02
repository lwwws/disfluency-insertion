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
    - Batch size: 2
    - Gradient accumulation steps: 4
    - Learning rate: 2e-4
    - Save strategy: steps (every 500 steps)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
from utils.models import BaseDisfluencyModel, _load_model_registry
from utils.swbd_disfluency_loader import SwbdDisfluencyDataset, Config as DataConfig

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


# ================================================================================
# Training Configuration
# ================================================================================
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,  # simulates batch size of 8
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
    "save_strategy": "steps",
    "eval_strategy": "steps",
    "eval_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "seed": 42,
}


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
) -> tuple:
    """
    Load the Switchboard disfluency dataset and return train/eval/test splits.

    Args:
        seed: Random seed for reproducibility
        train_ratio, eval_ratio, test_ratio: Split ratios (must sum to ~1.0)
        min_len: Minimum number of tokens for a record to be kept
        add_context: If True, add previous context turns to each record
        context_k: Number of context turns to include

    Returns:
        (train_records, eval_records, test_records) where each is a list of dicts
        Each dict has keys: 'fluent', 'disfluent', 'context' (if add_context=True), etc.
    """
    logger.info("Loading Switchboard disfluency dataset...")

    cfg = DataConfig()
    dataset = SwbdDisfluencyDataset(cfg).build()

    if add_context:
        logger.info(f"Adding context (k={context_k} turns)...")
        dataset.add_context(k=context_k, source="fluent", include_other_speaker=True)

    # Filter by minimum length
    logger.info(f"Filtering records with minimum length {min_len}...")
    dataset.filter_records(by="fluent", min_len=min_len)

    # Add splits
    logger.info(f"Splitting data: train={train_ratio}, eval={eval_ratio}, test={test_ratio}...")
    dataset.add_splits(
        ratios=(train_ratio, eval_ratio, test_ratio),
        seed=seed,
        by="conv",  # split by conversation to avoid speaker leakage
    )

    # Partition by split
    train_records = [r for r in dataset.records if r.get("split") == "train"]
    eval_records = [r for r in dataset.records if r.get("split") == "eval"]
    test_records = [r for r in dataset.records if r.get("split") == "test"]

    logger.info(f"Dataset splits: train={len(train_records)}, eval={len(eval_records)}, test={len(test_records)}")

    return train_records, eval_records, test_records


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

        if use_context and "context" in r:
            context = r.get("context", "").strip()
            if context:
                input_text = f"{context}\n\nCurrent: {fluent}"
            else:
                input_text = fluent
        else:
            input_text = fluent

        pairs.append((input_text, disfluent))

    logger.info(f"Prepared {len(pairs)} training pairs")
    return pairs


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
        **training_kwargs: Additional training arguments (learning_rate, num_epochs, etc.)

    Returns:
        Dict with training results including model name, best metrics, etc.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_key} ({model_id})")
    logger.info(f"{'='*80}")

    output_path = Path(output_dir) / model_key
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    final_model_dir = output_path / "final_model"

    try:
        # Load model
        logger.info(f"Loading model {model_key}...")
        model = BaseDisfluencyModel.from_config(model_key)
        model.load()
        logger.info(f"Model loaded successfully. Task type: {model.task_type}")

        # Enable LoRA
        logger.info("Enabling LoRA adapters...")
        lora_cfg = LORA_CONFIGS[task_type]
        logger.info(f"LoRA config: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, dropout={lora_cfg['lora_dropout']}")
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

        # Set device-specific flags
        if torch.cuda.is_available():
            train_config["fp16"] = True
            logger.info("Using FP16 training (CUDA available)")
        else:
            logger.warning("CUDA not available; using CPU training (slower)")

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


# ================================================================================
# Main Orchestration
# ================================================================================
def main(
        output_dir: str = "outputs/llm_runs",
        seed: int = 42,
        skip_models: Optional[List[str]] = None,
        max_models: Optional[int] = None,
) -> None:
    """
    Main training orchestration.
    """
    skip_models = skip_models or []
    logger.info(f"Starting LLM fine-tuning with LoRA")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seed: {seed}")

    # Load dataset (load with context globally so we have the option to use it)
    train_records, eval_records, test_records = load_disfluency_dataset(
        seed=seed,
        train_ratio=0.8,
        eval_ratio=0.1,
        test_ratio=0.1,
        add_context=True,
        context_k=3,
    )

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
        eval_data = prepare_training_data(eval_records, use_context=use_context) if eval_records else None

        # Create a specific output directory for this context state
        current_output_dir = str(Path(output_dir) / context_label)

        # Train each model
        for model_key, model_id in models_to_train:
            # We append the context label to the model key so we can distinguish them in logs/folders
            run_key = f"{model_key}_{context_label}"

            from utils.models import _infer_task_type
            task_type = _infer_task_type(model_key, model_id)

            result = train_model(
                model_key=run_key,  # Use the distinct run key
                model_id=model_id,
                train_data=train_data,
                eval_data=eval_data,
                output_dir=current_output_dir,
                task_type=task_type,
                seed=seed,
            )
            # Add the context state to the results dictionary for tracking
            result["used_context"] = use_context
            results.append(result)

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

    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        seed=args.seed,
        skip_models=args.skip_models,
        max_models=args.max_models,
    )

