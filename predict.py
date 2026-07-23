"""Generate disfluency predictions from a FINE-TUNED model for arbitrary text.

Interactive/ad-hoc counterpart to ``evaluate.py`` (which only runs the fixed test
split). Load a fine-tuned adapter ONCE, then rewrite any fluent sentence - with or
without context, greedy or sampled at any temperature.

Notebook / import:
    from predict import FinetunedPredictor
    p = FinetunedPredictor("L3.1-8B-Instruct", context_mode="with_context")
    p.predict("i listen to it all the time in my car",
              context="A: what kind of music do you like B: i mostly listen to popular music")
    p.predict("i listen to it all the time in my car", do_sample=True, temperature=1.0, n=3)

CLI:
    python predict.py --model L3.1-8B-Instruct --context-mode with_context \
        --text "i listen to it all the time in my car" \
        --context "A: what kind of music do you like B: i mostly listen to popular music"
    python predict.py --model L3.1-8B-Instruct --text "..." --sample --temperature 1.0 --num-samples 3

Notes
-----
* This is the FINE-TUNED path: it feeds ``build_input_text`` (the exact
  ``context\\n\\nCurrent: {fluent}`` convention the models were trained on) and reads
  each adapter's ``run_config.json`` for the right ``prompt_style``/``added_tokens``.
  Do NOT mix in the prompt-engineering few-shot machinery - that's for the untrained
  baseline (see prompt_eng.py).
* ``context_mode`` selects WHICH adapter loads and whether context is used; the
  ``with_context`` model was trained to read context, the ``no_context`` one was not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from dotenv import load_dotenv

from utils.models import BaseDisfluencyModel, _load_model_registry, _infer_task_type
from llm_training import build_input_text

try:  # reuse HF login so gated base models can download; harmless if unavailable
    from evaluate import _hf_login
except Exception:  # pragma: no cover
    def _hf_login() -> None:  # type: ignore
        pass

load_dotenv()


class FinetunedPredictor:
    """A loaded fine-tuned (base + LoRA) model for on-demand disfluency insertion.

    Construct once (loads the base model + adapter), then call ``predict`` freely.
    """

    def __init__(
        self,
        model_key: str,
        *,
        context_mode: str = "with_context",
        output_dir: str = "outputs/llm_runs_enriched",
        quantization: str = "4bit",
    ) -> None:
        if context_mode not in ("with_context", "no_context"):
            raise ValueError("context_mode must be 'with_context' or 'no_context'")
        self.model_key = model_key
        self.context_mode = context_mode
        self.use_context = context_mode == "with_context"

        adapter = Path(output_dir) / context_mode / f"{model_key}_{context_mode}" / "final_model"
        if not adapter.is_dir():
            raise FileNotFoundError(
                f"No fine-tuned adapter at {adapter}. Check --model/--context-mode/--output-dir."
            )
        self.adapter_dir = adapter

        cfg: dict = {}
        rc = adapter / "run_config.json"
        if rc.exists():
            cfg = json.loads(rc.read_text())
        model_id = cfg.get("model_id") or _load_model_registry().get(model_key, model_key)

        _hf_login()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = BaseDisfluencyModel.from_config(
            model_key,
            model_id=model_id,
            task_type=_infer_task_type(model_key, model_id),
            device_map="auto",
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            torch_dtype=dtype,
            prompt_style=cfg.get("prompt_style", "plain"),
            added_tokens=cfg.get("added_tokens") or None,
        )
        self.model.load()
        self.model.load_adapter(str(adapter))  # reloads adapter tokenizer + resizes embeddings

    # ------------------------------------------------------------------
    def _input(self, fluent: str, context: Optional[str]) -> str:
        return build_input_text(
            {"fluent": fluent, "context": context or ""}, use_context=self.use_context
        )

    def predict(
        self,
        fluent: str,
        context: Optional[str] = None,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        n: int = 1,
        max_new_tokens: int = 128,
        seed: int = 42,
    ) -> Union[str, List[str]]:
        """Disfluent rewrite of ``fluent``.

        Returns a single string, or a list of ``n`` drafts when ``do_sample`` and
        ``n > 1``. ``context`` is used only if this predictor was loaded with
        ``context_mode='with_context'``. Greedy (default) is deterministic.
        """
        torch.manual_seed(seed)
        k = n if do_sample else 1
        outs = self.model.generate(
            [self._input(fluent, context)] * k,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            max_new_tokens=max_new_tokens,
        )
        return outs if (do_sample and n > 1) else outs[0]

    def predict_batch(
        self,
        fluents: Sequence[str],
        contexts: Optional[Sequence[Optional[str]]] = None,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128,
        seed: int = 42,
    ) -> List[str]:
        """One prediction per fluent sentence (single draw each), generated as a batch."""
        contexts = list(contexts) if contexts is not None else [None] * len(fluents)
        torch.manual_seed(seed)
        return self.model.generate(
            [self._input(f, c) for f, c in zip(fluents, contexts)],
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            max_new_tokens=max_new_tokens,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Disfluency prediction from a fine-tuned model")
    ap.add_argument("--model", required=True, help="Model key, e.g. L3.1-8B-Instruct")
    ap.add_argument("--context-mode", default="with_context", choices=["with_context", "no_context"])
    ap.add_argument("--output-dir", default="outputs/llm_runs_enriched")
    ap.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--text", required=True, help="Fluent sentence to rewrite")
    ap.add_argument("--context", default=None, help="Prior turns (used only with --context-mode with_context)")
    ap.add_argument("--sample", action="store_true", help="Sample instead of greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-samples", type=int, default=1, help="Drafts when --sample is set")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p = FinetunedPredictor(
        args.model, context_mode=args.context_mode,
        output_dir=args.output_dir, quantization=args.quantization,
    )
    out = p.predict(
        args.text, context=args.context,
        do_sample=args.sample, temperature=args.temperature, top_p=args.top_p,
        n=args.num_samples, max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    if isinstance(out, list):
        for i, o in enumerate(out):
            print(f"[draft {i}] {o}")
    else:
        print(out)


if __name__ == "__main__":
    main()
