"""Model utilities for disfluency insertion experiments.

This module provides a lightweight wrapper around Hugging Face models for:
- loading models and tokenizers
- constructing prompts
- generating text
- supervised fine-tuning
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from transformers import (
	AutoModelForCausalLM,
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	Trainer,
	TrainingArguments,
	default_data_collator,
)

from abc import ABC, abstractmethod


def _load_model_registry(config_path: Optional[Path] = None) -> Dict[str, str]:
	"""Load model mappings from config.yaml. For example: {"Flan-T5-L": "google/flan-t5-large", ...}"""
	try:
		import yaml
	except ImportError:
		return {}

	path = config_path or (Path(__file__).resolve().parent.parent / "config.yaml")
	if not path.exists():
		return {}

	try:
		parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
		models = parsed.get("models", {}) if isinstance(parsed, dict) else {}
		return {str(k): str(v) for k, v in models.items()} if isinstance(models, dict) else {}
	except Exception:
		return {}


def _infer_task_type(model_name: str, model_id: str, explicit_task: Optional[str] = None) -> str:
	"""Infer whether model is seq2seq or causal."""
	if explicit_task:
		task = explicit_task.lower().strip()
		if task not in {"causal", "seq2seq"}:
			raise ValueError(f"task_type must be 'causal' or 'seq2seq', got {explicit_task!r}")
		return task

	joined = f"{model_name} {model_id}".lower()
	return "seq2seq" if any(t in joined for t in ("t5", "seq2seq", "flan")) else "causal"


def _ensure_dependencies() -> None:
	"""Raise if torch or transformers tokenizer is missing.

	This is a minimal check used before operations that require HF/torch.
	"""
	missing = []
	if torch is None:
		missing.append("torch")
	try:
		# AutoTokenizer should be available from transformers import above
		_ = AutoTokenizer
	except Exception:
		missing.append("transformers")
	if missing:
		raise ImportError(
			"Required dependencies are missing for this operation: "
			+ ", ".join(missing)
			+ ". Install them before loading or training a model."
		)


class BaseDisfluencyModel(ABC):
	"""Abstract base class for disfluency models.

	Concrete subclasses must implement ``_load_model`` and ``_post_generate`` to
	customize loading/generation behavior for seq2seq vs causal families.
	"""

	def __init__(
		self,
		model_name: str,
		model_id: Optional[str] = None,
		task_type: Optional[str] = None,
		config_path: Optional[str] = None,
		system_prompt: str = (
			"You are an expert assistant that rewrites fluent text into natural "
			"disfluent speech while preserving meaning."
		),
		auto_load: bool = False,
		load_in_4bit: bool = False,
		load_in_8bit: bool = False,
		device_map: Optional[str] = None,
		torch_dtype: Any = None,
	) -> None:
		self.model_name = model_name
		self.model_id = model_id
		self.task_type = task_type
		self.config_path = config_path
		self.system_prompt = system_prompt
		self.auto_load = auto_load
		self.load_in_4bit = load_in_4bit
		self.load_in_8bit = load_in_8bit
		self.device_map = device_map
		self.torch_dtype = torch_dtype

		self.tokenizer: Any = None
		self.model: Any = None

		# LoRA/PEFT state
		self._peft_enabled = False
		self._peft_config: Optional[dict] = None

		registry = _load_model_registry(Path(self.config_path) if self.config_path else None)
		resolved_id = self.model_id or registry.get(self.model_name) or self.model_name
		self.model_id = resolved_id
		self.task_type = _infer_task_type(self.model_name, resolved_id, self.task_type)

		if self.auto_load:
			self.load()

	@classmethod
	def from_config(cls, model_key: str, **kwargs: Any) -> "BaseDisfluencyModel":
		"""Factory: choose the correct concrete subclass based on model metadata."""

		registry = _load_model_registry(Path(kwargs.get("config_path") or ""))
		model_id = kwargs.get("model_id") or registry.get(model_key) or model_key
		task = _infer_task_type(model_key, model_id, kwargs.get("task_type"))
		if task == "seq2seq":
			return Seq2SeqDisfluencyModel(model_key, **kwargs)
		return CausalDisfluencyModel(model_key, **kwargs)

	@property
	def is_loaded(self) -> bool:
		return self.model is not None and self.tokenizer is not None

	def _ensure_loaded(self) -> None:
		if not self.is_loaded:
			raise RuntimeError("Model not loaded. Call .load() first.")

	def load(self) -> "BaseDisfluencyModel":
		"""Load tokenizer and model; concrete model creation delegated to subclass."""
		_ensure_dependencies()

		model_id = str(self.model_id)
		self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

		if not getattr(self.tokenizer, "pad_token", None):
			self.tokenizer.pad_token = (
				getattr(self.tokenizer, "eos_token", None) or getattr(self.tokenizer, "unk_token", None)
			)

		model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
		if self.device_map:
			model_kwargs["device_map"] = self.device_map  # type: ignore
		if self.load_in_4bit:
			model_kwargs["load_in_4bit"] = True
		if self.load_in_8bit:
			model_kwargs["load_in_8bit"] = True
		if self.torch_dtype is not None:
			model_kwargs["torch_dtype"] = self.torch_dtype

		# Delegate constructing the HF model to the concrete class
		self.model = self._load_model(model_id, model_kwargs)

		if hasattr(self.model, "config") and self.tokenizer.pad_token_id:
			self.model.config.pad_token_id = self.tokenizer.pad_token_id

		if not self.device_map and torch is not None:
			try:
				self.model = self.model.to(self.device)
			except Exception:
				pass

		try:
			self.model.eval()
		except Exception:
			pass

		return self

	@property
	def device(self) -> Any:
		if self.model is not None:
			try:
				return next(self.model.parameters()).device
			except Exception:
				pass
		return "cuda" if torch and torch.cuda.is_available() else "cpu"

	def _format_examples(self, examples: Optional[Sequence[Mapping[str, str]]]) -> str:
		if not examples:
			return ""
		lines: list[str] = []
		for ex in examples:
			inp = str(ex.get("input", "")).strip()
			if inp:
				lines.append(f"Input: {inp}\nOutput: {ex.get('output', '')}\n")
		return "".join(lines).strip()

	def build_prompt(
		self,
		input_text: str,
		*,
		instruction: Optional[str] = None,
		examples: Optional[Sequence[Mapping[str, str]]] = None,
		system_prompt: Optional[str] = None,
	) -> str:
		instruction = instruction or "Insert natural disfluencies while preserving the meaning of the text."
		system_prompt = system_prompt or self.system_prompt
		ex_block = self._format_examples(examples)

		if self.task_type == "seq2seq":
			parts = [instruction.strip()]
			if ex_block:
				parts.append("Examples:\n" + ex_block)
			parts.append(f"Input: {input_text.strip()}")
			return "\n\n".join(parts)

		parts = [f"System: {system_prompt.strip()}", f"User: {instruction.strip()}"]
		if ex_block:
			parts.append("Examples:\n" + ex_block)
		parts.append(f"Input: {input_text.strip()}\nAssistant:")
		return "\n\n".join(parts)

	def enable_lora(self, lora_config: dict) -> None:
		"""Enable LoRA/PEFT adapters for the currently loaded model.

		This method requires the `peft` package. Call after ``load()``. The
		provided ``lora_config`` should be compatible with ``peft.LoraConfig``.
		"""
		self._ensure_loaded()
		try:
			from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
		except Exception as exc:  # pragma: no cover - optional dependency
			raise ImportError("peft is required for LoRA support: pip install peft") from exc

		# If loaded in 4bit, prepare model for k-bit training
		if getattr(self, "load_in_4bit", False):
			try:
				self.model = prepare_model_for_kbit_training(self.model)
			except Exception:
				pass

		# Build a LoraConfig if a dict was passed
		if not isinstance(lora_config, dict):
			raise ValueError("lora_config must be a dict")

		cfg = LoraConfig(**lora_config)
		self.model = get_peft_model(self.model, cfg)
		self._peft_enabled = True
		self._peft_config = lora_config

	def is_peft(self) -> bool:
		return self._peft_enabled

	@abstractmethod
	def _load_model(self, model_id: str, model_kwargs: Dict[str, Any]) -> Any:
		raise NotImplementedError

	@abstractmethod
	def generate(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - implemented in subclasses
		raise NotImplementedError

	def fine_tune(
		self,
		train_dataset: Any,
		*,
		eval_dataset: Optional[Any] = None,
		output_dir: str = "outputs/disfluency_model",
		num_train_epochs: float = 3.0,
		per_device_train_batch_size: int = 4,
		learning_rate: float = 5e-5,
		warmup_steps: int = 100,
		logging_steps: int = 50,
		save_strategy: str = "epoch",
		**training_kwargs: Any,
	) -> Any:
		"""Fine-tune the model using a prepared dataset.

		Note: data tokenization/formatting is intentionally expected to be done
		by a separate dataloader module. ``train_dataset`` and ``eval_dataset``
		must be ready to be consumed by ``transformers.Trainer``.
		"""
		self._ensure_loaded()
		_ensure_dependencies()

		training_args = TrainingArguments(
			output_dir=output_dir,
			num_train_epochs=num_train_epochs,
			per_device_train_batch_size=per_device_train_batch_size,
			per_device_eval_batch_size=per_device_train_batch_size,
			learning_rate=learning_rate,
			warmup_steps=warmup_steps,
			logging_steps=logging_steps,
			save_strategy=save_strategy,
			evaluation_strategy="epoch" if eval_dataset is not None else "no",
			remove_unused_columns=False,
			report_to=[],
			**training_kwargs,
		)

		collator = (
			DataCollatorForSeq2Seq(self.tokenizer, self.model, padding=True)
			if self.task_type == "seq2seq"
			else default_data_collator
		)

		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			tokenizer=self.tokenizer,
			data_collator=collator,
		)

		trainer.train()

		# Save: if LoRA/PEFT is used, save adapters; Trainer.save_model saves full model
		try:
			trainer.save_model(output_dir)
		except Exception:
			pass

		# If peft is enabled, most peft models expose `save_pretrained` to save adapters
		if self._peft_enabled:
			try:
				self.model.save_pretrained(output_dir)
			except Exception:
				pass

		# Always save tokenizer
		try:
			self.tokenizer.save_pretrained(output_dir)
		except Exception:
			pass

		return trainer

	def save(self, output_dir: str) -> None:
		self._ensure_loaded()
		self.model.save_pretrained(output_dir)
		self.tokenizer.save_pretrained(output_dir)

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(name={self.model_name!r}, id={self.model_id!r}, type={self.task_type!r}, loaded={self.is_loaded}, peft={self._peft_enabled})"


class Seq2SeqDisfluencyModel(BaseDisfluencyModel):
	def _load_model(self, model_id: str, model_kwargs: Dict[str, Any]) -> Any:
		return AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)

	def generate(
		self,
		input_text: str,
		*,
		instruction: Optional[str] = None,
		examples: Optional[Sequence[Mapping[str, str]]] = None,
		system_prompt: Optional[str] = None,
		max_new_tokens: int = 128,
		temperature: float = 0.7,
		top_p: float = 0.9,
		do_sample: bool = True,
		num_beams: int = 1,
		**generate_kwargs: Any,
	) -> str:
		self._ensure_loaded()
		prompt = self.build_prompt(input_text, instruction=instruction, examples=examples, system_prompt=system_prompt)
		inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				top_p=top_p,
				do_sample=do_sample,
				num_beams=num_beams,
				**generate_kwargs,
			)
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class CausalDisfluencyModel(BaseDisfluencyModel):
	def _load_model(self, model_id: str, model_kwargs: Dict[str, Any]) -> Any:
		return AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

	def generate(
		self,
		input_text: str,
		*,
		instruction: Optional[str] = None,
		examples: Optional[Sequence[Mapping[str, str]]] = None,
		system_prompt: Optional[str] = None,
		max_new_tokens: int = 128,
		temperature: float = 0.7,
		top_p: float = 0.9,
		do_sample: bool = True,
		num_beams: int = 1,
		**generate_kwargs: Any,
	) -> str:
		self._ensure_loaded()
		prompt = self.build_prompt(input_text, instruction=instruction, examples=examples, system_prompt=system_prompt)

		# If tokenizer supports chat templates later, callers can pre-format. For
		# now we simply send the assembled prompt and slice off prompt tokens.
		inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
		input_len = inputs["input_ids"].shape[-1]
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				top_p=top_p,
				do_sample=do_sample,
				num_beams=num_beams,
				**generate_kwargs,
			)
		return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

