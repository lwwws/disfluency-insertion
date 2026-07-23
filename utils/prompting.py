"""Few-shot example selection + prompt preview for the prompt-engineering baseline.

This is the *reviewable* half of the prompt-engineering experiment: it holds the
convention-describing instruction, loads the curated example pool
(``prompt_eng/examples.yaml``), selects a diverse shot suite, and can render the
EXACT prompt a model will see — so the example suite can be eyeballed before any
GPU time is spent.

Pure/CPU-only (no model weights): ``preview`` optionally loads just a tokenizer
to render the native chat template. Nothing here imports torch.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_POOL = _REPO_ROOT / "prompt_eng" / "examples.yaml"

SHOTS = ("zero", "one", "few")
CONTEXTS = {"no": False, "with": True}

# All five categories the dataset distinguishes (utils/swbd_disfluency_loader.py).
ALL_TYPES = ("filled_pause", "silent_pause", "reparandum", "fragment", "insertion")

# The instruction describes the OUTPUT CONVENTION explicitly. Without this a
# zero-shot model has no way to know it should emit the literal "<sil>" token or
# which markers count as disfluencies. Wording is faithful to the taxonomy in
# utils/swbd_disfluency_loader.py (filler_words / insertion_words / reparandum).
INSTRUCTION = (
    "Your task: take the sentence on the line marked \"Fluent:\" and rewrite it as "
    "natural spontaneous speech by inserting disfluencies — the fillers, pauses, "
    "repetitions and repairs people produce when speaking.\n"
    "- Rewrite ONLY the \"Fluent:\" sentence. If a \"Context:\" block is present it is "
    "the preceding conversation, given as background ONLY — do not rewrite it.\n"
    "- Keep every original word of the Fluent sentence, in the same order, and "
    "preserve the meaning. Only ADD disfluencies between the existing words; never "
    "delete, replace, or reorder them, and do not add content of your own.\n"
    "Disfluencies you may insert:\n"
    "- Filled pauses: uh, um, eh, ah, hm, er.\n"
    "- Silent pauses: written as the literal token <sil>.\n"
    "- Repetitions and repairs: repeat or restart a word or short phrase "
    "(e.g. \"to <sil> to return\", \"no <sil> no no real\"); a repair may start a "
    "word, cut it off, and then say it in full (e.g. \"wro- <sil> he wrote\").\n"
    "- Insertions: the discourse markers well, like, so.\n"
    "Output ONLY the rewritten disfluent sentence — no \"Fluent:\" label, no "
    "explanation, nothing else."
)


# ----------------------------------------------------------------------------
# Pool loading
# ----------------------------------------------------------------------------
def load_pool(path: str | Path = _DEFAULT_POOL) -> List[Dict[str, Any]]:
    """Load the curated example pool from YAML (list under the ``examples`` key)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    examples = data.get("examples", []) if isinstance(data, dict) else data
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return list(examples)


# ----------------------------------------------------------------------------
# Input formatting (mirror of llm_training.build_input_text, kept dependency-free)
# ----------------------------------------------------------------------------
def format_input(fluent: str, context: Optional[str], with_context: bool) -> str:
    """Build the LABELLED model-facing input string for one row.

    The sentence to rewrite is marked ``Fluent:`` and (when using context) the
    preceding turns are put under a ``Context:`` header, so the demonstrations and
    the final query both read as "here is a fluent sentence (with some background)
    -> rewrite it". The same helper formats the few-shot example inputs AND the
    live test-row query, so they always match. Output is the (unlabelled) disfluent
    rewrite, produced by the assistant turn.
    """
    fluent = (fluent or "").strip()
    if with_context and context and context.strip():
        return f"Context:\n{context.strip()}\n\nFluent: {fluent}"
    return f"Fluent: {fluent}"


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
def _entry_to_shot(entry: Mapping[str, Any], with_context: bool) -> Dict[str, str]:
    """Map a pool entry to the {id, input, output} shape build_prompt consumes."""
    return {
        "id": str(entry.get("id", "")),
        "input": format_input(entry.get("fluent", ""), entry.get("context"), with_context),
        "output": str(entry.get("disfluent", "")).strip(),
    }


def select(
    pool: Sequence[Mapping[str, Any]],
    shot: str,
    with_context: bool,
    *,
    few_k: Optional[int] = None,
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Return the examples for a (shot, context) condition.

    examples.yaml IS the few-shot suite; flags choose what each condition uses:

    - ``zero`` -> ``[]``  (caller passes ``examples=None`` to build_prompt).
    - ``few``  -> every entry EXCEPT those tagged ``few_shot: false`` (file order,
      optionally capped by ``few_k``). By default every entry is a few-shot member.
    - ``one``  -> the entry tagged ``one_shot: true`` (first wins; else pool[0]).
    """
    if shot not in SHOTS:
        raise ValueError(f"shot must be one of {SHOTS}, got {shot!r}")
    if shot == "zero":
        return []
    if shot == "one":
        rep = next((e for e in pool if e.get("one_shot")), pool[0])
        return [_entry_to_shot(rep, with_context)]
    # few: all entries not explicitly excluded (cap only if the caller asks)
    chosen = [e for e in pool if e.get("few_shot", True) is not False]
    if few_k is not None:
        chosen = chosen[:few_k]
    return [_entry_to_shot(e, with_context) for e in chosen]


def pool_summary(pool_path: str | Path = _DEFAULT_POOL) -> None:
    """Print the few-shot suite; mark the one-shot pick — for review before running."""
    pool = load_pool(pool_path)
    one = [e["id"] for e in pool if e.get("one_shot")]
    print(f"few-shot suite = all {len(pool)} entries | one-shot = {one}\n")
    for e in pool:
        tag = "[ONE-SHOT]" if e.get("one_shot") else "[  few   ]"
        print(f"{tag} {e['id']:16} {e.get('length','?'):5} "
              f"{','.join(e.get('types', []))}")


def example_ids(examples: Sequence[Mapping[str, Any]]) -> List[str]:
    """Ids of the selected examples, for run_config provenance."""
    return [str(e.get("id", "")) for e in examples]


# ----------------------------------------------------------------------------
# Candidate browsing (for curating / expanding examples.yaml)
# ----------------------------------------------------------------------------
# Backchannels: if these are the ENTIRE last context turn, the context carries no
# real information, so such rows are skipped when require_meaningful_context=True.
BACKCHANNEL = {
    "um-hum", "uh-huh", "yeah", "okay", "right", "mm-hmm", "uh", "oh", "huh",
    "um", "sure", "yes", "no", "hum", "well", "mhm", "mm", "hmm", "ok",
}

# Length buckets (fluent-token counts). "short" ~ the train median (14).
LENGTH_BUCKETS = {"short": (7, 13), "long": (16, 24)}


def _last_context_turn(record: Mapping[str, Any]) -> str:
    turns = record.get("context_turns") or []
    return str(turns[-1].get("text", "")).strip() if turns else ""


def _context_is_meaningful(record: Mapping[str, Any], min_words: int = 4) -> bool:
    """True if the most recent context turn is substantive (not a backchannel)."""
    last = _last_context_turn(record).lower()
    return len(last.split()) >= min_words and last not in BACKCHANNEL


def _record_to_candidate(record: Mapping[str, Any], length: str) -> Dict[str, Any]:
    """Shape a train row into a copy-ready examples.yaml-style candidate entry."""
    ctx_turns = (record.get("context_turns") or [])[-2:]
    return {
        "id": f"cand_{record.get('conv')}_{record.get('turn')}_{record.get('sent_num')}",
        "length": length,
        "fluent": record.get("fluent"),
        "disfluent": record.get("disfluent"),
        "context": " ".join(f"{t.get('speaker')}: {t.get('text')}" for t in ctx_turns),
        "types": sorted(t for t in set(record.get("disfl_labels") or []) if t),
        "n_tokens_fluent": record.get("n_tokens_fluent"),
    }


def _tokens_no_sil(record: Mapping[str, Any]) -> List[str]:
    return [t for t in (record.get("disfluent_tokens") or []) if t != "<sil>"]


def _has_completed_fragment(record: Mapping[str, Any]) -> bool:
    """A cut-off word ('ad-') whose stem is completed by a nearby later word
    ('adequate'). Such fragments read clearly as false starts, not truncations."""
    toks = _tokens_no_sil(record)
    for i, t in enumerate(toks):
        if t.endswith("-") and len(t) >= 3:
            stem = t[:-1].lower()
            for j in range(i + 1, min(i + 4, len(toks))):
                nxt = toks[j].lower()
                if nxt.startswith(stem) and not nxt.endswith("-") and len(nxt) > len(stem):
                    return True
    return False


def _repeat_bonus(record: Mapping[str, Any]) -> int:
    """How clearly a row shows a repetition. A repeat with a silent pause between
    the copies ('to <sil> to') reads most naturally as a spoken disfluency (=2);
    a bare adjacent doubling ('was was') is clear but can look like a typo (=1);
    otherwise 0."""
    raw = record.get("disfluent_tokens") or []
    for i in range(len(raw) - 2):
        w = raw[i].lower()
        if raw[i + 1] == "<sil>" and w == raw[i + 2].lower() and w.isalpha() and len(w) > 1:
            return 2
    toks = _tokens_no_sil(record)
    for i in range(len(toks) - 1):
        w = toks[i].lower()
        if w == toks[i + 1].lower() and w.isalpha() and len(w) > 1:
            return 1
    return 0


def _clarity_score(record: Mapping[str, Any], dtype: str) -> tuple:
    """Rank key (higher first) for how cleanly a row demonstrates ``dtype``.

    Type-specific bonus (does it show the phenomenon in its clearest form?), then
    a preference for FEWER co-occurring disfluency types (a focused example reads
    better than a busy one). Deterministic; ties fall back to mine order.
    """
    n_types = len(set(t for t in (record.get("disfl_labels") or []) if t))
    if dtype == "fragment":
        bonus = 1 if _has_completed_fragment(record) else 0
    elif dtype == "reparandum":
        bonus = _repeat_bonus(record)  # 2 = 'to <sil> to', 1 = 'was was'
    else:
        bonus = 0
    return (bonus, -n_types)


def mine_candidates(
    train_path: str | Path,
    *,
    n_per_cell: int = 4,
    lengths: Optional[Mapping[str, tuple]] = None,
    require_meaningful_context: bool = True,
    seed: int = 0,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Reproducibly mine copy-ready example candidates from the TRAIN split.

    This is exactly how the shipped ``examples.yaml`` pool was built. It groups
    real gold train rows into a TYPE x LENGTH grid and returns the ``n_per_cell``
    CLEANEST candidates per cell (ranked by ``_clarity_score`` — fragments that
    get completed, repetitions with an actual repeated word, and focused examples
    over busy ones), each with a meaningful (non-backchannel) context. So you get
    short AND long, clearly-demonstrated examples of every disfluency type. Copy
    the good ones into ``examples.yaml`` and flag them.

    Returns ``{type: {length: [candidate, ...]}}``. Never touches the test split.
    """
    length_ranges = dict(lengths or LENGTH_BUCKETS)
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng.shuffle(rows)  # deterministic under seed -> reproducible tie-breaks

    # Collect ALL qualifying rows per (type, length), keeping the source record so
    # we can rank by clarity, then keep the top n_per_cell cleanest per cell.
    pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        t: {L: [] for L in length_ranges} for t in ALL_TYPES
    }
    for r in rows:
        if r.get("n_context_turns", 0) < 1:
            continue
        if require_meaningful_context and not _context_is_meaningful(r):
            continue
        n = r.get("n_tokens_fluent", 999)
        length = next((L for L, (lo, hi) in length_ranges.items() if lo <= n <= hi), None)
        if length is None:
            continue
        for t in set(x for x in (r.get("disfl_labels") or []) if x):
            if t in pools:
                pools[t][length].append(r)

    grid: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        t: {L: [] for L in length_ranges} for t in ALL_TYPES
    }
    for t in ALL_TYPES:
        for L in length_ranges:
            ranked = sorted(pools[t][L], key=lambda r: _clarity_score(r, t), reverse=True)
            grid[t][L] = [_record_to_candidate(r, L) for r in ranked[:n_per_cell]]
    return grid


# ----------------------------------------------------------------------------
# Prompt preview (renders the EXACT prompt; CPU-only)
# ----------------------------------------------------------------------------
def build_examples_arg(
    pool: Sequence[Mapping[str, Any]], shot: str, with_context: bool, **kw: Any
) -> Optional[List[Dict[str, str]]]:
    """examples= argument for build_prompt/generate: None for zero-shot, else list."""
    ex = select(pool, shot, with_context, **kw)
    return ex or None


def render_prompt(
    model_key: str,
    input_text: str,
    examples: Optional[Sequence[Mapping[str, str]]],
    *,
    use_chat: bool = True,
    config_path: Optional[str] = None,
) -> str:
    """Render the exact prompt STRING for one input.

    ``use_chat=True`` loads only the tokenizer (no model weights, no GPU) and
    renders the native chat template — exactly what the runner sends. Set
    ``use_chat=False`` for a tokenizer-free view of the plain scaffold.
    """
    from utils.models import BaseDisfluencyModel, _load_model_registry

    registry = _load_model_registry(Path(config_path) if config_path else None)
    model_id = registry.get(model_key, model_key)
    m = BaseDisfluencyModel.from_config(
        model_key, model_id=model_id, prompt_style="chat" if use_chat else "plain",
        auto_load=False, config_path=config_path,
    )
    if not use_chat:
        return m.build_prompt(input_text, instruction=INSTRUCTION, examples=examples)

    from transformers import AutoTokenizer

    m.tokenizer = AutoTokenizer.from_pretrained(str(m.model_id), trust_remote_code=True)
    messages = m._chat_messages(input_text, instruction=INSTRUCTION, examples=examples)
    return m._apply_chat(messages, add_generation_prompt=True)


def preview(
    model_key: str,
    shot: str,
    with_context: bool,
    *,
    pool_path: str | Path = _DEFAULT_POOL,
    fluent: str = "i listen to it all the time in my car",
    context: str = "A: what kind of music do you like B: i mostly listen to popular music",
    few_k: int = 5,
    seed: int = 0,
    use_chat: bool = True,
    config_path: Optional[str] = None,
) -> str:
    """Print (and return) the exact prompt for a (model, shot, context) condition.

    Uses a sample fluent/context row so you can verify — before inference —
    that the instruction, the selected examples, `<sil>`, and (for with-context)
    the `context ... Current:` block all render correctly for this model.
    """
    pool = load_pool(pool_path)
    examples = build_examples_arg(pool, shot, with_context, few_k=few_k, seed=seed)
    ids = example_ids(examples or [])
    input_text = format_input(fluent, context, with_context)
    prompt = render_prompt(
        model_key, input_text, examples, use_chat=use_chat, config_path=config_path,
    )
    print(f"# model={model_key} shot={shot} context={'with' if with_context else 'no'} "
          f"examples={ids}\n{'=' * 78}")
    print(prompt)
    print("=" * 78)
    return prompt
