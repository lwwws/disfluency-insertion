"""Reproducibly mine few-shot example CANDIDATES from the train split.

This is the documented, one-command way the ``examples.yaml`` pool was built.
It selects real gold rows from the TRAIN split (never test), arranged as a
TYPE x LENGTH grid with meaningful (non-backchannel) context, and writes them to
``prompt_eng/candidates.yaml`` to browse and copy the good ones into
``examples.yaml`` (then flag them with one_shot:/few_shot:).

    # regenerate the candidate pool (deterministic under --seed):
    python prompt_eng/mine_candidates.py

    # more per cell, allow weak contexts, different seed:
    python prompt_eng/mine_candidates.py --n-per-cell 6 --allow-weak-context --seed 1
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Allow running as a script from the subdir: put the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import prompting


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine few-shot example candidates from the train split")
    parser.add_argument("--train-split", default="outputs/llm_runs_enriched/splits/train.jsonl")
    parser.add_argument("--out", default="prompt_eng/candidates.yaml")
    parser.add_argument("--n-per-cell", type=int, default=4,
                        help="Candidates per (type x length) cell")
    parser.add_argument("--allow-weak-context", action="store_true",
                        help="Keep rows whose context is only backchannels (uh-huh, yeah, ...)")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed -> reproducible pool")
    args = parser.parse_args()

    grid = prompting.mine_candidates(
        args.train_split,
        n_per_cell=args.n_per_cell,
        require_meaningful_context=not args.allow_weak_context,
        seed=args.seed,
    )

    # Flatten to a de-duplicated list (a row can cover several types) and dump.
    seen, candidates = set(), []
    for by_len in grid.values():
        for entries in by_len.values():
            for e in entries:
                if e["id"] not in seen:
                    seen.add(e["id"])
                    candidates.append(e)

    header = (
        "# Auto-mined few-shot candidates (reproducible via prompt_eng/mine_candidates.py).\n"
        f"# seed={args.seed}  n_per_cell={args.n_per_cell}  "
        f"meaningful_context={not args.allow_weak_context}\n"
        "# Browse these and copy whatever into examples.yaml ok???\n"
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + yaml.safe_dump({"examples": candidates}, sort_keys=False,
                                                allow_unicode=True, width=1000))

    n_cells = sum(1 for by_len in grid.values() for L in by_len)
    print(f"Wrote {len(candidates)} candidates ({n_cells} type x length cells) -> {out_path}")
    for t, by_len in grid.items():
        print(f"  {t:14} " + "  ".join(f"{L}:{len(v)}" for L, v in by_len.items()))


if __name__ == "__main__":
    main()
