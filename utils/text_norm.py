"""Normalize orthographic dialogue text into the Switchboard style the models saw.

The fine-tuned adapters were trained on Switchboard transcripts, where ``fluent``
looks like ``i listen to it all the time in my car``: lowercase, no sentence
punctuation, apostrophes kept, numbers already spelled out. The hand-written
study conversations under ``convos/`` are ordinary orthography
(``I'm worried about my child's performance in math.``), which is off-distribution
for the adapters and would also leak punctuation into the TTS text.

``normalize_swbd`` bridges the two. Apply it ONCE, up front, so the fluent
condition, the model input, and the TTS text all derive from the same string.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Sentence punctuation Switchboard transcripts do not carry.
_PUNCT = r"""[.,?!;:"“”„«»()\[\]{}…]"""

_SPEAKER_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$")


def normalize_swbd(text: str) -> str:
    """Lowercase, de-punctuate and de-hyphenate ``text`` into Switchboard style.

    Apostrophes are preserved (``it's``, ``don't``, ``child's``) because the
    training transcripts keep them. Hyphens joining two words become a space
    (``solution-oriented`` -> ``solution oriented``); a hyphen at the END of a
    token is kept, since that is Switchboard's cut-off/speech-error marker
    (``innova-``) and the models emit it.
    """
    txt = str(text)
    # Normalize unicode quotes to ASCII before anything strips them.
    txt = txt.replace("’", "'").replace("‘", "'")
    txt = txt.replace("–", "-").replace("—", "-")
    txt = txt.lower()
    # Word-internal hyphen -> space; a trailing hyphen (cut-off marker) survives.
    txt = re.sub(r"(?<=\w)-(?=\w)", " ", txt)
    txt = re.sub(_PUNCT, " ", txt)
    return " ".join(txt.split())


def find_digits(text: str) -> List[str]:
    """Digit runs in ``text``. Switchboard spells numbers out, so these need
    manual attention (``5`` -> ``five``) before the text is model/TTS ready."""
    return re.findall(r"\d+", str(text))


def parse_conversation(raw: str, normalize: bool = True) -> List[Dict[str, str]]:
    """Parse a ``SPEAKER: text`` transcript into ``[{"speaker", "text"}, ...]``.

    The dict shape matches the ``context_turns`` records produced by
    ``SwbdDisfluencyDataset.add_context``, so the result feeds straight into
    ``SwbdDisfluencyDataset.render_context``.

    Blank lines are skipped. A line without a ``SPEAKER:`` prefix is appended to
    the previous turn rather than silently dropped.
    """
    turns: List[Dict[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _SPEAKER_LINE.match(line)
        if m:
            speaker, text = m.group(1).strip().upper(), m.group(2).strip()
            turns.append({"speaker": speaker,
                          "text": normalize_swbd(text) if normalize else text})
        elif turns:
            extra = normalize_swbd(line) if normalize else line.strip()
            turns[-1]["text"] = f"{turns[-1]['text']} {extra}".strip()
        else:
            raise ValueError(f"Transcript does not start with a 'SPEAKER: ...' line: {line!r}")
    return turns


# ----------------------------------------------------------------------------
# Did the model actually insert anything?
# Shared by the Turing-test stimulus selection and the conversation pipeline.
# ----------------------------------------------------------------------------
def has_only_final_sil(disfluent_text: str, fluent_text: str) -> bool:
    """Check if the only disfluency is a single <sil> token at the end.

    Returns True if disfluent text with final <sil> removed matches fluent text.
    Examples:
      - has_only_final_sil("<sil>", "") -> True
      - has_only_final_sil("the dog <sil>", "the dog") -> True
      - has_only_final_sil("the <sil> dog", "the dog") -> False (sil not at end)
    """
    if not disfluent_text:
        return False

    disfluent_text = disfluent_text.strip()
    fluent_text = fluent_text.strip() if fluent_text else ""

    if disfluent_text.endswith("<sil>"):
        candidate = disfluent_text[:-5].strip()  # Remove "<sil>"
        if candidate == fluent_text:
            return True

    return False


def is_unchanged(disfluent_text: str, fluent_text: str) -> bool:
    """True when nothing meaningful was inserted: the output is byte-identical to
    the input, or differs only by a trailing ``<sil>``."""
    return (disfluent_text.strip() == (fluent_text or "").strip()
            or has_only_final_sil(disfluent_text, fluent_text))
