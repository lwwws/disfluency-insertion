"""
swbd_disfluency_loader.py
==================================================================================
Build fluent <-> disfluent utterance pairs from Switchboard, combining:

  * Zayats et al. (2019)  -> ATYPICAL disfluencies (reparandum / repair), as BIO
                             tags over the MS-State transcript (`ms_disfl`).
  * NXT  (Calhoun 2010)   -> TYPICAL disfluencies: filled pauses (uh, um, ...) and
                             silent pauses (<sil>), PLUS atypical "insertions"
                             (well, like, so) detected via the Treebank pos="UH".

How the two corpora are joined
------------------------------
Their phonword IDs are NOT a shared key (`uw*` vs `ms*` number independently), so
we align the two MS-State word sequences by CONTENT (difflib), merging Zayats
clitic splits (`it's` -> `it`,`'s`, flagged by `_a/_b` id suffixes) first. Some
conversations have the A/B channels swapped between releases; `build_align_best`
detects this and aligns to the correct speaker. Each alignment reports its own
orth-agreement, and enrichment is skipped below `Config.min_agreement`.

Token -> disfluency category (single source of truth: `token_category`)
-----------------------------------------------------------------------
  silent_pause   (typical)   the injected <sil> token
  reparandum     (atypical)  Zayats BIO in {BE,IE,IP,BE_IP,C_IE,C_IP}
  fragment       (atypical)  a cut-off word ending in '-'  (e.g. "h-")
  filled_pause   (typical)   pos="UH" word in Config.filler_words (NXT lexicon),
                             OR any token in Config.clear_fillers as a final
                             surface-form fallback (e.g. a Zayats 'C'-tagged "uh")
  insertion      (atypical)  pos="UH" word in Config.insertion_words (NXT lexicon)
  None           (fluent)    everything else, incl. repair 'C' (non-filler) and
                             plain 'O' tokens not matched by any rule above

Fluent  = drop every categorised token (typical + atypical) + drop NONLEXICAL marks.
Disfluent = keep every spoken token + injected <sil>, drop NONLEXICAL marks.

Precedence (token_category, in order): silent_pause -> reparandum -> fragment ->
NXT lexicon (filled_pause/insertion) -> clear_fillers fallback -> fluent (None).
==================================================================================
"""
from __future__ import annotations
import ast, os, re, glob, json, difflib, statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd

import random
from collections import defaultdict

NITE = "{http://nite.sourceforge.net/}"


# ================================================================
# Config
# ================================================================
@dataclass
class Config:
    zayats_tsv: str = "data/raw/zayats/switchboard_corrected_with_silver_reannotation.tsv"
    nxt_dir:    str = "data/raw/nxt/nxt_switchboard_ann/xml"
    verbose:    bool = True

    # POS="UH" lexicons (split per the paper: insertions are ATYPICAL, fillers TYPICAL)
    filler_words:    set = field(default_factory=lambda: {"uh","um","eh","ah","hm","hum","er"})
    insertion_words: set = field(default_factory=lambda: {"well","like","so"})

    # final surface sweep: tag these as filled_pause even without a POS, where untagged.
    # only unambiguous fillers (never like/well/so, which need the POS gate).
    clear_fillers:   set = field(default_factory=lambda: {"uh","um","eh","ah","hm","er"})

    reparandum_tags: set = field(default_factory=lambda: {"BE","IE","IP","BE_IP","C_IE","C_IP"})
    nonlexical:      set = field(default_factory=lambda:
                        {"--","-","//","[silence]","[noise]","[laughter]","[vocalized-noise]"})

    sil_token:     str   = "<sil>"
    min_agreement: float = 0.90 # don't enrich a conversation below this join quality
    swap_margin:   float = 0.20 # treat as A/B swap if cross-speaker beats self by this much

    # Zayats TSV list-valued columns (parsed with ast.literal_eval)
    list_cols: tuple = ("sentence","ms_sentence","comb_sentence",
                        "names","ms_names","comb_ann","ms_disfl")
    tokens_col: str = "ms_sentence"
    tags_col:   str = "ms_disfl"


# ================================================================
# Tiny helpers + NXT XML parsing
# ================================================================
def safe_float(x):
    try: return float(x)
    except (TypeError, ValueError): return None # "non-aligned" etc. -> None

def pw_key(idstr):
    """uwB2_pw3_a -> ('B',3,'a');  ms1A_pw1 -> ('A',1,'').  None/'None' -> None."""
    if idstr in (None, "None"): return None
    m_spk = re.search(r"(?:uw|ms\d*)([AB])", idstr)
    m_pw  = re.search(r"_pw(\d+)([A-Za-z_]*)$", idstr)
    if not (m_spk and m_pw): return None
    return (m_spk.group(1), int(m_pw.group(1)), m_pw.group(2).lstrip("_"))

FRAG_RE = re.compile(r"^[a-z]+[a-z']*-$") # cut-off word: "h-", "th-"
def is_fragment(t): return bool(FRAG_RE.match(t.lower()))

def _href(ch): return ch.get(f"{NITE}href") or ch.get("href") or ""

def parse_terminals(fp):
    """Ordered stream of dicts. kind in {word, sil}; a word carries orth/pos/phon
       where phon = the MS-State phonword id it points to (NXT's own cross-layer link)."""
    out = []
    for el in ET.parse(fp).getroot():
        tag = el.tag.split('}')[-1]
        if tag == "word":
            phon = None
            for ch in el:
                if ch.tag.split('}')[-1] == "pointer" and ch.get("role") == "phon":
                    m = re.search(r"#id\((.*?)\)", _href(ch)); phon = m.group(1) if m else None
            out.append(dict(kind="word", orth=el.get("orth"), pos=el.get("pos"), phon=phon))
        elif tag == "sil":
            out.append(dict(kind="sil", orth=None, pos=None, phon=None))
    return out

def parse_phonwords(fp):
    """Ordered list of dicts: id / orth (lowercased) = the MS-State transcript."""
    out = []
    for pw in ET.parse(fp).getroot():
        if pw.tag.split('}')[-1] == "phonword":
            out.append(dict(id=pw.get(f"{NITE}id"), orth=(pw.get("orth") or "").lower()))
    return out

def term_phon_path(conv, spk, nxt_dir):
    return (os.path.join(nxt_dir, "terminals", f"{conv}.{spk}.terminals.xml"),
            os.path.join(nxt_dir, "phonwords", f"{conv}.{spk}.phonwords.xml"))

def conv_speaker_ids(nxt_dir):
    ids = []
    for fp in glob.glob(os.path.join(nxt_dir, "terminals", "*.terminals.xml")):
        b = os.path.basename(fp).split("."); ids.append((b[0], b[1]))
    return sorted(set(ids))


# ================================================================
# Content alignment (NXT phonwords and Zayats tokens)
# ================================================================
def zayats_units(df, conv, spk, tokens_col):
    """Zayats tokens for (conv, spk) grouped into spoken-word UNITS (merging _a/_b
       clitic splits). Returns [(unit_orth_lower, [(row_id, pos), ...])] in order."""
    sub = df[(df["conv"] == conv) & (df["speaker"] == spk)]
    if "turn" in sub: sub = sub.sort_values(["turn", "sent_num"])
    units, cur_key, cur = [], object(), None
    def base(idstr):
        k = pw_key(idstr); return (k[0], k[1]) if k else None
    for rid, r in sub.iterrows():
        toks, names = r[tokens_col], r["ms_names"]
        if not (isinstance(toks, list) and isinstance(names, list)): continue
        for pos, (w, nm) in enumerate(zip(toks, names)):
            b = base(nm)
            if b is not None and b == cur_key:
                cur[0] += w.lower(); cur[1].append((rid, pos))
            else:
                if cur: units.append((cur[0], cur[1]))
                cur = [w.lower(), [(rid, pos)]]; cur_key = b if b is not None else object()
    if cur: units.append((cur[0], cur[1]))
    return units

def _align(phon, units):
    """Core difflib pass. Returns (phonid2row, phon_order, agreement, coverage)
       where phonid2row maps a phonword id -> the LAST Zayats (row,pos) of its unit."""
    a = [p["orth"] for p in phon]
    b = [u for u, _ in units]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    idx2pos, hit, tot = {}, 0, 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("equal", "replace"):
            for d in range(min(i2 - i1, j2 - j1)):
                idx2pos[i1 + d] = units[j1 + d][1][-1]
                tot += 1; hit += (a[i1 + d] == b[j1 + d])
    order = [p["id"] for p in phon]
    return ({order[i]: rc for i, rc in idx2pos.items()}, order,
            hit / tot if tot else 0.0, len(idx2pos) / len(a) if a else 0.0)

def build_align(df, conv, nxt_spk, cfg, zay_spk=None):
    """Align NXT (conv, nxt_spk) phonwords to Zayats rows of `zay_spk` (default = nxt_spk)."""
    zay_spk = zay_spk or nxt_spk
    fp_t, fp_p = term_phon_path(conv, nxt_spk, cfg.nxt_dir)
    if not (os.path.exists(fp_t) and os.path.exists(fp_p)): return None
    phon  = parse_phonwords(fp_p)
    units = zayats_units(df, conv, zay_spk, cfg.tokens_col)
    if not phon or not units: return None
    phonid2row, order, agree, cov = _align(phon, units)
    return dict(conv=conv, nxt_spk=nxt_spk, zay_spk=zay_spk, agreement=agree, coverage=cov,
                phonid2row=phonid2row, phon_order=order, terminals_path=fp_t, swapped=(zay_spk != nxt_spk))

def build_align_best(df, conv, nxt_spk, cfg):
    """Align to the correct Zayats speaker, auto-detecting an A/B channel swap."""
    other   = "A" if nxt_spk == "B" else "B"
    self_al = build_align(df, conv, nxt_spk, cfg)
    cross   = build_align(df, conv, nxt_spk, cfg, zay_spk=other)
    if self_al is None: return cross # only the cross-speaker pairing has rows
    if cross is None:   return self_al
    # prefer the swapped pairing only when it clearly wins (avoids spurious swaps)
    return cross if cross["agreement"] > self_al["agreement"] + cfg.swap_margin else self_al


# ================================================================
# Turn an alignment into per-token NXT tags + silence positions
# ================================================================
def nxt_tags_for_alignment(df, align, cfg):
    """From one alignment, return:
         typ_flags : {(row_id, pos): 'filled_pause' | 'insertion'}   (POS=UH lexicon)
         sil_after : {(row_id, pos)}  meaning a silent pause occurs AFTER that token
       A tag is only transferred when the mapped Zayats token's surface actually
       equals the NXT word (guards against replace-block mis-tagging)."""
    typ_flags, sil_after = {}, {}
    order_idx = {pid: i for i, pid in enumerate(align["phon_order"])}
    last_phon = None
    for t in parse_terminals(align["terminals_path"]):
        if t["kind"] == "word":
            if t["phon"]: last_phon = t["phon"]
            if t["pos"] != "UH" or not t["phon"]: continue
            orth = (t["orth"] or "").lower()
            cat = ("filled_pause" if orth in cfg.filler_words else
                   "insertion"    if orth in cfg.insertion_words else None)
            if cat is None: continue
            rc = align["phonid2row"].get(t["phon"])
            if not rc: continue
            rid, pos = rc
            toks = df.loc[rid, cfg.tokens_col]
            if pos < len(toks) and toks[pos].lower() == orth: # surface guard
                typ_flags[rc] = cat
        elif t["kind"] == "sil" and last_phon is not None:
            rc = align["phonid2row"].get(last_phon)
            if rc is None and last_phon in order_idx: # walk back to nearest mapped
                for i in range(order_idx[last_phon] - 1, -1, -1):
                    rc = align["phonid2row"].get(align["phon_order"][i])
                    if rc: break
            if rc: sil_after[rc] = True
    return typ_flags, sil_after


# ================================================================
# Classification, rendering, detokenization, loading
# ================================================================
def token_category(tok, bio_tag, typ_tag, cfg):
    """Single source of truth for a token's disfluency category (or None = fluent)."""
    if tok == cfg.sil_token:           return "silent_pause"   # typical
    if bio_tag in cfg.reparandum_tags: return "reparandum"     # atypical (Zayats wins)
    if is_fragment(tok):               return "fragment"       # atypical
    if typ_tag is not None:            return typ_tag          # filled_pause / insertion
    return None                                                # repair 'C' and plain 'O'

_CLITICS = {"'s","'re","'ve","'ll","'d","'m","n't","'t","'em","'all"}
_PUNCT   = {",",".","?","!",";",":","%",")","]","}","''"}
def detok(toks):
    """Join PTB tokens into a readable string (cosmetic only; does not affect counts)."""
    s = ""
    for j, tok in enumerate(toks):
        if j == 0: s = tok
        elif tok.lower() in _CLITICS or tok in _PUNCT or tok.startswith("'"): s += tok
        else: s += " " + tok
    for a, b in {"gon na":"gonna","wan na":"wanna","got ta":"gotta"}.items():
        s = s.replace(a, b)
    return s

def render_pair(tokens, bio, typ, sil_after, cfg):
    """Build fluent & disfluent token lists (+ per-disfluent-token labels) from one row."""
    sil_after = set(sil_after)
    flu, dis, dis_labels, n_disfl = [], [], [], 0
    for i, tok in enumerate(tokens):
        if tok not in cfg.nonlexical:
            cat = token_category(tok, bio[i], typ[i], cfg)
            dis.append(tok); dis_labels.append(cat)
            if cat is None: flu.append(tok)
            else: n_disfl += 1
        if i in sil_after:                              # silence = disfluent-only token
            dis.append(cfg.sil_token); dis_labels.append("silent_pause"); n_disfl += 1
    return flu, dis, dis_labels, n_disfl

def load_zayats(cfg):
    df = pd.read_csv(cfg.zayats_tsv, sep="\t")
    for c in cfg.list_cols:
        if c in df.columns: df[c] = df[c].apply(ast.literal_eval)
    df["conv"]  = df["file"].str.extract(r"(sw\d+)")
    return df


# ================================================================
# The dataset
# ================================================================
class SwbdDisfluencyDataset:
    """Build, hold, query and export fluent/disfluent records.

    Usage
    -----
        ds = SwbdDisfluencyDataset(Config(zayats_tsv=..., nxt_dir=...)).build()
        ds[0]                       # one record (dict)
        ds.to_pandas()              # DataFrame for analysis
        ds.save_jsonl("out.jsonl")  # persist; SwbdDisfluencyDataset.from_jsonl(...) to reload
    """
    def __init__(self, config: Config, df: pd.DataFrame | None = None):
        self.cfg = config
        self.df = df
        self.records: list[dict] = []
        self.align_report: list[dict] = []

    def build(self):
        """Build fluent/disfluent records."""
        cfg = self.cfg
        df = self.df if self.df is not None else load_zayats(cfg)
        if "conv" not in df:  df["conv"]  = df["file"].str.extract(r"(sw\d+)")
        self.df = df

        # align every NXT speaker file -> per-row NXT enrichment (handles swaps)
        enrich: dict = {}  # row_id -> {"typ": {pos:cat}, "sil": set(pos)}
        meta: dict = {}  # row_id -> alignment provenance
        speaker_ids = conv_speaker_ids(cfg.nxt_dir)

        for conv, nxt_spk in tqdm(speaker_ids, desc="Processing convos . . .", disable=not cfg.verbose):
            if (df["conv"] == conv).sum() == 0: continue
            al = build_align_best(df, conv, nxt_spk, cfg)
            if al is None: continue
            self.align_report.append(dict(conv=conv, nxt_spk=nxt_spk, zay_spk=al["zay_spk"],
                                          agreement=al["agreement"], swapped=al["swapped"]))
            ok = al["agreement"] >= cfg.min_agreement

            typ_flags, sil_after = nxt_tags_for_alignment(df, al, cfg) if ok else ({}, {})
            for (rid, pos), cat in typ_flags.items():
                enrich.setdefault(rid, {"typ": {}, "sil": set()})["typ"][pos] = cat
            for (rid, pos) in sil_after:
                enrich.setdefault(rid, {"typ": {}, "sil": set()})["sil"].add(pos)
            for rid in df[(df["conv"] == conv) & (df["speaker"] == al["zay_spk"])].index:
                meta[rid] = dict(nxt_speaker=al["nxt_spk"], align_agreement=round(al["agreement"], 3),
                                 swapped=al["swapped"], enriched=ok)

        # one record per Zayats row
        self.records = []
        for rid, r in df.iterrows():
            tokens, bio = r[cfg.tokens_col], r[cfg.tags_col]
            if not (isinstance(tokens, list) and isinstance(bio, list) and len(tokens) == len(bio)):
                continue  # skip mis-aligned rows
            e = enrich.get(rid, {"typ": {}, "sil": set()})
            m = meta.get(rid, dict(nxt_speaker=None, align_agreement=None, swapped=None, enriched=False))

            # NXT lexicon tags apply wherever Zayats does NOT mark a reparandum
            # (so a filler that Zayats folded into a repair 'C' is still stripped, reparandum tokens are
            # already removed and keep their structural identity)
            typ = [e["typ"].get(p) if bio[p] not in cfg.reparandum_tags else None
                   for p in range(len(tokens))]

            # final surface sweep: unambiguous fillers, only where still untagged
            for p, tok in enumerate(tokens):
                if (typ[p] is None and bio[p] not in cfg.reparandum_tags and not is_fragment(tok)
                        and tok.lower() in cfg.clear_fillers):
                    typ[p] = "filled_pause"
            sil_after = sorted(e["sil"])

            flu, dis, dis_labels, n_disfl = render_pair(tokens, bio, typ, sil_after, cfg)
            self.records.append({
                # identity / metadata
                "conv": r["conv"], "file": r["file"], "speaker": r["speaker"],
                "turn": int(r["turn"]) if "turn" in r else None,
                "sent_num": int(r["sent_num"]) if "sent_num" in r else None,
                "nxt_speaker": m["nxt_speaker"], "align_agreement": m["align_agreement"],
                "swapped": m["swapped"], "enriched": m["enriched"],

                # 1) raw Zayats tokens  2) Zayats BIO   3) NXT tags + silence positions
                "tokens": tokens, "bio": bio, "typ": typ, "sil_after": sil_after,
                # 4) clean (fluent) string  5) disfluent string (with <sil>)
                "fluent": detok(flu), "disfluent": detok(dis),

                # extras for analysis
                "fluent_tokens": flu, "disfluent_tokens": dis, "disfl_labels": dis_labels,
                "n_tokens_fluent": len(flu), "n_tokens_disfluent": len(dis),
                "n_disfl_tokens": n_disfl,
            })
        if self.cfg.verbose:
            ag = [a["agreement"] for a in self.align_report]
            print(f"built {len(self.records)} records from {len(df)} rows")
            if ag:
                print(f"alignments: {len(ag)} | median agreement {statistics.median(ag):.3f} | "
                      f"<{cfg.min_agreement} (skipped enrichment): {sum(a < cfg.min_agreement for a in ag)} | "
                      f"A/B swaps: {sum(a['swapped'] for a in self.align_report)}")
        return self
    
    # Shared record filtering...
    def _iter_records(self, split=None, enriched_only=False):
        """Single source of truth for split / enriched filtering, used by every
        exporter so they all behave identically."""
        for r in self.records:
            if split is not None and r.get("split") != split:
                continue
            if enriched_only and not r.get("enriched", False):
                continue
            yield r


    # Access
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]

    def to_pandas(self, split=None, enriched_only=False):
        """DataFrame of records."""
        import pandas as pd
        return pd.DataFrame(list(self._iter_records(split, enriched_only)))


    def to_hf_dataset(self, fields=("fluent", "disfluent"), split=None, enriched_only=False):
        """HuggingFace dataset of the chosen fields.
    
        fields : tuple of record keys, or None to keep every field. To include
                context: fields=("context", "fluent", "disfluent").
        split  : None  + splits assigned -> DatasetDict{train,val,test}
                None  + no splits        -> single Dataset of all records
                "train"/"val"/"test"     -> single Dataset for that split
        """
        if fields is not None and self.records:
            missing = [f for f in fields if f not in self.records[0]]
            if missing:
                raise KeyError(f"fields {missing} not on records -- did you call "
                            f"add_context()/add_splits() first?")
    
        def _select(r):
            return dict(r) if fields is None else {f: r.get(f) for f in fields}
    
        def _ds(spl):
            return Dataset.from_list([_select(r) for r in self._iter_records(spl, enriched_only)])
    
        if split is not None:
            return _ds(split)
    
        present = []
        for r in self.records:
            s = r.get("split")
            if s is not None and s not in present:
                present.append(s)
        if not present: # no splits assigned -> flat dataset
            return _ds(None)

        order = ["train", "val", "test"]
        ordered = [s for s in order if s in present] + [s for s in present if s not in order]
        return DatasetDict({s: _ds(s) for s in ordered})


    def pairs(self, split=None, enriched_only=False):
        """[(fluent, disfluent)] ready for fine-tuning."""
        return [(r["fluent"], r["disfluent"]) for r in self._iter_records(split, enriched_only)]
 
 
    def context_pairs(self, split=None, enriched_only=False, sep="\n",
                    context_template="{context}{sep}{target}"):
        """[(input, disfluent)] where input = context + current fluent utterance.
    
        Falls back to the bare fluent string when a record has no context attached
        (conversation-initial turns, or if add_context was never called)."""
        out = []
        for r in self._iter_records(split, enriched_only):
            ctx = r.get("context", "")
            inp = context_template.format(context=ctx, sep=sep, target=r["fluent"]) if ctx else r["fluent"]
            out.append((inp, r["disfluent"]))
        return out


    # Persistence
    def save_jsonl(self, path):
        with open(path, "w") as f:
            for r in self.records: f.write(json.dumps(r) + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path, config: Config | None = None):
        ds = cls(config or Config())
        with open(path) as f:
            ds.records = [json.loads(l) for l in f if l.strip()]
        return ds

    def filter_records(self, min_tokens_fluent=1, min_tokens_disfluent=1,
                       require_nonempty=True, custom_fn=None):
        """Drop records that don't meet basic quality thresholds.

        min_tokens_fluent / min_tokens_disfluent : minimum token counts
        require_nonempty   : drop rows where fluent or disfluent string is empty/whitespace
        custom_fn          : optional callable(record) -> bool, True = keep
        """
        before = len(self.records)
        kept = []
        for r in self.records:
            if r["n_tokens_fluent"] < min_tokens_fluent: continue
            if r["n_tokens_disfluent"] < min_tokens_disfluent: continue
            if require_nonempty and (not r["fluent"].strip() or not r["disfluent"].strip()): continue
            if custom_fn is not None and not custom_fn(r): continue
            kept.append(r)
        self.records = kept
        if self.cfg.verbose:
            print(f"filter_records: kept {len(kept)}/{before}")

        return self

    def add_splits(self, ratios=(0.8, 0.1, 0.1), names=("train", "val", "test"),
                seed=42, by="conv"):
        """Assign each record a `split` field, splitting by `by` (default: conversation).
    
        Splitting by conversation (NOT by utterance) prevents leakage of speaker
        style/topic across splits, and is REQUIRED once context is added so a test
        utterance's previous turns can't end up in train.
    
        ratios : (train, val, test). Use (0.9, 0.0, 0.1) to mirror the paper's
                90/10 train/test with no validation set. Set val>0 if you do any
                model selection / hyperparameter tuning (recommended).
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
        keys = sorted({r[by] for r in self.records})
        random.Random(seed).shuffle(keys)

        n = len(keys)
        b1 = int(round(ratios[0] * n))
        b2 = b1 + int(round(ratios[1] * n))
    
        split_of = {}
        for i, key in enumerate(keys):
            split_of[key] = names[0] if i < b1 else names[1] if i < b2 else names[2]
        for r in self.records:
            r["split"] = split_of[r[by]]
    
        if self.cfg.verbose:
            convs = {nm: sum(s == nm for s in split_of.values()) for nm in names}
            recs = {nm: sum(r["split"] == nm for r in self.records) for nm in names}
            print(f"add_splits (by {by}): convs {convs} | records {recs}")
        return self
 
 
    @staticmethod
    def render_context(turns, speaker_map=None, template="{speaker}: {text}",
                    join=" ", prefix=True):
        """Join structured context turns into a string, the caller's way.
    
        turns       : list of {"speaker": str, "text": str}  (record["context_turns"])
        speaker_map : optional {orig: new}, e.g. {"A": "Agent", "B": "Caller"}
        template    : per-turn format; available fields are {speaker} and {text}
        join        : separator between turns (" ", "\n", ...)
        prefix      : False -> drop speaker labels entirely (texts only)
        """
        out = []
        for t in turns:
            spk = (speaker_map or {}).get(t["speaker"], t["speaker"])
            out.append(template.format(speaker=spk, text=t["text"]) if prefix else t["text"])
        return join.join(out)


    def add_context(self, k=3, source="fluent", include_other_speaker=True,
                    speaker_prefix=True, join=" ", template="{speaker}: {text}",
                    speaker_map=None):
        """Attach the previous `k` turns of the same conversation to each record.
    
        Stores per record:
            context_turns   : list of {"speaker", "text"}  -- RAW, no prefix baked in;
                            reformat freely downstream via render_context(...)
            context         : ONE prepared string built with the args below
                            (a convenience; override anytime from context_turns)
            n_context_turns : how many turns were available (< k near conversation start)
    
        source : "fluent" (recommended) or "disfluent".
            Use "fluent": no disfluency labels enter the input, so there's no
            cross-datapoint label leakage and no train/deploy mismatch, and the
            model can't copy a speaker's disfluency *rate* from context.
            I reckon "disfluent" is leakage sensitive choice...
    
        include_other_speaker : True = full dialogue context (both speakers in true
            temporal order); False = only the target speaker's own prior turns.
    
        Blank turns (empty after rendering) are skipped so they don't consume a
        `k` slot, a real prior turn fills it instead.
    
        NOTE: call this BEFORE filter_records so short utterances remain available
        as context. Ordering assumes `turn` is a GLOBAL conversation index
        (increasing across both speakers) and `sent_num` orders sentences within a
        turn.
        """
        assert source in ("fluent", "disfluent")
    
        by_conv = defaultdict(list)
        for idx, r in enumerate(self.records):
            by_conv[r["conv"]].append(idx)
    
        for conv, idxs in by_conv.items():
            idxs.sort(key=lambda i: (self.records[i].get("turn") or 0,
                                    self.records[i].get("sent_num") or 0))
            for pos, i in enumerate(idxs):
                r = self.records[i]
                turns, j = [], pos - 1
                while j >= 0 and len(turns) < k:
                    pr = self.records[idxs[j]]
                    same_or_both = include_other_speaker or pr["speaker"] == r["speaker"]
                    if same_or_both and pr[source].strip():     # skip blank turns
                        turns.append({"speaker": pr["speaker"], "text": pr[source]})
                    j -= 1
                turns.reverse()  # chronological
    
                r["context_turns"] = turns # raw, reformattable
                r["n_context_turns"] = len(turns)
                r["context"] = self.render_context( # one prepared string
                    turns, speaker_map=speaker_map, template=template,
                    join=join, prefix=speaker_prefix)
    
        if self.cfg.verbose:
            avg = sum(r["n_context_turns"] for r in self.records) / max(len(self.records), 1)
            print(f"add_context: k={k} source={source} "
                f"other_speaker={include_other_speaker} | avg turns attached {avg:.2f}")
        return self



if __name__ == "__main__":
    cfg = Config()
    ds = SwbdDisfluencyDataset(cfg).build()
    ds.save_jsonl("swbd_disfluency.jsonl")
