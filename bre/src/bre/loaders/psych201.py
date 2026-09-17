"""Psych-201 natural-language transcripts -> DecisionEvent rows (CATALOG.md entry C).

Source: ``data/raw/psych201/<folder>/prompts.jsonl.zip`` (one JSON line per participant with
``text``, ``experiment`` (``<folder>/<file>.csv``), ``participant`` and, for spektor2024, ``RTs``)
and the sibling ``generate_prompts.py`` that documents the sentence templates. The loader parses
the transcript back into trials with regular expressions built from those templates and validates
the parse by regenerating every line from the parsed values and comparing it with the transcript
(the round-trip fraction is reported in the meta and asserted in the tests).

Three folders are supported:

``spektor2024lossaversion`` (Spektor, Kellen, Rieskamp & Klauer 2024; 282 participants)
    Two 50/50 lotteries per trial: ``The outcomes of lottery X are a with 50% chance and b with
    50% chance. The outcomes of lottery Y are c with 50% chance and d with 50% chance. You choose
    <<Z>>.`` Sessions are ``Session k:`` blocks. The letters are randomized per participant; the
    first-named lottery is the original left/X option, the second the right/Y option.
    ``response = 1`` iff the lower-variance lottery was chosen (variance of a 50/50 lottery is
    ``((a - b) / 2)**2``, so the smaller |a - b| wins; ties -> the first-named lottery is the
    safer one). ``context_tags = ["domain:gain"|"domain:loss"|"domain:mixed", "session:<k>"]``
    where gain = all four outcomes >= 0, loss = all four <= 0, mixed otherwise.
    ``loss_pct = min(outcome) / max |outcome|`` of the chosen lottery (null if both outcomes are
    0). ``response_time_ms`` = the participant's ``RTs`` entry of the trial (ms; the generator
    multiplied second-valued columns by 1000). ``session_id = "session<k>"``.

``spektor2019contexteffects`` (Spektor, Gluth, Fontanesi & Rieskamp 2019; 130 participants)
    Decisions from experience among 2 or 3 options with full feedback: ``You choose <<Z>>. Option
    X yields x points. Option Y yields y points.[ Option W yields w points.]`` inside ``Block k:``
    blocks (block 1 is the two-option training block). The decision on a line is made *before*
    the outcomes on that line are revealed.

``olschewski2024skewness`` (Olschewski, Spektor & Le Mens 2024; 1,300 participants)
    Studies 1-4 ("Broker Game" / ping-pong "Bag" variant): 30 sample lines ``Stock X yields x
    points. Stock Y yields y points.`` followed by one decision ``You choose Stock <<Z>>.`` per
    ``Stock market k:`` / ``Round k:`` block. Study-4 participants with the stock introduction
    get no block headers from the generator (its ``else`` branch is inside the study-4 test), so
    their blocks are delimited by the choice line and labelled ``market<k>`` implicitly, k
    counting from 1 in transcript order. Studies 5-7: repeated choices with feedback in the
    spektor2019 line format (2 or 3 stocks; an optional ``There are two|three stocks to choose
    from.`` header).

Experience tasks (spektor2019 and olschewski2024), rules chosen here and documented (the catalog
only names the ingredients):

* ``response = 1`` iff the chosen option is the *lower-variance option judged from the outcomes
  shown so far in that block*: the population variance (ddof = 0) of each option's outcomes
  revealed on the block's previous lines (all displayed options' outcomes are revealed after
  every choice; in studies 1-4 the 30 samples per option); the safer option is the one with
  the smallest variance among the options with at least one revealed outcome, ties (including
  the first trial of a block, where nothing has been shown, and the second, where every
  variance is 0) going to the earliest-named option. ``covariates["x_n_shown"]`` (number of
  previous lines in the block) lets a model drop uninformative early trials.
* ``context_tags = ["options:2"|"options:3"]`` plus ``exp:gain`` / ``exp:loss``: the *previous
  outcome of the chosen option* (the most recent revealed outcome of the option chosen on this
  line) compared with the *block's running mean* (the mean of every outcome revealed in the block
  before this decision, over all options): ``exp:gain`` if previous outcome >= running mean,
  ``exp:loss`` if below; no tag when the chosen option has no revealed outcome yet.
* ``loss_pct = null``; ``response_time_ms = null`` (no RTs in these transcripts).
* ``session_id`` = the block label (``block<k>``, ``market<k>``, ``round<k>``);
  ``position_in_session`` = 0-based decision index within the block (always 0 in studies 1-4);
  ``scenario_id = "<file stem>:<block label>"``.

Common to the three folders: ``dataset = "psych201:<folder>"``; ``subject_id = "<file
stem>:<participant>"`` (participant ids repeat across the experiment files of a folder, so the
file stem, e.g. ``exp2`` or ``study03``, is part of the id); ``covariates = {"x_experiment":
<file stem>, "x_chosen_position": 0-based position of the chosen option among the named options,
...}``; ``prior_question_ids = []`` (the within-block order is carried by
``position_in_session``); ``incentivized = True`` -- an ASSUMPTION for spektor2019 and the
olschewski studies whose instructions only say "maximize the points": the spektor2024 and the
olschewski study-4 instructions state a performance-contingent bonus explicitly, the remaining
transcripts do not, and the papers could not be fetched from the build session (CATALOG.md);
``is_synthetic = False``; ``source_row_ref = "<folder>/prompts.jsonl:<0-based line>:t<decision
index within the transcript>"``.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre.loaders._common import PROCESSED_DIR, RAW_DIR, assemble, dumps, summarize
from bre.schema import ValidationReport, empty_frame, write_events

DATASET_PREFIX = "psych201"
FOLDERS: tuple[str, ...] = ("spektor2024lossaversion", "spektor2019contexteffects", "olschewski2024skewness")
ZIP_NAME = "prompts.jsonl.zip"
JSONL_NAME = "prompts.jsonl"
INCENTIVIZED: dict[str, bool] = {f: True for f in FOLDERS}
"""See the module docstring: explicit for spektor2024, assumed for the experience tasks."""

_NUM = r"-?\d+(?:\.\d+)?"
NAMES_RE = re.compile(
    r"^The (?P<kind>lotteries|options|stocks|bags)' names are "
    r"(?:(?P<a>[A-Z]) and (?P<b>[A-Z])|(?P<a3>[A-Z]), (?P<b3>[A-Z]), and (?P<c3>[A-Z]))\.$"
)
HEADER_RE = re.compile(r"^(?P<kind>Session|Block|Stock market|Round) (?P<n>\d+):$")
COUNT_RE = re.compile(
    r"^There are (?P<num>two|three) (?P<kind>options|stocks) to choose from"
    r"(?:, (?P<a>[A-Z]) and (?P<b>[A-Z])|, (?P<a3>[A-Z]), (?P<b3>[A-Z]), and (?P<c3>[A-Z]))?\.$"
)
LOTTERY_RE = re.compile(
    rf"^The outcomes of lottery (?P<o0>[A-Z]) are (?P<a>{_NUM}) with 50% chance and (?P<b>{_NUM}) with 50% chance\. "
    rf"The outcomes of lottery (?P<o1>[A-Z]) are (?P<c>{_NUM}) with 50% chance and (?P<d>{_NUM}) with 50% chance\. "
    r"You choose <<(?P<choice>[A-Z])>>\.$"
)
YIELD_ITEM_RE = re.compile(rf"(?P<label>Option|Stock|Bag) (?P<opt>[A-Z]) yields (?P<x>{_NUM}) points\.")
YIELD_LINE_RE = re.compile(
    rf"^(?:You choose <<(?P<choice>[A-Z])>>\. )?(?P<items>(?:(?:Option|Stock|Bag) [A-Z] yields {_NUM} points\.)(?: (?:Option|Stock|Bag) [A-Z] yields {_NUM} points\.)*)$"
)
CHOICE_ONLY_RE = re.compile(r"^You choose (?P<label>Stock|Bag) <<(?P<choice>[A-Z])>>\.$")
_HEADER_SLUG = {"Session": "session", "Block": "block", "Stock market": "market", "Round": "round"}


@dataclass
class Decision:
    """One parsed decision with the block state at decision time."""

    line_idx: int
    block_label: str
    block_no: int
    options: list[str]
    choice: str
    outcomes_now: dict[str, str] | None  # this line's revealed outcomes (verbatim), experience tasks
    lottery: dict[str, tuple[str, str]] | None  # spektor2024: option -> (a, b) verbatim
    n_shown: int
    seen: dict[str, list[float]]  # option -> outcomes revealed before this decision
    all_seen: list[float]


@dataclass
class ParsedTranscript:
    line_no: int
    experiment: str
    participant: int
    options: list[str]
    decisions: list[Decision] = field(default_factory=list)
    n_lines_checked: int = 0
    n_roundtrip_ok: int = 0
    unrecognized: list[str] = field(default_factory=list)


def _regen_yield(choice: str | None, items: list[tuple[str, str, str]]) -> str:
    body = " ".join(f"{label} {opt} yields {x} points." for label, opt, x in items)
    return (f"You choose <<{choice}>>. " if choice else "") + body


def _regen_names(kind: str, options: list[str]) -> str:
    if len(options) == 2:
        return f"The {kind}' names are {options[0]} and {options[1]}."
    return f"The {kind}' names are {options[0]}, {options[1]}, and {options[2]}."


def parse_transcript(text: str, line_no: int = 0, experiment: str = "", participant: int = -1) -> ParsedTranscript:
    """Parse one transcript; every line after the names line is regenerated and compared.

    Lines that do not match any template are recorded in ``unrecognized`` and count as failed
    round-trips; the caller reports the fraction ``n_roundtrip_ok / n_lines_checked``.
    """
    lines = text.split("\n")
    names_idx, options, kind = None, [], ""
    for i, ln in enumerate(lines):
        m = NAMES_RE.match(ln)
        if m:
            names_idx, kind = i, m.group("kind")
            options = [m.group("a"), m.group("b")] if m.group("a") else [m.group("a3"), m.group("b3"), m.group("c3")]
            break
    parsed = ParsedTranscript(line_no=line_no, experiment=experiment, participant=participant, options=options)
    if names_idx is None:
        parsed.unrecognized.append("<no names line>")
        parsed.n_lines_checked = 1
        return parsed
    parsed.n_lines_checked += 1
    parsed.n_roundtrip_ok += int(_regen_names(kind, options) == lines[names_idx])

    block_label, block_no, block_options = "", 0, list(options)
    block_idx = 0  # blocks started so far (headers or implicit)
    seen: dict[str, list[float]] = {o: [] for o in options}
    all_seen: list[float] = []
    n_shown = 0

    def _reset() -> None:
        nonlocal seen, all_seen, n_shown, block_options
        block_options = list(options)
        seen = {o: [] for o in options}
        all_seen, n_shown = [], 0

    for i in range(names_idx + 1, len(lines)):
        ln = lines[i]
        if ln == "":
            continue
        parsed.n_lines_checked += 1
        m = HEADER_RE.match(ln)
        if m:
            block_idx += 1
            block_label = f"{_HEADER_SLUG[m.group('kind')]}{int(m.group('n'))}"
            block_no = int(m.group("n"))
            _reset()
            parsed.n_roundtrip_ok += int(f"{m.group('kind')} {m.group('n')}:" == ln)
            continue
        if block_label == "" and not LOTTERY_RE.match(ln):
            # olschewski study 4 (stock intro) prints no block headers: a block starts implicitly
            # with its first sample line and ends with its single "You choose ... <<X>>." line
            block_idx += 1
            block_label, block_no = f"market{block_idx}", block_idx
            _reset()
        m = COUNT_RE.match(ln)
        if m:
            num = 2 if m.group("num") == "two" else 3
            if m.group("a"):
                block_options = [m.group("a"), m.group("b")]
                regen = f"There are two {m.group('kind')} to choose from, {m.group('a')} and {m.group('b')}."
            elif m.group("a3"):
                block_options = [m.group("a3"), m.group("b3"), m.group("c3")]
                regen = f"There are three {m.group('kind')} to choose from, {m.group('a3')}, {m.group('b3')}, and {m.group('c3')}."
            else:
                block_options = list(options[:num])
                regen = f"There are {m.group('num')} {m.group('kind')} to choose from."
            parsed.n_roundtrip_ok += int(regen == ln)
            continue
        m = LOTTERY_RE.match(ln)
        if m:
            g = m.groupdict()
            regen = (
                f"The outcomes of lottery {g['o0']} are {g['a']} with 50% chance and {g['b']} with 50% chance. "
                f"The outcomes of lottery {g['o1']} are {g['c']} with 50% chance and {g['d']} with 50% chance. "
                f"You choose <<{g['choice']}>>."
            )
            ok = regen == ln and [g["o0"], g["o1"]] == options[:2] and g["choice"] in options
            parsed.n_roundtrip_ok += int(ok)
            if not ok:
                parsed.unrecognized.append(ln)
                continue
            parsed.decisions.append(
                Decision(i, block_label or "session1", block_no or 1, list(options), g["choice"], None,
                         {g["o0"]: (g["a"], g["b"]), g["o1"]: (g["c"], g["d"])}, 0, {}, [])
            )
            continue
        m = YIELD_LINE_RE.match(ln)
        if m:
            items = [(im.group("label"), im.group("opt"), im.group("x")) for im in YIELD_ITEM_RE.finditer(m.group("items"))]
            choice = m.group("choice")
            opts_here = [opt for _, opt, _ in items]
            ok = _regen_yield(choice, items) == ln and opts_here == options[: len(opts_here)] and (choice is None or choice in opts_here)
            parsed.n_roundtrip_ok += int(ok)
            if not ok:
                parsed.unrecognized.append(ln)
                continue
            block_options = opts_here
            if choice is not None:
                parsed.decisions.append(
                    Decision(i, block_label, block_no, list(opts_here), choice, {opt: x for _, opt, x in items}, None,
                             n_shown, {o: list(seen[o]) for o in opts_here}, list(all_seen))
                )
            for _, opt, x in items:
                seen.setdefault(opt, []).append(float(x))
                all_seen.append(float(x))
            n_shown += 1
            continue
        m = CHOICE_ONLY_RE.match(ln)
        if m:
            choice = m.group("choice")
            ok = f"You choose {m.group('label')} <<{choice}>>." == ln and choice in block_options
            parsed.n_roundtrip_ok += int(ok)
            if not ok:
                parsed.unrecognized.append(ln)
                continue
            parsed.decisions.append(
                Decision(i, block_label, block_no, list(block_options), choice, None, None,
                         n_shown, {o: list(seen[o]) for o in block_options}, list(all_seen))
            )
            # one decision per block in this format: the next block starts fresh (header or implicit)
            block_label = ""
            _reset()
            continue
        parsed.unrecognized.append(ln)
    return parsed


def read_raw(raw_dir: Path = RAW_DIR, folder: str = FOLDERS[0]) -> list[dict]:
    """Read ``<folder>/prompts.jsonl.zip`` into a list of records (with their 0-based line index)."""
    if folder not in FOLDERS:
        raise ValueError(f"folder must be one of {FOLDERS}; got {folder!r}")
    path = Path(raw_dir) / "psych201" / folder / ZIP_NAME
    with zipfile.ZipFile(path) as z, z.open(JSONL_NAME) as fh:
        lines = fh.read().decode("utf-8").splitlines()
    records = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        r = json.loads(ln)
        r["_line"] = i
        records.append(r)
    return records


def _stem(experiment: str) -> str:
    return Path(experiment).stem


def _domain(vals: list[float]) -> str:
    if min(vals) >= 0:
        return "domain:gain"
    if max(vals) <= 0:
        return "domain:loss"
    return "domain:mixed"


def _pvar(xs: list[float]) -> float:
    return float(np.var(np.asarray(xs, dtype=float))) if xs else float("nan")


def _rows_spektor2024(rec: dict, parsed: ParsedTranscript, folder: str, cols: dict[str, list], meta: dict) -> None:
    stem = _stem(rec["experiment"])
    rts = rec.get("RTs")
    rts_ok = isinstance(rts, list) and len(rts) == len(parsed.decisions)
    if not rts_ok:
        meta["n_rt_length_mismatch"] += 1
    pos_by_session: dict[str, int] = {}
    for k, d in enumerate(parsed.decisions):
        o0, o1 = d.options[0], d.options[1]
        a, b = (float(v) for v in d.lottery[o0])
        c, dd = (float(v) for v in d.lottery[o1])
        var0, var1 = ((a - b) / 2) ** 2, ((c - dd) / 2) ** 2
        safer = o1 if var1 < var0 else o0
        chosen = (a, b) if d.choice == o0 else (c, dd)
        mx = max(abs(chosen[0]), abs(chosen[1]))
        loss = min(chosen) / mx if mx > 0 else np.nan
        pos = pos_by_session.get(d.block_label, 0)
        pos_by_session[d.block_label] = pos + 1
        cols["subject_id"].append(f"{stem}:{rec['participant']}")
        cols["session_id"].append(d.block_label)
        cols["position_in_session"].append(pos)
        cols["scenario_id"].append(f"{d.lottery[o0][0]},{d.lottery[o0][1]}|{d.lottery[o1][0]},{d.lottery[o1][1]}")
        cols["loss_pct"].append(loss)
        cols["context_tags"].append([_domain([a, b, c, dd]), f"session:{d.block_no}"])
        cols["response"].append(1.0 if d.choice == safer else 0.0)
        cols["response_time_ms"].append(float(rts[k]) if rts_ok else np.nan)
        cols["covariates"].append(dumps({"x_experiment": stem, "x_chosen_position": d.options.index(d.choice)}))
        cols["source_row_ref"].append(f"{folder}/{JSONL_NAME}:{rec['_line']}:t{k}")


def _rows_experience(rec: dict, parsed: ParsedTranscript, folder: str, cols: dict[str, list], meta: dict) -> None:
    stem = _stem(rec["experiment"])
    pos_by_block: dict[str, int] = {}
    for k, d in enumerate(parsed.decisions):
        variances = {o: _pvar(d.seen.get(o, [])) for o in d.options}
        with_data = [o for o in d.options if d.seen.get(o)]
        if with_data:
            best = min(variances[o] for o in with_data)
            safer = next(o for o in d.options if o in with_data and variances[o] <= best + 1e-12)
        else:
            safer = d.options[0]
        tags = [f"options:{len(d.options)}"]
        prev = d.seen.get(d.choice, [])
        if prev and d.all_seen:
            tags.append("exp:gain" if prev[-1] >= float(np.mean(d.all_seen)) else "exp:loss")
        pos = pos_by_block.get(d.block_label, 0)
        pos_by_block[d.block_label] = pos + 1
        cols["subject_id"].append(f"{stem}:{rec['participant']}")
        cols["session_id"].append(d.block_label)
        cols["position_in_session"].append(pos)
        cols["scenario_id"].append(f"{stem}:{d.block_label}")
        cols["loss_pct"].append(np.nan)
        cols["context_tags"].append(tags)
        cols["response"].append(1.0 if d.choice == safer else 0.0)
        cols["response_time_ms"].append(np.nan)
        cols["covariates"].append(
            dumps({"x_experiment": stem, "x_chosen_position": d.options.index(d.choice), "x_n_options": len(d.options), "x_n_shown": int(d.n_shown)})
        )
        cols["source_row_ref"].append(f"{folder}/{JSONL_NAME}:{rec['_line']}:t{k}")


def load_psych201_with_meta(raw_dir: Path = RAW_DIR, folder: str = FOLDERS[0]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the DecisionEvent frame of one Psych-201 folder and the parser statistics."""
    records = read_raw(raw_dir, folder)
    meta: dict[str, Any] = {
        "dataset": f"{DATASET_PREFIX}:{folder}",
        "raw_file": f"{folder}/{ZIP_NAME}",
        "n_participants": len(records),
        "n_lines_checked": 0,
        "n_roundtrip_ok": 0,
        "n_unrecognized_lines": 0,
        "n_rt_length_mismatch": 0,
        "n_decisions_in_text": 0,
        "unrecognized_examples": [],
        "dropped": {},
        "notes": [],
    }
    cols: dict[str, list] = {k: [] for k in ("subject_id", "session_id", "position_in_session", "scenario_id", "loss_pct", "context_tags", "response", "response_time_ms", "covariates", "source_row_ref")}
    for rec in records:
        parsed = parse_transcript(rec["text"], rec["_line"], rec["experiment"], int(rec["participant"]))
        meta["n_lines_checked"] += parsed.n_lines_checked
        meta["n_roundtrip_ok"] += parsed.n_roundtrip_ok
        meta["n_unrecognized_lines"] += len(parsed.unrecognized)
        meta["n_decisions_in_text"] += rec["text"].count("<<")
        if parsed.unrecognized and len(meta["unrecognized_examples"]) < 5:
            meta["unrecognized_examples"].extend(parsed.unrecognized[:2])
        if folder == "spektor2024lossaversion":
            _rows_spektor2024(rec, parsed, folder, cols, meta)
        else:
            _rows_experience(rec, parsed, folder, cols, meta)
    n = len(cols["subject_id"])
    if n:
        frame = assemble(
            {**cols, "dataset": f"{DATASET_PREFIX}:{folder}", "elicitation_type": "lottery_choice", "incentivized": INCENTIVIZED[folder]},
            n,
        )
    else:
        frame = empty_frame()
    meta["roundtrip_fraction"] = meta["n_roundtrip_ok"] / meta["n_lines_checked"] if meta["n_lines_checked"] else 0.0
    n_lost = meta["n_decisions_in_text"] - n
    if n_lost:
        meta["dropped"][f"decision line not parsed ({meta['n_unrecognized_lines']} unrecognized line(s))"] = int(n_lost)
    meta["notes"].append(f"round-trip: {meta['n_roundtrip_ok']}/{meta['n_lines_checked']} lines regenerate exactly ({meta['roundtrip_fraction']:.4f})")
    meta["notes"].append("incentivized=True is explicit in the transcript instructions for spektor2024 and olschewski study 4 only (assumed for the rest)")
    if folder == "spektor2024lossaversion":
        meta["notes"].append(f"RT list length mismatches: {meta['n_rt_length_mismatch']} participant(s)")
    meta.update(summarize(frame))
    return frame, meta


def load_psych201(raw_dir: Path = RAW_DIR, folder: str = FOLDERS[0]) -> pd.DataFrame:
    """One Psych-201 folder as a DecisionEvent frame (see :func:`load_psych201_with_meta`)."""
    return load_psych201_with_meta(raw_dir, folder)[0]


def processed_name(folder: str) -> str:
    return f"{DATASET_PREFIX}_{folder}.parquet"


def write_psych201(out_dir: Path = PROCESSED_DIR, raw_dir: Path = RAW_DIR, folders: tuple[str, ...] = FOLDERS) -> dict[str, ValidationReport]:
    """Write ``psych201_<folder>.parquet`` (+ validation report) per folder; returns the reports."""
    reports = {}
    for folder in folders:
        reports[folder] = write_events(load_psych201(raw_dir, folder), Path(out_dir) / processed_name(folder))
    return reports
