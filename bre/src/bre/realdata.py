"""Real-data assembly for the fits on public data (PLAN.md section 6 mapping, section 8 item 6)
and the A -> B transfer protocol.

Everything here reads catalogued tables only (``data/CATALOG.md`` entries A, B, C through the
processed parquet files of ``data/processed/`` or, for the CPC18 pairs variant, the loader
itself), applies the pre-registered Phase 4 mapping and returns ``(frame, ModelData)`` pairs.
Nothing is fabricated: every row is a row of the source table (aggregate rows are computed from
the individual rows they summarize, and say so in ``source_row_ref``).

Datasets (``DATASETS``; ``load_real(name, subjects=None, seed=0)``):

``choices13k`` (A, aggregate)
    One ``choice_rate`` row per problem x feedback x block with the binomial weight ``n`` of the
    source; ``context_tags`` stay ``[feedback:on|off, block:k]``. **Loss magnitude**: the loader
    leaves ``loss_pct`` null (no portfolio framing). For the fits the relative worst outcome of the
    problem, ``x_worst_outcome_rel = min outcome / max |outcome|`` over both gambles (catalog A),
    becomes ``loss_pct = min(0, x_worst_outcome_rel)``: the magnitude of the worst possible
    relative loss of the problem, 0 when no outcome is negative. This is the only sense in which
    a described gamble carries a "loss size"; it is a problem property, not an experienced loss.
    The response is the share choosing gamble B (the CPC convention: A is always a two-outcome
    gamble, B carries the lottery shape), *not* the safe-choice analog of CPC18.
``cpc15`` (B, aggregate)
    Same layout; the respondent counts are not in the mirror (weight null -> 1). The worst
    outcome is computed from the mirror's lottery parameters (``cpc15_worst_outcome_rel``) with
    the organizers' distribution rule of :mod:`bre.loaders.cpc18`.
``cpc18`` / ``cpc18_pairs`` (B, individual; PLAN.md section 6 real-data mapping for split (b))
    Feedback blocks 2-5 only (rows tagged ``feedback:on``); subject = SubjID; scenario =
    problem; response = 1 iff the safer option was chosen; **context = the experienced-outcome
    tags only** (``exp:loss`` / ``exp:gain`` of the preceding trial, singles, or of the two
    preceding trials in chronological order, pairs); the ``feedback:*`` and ``block:*`` tags are
    dropped from ``context_tags`` (they are constant / bookkeeping inside blocks 2-5 and would
    otherwise fill the context positions). ``loss_pct`` = previous obtained payoff / largest
    absolute outcome of the problem (signed; the loader's ``max_abs`` rule), and it is set to
    null (``loss = 0``) on rows without an ``exp:*`` tag: the first trial of block 2, whose
    previous payoff was not displayed. Such rows are the ``none`` condition. In the pairs variant
    the second feedback trial carries one tag, every later trial two.
``psych201_spektor2024lossaversion`` (C, individual)
    Described 50/50 lotteries; context = the domain tag (``domain:gain|loss|mixed``); the
    ``session:k`` tag is dropped (the session is ``session_id``). **Loss magnitude**: the
    loader's ``loss_pct`` is the worst outcome of the *chosen* lottery, which depends on the
    response; here it is replaced by the choice-independent worst outcome on the table,
    ``min(a, b, c, d) / max |outcome|`` over both lotteries (parsed from ``scenario_id``),
    clipped at 0 like choices13k.
``psych201_spektor2019contexteffects``, ``psych201_olschewski2024skewness`` (C, experience)
    Context = ``exp:gain|loss`` (previous outcome of the chosen option vs the block's running
    mean); the ``options:n`` tag is dropped (it is in the covariates as ``x_n_options``);
    ``loss_pct`` is null (no described outcomes).

Common rules: rows keep ``consent_training = False`` (the flag is a property of the BRE intake
battery; the public studies never collected it) so ``build_model_data`` runs with
``require_consent=False`` for these tables — the catalog designates them for fitting;
``subject_key = "dataset/subject_id"``; ``sell_types = ("lottery_choice",)``; ``is_synthetic``
is False on every row and asserted. ``subjects=N`` keeps a deterministic random subsample of
``N`` subject ids (``numpy.random.default_rng(seed).choice`` over the sorted unique ids, without
replacement; the same ``(N, seed)`` always gives the same subjects); for the aggregate tables
the loader's pseudo-subjects (one per row) are what is subsampled.

Split (b) on CPC18 (:func:`split_pairs_by_exp_tags`): train on the rows with at most one
``exp:*`` tag (the ``none`` rows and the singles), test on the sell rows with two (the ordered
pairs), inside the pairs variant. Training on the singles *variant* and testing on the pairs
variant would score the same trials twice (a single-tag and a two-tag view of one response), so
the split is taken within one table. Split (c) (question order) does not exist for any real
table in hand.

Transfer protocol A -> B (:func:`run_transfer`, ``python -m bre.realdata transfer``): the
population parts of B1, B3 and Q2 are fitted on the choices13k rates (one pseudo-subject, the
encoder random effect ``u`` frozen at zero through :class:`PopulationOnly`, so only ``W``,
``b`` and the model's population parameters move; binomial weights; contexts = feedback/block
tags; loss = the worst relative outcome), then scored without refitting on the CPC15 rates and
on the CPC18 per-problem B-rates computed from the individual rows
(:func:`aggregate_cpc18_by_problem`). Each target's problems are split 80/20 (seeded); the
transferred fit, a refit of the same model on the target's training problems and a constant
rate are all scored on the held-out problems with the weighted NLL per respondent (binomial),
the Brier score and the correlation of predicted vs observed rates, with problem-level
bootstrap intervals. ``transfer_report`` renders the markdown; every number in it is computed
here.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bre import schema as S
from bre.artifact import jsonable
from bre.fit import DEFAULT_FIT_SETTINGS, _log, artifact_from_fit, fit_model, fit_settings, load_config, resolve_path
from bre.loaders import cpc18 as _cpc18
from bre.loaders._common import PROCESSED_DIR, RAW_DIR, assemble, dumps
from bre.models.data import ModelData, build_model_data
from bre.registry import MODEL_FAMILY, make_model

PROJECT_ROOT = S.PROJECT_ROOT
TRANSFER_REPORTS_DIR = PROJECT_ROOT / "reports" / "transfer"
TRANSFER_RUNS_DIR = PROJECT_ROOT / "runs" / "transfer"
CACHE_DIR = PROJECT_ROOT / "runs" / "realdata"
"""Parquet cache of the assembled / subsampled frames (:func:`assembled_frame`); gitignored."""

SELL_TYPES_INDIVIDUAL: tuple[str, ...] = ("lottery_choice",)
"""Elicitation types mapped to the sell likelihood for the individual tables (INTERFACE.md 1)."""

SUBJECT_KEY = "dataset/subject_id"
AGGREGATE_SUBJECT = "agg"
"""The single pseudo-subject of a collapsed aggregate table (``single_subject=True``)."""

DATASETS: dict[str, dict[str, Any]] = {
    "choices13k": {"catalog": "A", "file": "choices13k.parquet", "kind": "aggregate"},
    "cpc15": {"catalog": "B", "file": "cpc15.parquet", "kind": "aggregate"},
    "cpc18": {"catalog": "B", "file": "cpc18.parquet", "kind": "individual", "variant": "singles"},
    "cpc18_pairs": {"catalog": "B", "file": None, "kind": "individual", "variant": "pairs"},
    "psych201_spektor2024lossaversion": {"catalog": "C", "file": "psych201_spektor2024lossaversion.parquet", "kind": "individual", "contexts": "domain"},
    "psych201_spektor2019contexteffects": {"catalog": "C", "file": "psych201_spektor2019contexteffects.parquet", "kind": "individual", "contexts": "exp"},
    "psych201_olschewski2024skewness": {"catalog": "C", "file": "psych201_olschewski2024skewness.parquet", "kind": "individual", "contexts": "exp"},
}
"""Dataset key -> catalog entry, processed file (None: built by the loader on demand), kind."""

TRANSFER_MODELS: tuple[str, ...] = ("B1", "B3", "Q2")
"""Models of the A -> B transfer protocol (PLAN.md section 8, item 6)."""

TRANSFER_TARGETS: tuple[str, ...] = ("cpc15", "cpc18_agg")
TARGET_TEST_FRAC = 0.2
"""Share of a target's problems held out for scoring the transferred fit, the refit and the constant."""

TRANSFER_PROTOCOL_TEXT = (
    "Fit the population parts of B1, B3 and Q2 on the choices13k aggregate choice rates (one pseudo-subject, "
    "encoder random effect u frozen at 0, binomial weights = respondents per rate, contexts = feedback/block "
    "tags, loss magnitude = |min(0, worst outcome / max |outcome|)| of the problem, response = share choosing "
    "gamble B); score the fitted parameters without refitting on the CPC15 rates and on the CPC18 per-problem "
    "B-rates computed from the individual rows (per problem x block); compare with the same model refitted on "
    "the target's training problems and with a constant rate, all on the target's held-out 20% of problems; "
    "metrics: weighted NLL per respondent, Brier score, Pearson and Spearman correlation of predicted vs "
    "observed rates, problem-level bootstrap 95% intervals."
)


# ---------------------------------------------------------------------------------------------
# Reading and assembling the tables
# ---------------------------------------------------------------------------------------------


def _check_name(name: str) -> dict[str, Any]:
    if name not in DATASETS:
        raise KeyError(f"unknown real dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]


@functools.lru_cache(maxsize=1)
def read_frame(name: str) -> pd.DataFrame:
    """The loader's frame of ``name`` as stored (validated by ``read_events``; the CPC18 pairs
    variant is built by the loader on demand, which takes about 15 s). One full table is kept
    in memory at a time (the CPC18 tables are large; the build machine shares its memory with
    the recovery grid)."""
    spec = _check_name(name)
    if name == "cpc18_pairs":
        return _cpc18.load_cpc18(pairs=True)
    return S.read_events(PROCESSED_DIR / spec["file"])


def _cache_path(stem: str, subjects: int | None, seed: int) -> Path:
    return CACHE_DIR / f"{stem}-n{'all' if subjects is None else int(subjects)}-seed{int(seed)}.parquet"


def assembled_frame(name: str, subjects: int | None = None, seed: int = 0, *, cache: bool = True) -> pd.DataFrame:
    """The assembled (Phase 4 mapping) and subsampled frame of ``name``. With ``cache`` the
    result is written once to ``runs/realdata/<name>-n<subjects>-seed<seed>.parquet`` through
    ``bre.schema.write_events`` (strict validation) and read back by ``read_events`` (validated
    again) on later calls, so a subsample costs a second instead of a full-table load and the
    full table never has to stay in memory. ``runs/`` is gitignored; the cache holds real rows
    (``is_synthetic == False``) outside ``data/synthetic``, as the location rule requires."""
    path = _cache_path(name, subjects, seed)
    if cache and path.exists():
        return S.read_events(path)
    frame = subsample_subjects(assemble_frame(name, read_frame(name)), subjects, seed)
    if cache:
        S.write_events(frame, path)
        read_frame.cache_clear()  # release the full table; later calls with this key read the parquet
    return frame


def _covariate(text: object, key: str) -> Any:
    if S.is_null(text):
        return None
    return S.json_loads_strict(text).get(key)


def _tags(frame: pd.DataFrame) -> list[list[str]]:
    return frame["context_tags"].map(lambda v: list(v) if not S.is_null(v) else []).tolist()


@functools.lru_cache(maxsize=1)
def cpc15_worst_outcome_rel() -> dict[int, float]:
    """GameID -> ``min outcome / max |outcome|`` over both options (outcomes with p > 0) of the
    CPC15 problems, from the mirror's lottery parameters (``LotShape`` one-hots ``lot_shape__``,
    ``lot_shape_symm``, ``lot_shape_L``, ``lot_shape_R``) with the organizers' distribution
    rule (:func:`bre.loaders.cpc18.cpc18_distribution`)."""
    path = RAW_DIR / "cpc15" / "cpc15_thomas2024_aggregate.csv"
    df = pd.read_csv(path, index_col=0).drop_duplicates("GameID")

    def shape(row, side: str) -> str:
        if float(row[f"lot_shape_symm_{side}"]) == 1.0:
            return "Symm"
        if float(row[f"lot_shape_L_{side}"]) == 1.0:
            return "L-skew"
        if float(row[f"lot_shape_R_{side}"]) == 1.0:
            return "R-skew"
        return "-"

    out: dict[int, float] = {}
    for _, r in df.iterrows():
        _, _, min_a, max_a = _cpc18.option_moments(r["Ha"], r["pHa"], r["La"], shape(r, "A"), int(r["LotNumA"]))
        _, _, min_b, max_b = _cpc18.option_moments(r["Hb"], r["pHb"], r["Lb"], shape(r, "B"), int(r["LotNumB"]))
        lo, hi = min(min_a, min_b), max(max_a, max_b)
        m = max(abs(lo), abs(hi))
        out[int(r["GameID"])] = float(lo / m) if m > 0 else 0.0
    return out


@functools.lru_cache(maxsize=1)
def cpc18_worst_outcome_rel() -> dict[int, float]:
    """GameID -> ``min outcome / max |outcome|`` of the CPC18 problems (raw parameter table)."""
    raw = _cpc18.read_raw()
    problems = raw.drop_duplicates("GameID")[["GameID", "Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB"]]
    tab = _cpc18.problem_table(problems)
    return {int(g): (float(mn / ma) if ma > 0 else 0.0) for g, mn, ma in zip(tab["GameID"], tab["outcome_min"], tab["outcome_max_abs"])}


def loss_from_worst_outcome(wor: np.ndarray) -> np.ndarray:
    """``loss_pct = min(0, worst_outcome_rel)``: the signed relative worst loss of a described
    problem, 0 when no outcome is negative (module docstring)."""
    wor = np.asarray(wor, dtype=np.float64)
    return np.where(np.isfinite(wor), np.minimum(wor, 0.0), np.nan)


def _parse_spektor2024_outcomes(scenario_id: str) -> list[float]:
    """``"a,b|c,d"`` -> ``[a, b, c, d]``."""
    left, right = scenario_id.split("|")
    return [float(v) for part in (left, right) for v in part.split(",")]


def assemble_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the Phase 4 mapping of the module docstring to a loader frame; returns a new frame
    (row order kept, index reset)."""
    spec = _check_name(name)
    df = frame.copy()
    tags = _tags(df)
    if name in ("cpc18", "cpc18_pairs"):
        keep = np.asarray(["feedback:on" in t for t in tags], dtype=bool)
        df = df.loc[keep].copy()
        tags = [[x for x in t if x.startswith("exp:")] for t, k in zip(tags, keep) if k]
        loss = pd.to_numeric(df["loss_pct"], errors="raise").astype("float64").to_numpy()
        shown = np.asarray([len(t) > 0 for t in tags], dtype=bool)
        df["loss_pct"] = np.where(shown, loss, np.nan)
        df["context_tags"] = tags
    elif spec["kind"] == "aggregate":
        if name == "choices13k":
            wor = np.asarray([_covariate(c, "x_worst_outcome_rel") for c in df["covariates"].tolist()], dtype=np.float64)
        else:
            table = cpc15_worst_outcome_rel()
            wor = np.asarray([table[int(s)] for s in df["scenario_id"].tolist()], dtype=np.float64)
        df["loss_pct"] = loss_from_worst_outcome(wor)
    elif spec.get("contexts") == "domain":
        df["context_tags"] = [[x for x in t if x.startswith("domain:")] for t in tags]
        outs = [_parse_spektor2024_outcomes(str(s)) for s in df["scenario_id"].tolist()]
        wor = np.asarray([min(o) / max(abs(v) for v in o) if max(abs(v) for v in o) > 0 else 0.0 for o in outs], dtype=np.float64)
        df["loss_pct"] = loss_from_worst_outcome(wor)
    elif spec.get("contexts") == "exp":
        df["context_tags"] = [[x for x in t if x.startswith("exp:")] for t in tags]
    return df.reset_index(drop=True)


def subsample_subjects(frame: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Deterministic subject subsample (module docstring); the whole frame when ``n`` is None or
    not smaller than the number of subjects. Index reset."""
    if n is None:
        return frame.reset_index(drop=True)
    ids = np.asarray(sorted(frame["subject_id"].astype(str).unique()), dtype=object)
    if int(n) >= len(ids):
        return frame.reset_index(drop=True)
    if int(n) <= 0:
        raise ValueError("subjects must be a positive integer")
    rng = np.random.default_rng(int(seed))
    pick = set(rng.choice(ids, size=int(n), replace=False).tolist())
    keep = frame["subject_id"].astype(str).isin(pick).to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def collapse_to_single_subject(frame: pd.DataFrame) -> pd.DataFrame:
    """Every row of an aggregate table under one pseudo-subject ``agg`` (session ``agg``,
    ``position_in_session`` = row number, no prior questions): what the transfer protocol fits
    with the random effect frozen (one ``u`` row, held at zero)."""
    df = frame.copy().reset_index(drop=True)
    df["subject_id"] = AGGREGATE_SUBJECT
    df["session_id"] = AGGREGATE_SUBJECT
    df["position_in_session"] = np.arange(len(df), dtype="int64")
    df["prior_question_ids"] = [[] for _ in range(len(df))]
    return df


def load_real(
    name: str,
    subjects: int | None = None,
    seed: int = 0,
    *,
    single_subject: bool = False,
    ctx_vocab: tuple[str, ...] | None = None,
    validate: bool = True,
    n_ctx_positions: int = 2,
    cache: bool = True,
) -> tuple[pd.DataFrame, ModelData]:
    """``(frame, ModelData)`` of a real dataset after the Phase 4 mapping (module docstring).

    ``subjects`` / ``seed``: deterministic subject subsample. ``single_subject`` collapses an
    aggregate table under one pseudo-subject (transfer protocol). ``ctx_vocab`` fixes the
    context vocabulary (apply-to-new-data mode; unknown tags raise). ``validate`` runs the
    strict schema validation on the assembled frame (about 10 s on the full CPC18; workers
    that re-load a table the driver already validated pass False; a frame served from the
    parquet cache of :func:`assembled_frame` was validated when written and when read).
    ``cache`` uses that parquet cache. Every row must be real (``is_synthetic == False``),
    which is asserted.
    """
    spec = _check_name(name)
    frame = assembled_frame(name, subjects, seed, cache=cache)
    if single_subject:
        if spec["kind"] != "aggregate":
            raise ValueError("single_subject applies to the aggregate tables only")
        frame = collapse_to_single_subject(frame)
    if bool(frame["is_synthetic"].map(bool).any()):
        raise S.SchemaError(f"{name}: a real dataset carries is_synthetic=True rows")
    if validate and not (cache and not single_subject):
        S.validate_frame(frame, expect_synthetic=False, strict=True)
    data = build_model_data(
        frame,
        n_ctx_positions=n_ctx_positions,
        ctx_vocab=ctx_vocab,
        sell_types=SELL_TYPES_INDIVIDUAL,
        subject_key=SUBJECT_KEY,
        require_consent=False,
    )
    return frame, data


def exp_tag_count(data: ModelData) -> np.ndarray:
    """``(n,)`` number of ``exp:*`` context tags of every row."""
    is_exp = np.asarray([t.startswith("exp:") for t in data.ctx_vocab], dtype=bool)
    idx = np.clip(data.ctx_idx, 0, len(data.ctx_vocab) - 1)
    return (is_exp[idx] & data.ctx_mask).sum(axis=1).astype(np.int64)


def split_pairs_by_exp_tags(data: ModelData) -> tuple[np.ndarray, np.ndarray]:
    """Split (b) on the CPC18 pairs variant (module docstring): train rows = every row with at
    most one ``exp:*`` tag (``none`` and singles), test rows = the sell rows with two."""
    n = exp_tag_count(data)
    test = np.flatnonzero((n == 2) & data.sell_mask())
    train = np.flatnonzero(n <= 1)
    return train, test


# ---------------------------------------------------------------------------------------------
# CPC18 aggregate-by-problem (transfer target)
# ---------------------------------------------------------------------------------------------


def aggregate_cpc18_by_problem(subjects: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Per-problem x block B-rates of CPC18 computed from the individual rows of the processed
    singles table (``x_chose_b``): one ``choice_rate`` row per ``(GameID, block)`` with
    ``response`` = share choosing B, ``weight`` = number of individual choices behind it,
    ``context_tags = [feedback:on|off, block:k]`` (as choices13k / CPC15), ``loss_pct`` =
    ``min(0, worst outcome / max |outcome|)`` of the problem (:func:`cpc18_worst_outcome_rel`),
    ``subject_id = "agg:<GameID>:<block>"``, ``source_row_ref`` naming the aggregation.
    ``subjects`` subsamples the individual subjects first (same rule as :func:`load_real`).
    The result is cached under ``runs/realdata/`` like :func:`assembled_frame`."""
    path = _cache_path("cpc18_agg", subjects, seed)
    if path.exists():
        return S.read_events(path)
    frame = subsample_subjects(read_frame("cpc18"), subjects, seed)
    tags = _tags(frame)
    fb = [next(t for t in tt if t.startswith("feedback:")) for tt in tags]
    block = [next(t for t in tt if t.startswith("block:")) for tt in tags]
    chose_b = np.asarray([int(_covariate(c, "x_chose_b")) for c in frame["covariates"].tolist()], dtype=np.float64)
    key = pd.DataFrame({"game": frame["scenario_id"].astype(str).to_numpy(), "feedback": fb, "block": block, "b": chose_b, "subject": frame["subject_id"].astype(str).to_numpy()})
    g = key.groupby(["game", "block", "feedback"], sort=True)
    agg = g.agg(rate=("b", "mean"), n=("b", "size"), n_subjects=("subject", "nunique")).reset_index()
    wor = cpc18_worst_outcome_rel()
    w_rel = np.asarray([wor[int(gm)] for gm in agg["game"].tolist()], dtype=np.float64)
    n = len(agg)
    out = assemble(
        {
            "subject_id": [f"agg:{gm}:{b.split(':')[1]}" for gm, b in zip(agg["game"], agg["block"])],
            "dataset": _cpc18.DATASET,
            "session_id": AGGREGATE_SUBJECT,
            "position_in_session": np.zeros(n, dtype="int64"),
            "scenario_id": agg["game"].tolist(),
            "loss_pct": loss_from_worst_outcome(w_rel),
            "context_tags": [[f, b] for f, b in zip(agg["feedback"], agg["block"])],
            "elicitation_type": "choice_rate",
            "response": agg["rate"].to_numpy(dtype=float),
            "covariates": [dumps({"weight": int(k), "x_worst_outcome_rel": float(x), "x_n_subjects": int(s)}) for k, x, s in zip(agg["n"], w_rel, agg["n_subjects"])],
            "incentivized": True,
            "source_row_ref": [f"{_cpc18.RAW_FILE}:aggregated-by-problem:{gm}:{b}" for gm, b in zip(agg["game"], agg["block"])],
        },
        n,
    )
    S.write_events(out, path)
    read_frame.cache_clear()
    return out


def load_transfer_target(name: str, ctx_vocab: tuple[str, ...], *, subjects_b: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, ModelData]:
    """A transfer target (``cpc15`` or ``cpc18_agg``) as a single-subject rate table on the
    source vocabulary."""
    if name == "cpc15":
        return load_real("cpc15", single_subject=True, ctx_vocab=ctx_vocab, seed=seed)
    if name == "cpc18_agg":
        frame = collapse_to_single_subject(aggregate_cpc18_by_problem(subjects=subjects_b, seed=seed))
        S.validate_frame(frame, expect_synthetic=False, strict=True)
        data = build_model_data(frame, ctx_vocab=ctx_vocab, sell_types=SELL_TYPES_INDIVIDUAL, subject_key=SUBJECT_KEY, require_consent=False)
        return frame, data
    raise KeyError(f"unknown transfer target {name!r}; known: {TRANSFER_TARGETS}")


# ---------------------------------------------------------------------------------------------
# Population-only fits (random effect frozen at zero)
# ---------------------------------------------------------------------------------------------


class PopulationOnly:
    """Proxy of a gradient-fitted model whose encoder random effect ``enc.u`` is held at zero:
    ``objective``, ``log_lik``, ``predict_proba``, ``log_prior`` and ``interference_terms`` see
    ``u = 0`` whatever the parameter pytree holds, so the gradient with respect to ``u`` is zero
    and ``u`` never leaves its zero initialization; ``n_params`` reports the population count
    (``include_random_effects=False``). Everything else is forwarded to the wrapped model. Not
    for B2 (its own random effects are sampled) or the externally fitted B5."""

    def __init__(self, model: Any) -> None:
        if getattr(model, "requires_external_fit", False) or getattr(model, "name", "") == "B2":
            raise TypeError("PopulationOnly wraps the gradient-fitted models only")
        self._model = model
        self.name = model.name
        self.family = model.family
        self.requires_external_fit = False
        if hasattr(model, "interference_terms"):
            self.interference_terms = lambda params, data, **kw: model.interference_terms(self._zero_u(params), data, **kw)

    @staticmethod
    def _zero_u(params: Any) -> Any:
        import jax.numpy as jnp

        enc = params.get("enc") if isinstance(params, dict) else None
        if isinstance(enc, dict) and "u" in enc:
            return {**params, "enc": {**enc, "u": jnp.zeros_like(jnp.asarray(enc["u"]))}}
        return params

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Any:
        return self._zero_u(self._model.init_params(rng_key, data, restart))

    def objective(self, params, data: ModelData):
        return self._model.objective(self._zero_u(params), data)

    def log_lik(self, params, data: ModelData):
        return self._model.log_lik(self._zero_u(params), data)

    def predict_proba(self, params, data: ModelData):
        return self._model.predict_proba(self._zero_u(params), data)

    def log_prior(self, params):
        return self._model.log_prior(self._zero_u(params))

    def n_params(self, params, include_random_effects: bool = True) -> int:
        try:
            return int(self._model.n_params(params, include_random_effects=False))
        except TypeError:
            return int(self._model.n_params(params))

    def __getattr__(self, item: str) -> Any:
        if item == "_model":
            raise AttributeError(item)
        return getattr(self._model, item)


def population_model(name: str, data: ModelData, **kwargs: Any) -> PopulationOnly:
    """:class:`PopulationOnly` around ``make_model(name, data, **kwargs)``."""
    return PopulationOnly(make_model(name, data, **kwargs))


# ---------------------------------------------------------------------------------------------
# Rate metrics with problem-level bootstrap
# ---------------------------------------------------------------------------------------------


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    mx, my = np.sum(w * x) / sw, np.sum(w * y) / sw
    cov = np.sum(w * (x - mx) * (y - my))
    vx, vy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def rate_metrics(p, y, w, group_idx, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Metrics of predicted rates ``p`` against observed rates ``y`` with binomial weights
    ``w`` (respondents; 1 when unknown): ``nll_per_respondent`` = ``-sum(w ll) / sum(w)`` with
    ``ll`` the per-rate Bernoulli log-likelihood (the binomial log-likelihood per respondent up
    to the combinatorial constant), ``nll_per_row`` = ``-mean(w ll)`` (the ``bre.eval``
    convention, weights left in), weighted Brier, Pearson ``r`` and Spearman ``rho`` (rows
    unweighted). ``group_idx`` (problem per row) is the bootstrap unit: ``n_boot`` resamples of
    the problems, 2.5/97.5 percentiles."""
    from scipy.stats import spearmanr

    from bre.eval import subject_bootstrap

    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    y, w = np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)  # per respondent
    n = int(p.shape[0])
    if n == 0:
        return {"n_rows": 0}

    def stats(m: np.ndarray) -> tuple[float, float, float, float]:
        wm = w * m
        sw = wm.sum()
        if sw <= 0 or m.sum() <= 0:
            return (float("nan"),) * 4
        return (
            float(-np.sum(wm * ll) / sw),
            float(-np.sum(m * w * ll) / m.sum()),
            float(np.sum(wm * (p - y) ** 2) / sw),
            _weighted_pearson(p, y, m),
        )

    ones = np.ones(n)
    nll_r, nll_row, brier, r = stats(ones)
    out: dict[str, Any] = {
        "n_rows": n,
        "n_problems": int(len(np.unique(group_idx))),
        "sum_weights": float(w.sum()),
        "nll_per_respondent": nll_r,
        "nll_per_row": nll_row,
        "brier": brier,
        "pearson_r": r,
        "spearman_rho": float(spearmanr(p, y).correlation) if (n > 2 and np.ptp(p) > 0 and np.ptp(y) > 0) else float("nan"),
        "mean_observed_rate": float(np.sum(w * y) / w.sum()),
        "mean_predicted_rate": float(np.sum(w * p) / w.sum()),
    }
    if n_boot > 0:
        for j, key in enumerate(("nll_per_respondent", "nll_per_row", "brier", "pearson_r")):
            lo, hi, _ = subject_bootstrap(lambda m, j=j: stats(m)[j], np.asarray(group_idx), n_boot=int(n_boot), seed=int(seed))
            out[f"{key}_ci95"] = [lo, hi]
        out["bootstrap"] = {"n_draws": int(n_boot), "seed": int(seed), "unit": "problem"}
    return out


def split_problems(frame: pd.DataFrame, test_frac: float = TARGET_TEST_FRAC, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(train_rows, test_rows, group_idx)``: the problems (``scenario_id``) of a rate table
    split at random (seeded) with ``test_frac`` of them held out; every row of a problem is on
    the same side."""
    codes, uniques = pd.factorize(frame["scenario_id"].astype(str), sort=True)
    rng = np.random.default_rng(int(seed))
    n_test = max(1, int(round(test_frac * len(uniques)))) if len(uniques) >= 2 else 0
    test_problems = set(rng.permutation(len(uniques))[:n_test].tolist())
    is_test = np.asarray([c in test_problems for c in codes], dtype=bool)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), np.asarray(codes, dtype=np.int64)


def _population_summary(model_name: str, model: Any, params: dict[str, Any], data: ModelData) -> dict[str, Any]:
    """Named population parameters of a fit for provenance (what the demo mode may import)."""
    out: dict[str, Any] = {"ctx_vocab": list(data.ctx_vocab)}
    inner = getattr(model, "_model", model)
    if model_name == "B1":
        out["coefficients"] = {str(k): float(v) for k, v in inner.coefficients(params).items()}
    elif hasattr(inner, "context_table"):
        out["theta_ctx"] = np.asarray(inner.context_table(params)).tolist()
        out["theta_L"] = np.asarray(params["theta_L"]).tolist()
        out["phi"] = float(np.asarray(params["phi"]))
        out["encoder_b"] = np.asarray(params["enc"]["b"]).tolist()
    elif hasattr(inner, "natural_params"):
        out["natural"] = jsonable(inner.natural_params(params))
    if "enc" in params and "u" in params["enc"]:
        out["u_max_abs"] = float(np.max(np.abs(np.asarray(params["enc"]["u"])))) if np.asarray(params["enc"]["u"]).size else 0.0
    return out


# ---------------------------------------------------------------------------------------------
# The transfer protocol
# ---------------------------------------------------------------------------------------------


def run_transfer(
    models: tuple[str, ...] = TRANSFER_MODELS,
    *,
    rows_a: int | None = None,
    subjects_b: int | None = None,
    seed: int = 0,
    settings: dict[str, Any] | None = None,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
    n_boot: int = 1000,
    out_dir: str | Path = TRANSFER_REPORTS_DIR,
    runs_dir: str | Path = TRANSFER_RUNS_DIR,
    name: str = "A_to_B",
    targets: tuple[str, ...] = TRANSFER_TARGETS,
    save_artifacts: bool = True,
    echo: bool = True,
) -> tuple[dict[str, Any], Path, Path]:
    """Run the A -> B transfer protocol (module docstring); writes ``<name>.json`` and
    ``<name>.md`` under ``out_dir`` and returns ``(results, json_path, md_path)``.

    ``rows_a`` subsamples the choices13k rows (pseudo-subjects) deterministically, ``subjects_b``
    the CPC18 subjects behind the per-problem rates (tests); ``settings`` overrides
    :data:`bre.fit.DEFAULT_FIT_SETTINGS`; ``n_samples`` is forced to 0 (no draws are needed).
    """
    t_start = time.time()
    out_dir, runs_dir = Path(out_dir), Path(runs_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "transfer.log"
    fit_cfg = {**DEFAULT_FIT_SETTINGS, **(settings or {}), "n_samples": 0}
    models = tuple(models)
    for m in models:
        if m not in MODEL_FAMILY:
            raise KeyError(f"unknown model {m!r}")
        if m == "B2" or m == "B5":
            raise ValueError("the transfer protocol fits the gradient models with a frozen random effect; B2 and B5 are not covered")
    _log(log_path, f"[transfer] {name}: models={list(models)} rows_a={rows_a} subjects_b={subjects_b} seed={seed} settings={json.dumps(jsonable(fit_cfg))}", echo)
    frame_a, data_a = load_real("choices13k", subjects=rows_a, seed=seed, single_subject=True)
    vocab = tuple(data_a.ctx_vocab)
    results: dict[str, Any] = {
        "name": name,
        "protocol": TRANSFER_PROTOCOL_TEXT,
        "is_synthetic": bool(np.asarray(data_a.is_synthetic).any()),
        "source": {"dataset": "choices13k", "catalog": "A", "n_rows": int(data_a.n), "sum_weights": float(data_a.w.sum()), "ctx_vocab": list(vocab), "rows_option": rows_a, "seed": int(seed), "loss_rule": "loss_pct = min(0, x_worst_outcome_rel)", "response": "share choosing gamble B"},
        "targets": {},
        "fit_settings": jsonable(fit_cfg),
        "models": {},
        "test_frac_problems": TARGET_TEST_FRAC,
    }
    target_data: dict[str, tuple[pd.DataFrame, ModelData, np.ndarray, np.ndarray, np.ndarray]] = {}
    for t in targets:
        fr, dt = load_transfer_target(t, vocab, subjects_b=subjects_b, seed=seed)
        tr, te, grp = split_problems(fr, TARGET_TEST_FRAC, seed)
        target_data[t] = (fr, dt, tr, te, grp)
        results["targets"][t] = {
            "dataset": "cpc15" if t == "cpc15" else "cpc18 (per-problem B-rates from the individual rows)",
            "catalog": "B",
            "n_rows": int(dt.n),
            "n_problems": int(len(np.unique(grp))),
            "n_train_rows": int(len(tr)),
            "n_test_rows": int(len(te)),
            "n_test_problems": int(len(np.unique(grp[te]))),
            "sum_weights": float(dt.w.sum()),
            "weights": "respondents per rate" if t != "cpc15" else "not in the mirror: every rate weighs 1",
            "subjects_option": subjects_b if t == "cpc18_agg" else None,
        }
        _log(log_path, f"[transfer] target {t}: {dt.n} rows, {results['targets'][t]['n_problems']} problems, {len(te)} test rows", echo)

    def score(model: Any, params: dict[str, Any], dt: ModelData, rows: np.ndarray, grp: np.ndarray) -> dict[str, Any]:
        sub = dt.subset(rows)
        p = np.asarray(model.predict_proba(params, sub), dtype=np.float64)
        return rate_metrics(p, sub.y, sub.w, grp[rows], n_boot=n_boot, seed=seed)

    for m in models:
        kw = dict((model_kwargs or {}).get(m, {}))
        t0 = time.time()
        fit_a = fit_model(m, data_a, model=population_model(m, data_a, **kw), log_path=log_path, echo=echo, model_kwargs=kw, **fit_cfg)
        params_a = fit_a.params
        assert float(np.max(np.abs(np.asarray(params_a["enc"]["u"])))) == 0.0 if "enc" in params_a else True
        entry: dict[str, Any] = {
            "family": MODEL_FAMILY[m],
            "n_params": int(fit_a.n_params),
            "n_params_population": int(fit_a.n_params_population),
            "fit_on_A": {"train_nll_per_row": float(fit_a.train_nll), "val_nll_per_row": float(fit_a.val_nll), "best_restart": int(fit_a.best_restart), "wall_time_s": float(fit_a.wall_time), "restarts": jsonable(fit_a.restarts_table.to_dict(orient="records"))},
            "population_summary": _population_summary(m, fit_a.model, params_a, data_a),
            "targets": {},
        }
        if "interference_train" in fit_a.extra:
            entry["interference_train"] = fit_a.extra["interference_train"]
        if save_artifacts:
            art = artifact_from_fit(fit_a, data_a, training_data_refs=["data/processed/choices13k.parquet"], metrics={"transfer": {"role": "source fit on A (population only, u = 0)"}}, notes="A -> B transfer protocol: population parts fitted on choices13k rates with the encoder random effect frozen at zero.")
            art.save(runs_dir / "fit_A" / m)
        for t, (fr, dt, tr, te, grp) in target_data.items():
            tt = {"transfer_test": score(fit_a.model, params_a, dt, te, grp), "transfer_all": score(fit_a.model, params_a, dt, np.arange(dt.n), grp)}
            fit_b = fit_model(m, dt, rows=tr, model=population_model(m, dt, **kw), log_path=log_path, echo=echo, model_kwargs=kw, **fit_cfg)
            tt["refit_test"] = score(fit_b.model, fit_b.params, dt, te, grp)
            tt["refit_fit"] = {"train_nll_per_row": float(fit_b.train_nll), "val_nll_per_row": float(fit_b.val_nll), "wall_time_s": float(fit_b.wall_time), "n_params": int(fit_b.n_params)}
            tt["refit_population_summary"] = _population_summary(m, fit_b.model, fit_b.params, dt)
            w_tr = dt.w[tr]
            const = float(np.sum(w_tr * dt.y[tr]) / np.sum(w_tr))
            tt["constant_test"] = rate_metrics(np.full(len(te), const), dt.y[te], dt.w[te], grp[te], n_boot=n_boot, seed=seed)
            tt["constant_rate"] = const
            entry["targets"][t] = tt
            _log(log_path, f"[transfer] {m} -> {t}: transfer nll/resp={tt['transfer_test']['nll_per_respondent']:.4f} r={tt['transfer_test']['pearson_r']:.3f}; refit nll/resp={tt['refit_test']['nll_per_respondent']:.4f} r={tt['refit_test']['pearson_r']:.3f}; constant nll/resp={tt['constant_test']['nll_per_respondent']:.4f}", echo)
        entry["wall_time_s"] = time.time() - t0
        results["models"][m] = entry
    results["wall_time_s"] = time.time() - t_start
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    json_path.write_text(json.dumps(jsonable(results), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(transfer_report(results), encoding="utf-8")
    _log(log_path, f"[transfer] wrote {json_path} and {md_path} ({results['wall_time_s']:.0f}s)", echo)
    return results, json_path, md_path


def _fmt(v: Any, fmt: str = ".4f") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if not np.isfinite(f) else format(f, fmt)


def _ci(m: dict[str, Any], key: str, fmt: str = ".4f") -> str:
    ci = m.get(f"{key}_ci95")
    if not ci:
        return _fmt(m.get(key), fmt)
    return f"{_fmt(m.get(key), fmt)} [{_fmt(ci[0], fmt)}, {_fmt(ci[1], fmt)}]"


def transfer_report(results: dict[str, Any]) -> str:
    """Markdown of :func:`run_transfer`'s results (every number from the results dict)."""
    src = results["source"]
    lines = [f"# A -> B transfer: {results['name']}", ""]
    lines.append(f"**{'SYNTHETIC' if results.get('is_synthetic') else 'Real'} data.** Generated by `bre.realdata`; no hand-typed numbers.")
    lines.append("")
    lines.append(f"Protocol: {results['protocol']}")
    lines.append("")
    lines.append(f"Source A: choices13k (catalog A), {src['n_rows']} rate rows" + (f" (subsample option {src['rows_option']})" if src.get("rows_option") else "") + f", {src['sum_weights']:.0f} respondent-choices; contexts {src['ctx_vocab']}; {src['loss_rule']}; response = {src['response']}.")
    for t, info in results["targets"].items():
        lines.append(f"Target {t}: {info['dataset']} (catalog {info['catalog']}), {info['n_rows']} rate rows over {info['n_problems']} problems; held-out {info['n_test_problems']} problems ({info['n_test_rows']} rows); weights: {info['weights']}" + (f"; CPC18 subjects option {info['subjects_option']}" if info.get("subjects_option") else "") + ".")
    fs = results.get("fit_settings", {})
    lines.append(f"Fit settings: restarts {fs.get('restarts')}, steps {fs.get('steps')}, lr {fs.get('lr')}, L-BFGS polish {fs.get('lbfgs_steps')} it, seed {fs.get('seed')}; bootstrap over problems.")
    lines.append("")
    for t in results["targets"]:
        lines.append(f"## Target {t}: held-out problems")
        lines.append("")
        lines.append("| model | family | n_params | transfer NLL/respondent [CI95] | refit-on-B NLL/respondent [CI95] | constant NLL/respondent | transfer Brier | refit Brier | transfer r | refit r | transfer rho | refit rho |")
        lines.append("|---|---|---:|---|---|---|---|---|---|---|---|---|")
        for m, e in results["models"].items():
            tt = e["targets"][t]
            a, b, c = tt["transfer_test"], tt["refit_test"], tt["constant_test"]
            lines.append(f"| {m} | {e['family']} | {e['n_params']} | {_ci(a, 'nll_per_respondent')} | {_ci(b, 'nll_per_respondent')} | {_fmt(c['nll_per_respondent'])} | {_fmt(a['brier'])} | {_fmt(b['brier'])} | {_ci(a, 'pearson_r', '.3f')} | {_ci(b, 'pearson_r', '.3f')} | {_fmt(a['spearman_rho'], '.3f')} | {_fmt(b['spearman_rho'], '.3f')} |")
        lines.append("")
        lines.append("Transfer on every row of the target (fit on A, no refit):")
        lines.append("")
        lines.append("| model | NLL/respondent | Brier | r | rho | mean observed rate | mean predicted rate |")
        lines.append("|---|---|---|---|---|---|---|")
        for m, e in results["models"].items():
            a = e["targets"][t]["transfer_all"]
            lines.append(f"| {m} | {_fmt(a['nll_per_respondent'])} | {_fmt(a['brier'])} | {_fmt(a['pearson_r'], '.3f')} | {_fmt(a['spearman_rho'], '.3f')} | {_fmt(a['mean_observed_rate'], '.3f')} | {_fmt(a['mean_predicted_rate'], '.3f')} |")
        lines.append("")
    lines.append("## Fits on A")
    lines.append("")
    lines.append("| model | n_params (population) | train NLL/row | val NLL/row | best restart | seconds |")
    lines.append("|---|---:|---|---|---:|---:|")
    for m, e in results["models"].items():
        f = e["fit_on_A"]
        lines.append(f"| {m} | {e['n_params']} | {_fmt(f['train_nll_per_row'])} | {_fmt(f['val_nll_per_row'])} | {f['best_restart']} | {_fmt(f['wall_time_s'], '.0f')} |")
    lines.append("")
    lines.append("Reading: a transferred fit that scores no better than the constant rate on a target carries no information across datasets; a refit that beats the transfer shows how much of the gap is population shift rather than model form. The response is the share choosing gamble B (CPC convention), so these fits do not speak to the safe-choice analog of the Phase 4 evaluation; the population parameters are provenance for the demo mode only when the transfer beats the constant on the held-out problems.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _transfer_from_config(cfg: dict[str, Any], overrides: dict[str, Any], echo: bool) -> tuple[dict[str, Any], Path, Path]:
    fit_cfg = fit_settings(cfg) if cfg.get("fit") else dict(DEFAULT_FIT_SETTINGS)
    for k in ("steps", "restarts"):
        if overrides.get(k) is not None:
            fit_cfg[k] = overrides[k]
    src = dict(cfg.get("source") or {})
    tgt = dict(cfg.get("targets") or {})
    out = dict(cfg.get("output") or {})
    return run_transfer(
        tuple(overrides.get("models") or cfg.get("models") or TRANSFER_MODELS),
        rows_a=overrides.get("rows_a", src.get("rows")),
        subjects_b=overrides.get("subjects_b", tgt.get("cpc18_subjects")),
        seed=int(overrides.get("seed") if overrides.get("seed") is not None else cfg.get("seed", 0)),
        settings=fit_cfg,
        model_kwargs=dict(cfg.get("model_kwargs") or {}),
        n_boot=int(overrides.get("n_boot") or (cfg.get("bootstrap") or {}).get("n_draws", 1000)),
        out_dir=resolve_path(overrides.get("out") or out.get("dir", TRANSFER_REPORTS_DIR)),
        runs_dir=resolve_path(out.get("runs_dir", TRANSFER_RUNS_DIR)),
        name=str(overrides.get("name") or cfg.get("name", "A_to_B")),
        echo=echo,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bre.realdata", description="Real-data assembly and the A -> B transfer protocol.")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info", help="row / subject counts of a real dataset after the Phase 4 mapping")
    info.add_argument("dataset", choices=sorted(DATASETS))
    info.add_argument("--subjects", type=int, default=None)
    info.add_argument("--seed", type=int, default=0)
    tr = sub.add_parser("transfer", help="run the A -> B transfer protocol (default config: experiments/p3-transfer-A-to-B.yaml)")
    tr.add_argument("config", nargs="?", default=str(PROJECT_ROOT / "experiments" / "p3-transfer-A-to-B.yaml"))
    tr.add_argument("--models", nargs="+", default=None)
    tr.add_argument("--rows-a", type=int, default=None, help="subsample of choices13k rows (pseudo-subjects)")
    tr.add_argument("--subjects-b", type=int, default=None, help="subsample of CPC18 subjects behind the per-problem rates")
    tr.add_argument("--steps", type=int, default=None)
    tr.add_argument("--restarts", type=int, default=None)
    tr.add_argument("--seed", type=int, default=None)
    tr.add_argument("--n-boot", type=int, default=None)
    tr.add_argument("--out", default=None)
    tr.add_argument("--name", default=None)
    tr.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "info":
        frame, data = load_real(args.dataset, subjects=args.subjects, seed=args.seed)
        print(json.dumps(jsonable({"dataset": args.dataset, **data.summary()}), indent=1))
        return 0
    cfg = load_config(args.config) if Path(resolve_path(args.config)).exists() else {"name": "A_to_B"}
    overrides = {"models": args.models, "rows_a": args.rows_a, "subjects_b": args.subjects_b, "steps": args.steps, "restarts": args.restarts, "seed": args.seed, "n_boot": args.n_boot, "out": args.out, "name": args.name}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    _transfer_from_config(cfg, overrides, echo=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASETS",
    "PopulationOnly",
    "TRANSFER_MODELS",
    "TRANSFER_PROTOCOL_TEXT",
    "TRANSFER_TARGETS",
    "aggregate_cpc18_by_problem",
    "assemble_frame",
    "assembled_frame",
    "collapse_to_single_subject",
    "cpc15_worst_outcome_rel",
    "cpc18_worst_outcome_rel",
    "exp_tag_count",
    "load_real",
    "load_transfer_target",
    "loss_from_worst_outcome",
    "population_model",
    "rate_metrics",
    "read_frame",
    "run_transfer",
    "split_pairs_by_exp_tags",
    "split_problems",
    "subsample_subjects",
    "transfer_report",
]
