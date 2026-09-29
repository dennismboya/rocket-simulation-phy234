"""The synthetic demo book: the investors the dashboard and the API serve in demo mode (Phase 5).

Everything in this module is synthetic and labelled so (CLAUDE.md rule 3): every response row has
``is_synthetic = True``, every client row has ``is_synthetic = True`` and a display label of the
form ``"Demo client 07"`` or ``"Demo archetype: panic-prone"``. No number here is fitted to, or
quoted from, any dataset.

The book has two parts:

* :data:`N_GENERATED` investors drawn from the G_Q generator (:mod:`bre.sim.gq`, its labelled
  synthetic population defaults), each answering its own randomized, balanced intake battery of
  :data:`BATTERY_ITEMS` items (:func:`bre.design.battery_subset`, the full form) rather than the
  170-item design, so the book looks like intake data would;
* five hand-written archetypes (:data:`ARCHETYPES`: steady, panic-prone, gain-chaser,
  news-reactive, socially-driven) whose G_Q latents — the state pre-image ``v`` (hence the Bloch
  angle of ``psi``), the four context vectors ``theta_c``, the loss rotation ``theta_L`` and the
  dephasing rate ``gamma`` — are chosen explicitly and documented as synthetic design choices.
  They are produced by the same generator with the population spread switched off (``W = 0``,
  ``sigma_v = sigma_c = sigma_gamma = 0``), so each archetype's latents are exactly the listed
  numbers, never fitted values.

:func:`build_demo_book` returns ``(clients, responses)``: a client table (one row per investor,
covariates as canonical JSON text with the six product keys, the generator truth in ``truth_*``
columns) and a DecisionEvent frame (:mod:`bre.schema`) with ``subject_id == client_id``,
``dataset = "synthetic_demo_book"``, ``consent_training = True``. ``db.demo_seed.seed_demo``
loads both into the demo SQLite database under ``data/synthetic/`` and fits the demo model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre import design as D
from bre import schema as S
from bre.sim import common, gq

N_GENERATED = 60
"""Number of G_Q-generated investors in the book (plus the five archetypes)."""

BATTERY_ITEMS = D.BATTERY_FORMS["full"][1]
"""Items per client: the largest full-form intake battery (16), PLAN.md section 3."""

BATTERY_FORM = "full"

DATASET = "synthetic_demo_book"
"""``dataset`` of every response row of the book."""

SESSION_ID = "intake"
"""``session_id`` of every response row (one intake session per client)."""

CLIENT_PREFIX = "DEMO-C"
"""Generated clients are ``DEMO-C01`` ... ``DEMO-C60`` with labels ``"Demo client 01"``."""

ARCHETYPE_PREFIX = "DEMO-A-"
"""Archetype clients are ``DEMO-A-<name>`` with labels ``"Demo archetype: <name>"``."""

ARCHETYPE_SOURCE = "synthetic design choice (hand-written archetype), not a fitted value"
"""Provenance label of every archetype parameter below."""

GENERATED_SOURCE = "synthetic: G_Q generator with its labelled default population"

TRUTH_COLUMNS: tuple[str, ...] = (
    "truth_bloch_theta",
    "truth_bloch_phi",
    "truth_p_sell_base",
    "truth_p_sell_mean",
    "truth_gamma",
    "truth_source",
)
"""Generator latents carried in the client table (synthetic ground truth for the transparency
page; never shown as a fitted quantity)."""

# ---------------------------------------------------------------------------------------------
# Archetypes: explicit G_Q latents (synthetic design choices)
# ---------------------------------------------------------------------------------------------

_DEFAULT_THETA_L = tuple(float(x) for x in gq.POPULATION_DEFAULTS["theta_L"])

ARCHETYPES: dict[str, dict[str, Any]] = {
    "steady": {
        "description": (
            "Rarely sells: low resting sell amplitude, weak context rotations, half the population "
            "loss rotation, strong dephasing (context effects compose like a classical chain)."
        ),
        # state pre-image (re_hold, re_sell, im_hold, im_sell): P(sell | L = 0) = |psi_1|^2 ~ 0.05
        "v": (1.0, 0.22, 0.0, 0.05),
        "theta_L": tuple(0.5 * x for x in _DEFAULT_THETA_L),
        "theta_ctx": {
            "news:recession": (0.06, 0.02, 0.03),
            "news:technical": (-0.03, 0.04, 0.02),
            "social:friend_sells": (0.04, -0.05, 0.01),
            "market:recovered_5pct": (-0.05, -0.02, 0.04),
        },
        "gamma": 3.0,
        "covariates": {
            "age_band": "45-59", "wealth_band": "250k-1M", "invest_experience_yrs": 22,
            "self_reported_risk_tolerance": 4, "financial_literacy_score": 5, "education": "graduate",
        },
    },
    "panic-prone": {
        "description": (
            "Sells readily after losses: high resting sell amplitude, 1.5x the population loss "
            "rotation, a recession frame and a selling friend both rotate towards sell, near-zero "
            "dephasing (full interference: order effects and LTP violations)."
        ),
        "v": (1.0, 0.75, 0.0, 0.30),  # P(sell | L = 0) ~ 0.40
        "theta_L": tuple(1.5 * x for x in _DEFAULT_THETA_L),
        "theta_ctx": {
            "news:recession": (1.0, 0.3, 0.3),
            "news:technical": (0.3, 0.2, 0.2),
            "social:friend_sells": (0.6, -0.6, 0.1),
            "market:recovered_5pct": (-0.3, -0.1, 0.4),
        },
        "gamma": 0.05,
        "covariates": {
            "age_band": "30-44", "wealth_band": "<50k", "invest_experience_yrs": 2,
            "self_reported_risk_tolerance": 3, "financial_literacy_score": 2, "education": "some_college",
        },
    },
    "gain-chaser": {
        "description": (
            "Locks in gains: low resting sell amplitude and a small loss rotation, but a partial "
            "recovery rotates strongly towards sell; news and social cues barely move the state."
        ),
        "v": (1.0, 0.35, 0.0, 0.10),  # P(sell | L = 0) ~ 0.11
        "theta_L": (0.6, 0.2, 0.1),
        "theta_ctx": {
            "news:recession": (0.15, 0.05, 0.1),
            "news:technical": (-0.1, 0.1, 0.05),
            "social:friend_sells": (0.1, -0.1, 0.05),
            "market:recovered_5pct": (-1.0, -0.4, 0.4),
        },
        "gamma": 0.5,
        "covariates": {
            "age_band": "18-29", "wealth_band": "50-250k", "invest_experience_yrs": 4,
            "self_reported_risk_tolerance": 6, "financial_literacy_score": 3, "education": "bachelor",
        },
    },
    "news-reactive": {
        "description": (
            "Reacts to the stated cause of a loss: a recession frame rotates strongly towards "
            "sell and a technical-fault frame strongly away from it; social cue and recovery "
            "are weak; population loss rotation; moderate dephasing."
        ),
        "v": (1.0, 0.45, 0.0, 0.20),  # P(sell | L = 0) ~ 0.19
        "theta_L": _DEFAULT_THETA_L,
        "theta_ctx": {
            "news:recession": (1.2, 0.4, 0.3),
            "news:technical": (-0.8, 0.5, 0.2),
            "social:friend_sells": (0.1, -0.1, 0.05),
            "market:recovered_5pct": (-0.2, -0.1, 0.2),
        },
        "gamma": 0.3,
        "covariates": {
            "age_band": "60+", "wealth_band": ">1M", "invest_experience_yrs": 30,
            "self_reported_risk_tolerance": 3, "financial_literacy_score": 4, "education": "graduate",
        },
    },
    "socially-driven": {
        "description": (
            "Follows the crowd: a friend who sells rotates strongly towards sell; news frames and "
            "recovery are weak; population loss rotation; low dephasing."
        ),
        "v": (1.0, 0.45, 0.0, 0.20),  # P(sell | L = 0) ~ 0.19
        "theta_L": _DEFAULT_THETA_L,
        "theta_ctx": {
            "news:recession": (0.15, 0.05, 0.1),
            "news:technical": (-0.1, 0.1, 0.05),
            "social:friend_sells": (0.8, -1.0, 0.1),
            "market:recovered_5pct": (-0.15, -0.05, 0.1),
        },
        "gamma": 0.15,
        "covariates": {
            "age_band": "30-44", "wealth_band": "50-250k", "invest_experience_yrs": 8,
            "self_reported_risk_tolerance": 4, "financial_literacy_score": 2, "education": "hs",
        },
    },
}
"""The five archetypes. Every number is a synthetic design choice (:data:`ARCHETYPE_SOURCE`):
``v`` is the state pre-image fed to ``prepare_state`` (so ``P(sell | L = 0) = |psi_1|^2``),
``theta_L`` the loss rotation, ``theta_ctx`` the four context vectors in the su(2) basis,
``gamma`` the dephasing rate (near 0 = coherent, large = classical), ``covariates`` a plausible
intake record (it does not drive the state: the archetype population has ``W = 0``)."""

ARCHETYPE_NAMES: tuple[str, ...] = tuple(ARCHETYPES)


def archetype_client_id(name: str) -> str:
    return ARCHETYPE_PREFIX + name.replace("-", "_")


def archetype_population(name: str) -> dict[str, Any]:
    """The G_Q population override that reproduces an archetype's latents exactly: ``W = 0``
    and ``sigma_v = 0`` (state = ``prepare_state(v)``), ``mu_c`` = the context vectors with
    ``sigma_c = 0``, the archetype's ``theta_L``, ``mu_gamma = log gamma`` with
    ``sigma_gamma = 0``; ``phi`` keeps the generator default."""
    spec = ARCHETYPES[name]
    mu_c = np.asarray([spec["theta_ctx"][c] for c in D.NONNULL_CONTEXTS], dtype=np.float64)
    return {
        "SOURCE": f"{ARCHETYPE_SOURCE}: {name}",
        "W": np.zeros_like(gq.POPULATION_DEFAULTS["W"]),
        "b": np.asarray(spec["v"], dtype=np.float64),
        "sigma_v": 0.0,
        "theta_L": np.asarray(spec["theta_L"], dtype=np.float64),
        "mu_c": mu_c,
        "sigma_c": 0.0,
        "mu_gamma": float(np.log(spec["gamma"])),
        "sigma_gamma": 0.0,
    }


# ---------------------------------------------------------------------------------------------
# Book construction
# ---------------------------------------------------------------------------------------------


def _client_seed(seed: int, index: int) -> int:
    """Per-client generator seed: deterministic in ``(seed, index)``, disjoint across clients."""
    return int(seed) * 100_000 + int(index)


def _one_client(
    client_id: str,
    seed: int,
    index: int,
    n_items: int,
    population: dict[str, Any] | None,
    covariates: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Generate one client: a fresh battery subset, then G_Q on it. Returns the schema frame
    (subject/dataset/session/refs rewritten to the book's ids), the truth row and the
    battery assignment record."""
    client_seed = _client_seed(seed, index)
    battery_rng = np.random.default_rng([int(seed), int(index)])
    subset, assignment = D.battery_subset(battery_rng, n_items, BATTERY_FORM)
    df, truth = gq.generate(1, client_seed, population=population, design=subset, mixed_population=True)
    generated_id = str(truth["subjects"]["subject_id"][0])
    df = df.copy()
    df["subject_id"] = client_id
    df["dataset"] = DATASET
    df["session_id"] = SESSION_ID
    df["source_row_ref"] = df["source_row_ref"].str.replace(f":{generated_id}:", f":{client_id}:", regex=False)
    if covariates is not None:
        df["covariates"] = common.covariates_json(covariates)
    cov_text = str(df["covariates"].iloc[0])
    subj = truth["subjects"]
    p_sell = np.asarray(truth["items"]["p_sell"], dtype=np.float64)[0]
    truth_row = {
        "covariates": cov_text,
        "truth_bloch_theta": float(subj["bloch_theta"][0]),
        "truth_bloch_phi": float(subj["bloch_phi"][0]),
        "truth_p_sell_base": float(np.sin(float(subj["bloch_theta"][0]) / 2.0) ** 2),
        "truth_p_sell_mean": float(p_sell.mean()),
        "truth_gamma": float(subj["gamma"][0]),
        "seed": client_seed,
    }
    return df, truth_row, assignment.to_dict()


def build_demo_book(
    seed: int = 0,
    n_generated: int = N_GENERATED,
    n_items: int = BATTERY_ITEMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the demo book: ``n_generated`` G_Q investors plus the five archetypes.

    Returns ``(clients, responses)``. ``clients`` has one row per investor with ``client_id``,
    ``display_label``, ``archetype`` (None for generated clients), ``covariates`` (canonical JSON
    text, six product keys), ``is_synthetic`` (always True), ``consent_training`` (always True),
    ``source``, ``seed``, ``n_items``, ``battery_form``, ``battery_assignment`` (JSON text of
    the :class:`bre.design.BatteryAssignment`) and the :data:`TRUTH_COLUMNS`. ``responses`` is a
    validated DecisionEvent frame (``2 * n_items`` rows per client: the sell row and the tolerance
    row of every item, in the item's question order) with ``subject_id == client_id``.
    Deterministic in ``(seed, n_generated, n_items)``.
    """
    if n_generated < 0:
        raise ValueError("n_generated must be >= 0")
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for i in range(n_generated):
        cid = f"{CLIENT_PREFIX}{i + 1:02d}"
        df, truth_row, assignment = _one_client(cid, seed, i, n_items, None, None)
        frames.append(df)
        rows.append(
            {
                "client_id": cid,
                "display_label": f"Demo client {i + 1:02d}",
                "archetype": None,
                "source": GENERATED_SOURCE,
                "truth_source": gq.SOURCE,
                "battery_assignment": json.dumps(assignment, sort_keys=True),
                **truth_row,
            }
        )
    for j, name in enumerate(ARCHETYPE_NAMES):
        cid = archetype_client_id(name)
        spec = ARCHETYPES[name]
        df, truth_row, assignment = _one_client(
            cid, seed, n_generated + j, n_items, archetype_population(name), spec["covariates"]
        )
        frames.append(df)
        rows.append(
            {
                "client_id": cid,
                "display_label": f"Demo archetype: {name}",
                "archetype": name,
                "source": f"{ARCHETYPE_SOURCE}: {name}",
                "truth_source": ARCHETYPE_SOURCE,
                "battery_assignment": json.dumps(assignment, sort_keys=True),
                **truth_row,
            }
        )
    responses = common.concat_events(frames)
    S.validate_frame(responses, expect_synthetic=True, strict=True)
    clients = pd.DataFrame(rows)
    clients["is_synthetic"] = True
    clients["consent_training"] = True
    clients["n_items"] = int(n_items)
    clients["battery_form"] = BATTERY_FORM
    order = [
        "client_id", "display_label", "archetype", "covariates", "is_synthetic", "consent_training",
        "source", "seed", "n_items", "battery_form", "battery_assignment", *TRUTH_COLUMNS,
    ]
    clients = clients[order].reset_index(drop=True)
    assert set(clients["client_id"]) == set(responses["subject_id"])
    return clients, responses


# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------

BOOK_STEM = "demo_book"


def write_demo_book(clients: pd.DataFrame, responses: pd.DataFrame, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``demo_book.parquet`` (through :func:`bre.schema.write_events`, which enforces the
    ``data/synthetic`` location) and ``demo_book.clients.json`` next to it. Returns both paths."""
    out = S.SYNTHETIC_DIR if out_dir is None else Path(out_dir)
    pq_path = out / f"{BOOK_STEM}.parquet"
    S.write_events(responses, pq_path)
    clients_path = out / f"{BOOK_STEM}.clients.json"
    payload = {
        "is_synthetic": True,
        "dataset": DATASET,
        "design_version": D.DESIGN_VERSION,
        "archetype_source": ARCHETYPE_SOURCE,
        "clients": json.loads(clients.to_json(orient="records")),
    }
    clients_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return pq_path, clients_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic demo book (parquet + clients JSON).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=N_GENERATED, help="generated investors (plus 5 archetypes)")
    parser.add_argument("--items", type=int, default=BATTERY_ITEMS, help="battery items per client (12-16)")
    parser.add_argument("--out", default=None, help="output directory (must be under data/synthetic)")
    args = parser.parse_args(argv)
    clients, responses = build_demo_book(args.seed, args.n, args.items)
    pq_path, clients_path = write_demo_book(clients, responses, args.out)
    print(f"wrote {pq_path} ({len(responses)} rows, {len(clients)} clients) and {clients_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHETYPES",
    "ARCHETYPE_NAMES",
    "ARCHETYPE_PREFIX",
    "ARCHETYPE_SOURCE",
    "BATTERY_FORM",
    "BATTERY_ITEMS",
    "BOOK_STEM",
    "CLIENT_PREFIX",
    "DATASET",
    "GENERATED_SOURCE",
    "N_GENERATED",
    "SESSION_ID",
    "TRUTH_COLUMNS",
    "archetype_client_id",
    "archetype_population",
    "build_demo_book",
    "write_demo_book",
]
