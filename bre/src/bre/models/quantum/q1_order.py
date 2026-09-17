"""Q1 — the Wang–Busemeyer order-effect projector model in R^2 and the QQ-equality test.

This is PLAN.md §4 model Q1: a quantum-probability model (state vectors in a Hilbert space,
non-commuting projectors for questions, Born-rule probabilities with Lüders update). Everything
runs on a classical CPU with real 2×2 matrices; there is no quantum hardware anywhere.

Model
-----
* Decision state ``psi = (cos alpha, sin alpha)`` in R^2 (real amplitudes suffice: every cell is a
  product of squared cosines/sines, so a complex phase on ``psi`` changes nothing).
* Question A: ``|a> = e0``; ``P_Ay = |a><a|``, ``P_An = I - P_Ay``.
* Question B at relative angle ``phi``: ``|b> = (cos phi, sin phi)``; ``P_By = |b><b|``,
  ``P_Bn = I - P_By``.
* Sequential answers are Born-rule probabilities with Lüders update: measuring A first and
  observing ``y`` collapses ``psi`` onto ``P_Ay psi / ||P_Ay psi||``; the joint probability is
  ``P(Ay, By) = ||P_By P_Ay psi||^2`` and likewise for the other seven cells.

Cell naming follows ``data/raw/order_effects/question_order_tables.json``: the first letter is the
question asked first, ``y``/``n`` the answer; ``AyBn`` = P(A asked first and answered yes, then B
answered no); ``ByAn`` = P(B first yes, then A no).

Closed form (used by the fit; the Born-rule matrix version in :func:`q1_cells` is tested against it)
with ``c1 = cos^2 alpha``, ``c2 = cos^2(alpha - phi)`` and ``s = sin^2 phi``::

    AyBy = (1-s) c1    AyBn = s c1    AnBy = s (1-c1)    AnBn = (1-s)(1-c1)
    ByAy = (1-s) c2    ByAn = s c2    BnAy = s (1-c2)    BnAn = (1-s)(1-c2)

QQ equality
-----------
Wang & Busemeyer (2013, Topics in Cognitive Science 5:689–710) and Wang, Solloway, Shiffrin &
Busemeyer (2014, PNAS 111:9431–9436) show that for *any* state (pure or mixed), *any* dimension
and *any* pair of projectors, the "disagreement" mass is the same in both orders::

    q := [P(AyBn) + P(AnBy)] - [P(ByAn) + P(BnAy)] = 0.

Proof sketch (general): with ``P_An = I - P_Ay`` and ``P_Bn = I - P_By`` the A-first disagreement
operator ``P_Ay P_Bn P_Ay + P_An P_By P_An`` expands to ``P_Ay + P_By - P_Ay P_By - P_By P_Ay``,
which is symmetric under A <-> B, hence equals the B-first disagreement operator; the difference
of their expectations vanishes. In R^2 it is immediate from the closed form: both disagreement
sums equal ``s = sin^2 phi`` (``s (c1 + 1 - c1) = s (c2 + 1 - c2)``). The equality is therefore
parameter-free: it is a prediction of the model class, not of a fitted parameter.

Two equivalent forms are provided. :func:`qq_statistic` is ``q`` above (equivalently the
PLAN.md equality ``P(Ay,Bn) + P(An,By) = P(By,An) + P(Bn,Ay)``, see
:func:`qq_equality_sides`). :func:`qq_statistic_complement` is
``[P(ByAy) + P(BnAn)] - [P(AyBy) + P(AnBn)]``. They coincide for every pair of normalised tables,
model-generated or observed, because each order's four cells sum to one::

    P(ByAy) + P(BnAn) = 1 - [P(ByAn) + P(BnAy)]
    P(AyBy) + P(AnBn) = 1 - [P(AyBn) + P(AnBy)]
    => complement = [P(AyBn) + P(AnBy)] - [P(ByAn) + P(BnAy)] = q.

What Q1 in R^2 does and does not constrain
------------------------------------------
Two 2×2 tables have 6 free numbers; Q1 has 2 parameters. Besides the QQ equality the rank-1 R^2
model forces all four "agreement" conditionals to equal ``cos^2 phi``
(``P(By|Ay) = P(Bn|An) = P(Ay|By) = P(An|Bn)``), so a nonzero least-squares residual on a table
that satisfies the QQ equality is expected and is *not* evidence against the equality. The QQ
equality is the only one of these constraints that survives in higher dimensions and with mixed
states; that is why the 70-survey test in Wang et al. (2014) tests it and nothing else.

Information variant ("news frame", PLAN.md dataset D note)
----------------------------------------------------------
:func:`q1_cells_with_unitary` inserts a rotation ``U(theta)`` between the two questions in both
orders (information shown between the questions changes the state before the second question is
measured). Then ``s`` splits into ``s_A = sin^2(theta - phi)`` (A first) and
``s_B = sin^2(theta + phi)`` (B first) and ``q = s_A - s_B = -sin(2 theta) sin(2 phi)``, which is
nonzero unless ``theta`` or ``phi`` is a multiple of ``pi/2``. ``theta = 0`` recovers Q1.

Identifiability
---------------
The cells depend on ``(alpha, phi[, theta])`` only through ``cos^2 alpha``, ``cos^2(alpha - phi)``,
``sin^2(theta -/+ phi)``. Exact symmetries of the cells (all three verified in ``tests/test_q1.py``):

* ``alpha -> alpha + pi`` (``psi -> -psi``), ``phi -> phi + pi`` (``|b> -> -|b>``),
  ``theta -> theta + pi`` (``U -> -U``);
* the joint reflection ``(alpha, phi, theta) -> (-alpha, -phi, -theta)``;
* their composition ``(alpha, phi, theta) -> (pi - alpha, pi - phi, pi - theta)``.

Note that ``phi -> -phi`` *alone* is not a symmetry (``cos^2(alpha + phi) != cos^2(alpha - phi)``
in general); it is a symmetry only together with ``alpha -> -alpha``. :func:`canonicalize` maps
every orbit to the representative ``phi in [0, pi/2]``, ``alpha in [0, pi)``,
``theta in (-pi/2, pi/2]``, which is unique for generic parameters (non-uniqueness remains only on
the measure-zero set ``alpha in {0, pi/2}`` or ``phi in {0, pi/2}`` mod ``pi``, where the two
questions commute or the state is an eigenvector of A). All fitted parameters are reported in
canonical form.

Fitting
-------
:func:`fit_q1` minimises the sum of squared cell residuals (``scipy.optimize.least_squares`` with
the analytic Jacobian, restarts from a deterministic Halton grid, best restart kept). The model is
closed-form with two parameters, so the JAX/optax stack of the encoder models (PLAN.md §4) is not
needed here. A weighted multinomial negative log-likelihood is available through ``weights``
(respondent counts per order); the published tables carry no sample sizes, so it is implemented,
unit-tested, and unused in the report.

Command line: ``python -m bre.models.quantum.q1_order --report`` fits the three published tables
and writes ``reports/q1_qq_equality.md`` plus one figure per table under ``reports/figures/``.
Every number in that report is produced here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize

# ---------------------------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------------------------

CELLS: tuple[str, ...] = ("AyBy", "AyBn", "AnBy", "AnBn", "ByAy", "ByAn", "BnAy", "BnAn")
"""The eight cells of a question-order table, A-first block then B-first block."""

A_FIRST_CELLS: tuple[str, ...] = CELLS[:4]
B_FIRST_CELLS: tuple[str, ...] = CELLS[4:]

ORDERS: tuple[str, str] = ("A_first", "B_first")
"""Keys accepted by the ``weights`` argument of the fits (respondent counts per order)."""

_HERE = Path(__file__).resolve()
# src/bre/models/quantum/q1_order.py -> parents[4] is the bre/ project root (src layout).
PROJECT_ROOT: Path = _HERE.parents[4] if len(_HERE.parents) > 4 else Path.cwd()
DEFAULT_TABLES_PATH: Path = PROJECT_ROOT / "data" / "raw" / "order_effects" / "question_order_tables.json"
DEFAULT_REPORT_PATH: Path = PROJECT_ROOT / "reports" / "q1_qq_equality.md"
DEFAULT_FIGURE_DIR: Path = PROJECT_ROOT / "reports" / "figures"

UNVERIFIED_LABEL = "UNVERIFIED secondary transcription"
"""Label attached to every table whose ``verification`` field does not start with ``VERIFIED``."""

_EPS = 1e-12  # probability floor inside the multinomial log-likelihood


# ---------------------------------------------------------------------------------------------
# Hilbert-space primitives (real R^2)
# ---------------------------------------------------------------------------------------------


def state(alpha: float) -> np.ndarray:
    """Unit state vector ``psi = (cos alpha, sin alpha)``."""
    return np.array([math.cos(alpha), math.sin(alpha)], dtype=float)


def rotation(theta: float) -> np.ndarray:
    """Real rotation ``U(theta)`` of R^2 (an orthogonal matrix, ``U^T U = I``, ``det U = 1``)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def projectors(phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(P_Ay, P_An, P_By, P_Bn)`` for ``|a> = e0`` and ``|b> = (cos phi, sin phi)``."""
    a = np.array([1.0, 0.0])
    b = np.array([math.cos(phi), math.sin(phi)])
    eye = np.eye(2)
    p_ay = np.outer(a, a)
    p_by = np.outer(b, b)
    return p_ay, eye - p_ay, p_by, eye - p_by


def sequential_probability(
    psi: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    unitary: np.ndarray | None = None,
) -> float:
    """Born-rule joint probability ``||second · U · first · psi||^2`` (Lüders update between).

    Measuring ``first`` and observing its outcome leaves the state ``first psi / ||first psi||``;
    the conditional probability of the second outcome is ``||second U first psi||^2 /
    ||first psi||^2``; the joint is the product, i.e. the expression above. ``unitary`` defaults to
    the identity (no information between the questions).
    """
    v = first @ psi
    if unitary is not None:
        v = unitary @ v
    v = second @ v
    return float(v @ v)


# ---------------------------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------------------------


def q1_cells_with_unitary(alpha: float, phi: float, theta: float) -> dict[str, float]:
    """The eight cells when a rotation ``U(theta)`` acts between the two questions (both orders).

    ``theta = 0`` gives :func:`q1_cells`. Computed with explicit projector matrices and the Born
    rule; the closed form in :func:`_cells_vector` is tested against this function.
    """
    psi = state(alpha)
    p_ay, p_an, p_by, p_bn = projectors(phi)
    u = rotation(theta)
    return {
        "AyBy": sequential_probability(psi, p_ay, p_by, u),
        "AyBn": sequential_probability(psi, p_ay, p_bn, u),
        "AnBy": sequential_probability(psi, p_an, p_by, u),
        "AnBn": sequential_probability(psi, p_an, p_bn, u),
        "ByAy": sequential_probability(psi, p_by, p_ay, u),
        "ByAn": sequential_probability(psi, p_by, p_an, u),
        "BnAy": sequential_probability(psi, p_bn, p_ay, u),
        "BnAn": sequential_probability(psi, p_bn, p_an, u),
    }


def q1_cells(alpha: float, phi: float) -> dict[str, float]:
    """The eight Q1 cells ``P(first answer, second answer)`` for both question orders.

    Each order's four cells sum to one; all cells are non-negative; the QQ statistic is exactly
    zero for every ``(alpha, phi)``.
    """
    return q1_cells_with_unitary(alpha, phi, 0.0)


def _block(s: float, c: float, ds: np.ndarray, dc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Four cells ``[(1-s)c, s c, s(1-c), (1-s)(1-c)]`` and their Jacobian rows."""
    cells = np.array([(1 - s) * c, s * c, s * (1 - c), (1 - s) * (1 - c)])
    jac = np.stack(
        [
            -c * ds + (1 - s) * dc,
            c * ds + s * dc,
            (1 - c) * ds - s * dc,
            -(1 - c) * ds - (1 - s) * dc,
        ]
    )
    return cells, jac


def _cells_vector(params: np.ndarray, with_unitary: bool) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form cell vector (8,) in :data:`CELLS` order and its Jacobian (8, k).

    ``params`` is ``(alpha, phi)`` or ``(alpha, phi, theta)``; ``k = len(params)``.
    """
    alpha, phi = float(params[0]), float(params[1])
    theta = float(params[2]) if with_unitary else 0.0
    k = 3 if with_unitary else 2
    e = np.eye(k)
    d_alpha, d_phi = e[0], e[1]
    d_theta = e[2] if with_unitary else np.zeros(k)

    c1 = math.cos(alpha) ** 2
    dc1 = -math.sin(2 * alpha) * d_alpha
    c2 = math.cos(alpha - phi) ** 2
    dc2 = -math.sin(2 * (alpha - phi)) * (d_alpha - d_phi)
    s_a = math.sin(theta - phi) ** 2
    ds_a = math.sin(2 * (theta - phi)) * (d_theta - d_phi)
    s_b = math.sin(theta + phi) ** 2
    ds_b = math.sin(2 * (theta + phi)) * (d_theta + d_phi)

    cells_a, jac_a = _block(s_a, c1, ds_a, dc1)
    cells_b, jac_b = _block(s_b, c2, ds_b, dc2)
    return np.concatenate([cells_a, cells_b]), np.vstack([jac_a, jac_b])


def cells_to_vector(table: Mapping[str, float]) -> np.ndarray:
    """Cells dict -> vector in :data:`CELLS` order (raises ``KeyError`` on a missing cell)."""
    return np.array([float(table[c]) for c in CELLS], dtype=float)


def vector_to_cells(vec: np.ndarray) -> dict[str, float]:
    """Vector in :data:`CELLS` order -> cells dict."""
    return {c: float(v) for c, v in zip(CELLS, vec, strict=True)}


# ---------------------------------------------------------------------------------------------
# Tables (published data) and the statistics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderTable:
    """One published 2×2×2 question-order table with its provenance and verification status."""

    name: str
    description: str
    verification: str
    """Verbatim ``verification`` field of the JSON."""
    verified: bool
    """True iff ``verification`` starts with ``VERIFIED``; False for secondary transcriptions."""
    intervening_information: bool
    cells: dict[str, float]
    provenance: str
    source_path: str

    @property
    def status_label(self) -> str:
        """Short label that must accompany the table everywhere it appears."""
        return "VERIFIED" if self.verified else UNVERIFIED_LABEL


def _as_cells(table: Mapping[str, float] | OrderTable) -> dict[str, float]:
    if isinstance(table, OrderTable):
        return dict(table.cells)
    return {c: float(table[c]) for c in CELLS}


def load_tables(path: str | Path | None = None, *, tol: float = 5e-4) -> dict[str, OrderTable]:
    """Load the catalogued question-order tables from JSON.

    Keys starting with ``_`` are metadata (``_provenance``). Every table must carry the eight
    cells, a ``verification`` string and an ``intervening_information`` flag, and each order's
    cells must sum to one within ``tol`` (the published proportions are rounded to 4 decimals).
    """
    p = Path(path) if path is not None else DEFAULT_TABLES_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    provenance = str(raw.get("_provenance", ""))
    out: dict[str, OrderTable] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        missing = [c for c in CELLS if c not in entry]
        if missing:
            raise ValueError(f"table {name!r} lacks cells {missing}")
        cells = {c: float(entry[c]) for c in CELLS}
        for block, label in ((A_FIRST_CELLS, "A-first"), (B_FIRST_CELLS, "B-first")):
            total = sum(cells[c] for c in block)
            if abs(total - 1.0) > tol:
                raise ValueError(f"table {name!r}: {label} cells sum to {total}, not 1")
        if any(v < 0 for v in cells.values()):
            raise ValueError(f"table {name!r} has a negative cell")
        verification = str(entry["verification"])
        out[name] = OrderTable(
            name=name,
            description=str(entry.get("description", "")),
            verification=verification,
            verified=verification.startswith("VERIFIED"),
            intervening_information=bool(entry["intervening_information"]),
            cells=cells,
            provenance=provenance,
            source_path=str(p),
        )
    return out


def qq_equality_sides(table: Mapping[str, float] | OrderTable) -> tuple[float, float]:
    """``(P(Ay,Bn) + P(An,By), P(By,An) + P(Bn,Ay))`` — the two sides of the QQ equality."""
    t = _as_cells(table)
    return t["AyBn"] + t["AnBy"], t["ByAn"] + t["BnAy"]


def qq_statistic(table: Mapping[str, float] | OrderTable) -> float:
    """``q = [P(AyBn) + P(AnBy)] - [P(ByAn) + P(BnAy)]``; zero under Q1 for every parameter."""
    lhs, rhs = qq_equality_sides(table)
    return lhs - rhs


def qq_statistic_complement(table: Mapping[str, float] | OrderTable) -> float:
    """``[P(ByAy) + P(BnAn)] - [P(AyBy) + P(AnBn)]``; equals :func:`qq_statistic` (see module doc)."""
    t = _as_cells(table)
    return (t["ByAy"] + t["BnAn"]) - (t["AyBy"] + t["AnBn"])


def qq_equality_holds(table: Mapping[str, float] | OrderTable, tol: float = 1e-9) -> bool:
    """True iff ``|q| <= tol``."""
    return abs(qq_statistic(table)) <= tol


def agreement_conditionals(table: Mapping[str, float] | OrderTable) -> dict[str, float]:
    """``P(By|Ay), P(Bn|An), P(Ay|By), P(An|Bn)``: the second answer agreeing with the first.

    Under the rank-1 R^2 model all four equal ``cos^2 phi`` (``|<a|b>|^2``); observed tables need
    not satisfy this, which is the usual source of the Q1 least-squares residual.
    """
    t = _as_cells(table)

    def cond(joint: float, marg: float) -> float:
        return joint / marg if marg > 0 else math.nan

    return {
        "P_By_given_Ay": cond(t["AyBy"], t["AyBy"] + t["AyBn"]),
        "P_Bn_given_An": cond(t["AnBn"], t["AnBy"] + t["AnBn"]),
        "P_Ay_given_By": cond(t["ByAy"], t["ByAy"] + t["ByAn"]),
        "P_An_given_Bn": cond(t["BnAn"], t["BnAy"] + t["BnAn"]),
    }


def order_effect(table: Mapping[str, float] | OrderTable) -> dict[str, float]:
    """Marginal 'yes' rates of each question in each position and their differences.

    Returns ``P_A_yes_first``, ``P_A_yes_second``, ``delta_A = first - second`` and the same for
    B. Under Q1 the deltas vanish when the projectors commute (``phi`` a multiple of ``pi/2``).
    """
    t = _as_cells(table)
    a_first = t["AyBy"] + t["AyBn"]
    a_second = t["ByAy"] + t["BnAy"]
    b_first = t["ByAy"] + t["ByAn"]
    b_second = t["AyBy"] + t["AnBy"]
    return {
        "P_A_yes_first": a_first,
        "P_A_yes_second": a_second,
        "delta_A": a_first - a_second,
        "P_B_yes_first": b_first,
        "P_B_yes_second": b_second,
        "delta_B": b_first - b_second,
    }


def qq_statistic_unitary_closed_form(phi: float, theta: float) -> float:
    """``q = -sin(2 theta) sin(2 phi)`` predicted by the information variant (independent of alpha)."""
    return -math.sin(2 * theta) * math.sin(2 * phi)


# ---------------------------------------------------------------------------------------------
# Identifiability
# ---------------------------------------------------------------------------------------------


def canonicalize(alpha: float, phi: float, theta: float = 0.0) -> tuple[float, float, float]:
    """Map ``(alpha, phi, theta)`` to the canonical representative of its symmetry orbit.

    Uses the exact cell symmetries listed in the module docstring. Result:
    ``phi in [0, pi/2]``, ``alpha in [0, pi)``, ``theta in (-pi/2, pi/2]``.
    """
    alpha, phi, theta = alpha % math.pi, phi % math.pi, theta % math.pi
    if phi > math.pi / 2:  # (alpha, phi, theta) -> (pi - alpha, pi - phi, pi - theta)
        alpha, phi, theta = (math.pi - alpha) % math.pi, math.pi - phi, (math.pi - theta) % math.pi
    if theta > math.pi / 2:
        theta -= math.pi
    return float(alpha), float(phi), float(theta)


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Q1Fit:
    """Result of :func:`fit_q1` / :func:`fit_q1_with_unitary` (parameters in canonical form)."""

    alpha: float
    phi: float
    theta: float
    with_unitary: bool
    loss: float
    loss_kind: str
    l2_distance: float
    max_abs_residual: float
    observed: dict[str, float]
    predicted: dict[str, float]
    residual: dict[str, float]
    qq_observed: float
    qq_predicted: float
    n_restarts: int
    n_converged: int
    # provenance flags propagated from an OrderTable input (None for a bare cells dict)
    table_name: str | None = None
    verified: bool | None = None
    verification: str | None = None
    intervening_information: bool | None = None

    @property
    def params(self) -> tuple[float, ...]:
        return (self.alpha, self.phi, self.theta) if self.with_unitary else (self.alpha, self.phi)

    @property
    def status_label(self) -> str | None:
        if self.verified is None:
            return None
        return "VERIFIED" if self.verified else UNVERIFIED_LABEL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _halton(n: int, dims: int) -> np.ndarray:
    """First ``n`` points of the Halton sequence in ``[0, 1)^dims`` (deterministic, space-filling)."""
    primes = (2, 3, 5, 7, 11)
    out = np.zeros((n, dims))
    for j in range(dims):
        base = primes[j]
        for i in range(n):
            x, f, k = 0.0, 1.0 / base, i + 1
            while k > 0:
                k, r = divmod(k, base)
                x += r * f
                f /= base
            out[i, j] = x
    return out


def _restart_grid(n_restarts: int, with_unitary: bool) -> np.ndarray:
    """Starting points covering the canonical domain (plus theta in (-pi/2, pi/2) if used)."""
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    dims = 3 if with_unitary else 2
    lo = np.array([0.0, 0.0, -math.pi / 2])[:dims]
    hi = np.array([math.pi, math.pi / 2, math.pi / 2])[:dims]
    u = _halton(n_restarts, dims)
    return lo + (hi - lo) * (0.05 + 0.9 * u)  # keep away from the degenerate boundaries


def _normalise_weights(weights: Mapping[str, float] | float | None) -> np.ndarray | None:
    if weights is None:
        return None
    if isinstance(weights, int | float):
        w = np.array([float(weights), float(weights)])
    else:
        w = np.array([float(weights[o]) for o in ORDERS])
    if np.any(w <= 0):
        raise ValueError("weights (respondent counts) must be positive")
    return w


def _fit(
    table: Mapping[str, float] | OrderTable,
    *,
    with_unitary: bool,
    n_restarts: int,
    weights: Mapping[str, float] | float | None,
) -> Q1Fit:
    observed = _as_cells(table)
    obs = cells_to_vector(observed)
    w = _normalise_weights(weights)
    starts = _restart_grid(n_restarts, with_unitary)

    best_x: np.ndarray | None = None
    best_loss = math.inf
    n_converged = 0

    if w is None:
        loss_kind = "least_squares"

        def resid(p: np.ndarray) -> np.ndarray:
            return _cells_vector(p, with_unitary)[0] - obs

        def jac(p: np.ndarray) -> np.ndarray:
            return _cells_vector(p, with_unitary)[1]

        for x0 in starts:
            r = least_squares(
                resid, x0, jac=jac, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000
            )
            n_converged += int(r.status > 0)
            loss = float(np.sum(r.fun**2))
            if loss < best_loss:
                best_loss, best_x = loss, r.x
    else:
        loss_kind = "weighted_multinomial"
        # per-cell respondent weight: n_A for the A-first block, n_B for the B-first block
        cell_w = np.repeat(w, 4) * obs

        def nll(p: np.ndarray) -> tuple[float, np.ndarray]:
            pred, jac_ = _cells_vector(p, with_unitary)
            safe = np.maximum(pred, _EPS)
            value = -float(np.sum(cell_w * np.log(safe)))
            grad = -(cell_w / safe) @ jac_
            return value, grad

        for x0 in starts:
            r = minimize(nll, x0, jac=True, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
            n_converged += int(r.success)
            if r.fun < best_loss:
                best_loss, best_x = float(r.fun), r.x

    assert best_x is not None
    alpha, phi, theta = canonicalize(best_x[0], best_x[1], best_x[2] if with_unitary else 0.0)
    pred_vec = _cells_vector(np.array([alpha, phi, theta])[: 3 if with_unitary else 2], with_unitary)[0]
    predicted = vector_to_cells(pred_vec)
    residual = {c: predicted[c] - observed[c] for c in CELLS}
    res_vec = pred_vec - obs

    meta: dict[str, Any] = {}
    if isinstance(table, OrderTable):
        meta = {
            "table_name": table.name,
            "verified": table.verified,
            "verification": table.verification,
            "intervening_information": table.intervening_information,
        }
    return Q1Fit(
        alpha=alpha,
        phi=phi,
        theta=theta,
        with_unitary=with_unitary,
        loss=best_loss,
        loss_kind=loss_kind,
        l2_distance=float(np.sqrt(np.sum(res_vec**2))),
        max_abs_residual=float(np.max(np.abs(res_vec))),
        observed=observed,
        predicted=predicted,
        residual=residual,
        qq_observed=qq_statistic(observed),
        qq_predicted=qq_statistic(predicted),
        n_restarts=n_restarts,
        n_converged=n_converged,
        **meta,
    )


def fit_q1(
    table: Mapping[str, float] | OrderTable,
    n_restarts: int = 20,
    weights: Mapping[str, float] | float | None = None,
) -> Q1Fit:
    """Fit ``(alpha, phi)`` to the eight cells by least squares (default) or weighted multinomial.

    ``weights`` are respondent counts per order (``{"A_first": n_AB, "B_first": n_BA}`` or one
    number for both); when given, the loss is the negative multinomial log-likelihood
    ``-sum_orders n_order sum_cells obs · log pred``. The published tables carry no counts, so the
    report uses least squares. Restarts start from a deterministic Halton grid over the canonical
    domain; the best restart is returned with parameters canonicalised (see :func:`canonicalize`).
    """
    return _fit(table, with_unitary=False, n_restarts=n_restarts, weights=weights)


def fit_q1_with_unitary(
    table: Mapping[str, float] | OrderTable,
    n_restarts: int = 20,
    weights: Mapping[str, float] | float | None = None,
) -> Q1Fit:
    """Fit ``(alpha, phi, theta)`` of the information variant (rotation between the questions)."""
    return _fit(table, with_unitary=True, n_restarts=n_restarts, weights=weights)


def q1_distance(table: Mapping[str, float] | OrderTable, n_restarts: int = 20) -> float:
    """Descriptive distance to the nearest Q1 table: ``min_(alpha, phi) ||pred - obs||_2``.

    Chi-square-free (no sample sizes needed); zero iff the table lies in the Q1 family.
    """
    return fit_q1(table, n_restarts=n_restarts).l2_distance


# ---------------------------------------------------------------------------------------------
# Analysis of the published tables, figures and report
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TableAnalysis:
    table: OrderTable
    qq_observed: float
    qq_complement_observed: float
    qq_sides: tuple[float, float]
    order_effects: dict[str, float]
    agreement: dict[str, float]
    fit: Q1Fit
    fit_unitary: Q1Fit | None
    figure_path: str | None = None


def analyze_tables(tables: Mapping[str, OrderTable], n_restarts: int = 20) -> dict[str, TableAnalysis]:
    """Compute statistics and fits for every table; the unitary variant where information intervened."""
    out: dict[str, TableAnalysis] = {}
    for name, t in tables.items():
        out[name] = TableAnalysis(
            table=t,
            qq_observed=qq_statistic(t),
            qq_complement_observed=qq_statistic_complement(t),
            qq_sides=qq_equality_sides(t),
            order_effects=order_effect(t),
            agreement=agreement_conditionals(t),
            fit=fit_q1(t, n_restarts=n_restarts),
            fit_unitary=fit_q1_with_unitary(t, n_restarts=n_restarts) if t.intervening_information else None,
        )
    return out


# Chart chrome (reference palette of the dataviz skill; series slots 1-3, validated light mode).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_table(analysis: TableAnalysis, out_path: str | Path) -> Path:
    """Grouped bars, observed vs predicted cells, saved as PNG (headless Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, fit, fit_u = analysis.table, analysis.fit, analysis.fit_unitary
    series: list[tuple[str, np.ndarray]] = [
        ("Observed", cells_to_vector(t.cells)),
        (f"Q1 fit (alpha={fit.alpha:.3f}, phi={fit.phi:.3f})", cells_to_vector(fit.predicted)),
    ]
    if fit_u is not None:
        series.append(
            (
                f"Q1 + U(theta) fit (alpha={fit_u.alpha:.3f}, phi={fit_u.phi:.3f}, theta={fit_u.theta:.3f})",
                cells_to_vector(fit_u.predicted),
            )
        )
    n_series = len(series)
    x = np.arange(len(CELLS))
    group = 0.62
    width = group / n_series
    gap = 0.03  # surface gap between adjacent bars

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for i, (label, values) in enumerate(series):
        offs = (i - (n_series - 1) / 2) * width
        ax.bar(x + offs, values, width=width - gap, color=_SERIES[i], label=label, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c[:2]},{c[2:]}" for c in CELLS], color=_INK2)
    ax.set_ylabel("Joint probability", color=_INK2)
    ymax = min(1.0, 1.25 * max(float(v.max()) for _, v in series))
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, length=0)
    ax.axvline(3.5, color=_AXIS, linewidth=1)
    ax.text(1.5, 0.97 * ymax, "A asked first", ha="center", va="top", color=_MUTED, fontsize=9)
    ax.text(5.5, 0.97 * ymax, "B asked first", ha="center", va="top", color=_MUTED, fontsize=9)
    title = f"{t.name} — {t.status_label}"
    sub = f"QQ statistic q = {analysis.qq_observed:+.4f}; Q1 L2 distance = {fit.l2_distance:.4f}"
    if fit_u is not None:
        sub += f"; Q1+U L2 distance = {fit_u.l2_distance:.4f}"
    ax.set_title(f"{title}\n{sub}", color=_INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n_series)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=_SURFACE)
    plt.close(fig)
    return out


def _fmt_cells_table(fit: Q1Fit) -> str:
    lines = ["| cell | observed | predicted | residual (pred − obs) |", "|---|---:|---:|---:|"]
    for c in CELLS:
        lines.append(f"| {c[:2]},{c[2:]} | {fit.observed[c]:.4f} | {fit.predicted[c]:.4f} | {fit.residual[c]:+.4f} |")
    return "\n".join(lines)


def _fmt_fit(fit: Q1Fit) -> str:
    deg = 180 / math.pi
    parts = [f"alpha = {fit.alpha:.4f} rad ({fit.alpha * deg:.2f}°)", f"phi = {fit.phi:.4f} rad ({fit.phi * deg:.2f}°)"]
    if fit.with_unitary:
        parts.append(f"theta = {fit.theta:.4f} rad ({fit.theta * deg:.2f}°)")
    return (
        ", ".join(parts)
        + f"; loss ({fit.loss_kind}, sum of squared residuals) = {fit.loss:.3e}; "
        f"L2 distance = {fit.l2_distance:.4f}; max |residual| = {fit.max_abs_residual:.4f}; "
        f"{fit.n_converged}/{fit.n_restarts} restarts converged; "
        f"QQ statistic of the fitted table = {fit.qq_predicted:+.4e}"
    )


def render_report(results: Mapping[str, TableAnalysis], *, today: str | None = None) -> str:
    """Markdown report; every number comes from ``results``."""
    today = today or _dt.date.today().isoformat()
    any_table = next(iter(results.values())).table
    lines: list[str] = []
    add = lines.append
    add("# Q1 order-effect projector model and the QQ equality on dataset D")
    add("")
    add(
        f"Generated {today} by `python -m bre.models.quantum.q1_order --report`. Every number below "
        f"is computed by that command from `{Path(any_table.source_path).name}` "
        "(`data/raw/order_effects/`); nothing is typed in by hand."
    )
    add("")
    add("## Model and statistic")
    add("")
    add(
        "Q1 (PLAN.md §4) is a quantum-probability model on a classical CPU: state "
        "psi = (cos alpha, sin alpha) in R^2, question A = projector onto e0, question B = projector "
        "onto (cos phi, sin phi), sequential answers by the Born rule with Lüders update, "
        "P(Ay,By) = ||P_By P_Ay psi||^2 and so on for the eight cells. "
        "The QQ statistic is q = [P(AyBn) + P(AnBy)] − [P(ByAn) + P(BnAy)] (Wang & Busemeyer 2013; "
        "Wang, Solloway, Shiffrin & Busemeyer 2014); the equality q = 0 holds for every state, dimension "
        "and pair of projectors, so it is parameter-free. Cell key: first letter = question asked first, "
        "y/n = its answer; AyBn = P(A first, yes; then B, no). The fitted parameters are the canonical "
        "representatives (phi in [0, pi/2], alpha in [0, pi), theta in (−pi/2, pi/2]) of the symmetry "
        "orbits documented in the module. The fit is least squares on the eight cells with 20 restarts "
        "from a deterministic grid. In R^2 the model also forces the four agreement conditionals to equal "
        "cos^2 phi, so a nonzero residual is expected even when q = 0; the L2 distance is a descriptive "
        "measure of that, not a test."
    )
    add("")
    add("## Provenance")
    add("")
    add(f"> {any_table.provenance}")
    add("")
    add("## Results")
    add("")
    for name, r in results.items():
        t = r.table
        add(f"### {name} — {t.status_label}")
        add("")
        if not t.verified:
            add(
                "**This table is an UNVERIFIED secondary transcription.** It appears here only as an "
                "implementation check of the code path, not as a reported result; the JSON verification "
                "note reproduced below is binding."
            )
            add("")
        add(f"* Description: {t.description}")
        add(f'* Verification (verbatim from the JSON): "{t.verification}"')
        add(f"* Intervening information between the questions: {'yes' if t.intervening_information else 'no'}")
        oe = r.order_effects
        add(
            f"* Observed marginals: P(A yes | A first) = {oe['P_A_yes_first']:.4f}, "
            f"P(A yes | A second) = {oe['P_A_yes_second']:.4f}, order effect delta_A = {oe['delta_A']:+.4f}; "
            f"P(B yes | B first) = {oe['P_B_yes_first']:.4f}, P(B yes | B second) = {oe['P_B_yes_second']:.4f}, "
            f"delta_B = {oe['delta_B']:+.4f}"
        )
        lhs, rhs = r.qq_sides
        add(
            f"* Observed QQ statistic: q = {r.qq_observed:+.4f} "
            f"(P(AyBn) + P(AnBy) = {lhs:.4f} vs P(ByAn) + P(BnAy) = {rhs:.4f}; "
            f"complementary form [P(ByAy) + P(BnAn)] − [P(AyBy) + P(AnBn)] = {r.qq_complement_observed:+.4f})"
        )
        ag = r.agreement
        add(
            f"* Observed agreement conditionals: P(By|Ay) = {ag['P_By_given_Ay']:.4f}, "
            f"P(Bn|An) = {ag['P_Bn_given_An']:.4f}, P(Ay|By) = {ag['P_Ay_given_By']:.4f}, "
            f"P(An|Bn) = {ag['P_An_given_Bn']:.4f}; the rank-1 R^2 model forces all four to equal "
            f"cos^2 phi, which the fit puts at {math.cos(r.fit.phi) ** 2:.4f}"
        )
        add(f"* Q1 fit (theta = 0): {_fmt_fit(r.fit)}")
        add("")
        add(_fmt_cells_table(r.fit))
        add("")
        if r.fit_unitary is not None:
            fu = r.fit_unitary
            add(
                "Information variant (rotation U(theta) applied between the questions in both orders, "
                "because information intervened): "
                f"{_fmt_fit(fu)}. Closed-form q of this variant, −sin(2 theta) sin(2 phi) = "
                f"{qq_statistic_unitary_closed_form(fu.phi, fu.theta):+.4f}, versus observed q = {r.qq_observed:+.4f}."
            )
            add("")
            add(_fmt_cells_table(fu))
            add("")
        if r.figure_path:
            rel = Path(r.figure_path)
            try:
                rel = rel.relative_to(PROJECT_ROOT / "reports")
            except ValueError:
                pass
            add(f"Figure: `{rel.as_posix()}` (observed vs predicted cells).")
            add("")
    add("## Uncertainty")
    add("")
    add(
        "The JSON carries proportions only, without respondent counts, so no confidence interval or "
        "significance test is computed for any statistic above; the weighted-multinomial fit option "
        "exists in the code but is unused."
    )
    add("")
    add("## Interpretation")
    add("")
    verified = [n for n, r in results.items() if r.table.verified]
    unverified = [n for n, r in results.items() if not r.table.verified]
    qv = ", ".join(f"{n}: q = {results[n].qq_observed:+.4f}" for n in verified)
    qu = ", ".join(f"{n}: q = {results[n].qq_observed:+.4f}" for n in unverified)
    add(
        f"What this shows: the Q1 code reproduces the QQ equality exactly on model-generated tables "
        f"(unit tests, |q| < 1e-12 for random parameters) and computes the statistic on the one verified "
        f"published table ({qv}). What it does not show: this is an implementation check on a single "
        f"verified table, not a replication of the 70-survey result of Wang et al. (2014); without sample "
        f"sizes the observed q cannot be compared with its sampling error, so no claim that the equality "
        f"holds or fails on real data is made here. The unverified tables ({qu}) are shown only to "
        f"exercise the code path, including the information variant for the table with intervening "
        f"information; their values are not results until the owner verifies the transcription "
        f"(SOURCE.txt lists the action). The fitted (alpha, phi) are descriptive: two parameters against "
        f"six free cell values, with the R^2 model imposing constraints beyond the QQ equality, so the "
        f"residuals say how far each table is from the rank-1 R^2 family, nothing about causality or about "
        f"whether a quantum-probability model is preferable to a classical one (that is the Phase 4 question)."
    )
    add("")
    return "\n".join(lines)


def write_report(
    tables_path: str | Path | None = None,
    report_path: str | Path | None = None,
    figure_dir: str | Path | None = None,
    n_restarts: int = 20,
    today: str | None = None,
) -> tuple[Path, dict[str, TableAnalysis]]:
    """Run the analysis on the catalogued tables, save the figures and the Markdown report."""
    tables = load_tables(tables_path)
    results = analyze_tables(tables, n_restarts=n_restarts)
    fig_dir = Path(figure_dir) if figure_dir is not None else DEFAULT_FIGURE_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    with_figs: dict[str, TableAnalysis] = {}
    for name, r in results.items():
        fig_path = plot_table(r, fig_dir / f"q1_{name}.png")
        with_figs[name] = TableAnalysis(**{**asdict_shallow(r), "figure_path": str(fig_path)})
    out = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(with_figs, today=today), encoding="utf-8")
    return out, with_figs


def asdict_shallow(r: TableAnalysis) -> dict[str, Any]:
    """Field dict of a TableAnalysis without recursing into the nested dataclasses."""
    return {k: getattr(r, k) for k in r.__dataclass_fields__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="fit the catalogued tables and write the report")
    parser.add_argument("--data", default=None, help=f"tables JSON (default {DEFAULT_TABLES_PATH})")
    parser.add_argument("--out", default=None, help=f"report path (default {DEFAULT_REPORT_PATH})")
    parser.add_argument("--fig-dir", default=None, help=f"figure directory (default {DEFAULT_FIGURE_DIR})")
    parser.add_argument("--n-restarts", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.report:
        parser.print_help()
        return 2
    out, results = write_report(args.data, args.out, args.fig_dir, n_restarts=args.n_restarts)
    for name, r in results.items():
        line = (
            f"{name} [{r.table.status_label}]: q = {r.qq_observed:+.4f}; Q1 alpha = {r.fit.alpha:.4f}, "
            f"phi = {r.fit.phi:.4f}, L2 = {r.fit.l2_distance:.4f}"
        )
        if r.fit_unitary is not None:
            fu = r.fit_unitary
            line += f"; Q1+U alpha = {fu.alpha:.4f}, phi = {fu.phi:.4f}, theta = {fu.theta:.4f}, L2 = {fu.l2_distance:.4f}"
        print(line)
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
