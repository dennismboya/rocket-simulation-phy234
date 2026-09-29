"""Tests for bre.models.quantum.q1_order: the Q1 order-effect projector model and the QQ equality.

Math checks required by PLAN.md §4: projector algebra, probabilities sum to one, the QQ equality
exact on Q1-generated data, Lüders collapse idempotent, recovery of known parameters. Real-table
checks: the catalogued JSON loads, the statistic is computed, the verification status propagates.
The two unverified tables are used here only with the 'unverified' flag asserted, as their JSON
verification note requires.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from bre.models.quantum import q1_order as q1

PI = math.pi
RANDOM_PARAMS = [tuple(x) for x in np.random.default_rng(1234).uniform(-2 * PI, 2 * PI, size=(12, 3))]
"""Twelve (alpha, phi, theta) triples over four periods; fixed seed so the run is reproducible."""

GENERIC_PARAMS = [(0.35, 0.6), (1.1, 0.7), (2.0, 1.2), (2.8, 0.25), (0.9, 1.45)]
"""Points inside the canonical domain and away from its degenerate boundaries (identifiable)."""


# ---------------------------------------------------------------------------------------------
# Hilbert-space primitives
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.3, PI / 4, 1.2, PI / 2, 2.5, -0.7])
def test_projectors_idempotent_symmetric(phi):
    for p in q1.projectors(phi):
        assert np.allclose(p @ p, p, atol=1e-14)  # Lüders collapse applied twice = once
        assert np.allclose(p, p.T, atol=1e-14)
        assert abs(np.trace(p) - 1.0) < 1e-14  # rank one


@pytest.mark.parametrize("phi", [0.0, 0.3, PI / 4, 1.2, PI / 2, 2.5])
def test_projectors_orthogonal_and_complete(phi):
    p_ay, p_an, p_by, p_bn = q1.projectors(phi)
    assert np.allclose(p_ay @ p_an, 0.0, atol=1e-14)
    assert np.allclose(p_by @ p_bn, 0.0, atol=1e-14)
    assert np.allclose(p_ay + p_an, np.eye(2), atol=1e-14)
    assert np.allclose(p_by + p_bn, np.eye(2), atol=1e-14)
    # A and B commute iff phi is a multiple of pi/2
    comm = p_ay @ p_by - p_by @ p_ay
    commuting = math.isclose(math.sin(2 * phi), 0.0, abs_tol=1e-12)
    assert np.allclose(comm, 0.0, atol=1e-12) == commuting


@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0, PI, -2.2])
def test_state_is_normalised(alpha):
    psi = q1.state(alpha)
    assert abs(psi @ psi - 1.0) < 1e-14


@pytest.mark.parametrize("theta", [0.0, 0.3, PI / 8, -1.0, 2.9])
def test_rotation_is_orthogonal_with_unit_determinant(theta):
    u = q1.rotation(theta)
    assert np.allclose(u.T @ u, np.eye(2), atol=1e-14)
    assert abs(np.linalg.det(u) - 1.0) < 1e-14


def test_sequential_probability_is_born_rule_with_lueders_update():
    psi = q1.state(0.6)
    p_ay, _, p_by, _ = q1.projectors(0.9)
    joint = q1.sequential_probability(psi, p_ay, p_by)
    p_first = float((p_ay @ psi) @ (p_ay @ psi))
    collapsed = p_ay @ psi / math.sqrt(p_first)
    p_second_given_first = float((p_by @ collapsed) @ (p_by @ collapsed))
    assert math.isclose(joint, p_first * p_second_given_first, abs_tol=1e-14)
    assert math.isclose(joint, math.cos(0.6) ** 2 * math.cos(0.9) ** 2, abs_tol=1e-14)


# ---------------------------------------------------------------------------------------------
# Cells: normalisation, closed form, QQ equality, order effects
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,phi,theta", RANDOM_PARAMS)
def test_each_order_sums_to_one_and_cells_nonnegative(alpha, phi, theta):
    for cells in (q1.q1_cells(alpha, phi), q1.q1_cells_with_unitary(alpha, phi, theta)):
        assert set(cells) == set(q1.CELLS)
        assert all(v >= -1e-15 for v in cells.values())
        assert math.isclose(sum(cells[c] for c in q1.A_FIRST_CELLS), 1.0, abs_tol=1e-12)
        assert math.isclose(sum(cells[c] for c in q1.B_FIRST_CELLS), 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("alpha,phi,theta", RANDOM_PARAMS)
def test_closed_form_matches_born_rule_matrices(alpha, phi, theta):
    born = q1.cells_to_vector(q1.q1_cells_with_unitary(alpha, phi, theta))
    closed, _ = q1._cells_vector(np.array([alpha, phi, theta]), True)
    assert np.allclose(born, closed, atol=1e-14)
    born0 = q1.cells_to_vector(q1.q1_cells(alpha, phi))
    closed0, _ = q1._cells_vector(np.array([alpha, phi]), False)
    assert np.allclose(born0, closed0, atol=1e-14)
    # the documented closed form itself
    c1, c2, s = math.cos(alpha) ** 2, math.cos(alpha - phi) ** 2, math.sin(phi) ** 2
    expected = [(1 - s) * c1, s * c1, s * (1 - c1), (1 - s) * (1 - c1), (1 - s) * c2, s * c2, s * (1 - c2), (1 - s) * (1 - c2)]
    assert np.allclose(born0, expected, atol=1e-14)


@pytest.mark.parametrize("alpha,phi,theta", RANDOM_PARAMS)
def test_analytic_jacobian_matches_finite_differences(alpha, phi, theta):
    p = np.array([alpha, phi, theta])
    _, jac = q1._cells_vector(p, True)
    eps = 1e-6
    for k in range(3):
        e = np.zeros(3)
        e[k] = eps
        fd = (q1._cells_vector(p + e, True)[0] - q1._cells_vector(p - e, True)[0]) / (2 * eps)
        assert np.allclose(jac[:, k], fd, atol=1e-8)


@pytest.mark.parametrize("alpha,phi,theta", RANDOM_PARAMS)
def test_qq_equality_exact_on_q1_generated_data(alpha, phi, theta):
    cells = q1.q1_cells(alpha, phi)
    assert abs(q1.qq_statistic(cells)) < 1e-12
    assert abs(q1.qq_statistic_complement(cells)) < 1e-12
    lhs, rhs = q1.qq_equality_sides(cells)
    assert math.isclose(lhs, rhs, abs_tol=1e-12)
    assert math.isclose(lhs, math.sin(phi) ** 2, abs_tol=1e-12)  # both sides equal sin^2 phi
    assert q1.qq_equality_holds(cells)


def test_qq_forms_coincide_on_any_normalised_tables(rng):
    """The two forms agree for arbitrary (non-model) tables: it only needs each order to sum to one."""
    for _ in range(50):
        a = rng.dirichlet(np.ones(4))
        b = rng.dirichlet(np.ones(4))
        table = dict(zip(q1.CELLS, np.concatenate([a, b]), strict=True))
        assert math.isclose(q1.qq_statistic(table), q1.qq_statistic_complement(table), abs_tol=1e-12)
        # and a generic random table violates the equality (the statistic is not identically zero)
    violating = {c: v for c, v in zip(q1.CELLS, [0.4, 0.1, 0.2, 0.3, 0.2, 0.3, 0.1, 0.4], strict=True)}
    assert abs(q1.qq_statistic(violating)) > 0.09
    assert not q1.qq_equality_holds(violating)


@pytest.mark.parametrize("phi", [0.0, PI / 2, PI, -PI / 2, 3 * PI / 2])
@pytest.mark.parametrize("alpha", [0.2, 0.9, 1.7, 2.6])
def test_order_effects_vanish_when_projectors_commute(alpha, phi):
    oe = q1.order_effect(q1.q1_cells(alpha, phi))
    assert abs(oe["delta_A"]) < 1e-12
    assert abs(oe["delta_B"]) < 1e-12


@pytest.mark.parametrize(
    "alpha,phi,expected_delta_a",
    [(0.0, PI / 4, 0.5), (0.0, PI / 3, 0.375), (PI / 6, PI / 3, 0.375)],
)
def test_order_effects_present_when_projectors_do_not_commute(alpha, phi, expected_delta_a):
    oe = q1.order_effect(q1.q1_cells(alpha, phi))
    assert math.isclose(oe["delta_A"], expected_delta_a, abs_tol=1e-12)
    assert max(abs(oe["delta_A"]), abs(oe["delta_B"])) > 0.1


@pytest.mark.parametrize("alpha,phi,theta", RANDOM_PARAMS[:6])
def test_order_effect_reports_both_marginals_in_each_order(alpha, phi, theta):
    cells = q1.q1_cells(alpha, phi)
    oe = q1.order_effect(cells)
    assert set(oe) == {"P_A_yes_first", "P_A_yes_second", "delta_A", "P_B_yes_first", "P_B_yes_second", "delta_B"}
    assert math.isclose(oe["P_A_yes_first"], math.cos(alpha) ** 2, abs_tol=1e-12)
    assert math.isclose(oe["P_B_yes_first"], math.cos(alpha - phi) ** 2, abs_tol=1e-12)
    assert math.isclose(oe["P_A_yes_second"], cells["ByAy"] + cells["BnAy"], abs_tol=1e-14)
    assert math.isclose(oe["P_B_yes_second"], cells["AyBy"] + cells["AnBy"], abs_tol=1e-14)
    assert math.isclose(oe["delta_A"], oe["P_A_yes_first"] - oe["P_A_yes_second"], abs_tol=1e-14)
    assert math.isclose(oe["delta_B"], oe["P_B_yes_first"] - oe["P_B_yes_second"], abs_tol=1e-14)


# ---------------------------------------------------------------------------------------------
# Information variant (rotation between the questions)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,phi,theta", RANDOM_PARAMS)
def test_unitary_variant_with_theta_zero_equals_q1(alpha, phi, theta):
    a = q1.cells_to_vector(q1.q1_cells_with_unitary(alpha, phi, 0.0))
    b = q1.cells_to_vector(q1.q1_cells(alpha, phi))
    assert np.allclose(a, b, atol=1e-14)
    assert abs(q1.qq_statistic(q1.q1_cells_with_unitary(alpha, phi, 0.0))) < 1e-12


@pytest.mark.parametrize("alpha", [0.0, 0.7, 2.1])
@pytest.mark.parametrize("phi,theta", [(PI / 4, PI / 8), (0.6, 0.3), (1.2, -0.4), (PI / 4, PI / 4)])
def test_unitary_variant_breaks_the_qq_equality(alpha, phi, theta):
    cells = q1.q1_cells_with_unitary(alpha, phi, theta)
    q = q1.qq_statistic(cells)
    assert abs(q) > 1e-3
    assert math.isclose(q, q1.qq_statistic_unitary_closed_form(phi, theta), abs_tol=1e-12)
    assert math.isclose(q, -math.sin(2 * theta) * math.sin(2 * phi), abs_tol=1e-12)
    assert math.isclose(q, q1.qq_statistic_complement(cells), abs_tol=1e-12)


@pytest.mark.parametrize("theta", [PI / 2, PI, -PI / 2])
def test_unitary_variant_keeps_the_equality_when_theta_is_a_multiple_of_half_pi(theta):
    assert abs(q1.qq_statistic(q1.q1_cells_with_unitary(0.7, 0.9, theta))) < 1e-12


# ---------------------------------------------------------------------------------------------
# Identifiability and canonical form
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,phi,theta", RANDOM_PARAMS)
def test_documented_symmetries_leave_the_cells_unchanged(alpha, phi, theta):
    ref = q1.cells_to_vector(q1.q1_cells_with_unitary(alpha, phi, theta))
    for a2, f2, t2 in [
        (alpha + PI, phi, theta),
        (alpha, phi + PI, theta),
        (alpha, phi, theta + PI),
        (-alpha, -phi, -theta),
        (PI - alpha, PI - phi, PI - theta),
    ]:
        assert np.allclose(q1.cells_to_vector(q1.q1_cells_with_unitary(a2, f2, t2)), ref, atol=1e-13)
    # phi -> -phi alone is NOT a symmetry in general
    alone = q1.cells_to_vector(q1.q1_cells_with_unitary(alpha, -phi, theta))
    if abs(math.sin(2 * alpha) * math.sin(2 * phi)) > 1e-3:
        assert not np.allclose(alone, ref, atol=1e-6)


@pytest.mark.parametrize("alpha,phi", GENERIC_PARAMS)
def test_canonicalize_maps_every_orbit_point_to_the_same_representative(alpha, phi):
    theta = 0.3
    ref = q1.canonicalize(alpha, phi, theta)
    assert ref == pytest.approx((alpha, phi, theta), abs=1e-12)
    for a2, f2, t2 in [(alpha + PI, phi, theta), (alpha - 3 * PI, phi + 2 * PI, theta + PI), (-alpha, -phi, -theta), (PI - alpha, PI - phi, PI - theta)]:
        assert q1.canonicalize(a2, f2, t2) == pytest.approx(ref, abs=1e-12)
    a_c, f_c, t_c = q1.canonicalize(alpha, phi, theta)
    assert 0.0 <= a_c < PI and 0.0 <= f_c <= PI / 2 and -PI / 2 < t_c <= PI / 2


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,phi", GENERIC_PARAMS)
def test_fit_recovers_known_parameters_from_generated_cells(alpha, phi):
    cells = q1.q1_cells(alpha, phi)
    fit = q1.fit_q1(cells, n_restarts=20)
    assert fit.with_unitary is False and fit.theta == 0.0
    assert fit.loss_kind == "least_squares"
    assert abs(fit.alpha - alpha) < 1e-6 and abs(fit.phi - phi) < 1e-6
    assert fit.loss < 1e-20 and fit.l2_distance < 1e-10 and fit.max_abs_residual < 1e-10
    assert np.allclose(q1.cells_to_vector(fit.predicted), q1.cells_to_vector(cells), atol=1e-10)
    assert abs(fit.qq_predicted) < 1e-12 and abs(fit.qq_observed) < 1e-12
    assert fit.n_restarts == 20 and 1 <= fit.n_converged <= 20
    assert fit.table_name is None and fit.verified is None and fit.status_label is None


def test_fit_is_deterministic_and_the_grid_is_reproducible():
    cells = q1.q1_cells(1.1, 0.7)
    a, b = q1.fit_q1(cells), q1.fit_q1(cells)
    assert a.params == b.params and a.loss == b.loss
    assert np.array_equal(q1._restart_grid(20, False), q1._restart_grid(20, False))
    assert q1._restart_grid(20, True).shape == (20, 3)
    with pytest.raises(ValueError):
        q1.fit_q1(cells, n_restarts=0)


@pytest.mark.parametrize("alpha,phi", GENERIC_PARAMS[:3])
def test_weighted_multinomial_option_recovers_parameters(alpha, phi):
    cells = q1.q1_cells(alpha, phi)
    fit = q1.fit_q1(cells, n_restarts=8, weights={"A_first": 150, "B_first": 250})
    assert fit.loss_kind == "weighted_multinomial"
    assert abs(fit.alpha - alpha) < 1e-6 and abs(fit.phi - phi) < 1e-6
    assert fit.l2_distance < 1e-6
    same = q1.fit_q1(cells, n_restarts=8, weights=200)
    assert abs(same.alpha - alpha) < 1e-6
    with pytest.raises(ValueError):
        q1.fit_q1(cells, weights={"A_first": 0, "B_first": 10})


@pytest.mark.parametrize("alpha,phi,theta", [(1.1, 0.7, 0.3), (0.5, 1.0, -0.6), (2.3, 0.4, 1.2)])
def test_unitary_fit_recovers_known_parameters(alpha, phi, theta):
    cells = q1.q1_cells_with_unitary(alpha, phi, theta)
    fit = q1.fit_q1_with_unitary(cells, n_restarts=20)
    a_c, f_c, t_c = q1.canonicalize(alpha, phi, theta)
    assert fit.with_unitary is True
    assert abs(fit.alpha - a_c) < 1e-6 and abs(fit.phi - f_c) < 1e-6 and abs(fit.theta - t_c) < 1e-6
    assert fit.l2_distance < 1e-10
    assert math.isclose(fit.qq_predicted, q1.qq_statistic(cells), abs_tol=1e-10)


def test_q1_distance_is_zero_in_family_and_positive_outside():
    assert q1.q1_distance(q1.q1_cells(0.9, 0.4)) < 1e-10
    outside = {c: v for c, v in zip(q1.CELLS, [0.4, 0.1, 0.2, 0.3, 0.2, 0.3, 0.1, 0.4], strict=True)}
    d = q1.q1_distance(outside)
    assert d > 0.05
    assert math.isclose(d, q1.fit_q1(outside).l2_distance, abs_tol=1e-12)


# ---------------------------------------------------------------------------------------------
# The catalogued published tables (dataset D)
# ---------------------------------------------------------------------------------------------


def test_real_tables_load_with_provenance_and_normalised_orders():
    assert q1.DEFAULT_TABLES_PATH.exists()
    tables = q1.load_tables()
    assert set(tables) == {"clinton_gore", "black_white", "rose_jackson"}
    for t in tables.values():
        assert set(t.cells) == set(q1.CELLS)
        assert math.isclose(sum(t.cells[c] for c in q1.A_FIRST_CELLS), 1.0, abs_tol=5e-4)
        assert math.isclose(sum(t.cells[c] for c in q1.B_FIRST_CELLS), 1.0, abs_tol=5e-4)
        assert "qinst_ozawa_khrennikov" in t.provenance
        assert t.verification


def test_real_table_verification_status_is_read_verbatim():
    tables = q1.load_tables()
    assert tables["clinton_gore"].verified is True
    assert tables["clinton_gore"].verification.startswith("VERIFIED against Ozawa & Khrennikov (2021)")
    assert tables["clinton_gore"].status_label == "VERIFIED"
    assert tables["clinton_gore"].intervening_information is False
    for name in ("black_white", "rose_jackson"):
        assert tables[name].verified is False
        assert tables[name].verification.startswith("UNVERIFIED secondary transcription")
        assert tables[name].status_label == q1.UNVERIFIED_LABEL
    assert tables["rose_jackson"].intervening_information is True
    assert tables["black_white"].intervening_information is False


def test_qq_statistic_on_real_tables_is_computed_from_the_cells():
    tables = q1.load_tables()
    for t in tables.values():
        c = t.cells
        expected = (c["AyBn"] + c["AnBy"]) - (c["ByAn"] + c["BnAy"])
        assert math.isclose(q1.qq_statistic(t), expected, abs_tol=1e-12)
        assert math.isclose(q1.qq_statistic_complement(t), expected, abs_tol=5e-4)  # rounding of published cells
        assert math.isfinite(q1.qq_statistic(t))
    # the table with intervening information shows the largest |q| of the three (a code check,
    # not a substantive claim: two of the tables are unverified transcriptions)
    qs = {n: abs(q1.qq_statistic(t)) for n, t in tables.items()}
    assert max(qs, key=qs.get) == "rose_jackson"


def test_unverified_flag_propagates_to_fits_and_analysis():
    tables = q1.load_tables()
    fit = q1.fit_q1(tables["black_white"], n_restarts=4)
    assert fit.verified is False and fit.status_label == q1.UNVERIFIED_LABEL
    assert fit.verification == tables["black_white"].verification
    assert fit.table_name == "black_white" and fit.intervening_information is False
    fit_v = q1.fit_q1(tables["clinton_gore"], n_restarts=4)
    assert fit_v.verified is True and fit_v.status_label == "VERIFIED"
    results = q1.analyze_tables(tables, n_restarts=4)
    assert results["rose_jackson"].fit_unitary is not None
    assert results["rose_jackson"].fit_unitary.verified is False
    assert results["clinton_gore"].fit_unitary is None and results["black_white"].fit_unitary is None
    d = fit.to_dict()
    assert d["verified"] is False and d["verification"].startswith("UNVERIFIED")


def test_load_tables_rejects_malformed_tables(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"x": {"verification": "VERIFIED", "intervening_information": false, "AyBy": 0.5, "AyBn": 0.5, "AnBy": 0.5, "AnBn": 0.5, "ByAy": 0.25, "ByAn": 0.25, "BnAy": 0.25, "BnAn": 0.25}}')
    with pytest.raises(ValueError, match="sum to"):
        q1.load_tables(bad)
    bad.write_text('{"x": {"verification": "VERIFIED", "intervening_information": false, "AyBy": 1.0}}')
    with pytest.raises(ValueError, match="lacks cells"):
        q1.load_tables(bad)


def test_report_command_writes_report_and_figures_with_labels(tmp_path):
    out = tmp_path / "q1.md"
    figs = tmp_path / "figs"
    rc = q1.main(["--report", "--out", str(out), "--fig-dir", str(figs), "--n-restarts", "4"])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    for name in ("clinton_gore", "black_white", "rose_jackson"):
        assert (figs / f"q1_{name}.png").stat().st_size > 0
        assert f"### {name}" in text
    assert "### clinton_gore — VERIFIED" in text
    assert f"### black_white — {q1.UNVERIFIED_LABEL}" in text
    assert f"### rose_jackson — {q1.UNVERIFIED_LABEL}" in text
    assert "no confidence interval" in text
    assert "not a replication" in text
    assert "advantage" not in text.lower()  # CLAUDE.md wording rule: no such claim anywhere
    tables = q1.load_tables()
    assert f'"{tables["clinton_gore"].verification}"' in text  # verbatim
    assert f"q = {q1.qq_statistic(tables['clinton_gore']):+.4f}" in text
    assert "theta =" in text  # the unitary-variant fit for rose_jackson is reported
    assert q1.main([]) == 2  # no --report: help only


def test_report_paths_default_into_the_project_reports_folder():
    assert q1.DEFAULT_REPORT_PATH == q1.PROJECT_ROOT / "reports" / "q1_qq_equality.md"
    assert q1.DEFAULT_FIGURE_DIR == q1.PROJECT_ROOT / "reports" / "figures"
    assert (q1.PROJECT_ROOT / "PLAN.md").exists()
    assert Path(q1.DEFAULT_TABLES_PATH).is_file()
