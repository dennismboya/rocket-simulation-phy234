"""Tests of the G_Q generator (``bre.sim.gq``) and the generator command line (``bre.sim.cli``).

Fast tests: output validates against the schema, is synthetic, has ``n_subjects x 170 x 2`` rows;
the truth file round-trips; seeds reproduce; the population defaults are labelled synthetic; the
probabilities recorded in the truth are the ones Q4 computes from the same latents (generator and
model share the core code path); ``population_from_fit`` maps a fit back to a population; the
CLI writes files, reports missing generators and refuses a non-synthetic output directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bre import design as D
from bre import schema as S
from bre.models import build_model_data
from bre.models.data import covariate_columns
from bre.models.quantum import core as C
from bre.models.quantum.q2_context_unitary import Q2
from bre.models.quantum.q4_open_system import Q4
from bre.sim import GENERATOR_NAMES, cli, common, gq


@pytest.fixture(scope="module")
def small():
    """Three subjects, full design, mixed population, seed 0."""
    df, truth = gq.generate(3, 0)
    return df, truth


@pytest.fixture(scope="module")
def small_pure():
    df, truth = gq.generate(4, 1, mixed_population=False)
    return df, truth


# ---------------------------------------------------------------------------------------------
# Population defaults
# ---------------------------------------------------------------------------------------------


def test_population_defaults_are_labelled_synthetic():
    pop = gq.POPULATION_DEFAULTS
    assert pop["SOURCE"] == "synthetic default, not fitted to data"
    assert gq.NAME == "gq" and "gq" in GENERATOR_NAMES
    cols = covariate_columns()
    assert tuple(pop["covariate_columns"]) == cols
    assert pop["W"].shape == (len(cols), 4) and pop["b"].shape == (4,) and pop["sigma_v"].shape == (4,)
    assert pop["theta_L"].shape == (3,) and pop["mu_c"].shape == (4, 3) and pop["sigma_c"].shape == (4, 3)
    assert tuple(pop["contexts"]) == tuple(D.NONNULL_CONTEXTS)
    assert 0 < pop["phi"] < np.pi / 2 and pop["sigma_gamma"] > 0
    # the resolved population carries the label through, and overrides are recorded
    assert gq.resolve_population()["SOURCE"] == pop["SOURCE"]
    res = gq.resolve_population({"phi": 0.3})
    assert res["phi"] == 0.3 and "phi" in res["SOURCE"] and "synthetic default" in res["SOURCE"]
    assert gq.resolve_population({"sigma_c": 0.0})["sigma_c"].shape == (4, 3)
    with pytest.raises(KeyError):
        gq.resolve_population({"nope": 1})
    with pytest.raises(ValueError):
        gq.resolve_population({"theta_L": [1.0, 2.0]})
    with pytest.raises(ValueError):
        gq.resolve_population({"sigma_gamma": -1.0})


# ---------------------------------------------------------------------------------------------
# Output layout and validation
# ---------------------------------------------------------------------------------------------


def test_generate_validates_and_has_expected_shape(small):
    df, truth = small
    report = S.validate_frame(df, expect_synthetic=True, strict=False)
    assert report.ok, report.errors
    assert len(df) == 3 * D.N_ITEMS * 2
    assert df["is_synthetic"].all() and (df["dataset"] == gq.DATASET).all()
    assert set(df["elicitation_type"]) == {"binary_sell", "binary_yes_no"}
    assert (df["elicitation_type"] == "binary_sell").sum() == 3 * D.N_ITEMS
    assert set(df["response"].unique()) <= {0.0, 1.0}
    assert df["source_row_ref"].str.startswith("sim:gq:0:").all()
    assert set(df["subject_id"]) == {"S0001", "S0002", "S0003"}
    # every sell row has a tolerance row in the same item, in the item's order
    data = build_model_data(df)
    assert data.n_subjects == 3 and set(data.row_kind) == {"sell", "tol"}
    sell = data.sell_mask()
    tf = sell & (data.order_flag == 1)
    assert np.isfinite(data.tol_answer[tf]).all()
    assert np.isnan(data.tol_answer[sell & (data.order_flag == 0)]).all()
    assert tf.sum() == 3 * D.N_ITEMS // 2
    assert data.ctx_vocab == tuple(D.NONNULL_CONTEXTS)


def test_truth_contents(small):
    _, truth = small
    assert truth["generator"] == "gq" and truth["seed"] == 0 and truth["n_subjects"] == 3
    assert truth["n_items"] == D.N_ITEMS and truth["mixed_population"] is True
    assert truth["population"]["SOURCE"] == gq.SOURCE
    sub = truth["subjects"]
    assert len(sub["subject_id"]) == 3
    for key, shape in [("v", (3, 4)), ("psi_re", (3, 2)), ("psi_im", (3, 2)), ("theta_ctx", (3, 4, 3)), ("gamma", (3,)), ("bloch_theta", (3,)), ("bloch_phi", (3,))]:
        assert np.asarray(sub[key]).shape == shape, key
    items = truth["items"]
    assert len(items["item_id"]) == D.N_ITEMS
    for key in ("p_sell", "p_tol", "sell", "tol"):
        arr = np.asarray(items[key], dtype=float)
        assert arr.shape == (3, D.N_ITEMS) and np.all((arr >= 0) & (arr <= 1)), key
    psi = np.asarray(sub["psi_re"]) + 1j * np.asarray(sub["psi_im"])
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0)
    assert np.allclose(np.imag(psi[:, 0]), 0.0) and np.all(np.real(psi[:, 0]) >= 0)
    assert np.allclose(np.sin(np.asarray(sub["bloch_theta"]) / 2) ** 2, np.abs(psi[:, 1]) ** 2)
    assert np.all(np.asarray(sub["gamma"]) > 0)
    # the tolerance-first answers were drawn from the initial state's P(yes)
    assert np.allclose(np.asarray(sub["p_yes_initial"]), [float(C.born(jnp.asarray(p), C.tolerance_projector(truth["population"]["phi"]))) for p in psi])


def test_truth_round_trips_and_write_enforces_location(small, tmp_path: Path):
    df, truth = small
    out = tmp_path / "data" / "synthetic"
    pq_path, t_path = common.write_synthetic(df, truth, "gq_n3_seed0", out)
    assert pq_path.exists() and t_path.exists() and (out / "gq_n3_seed0.validation.md").exists()
    df2, truth2 = common.read_synthetic("gq_n3_seed0", out)
    assert df2.equals(df)
    assert truth2["is_synthetic"] is True and truth2["file"] == "gq_n3_seed0.parquet"
    assert truth2["population"]["SOURCE"] == gq.SOURCE
    assert np.allclose(np.asarray(truth2["items"]["p_sell"]), np.asarray(truth["items"]["p_sell"]))
    assert np.allclose(np.asarray(truth2["subjects"]["theta_ctx"]), np.asarray(truth["subjects"]["theta_ctx"]))
    with pytest.raises(S.SchemaError):
        common.write_synthetic(df, truth, "gq_wrong_place", tmp_path / "data" / "processed")


def test_seed_reproducibility():
    a, ta = gq.generate(3, 7)
    b, tb = gq.generate(3, 7)
    c, _ = gq.generate(3, 8)
    assert a.equals(b)
    assert np.array_equal(np.asarray(ta["items"]["p_sell"]), np.asarray(tb["items"]["p_sell"]))
    assert np.array_equal(np.asarray(ta["subjects"]["theta_ctx"]), np.asarray(tb["subjects"]["theta_ctx"]))
    assert not a["response"].equals(c["response"])
    assert json.dumps(common.jsonable(ta), sort_keys=True) == json.dumps(common.jsonable(tb), sort_keys=True)


def test_mixed_population_flag(small, small_pure):
    _, mixed = small
    _, pure = small_pure
    assert pure["mixed_population"] is False
    assert np.all(np.asarray(pure["subjects"]["gamma"]) == 0.0)
    assert all(v is None for v in pure["subjects"]["log_gamma"])
    g = np.asarray(mixed["subjects"]["gamma"])
    assert np.all(g > 0) and np.all(np.isfinite(np.asarray(mixed["subjects"]["log_gamma"], dtype=float)))
    # the LogNormal spans both regimes over a larger sample
    _, big = gq.generate(60, 3)
    gb = np.asarray(big["subjects"]["gamma"])
    assert gb.min() < 0.1 and gb.max() > 1.0


def test_empty_and_subset_designs():
    df, truth = gq.generate(0, 0)
    assert len(df) == 0 and truth["n_subjects"] == 0
    subset, _ = D.battery_subset(np.random.default_rng(0), 12)
    df, truth = gq.generate(2, 0, design=subset)
    assert len(df) == 2 * 12 * 2 and truth["n_items"] == 12
    assert S.validate_frame(df, expect_synthetic=True, strict=False).ok


# ---------------------------------------------------------------------------------------------
# Generator and model share the code path
# ---------------------------------------------------------------------------------------------


def _truth_params(truth, subject: int, log_gamma) -> dict:
    """Q2/Q4 parameters that reproduce subject ``subject``'s latents exactly: ``u`` = the state
    noise, ``theta_ctx`` = that subject's own context vector, ``log_gamma`` as given."""
    pop, sub = truth["population"], truth["subjects"]
    enc = {
        "W": jnp.asarray(pop["W"]),
        "b": jnp.asarray(pop["b"]),
        "u": jnp.asarray(sub["state_noise"]),
        "log_sigma_u": jnp.zeros(4),
    }
    return {
        "enc": enc,
        "theta_L": jnp.asarray(pop["theta_L"]),
        "theta_ctx": jnp.asarray(sub["theta_ctx"][subject]),
        "phi": jnp.asarray(pop["phi"]),
        "log_gamma": jnp.asarray(log_gamma),
        "gamma_mu": jnp.asarray(0.0),
        "gamma_log_sigma": jnp.asarray(0.0),
    }


def _rows_probs(df, truth, data, subject: int):
    rows = np.flatnonzero((data.subject_idx == subject) & data.sell_mask())
    item_pos = {k: i for i, k in enumerate(truth["items"]["item_id"])}
    ids = [ref.split(":")[-1] for ref in df["source_row_ref"].iloc[rows]]
    p_ref = np.asarray([truth["items"]["p_sell"][subject][item_pos[k]] for k in ids])
    return rows, p_ref


def test_truth_probabilities_match_q4_with_truth_parameters(small):
    df, truth = small
    data = build_model_data(df)
    model = Q4(data)
    log_gamma = np.log(np.asarray(truth["subjects"]["gamma"]))
    for s in range(3):
        params = _truth_params(truth, s, log_gamma)
        rows, p_ref = _rows_probs(df, truth, data, s)
        p = np.asarray(model.predict_proba(params, data))[rows]
        assert np.allclose(p, p_ref, atol=1e-12)
        psi = np.asarray(model.states(params, data))[s]
        assert np.allclose(psi, np.asarray(truth["subjects"]["psi_re"][s]) + 1j * np.asarray(truth["subjects"]["psi_im"][s]))


def test_pure_truth_probabilities_match_q2(small_pure):
    df, truth = small_pure
    data = build_model_data(df)
    model = Q2(data)
    for s in range(data.n_subjects):
        params = _truth_params(truth, s, np.zeros(data.n_subjects))
        rows, p_ref = _rows_probs(df, truth, data, s)
        p = np.asarray(model.predict_proba(params, data))[rows]
        assert np.allclose(p, p_ref, atol=1e-12)


def test_responses_are_bernoulli_draws_of_recorded_probabilities():
    """Over many items the sell rate tracks the recorded probabilities (a sanity check of the
    draw, not a distributional test)."""
    _, truth = gq.generate(40, 5)
    p = np.asarray(truth["items"]["p_sell"])
    y = np.asarray(truth["items"]["sell"])
    assert abs(y.mean() - p.mean()) < 0.02
    hi, lo = p > 0.8, p < 0.2
    assert y[hi].mean() > 0.7 and y[lo].mean() < 0.3
    p_tol = np.asarray(truth["items"]["p_tol"])
    t = np.asarray(truth["items"]["tol"])
    assert abs(t.mean() - p_tol.mean()) < 0.03
    # scenario-first tolerance answers come from the collapsed post-sell state: sin^2 phi or cos^2 phi
    phi = truth["population"]["phi"]
    sf = np.asarray(truth["items"]["order_flag"]) == 0
    expected = np.where(y[:, sf] == 1, np.sin(phi) ** 2, np.cos(phi) ** 2)
    assert np.allclose(p_tol[:, sf], expected)


# ---------------------------------------------------------------------------------------------
# population_from_fit
# ---------------------------------------------------------------------------------------------


def test_population_from_fit_round_trip(small):
    df, truth = small
    data = build_model_data(df)
    q4 = Q4(data)
    params = q4.init_params(jax.random.PRNGKey(1), data, 0)
    pop = gq.population_from_fit(params, data.ctx_vocab, x_columns=data.X_columns, source="unit test fit")
    assert pop["SOURCE"] == "unit test fit"
    assert np.allclose(pop["W"], np.asarray(params["enc"]["W"])) and np.allclose(pop["b"], np.asarray(params["enc"]["b"]))
    assert np.allclose(pop["mu_c"], np.asarray(params["theta_ctx"])) and np.allclose(pop["sigma_c"], 0.0)
    assert np.allclose(pop["sigma_v"], np.exp(np.asarray(params["enc"]["log_sigma_u"])))
    assert pop["mu_gamma"] == float(params["gamma_mu"]) and pop["sigma_gamma"] == float(np.exp(params["gamma_log_sigma"]))
    df2, truth2 = gq.generate(2, 0, population=pop)
    assert len(df2) == 2 * D.N_ITEMS * 2 and truth2["population"]["SOURCE"] == "unit test fit"
    # a Q2 fit carries no dephasing population: the defaults are used and the label says so
    q2_params = {k: v for k, v in params.items() if k not in ("log_gamma", "gamma_mu", "gamma_log_sigma")}
    pop2 = gq.population_from_fit(q2_params, data.ctx_vocab, source="q2 fit")
    assert "mu_gamma" not in pop2 and "synthetic default" in pop2["SOURCE"]
    res = gq.resolve_population(pop2)
    assert res["mu_gamma"] == gq.POPULATION_DEFAULTS["mu_gamma"]
    with pytest.raises(ValueError):
        gq.population_from_fit(params, ("news:recession",))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def test_cli_writes_files_and_reports_missing_generators(tmp_path: Path, capsys):
    out = tmp_path / "data" / "synthetic"
    rc = cli.main(["--generators", "gq,zz_not_a_generator", "--n", "2", "--seeds", "2", "--out", str(out)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "missing" in captured and "zz_not_a_generator" in captured
    for seed in (0, 1):
        assert (out / f"gq_n2_seed{seed}.parquet").exists() and (out / f"gq_n2_seed{seed}.truth.json").exists()
    manifest = json.loads((out / cli.MANIFEST_NAME).read_text())
    assert manifest["is_synthetic"] is True and len(manifest["entries"]) == 2
    assert {e["seed"] for e in manifest["entries"]} == {0, 1}
    assert all(e["generator"] == "gq" and e["n_subjects"] == 2 and e["n_rows"] == 2 * D.N_ITEMS * 2 for e in manifest["entries"])
    df = S.read_events(out / "gq_n2_seed1.parquet")
    assert df["is_synthetic"].all() and len(df) == 2 * D.N_ITEMS * 2


def test_cli_seed_lists_dry_run_and_no_mixed(tmp_path: Path, capsys):
    out = tmp_path / "data" / "synthetic"
    assert cli.parse_seeds("3") == [0, 1, 2] and cli.parse_seeds("0,4") == [0, 4]
    rc = cli.main(["--generators", "gq", "--n", "1,2", "--seeds", "0,3", "--out", str(out), "--dry-run"])
    assert rc == 0 and not list(out.glob("*.parquet"))
    assert "would generate gq_n2_seed3" in capsys.readouterr().out
    rc = cli.main(["--generators", "gq", "--n", "2", "--seeds", "1", "--out", str(out), "--no-mixed"])
    assert rc == 0
    truth = json.loads((out / "gq_n2_seed0.truth.json").read_text())
    assert truth["mixed_population"] is False and all(g == 0.0 for g in truth["subjects"]["gamma"])


def test_cli_refuses_non_synthetic_output_and_all_missing(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.main(["--generators", "gq", "--n", "1", "--seeds", "1", "--out", str(tmp_path / "data" / "processed")])
    rc = cli.main(["--generators", "zz_none", "--n", "1", "--seeds", "1", "--out", str(tmp_path / "data" / "synthetic")])
    assert rc == 1
    assert cli.load_generator("zz_none") is None and cli.load_generator("gq") is gq
