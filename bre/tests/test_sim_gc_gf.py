"""Tests for the classical synthetic generators ``bre.sim.gc`` (G_C) and ``bre.sim.gf`` (G_F).

Fast checks: the frames validate under the schema (``is_synthetic`` True, two rows per item, the
real-vs-synthetic location rule through ``write_synthetic``), seed reproducibility, the analytic
structure of each generator from its truth (G_C: law of total probability exact, additive and
position-independent contexts; G_F: non-additive, order-asymmetric pair effects), the Markov and
population-mixture variants, and the round trip into ``ModelData``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bre import design as D
from bre import schema as S
from bre.models import build_model_data
from bre.sim import common, gc, gf

N_ITEMS = D.N_ITEMS  # 170


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


@pytest.fixture(scope="module")
def gc_small():
    return gc.generate(6, 11)


@pytest.fixture(scope="module")
def gf_small():
    return gf.generate(6, 12)


@pytest.fixture(scope="module")
def items():
    return common.design_items()


@pytest.fixture(scope="module")
def item_arr(items):
    return common.item_arrays(items)


# ---------------------------------------------------------------------------------------------
# Shared: schema, row counts, provenance labels, reproducibility
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("module", [gc, gf])
def test_generate_validates_and_has_two_rows_per_item(module, tmp_path):
    n = 3
    df, truth = module.generate(n, 5)
    assert len(df) == n * N_ITEMS * 2
    assert df["is_synthetic"].all()
    assert set(df["dataset"]) == {module.DATASET}
    assert (df["battery_version"] == D.DESIGN_VERSION).all()
    report = S.validate_frame(df, expect_synthetic=True, strict=True)
    assert report.ok and report.duplicate_keys == 0
    assert (df["elicitation_type"] == "binary_sell").sum() == n * N_ITEMS
    assert (df["elicitation_type"] == "binary_yes_no").sum() == n * N_ITEMS
    # every subject answers every item exactly once
    sell = df[df["elicitation_type"] == "binary_sell"]
    per_subject = sell.groupby("subject_id")["scenario_id"].nunique()
    assert (per_subject == D.N_LOSSES * D.N_CONDITIONS).all()
    assert set(sell["question_order_id"]) == set(D.QUESTION_ORDER_IDS)
    # the truth carries the provenance label and one entry per subject
    assert truth["population"]["SOURCE"] == module.SOURCE == "synthetic default, not fitted to data"
    assert truth["generator"] == module.NAME and truth["n_subjects"] == n and truth["seed"] == 5
    assert sorted(truth["subjects"]) == sorted(df["subject_id"].unique())
    # writes only under data/synthetic (schema rule) and reads back identically
    out = tmp_path / "data" / "synthetic"
    pq, tj = common.write_synthetic(df, truth, f"{module.NAME}_test", out)
    back, truth_back = common.read_synthetic(f"{module.NAME}_test", out)
    assert len(back) == len(df) and truth_back["is_synthetic"] is True
    assert truth_back["population"]["SOURCE"] == module.SOURCE
    with pytest.raises(S.SchemaError):
        common.write_synthetic(df, truth, "not_allowed", tmp_path / "data" / "processed")


@pytest.mark.parametrize("module", [gc, gf])
def test_seed_reproducibility(module):
    a, ta = module.generate(4, 3)
    b, tb = module.generate(4, 3)
    assert a.equals(b) and ta == tb
    c, _ = module.generate(4, 4)
    assert not a["response"].equals(c["response"])


@pytest.mark.parametrize("module", [gc, gf])
def test_population_override_is_labelled_and_unknown_keys_raise(module):
    _, truth = module.generate(2, 0, population={"beta_L": 2.0})
    assert truth["population"]["beta_L"] == 2.0
    assert truth["population"]["SOURCE"] != module.SOURCE and "beta_L" in truth["population"]["SOURCE"]
    with pytest.raises(ValueError):
        module.generate(2, 0, population={"no_such_parameter": 1.0})


def test_model_data_round_trip(gc_small):
    df, truth = gc_small
    data = build_model_data(df)
    n = truth["n_subjects"]
    assert data.n == n * N_ITEMS * 2 and data.n_subjects == n
    assert tuple(data.ctx_vocab) == tuple(D.NONNULL_CONTEXTS) == tuple(truth["context_vocab"])
    assert data.sell_mask().sum() == n * N_ITEMS
    sell = data.sell_mask()
    assert data.order_flag[sell].sum() == n * N_ITEMS // 2
    # every tolerance-first sell row sees its tolerance answer; no scenario-first row does
    observed = np.isfinite(data.tol_answer)
    assert observed[sell & (data.order_flag == 1)].all()
    assert not observed[sell & (data.order_flag == 0)].any()
    assert list(truth["covariate_columns"]) == list(data.X_columns)


def test_design_subset_is_accepted(rng):
    subset, _ = D.battery_subset(rng, 12)
    df, truth = gc.generate(2, 0, design=subset)
    assert len(df) == 2 * 12 * 2 and truth["n_items"] == 12
    df, truth = gf.generate(2, 0, design=subset)
    assert len(df) == 2 * 12 * 2 and truth["n_items"] == 12


# ---------------------------------------------------------------------------------------------
# G_C: latent types, law of total probability, additivity, Markov variant
# ---------------------------------------------------------------------------------------------


def test_gc_latent_types_and_intercepts(gc_small):
    _, truth = gc_small
    pop = truth["population"]
    K = pop["n_types"]
    for sid, rec in truth["subjects"].items():
        assert rec["type"] in range(K)
        assert abs(sum(rec["type_probs"]) - 1.0) < 1e-12 and min(rec["type_probs"]) >= 0
        assert rec["sell_intercept"] == pytest.approx(pop["intercept"] + pop["type_intercepts"][rec["type"]] + rec["u"])
        assert rec["tol_logit"] == pytest.approx(pop["tol_intercept"] + pop["tol_type_effects"][rec["type"]] + pop["tol_u_slope"] * rec["u"])
    assert sum(truth["type_counts"].values()) == truth["n_subjects"]


def test_gc_type_probabilities_are_softmax_of_covariates(gc_small):
    df, truth = gc_small
    data = build_model_data(df)
    pop = truth["population"]
    M = gc.type_map_matrix(pop, data.X_columns)
    logits = data.X @ M + np.asarray(pop["type_bias"])[None, :]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    for i, sid in enumerate(data.subject_ids):
        assert np.allclose(truth["subjects"][sid]["type_probs"], probs[i], atol=1e-12)


def test_gc_all_types_occur_in_a_moderate_sample():
    _, truth = gc.generate(120, 7)
    counts = truth["type_counts"]
    assert all(c > 0 for c in counts.values()), counts


def test_gc_mixed_population_false_is_one_type():
    _, truth = gc.generate(5, 0, mixed_population=False)
    types = {rec["type"] for rec in truth["subjects"].values()}
    assert types == {truth["population"]["n_types"] // 2}
    assert all(rec["type_probs"][rec["type"]] == 1.0 for rec in truth["subjects"].values())


def test_gc_law_of_total_probability_exact_from_truth(gc_small, items, item_arr):
    """PLAN.md section 4: ``delta_LTP = P(sell | scenario-first) - sum_a P(a, sell | tolerance-first)``
    is identically zero under G_C's defaults, for every subject and item; and the generator's
    probabilities agree with an independent recomputation of the log-odds."""
    _, truth = gc_small
    pop = truth["population"]
    assert pop["beta_order"] == 0.0 and pop["beta_tol"] == 0.0
    for sid, rec in truth["subjects"].items():
        ids = [sid] * len(items)
        delta = gc.ltp_interference(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"])
        assert np.abs(delta).max() < 1e-12
        # independent recomputation of eta for the scenario-first rows
        beta = np.asarray([pop["beta_ctx"][c] for c in D.NONNULL_CONTEXTS])
        ctx = np.array([sum(beta[D.NONNULL_CONTEXTS.index(t)] for t in tags) for tags in items["context_tags"]])
        eta = rec["sell_intercept"] + pop["beta_L"] * np.abs(items["loss_pct"].to_numpy()) + ctx
        p = gc.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], np.zeros(len(items), int))
        assert np.allclose(p, _sigmoid(eta), atol=1e-12)
        # and the tolerance-first marginal is the mixture over the answers (holds for any joint law)
        p_a = gc.tolerance_probability(truth, ids)
        ones = np.ones(len(items), int)
        p_yes = gc.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], ones, np.ones(len(items)))
        p_no = gc.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], ones, np.zeros(len(items)))
        assert np.allclose(p_a * p_yes + (1 - p_a) * p_no, p, atol=1e-12)


def test_gc_ltp_holds_in_expectation_empirically():
    """Sampled sell rates by question order for the same (subject, loss, condition) items differ
    only by Monte Carlo noise: 40 subjects x 85 item pairs per order, SE of the difference of
    two Bernoulli means ~ sqrt(2 * 0.25 / 3400) = 0.012."""
    df, _ = gc.generate(40, 21)
    sell = df[df["elicitation_type"] == "binary_sell"]
    by_order = sell.groupby("question_order_id")["response"].mean()
    assert abs(by_order[D.TOLERANCE_FIRST] - by_order[D.SCENARIO_FIRST]) < 0.05


def test_gc_classical_order_effect_variant_is_the_documented_displacement(items, item_arr):
    """With ``beta_tol != 0`` the LTP term equals ``sigmoid(eta) - sum_a P(a) sigmoid(eta + beta_tol a)``
    (a classical prior-answer effect, not interference)."""
    _, truth = gc.generate(3, 0, population={"beta_tol": 0.7, "beta_order": 0.2})
    sid = "S0001"
    ids = [sid] * len(items)
    delta = gc.ltp_interference(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"])
    zeros = np.zeros(len(items), int)
    eta = _logit(gc.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], zeros))
    p_a = _sigmoid(truth["subjects"][sid]["tol_logit"])
    expected = _sigmoid(eta) - (p_a * _sigmoid(eta + 0.2 + 0.7) + (1 - p_a) * _sigmoid(eta + 0.2))
    assert np.allclose(delta, expected, atol=1e-12)
    assert np.abs(delta).max() > 1e-3


def test_gc_contexts_additive_and_position_independent(gc_small, items, item_arr):
    _, truth = gc_small
    ids = ["S0002"] * len(items)
    lg = _logit(gc.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], item_arr["order_flag"]))
    cid = items["context_condition_id"].to_numpy()
    sel = (items["loss_pct"].to_numpy() == -0.10) & (items["question_order_id"].to_numpy() == D.SCENARIO_FIRST)
    by = {c: float(lg[sel & (cid == c)][0]) for c in np.unique(cid)}
    for a in D.NONNULL_CONTEXTS:
        for b in D.NONNULL_CONTEXTS:
            if a == b:
                continue
            pair = D.condition_id((a, b))
            rev = D.condition_id((b, a))
            single_a = D.condition_id((a,))
            single_b = D.condition_id((b,))
            assert by[pair] == pytest.approx(by[rev], abs=1e-12)
            assert by[pair] - by["none"] == pytest.approx((by[single_a] - by["none"]) + (by[single_b] - by["none"]), abs=1e-12)


def test_gc_markov_variant_adds_previous_answer_dependence():
    df0, t0 = gc.generate(30, 33, markov=False)
    df1, t1 = gc.generate(30, 33, markov=True)
    assert t0["beta_prev_effective"] == 0.0 and t1["beta_prev_effective"] == t1["population"]["beta_prev"] > 0
    assert t0["markov"] is False and t1["markov"] is True
    assert not df0["response"].equals(df1["response"])

    def lag_effect(df: pd.DataFrame) -> float:
        sell = df[df["elicitation_type"] == "binary_sell"].sort_values(["subject_id", "position_in_session"])
        prev = sell.groupby("subject_id")["response"].shift(1)
        ok = prev.notna()
        return float(sell["response"][ok & (prev == 1)].mean() - sell["response"][ok & (prev == 0)].mean())

    # the latent alone induces some dependence on the previous answer (subjects differ); the
    # Markov term adds to it. Each mean averages > 1500 rows, so 0.1 is far outside noise.
    assert lag_effect(df1) > lag_effect(df0) + 0.1


# ---------------------------------------------------------------------------------------------
# G_F: non-additive, order-asymmetric pair effects; population variants
# ---------------------------------------------------------------------------------------------


def test_gf_truth_matches_independent_recomputation(gf_small, items, item_arr):
    _, truth = gf_small
    pop = truth["population"]
    sid = "S0003"
    rec = truth["subjects"][sid]
    ids = [sid] * len(items)
    p = gf.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], item_arr["order_flag"], None)
    eta = np.empty(len(items))
    for j, row in enumerate(items.itertuples(index=False)):
        tags = list(row.context_tags)
        e = pop["intercept"] + rec["u"] + pop["beta_L"] * abs(row.loss_pct)
        for k, t in enumerate(tags):
            e += pop["beta_ctx_by_position"][k][t]
        if len(tags) == 2:
            e += rec["pair_scale"] * pop["beta_pair"][D.condition_id(tuple(tags))]
        e += pop["beta_order"] * (row.question_order_id == D.TOLERANCE_FIRST)
        eta[j] = e
    assert np.allclose(p, _sigmoid(eta), atol=1e-12)


def test_gf_pair_effects_are_non_additive_and_order_asymmetric(gf_small, items, item_arr):
    _, truth = gf_small
    pop = truth["population"]
    sid = "S0001"
    s_i = truth["subjects"][sid]["pair_scale"]
    ids = [sid] * len(items)
    lg = _logit(gf.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], item_arr["order_flag"], None))
    cid = items["context_condition_id"].to_numpy()
    sel = (items["loss_pct"].to_numpy() == -0.20) & (items["question_order_id"].to_numpy() == D.SCENARIO_FIRST)
    by = {c: float(lg[sel & (cid == c)][0]) for c in np.unique(cid)}
    n_nonadditive = n_asymmetric = 0
    for a in D.NONNULL_CONTEXTS:
        for b in D.NONNULL_CONTEXTS:
            if a == b:
                continue
            pair, rev = D.condition_id((a, b)), D.condition_id((b, a))
            single_a, single_b = D.condition_id((a,)), D.condition_id((b,))
            # the analytic decomposition of the pair displacement
            expected = pop["beta_ctx_by_position"][0][a] + pop["beta_ctx_by_position"][1][b] + s_i * pop["beta_pair"][pair]
            assert by[pair] - by["none"] == pytest.approx(expected, abs=1e-12)
            additive = (by[single_a] - by["none"]) + (by[single_b] - by["none"])
            n_nonadditive += abs((by[pair] - by["none"]) - additive) > 0.05
            n_asymmetric += abs(by[pair] - by[rev]) > 0.05
    assert n_nonadditive == 12 and n_asymmetric == 12


def test_gf_order_and_prior_answer_effects_are_classical_displacements(gf_small, items, item_arr):
    """G_F's defaults make ``beta_order`` and ``beta_tol`` non-zero: the tolerance-first sell
    log-odds are shifted by ``beta_order + beta_tol * a`` and nothing else (a classical,
    additive order effect; it makes delta_LTP != 0 and the module docstring says so)."""
    _, truth = gf_small
    pop = truth["population"]
    assert pop["beta_order"] != 0.0 and pop["beta_tol"] != 0.0
    ids = ["S0002"] * len(items)
    n = len(items)
    eta0 = _logit(gf.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], np.zeros(n, int), None))
    eta1 = _logit(gf.sell_probability(truth, ids, item_arr["loss"], item_arr["ctx_idx"], item_arr["ctx_mask"], np.ones(n, int), np.ones(n)))
    assert np.allclose(eta1 - eta0, pop["beta_order"] + pop["beta_tol"], atol=1e-10)


def test_gf_mixed_population_flag():
    _, t_mixed = gf.generate(6, 0, mixed_population=True)
    _, t_homog = gf.generate(6, 0, mixed_population=False)
    assert all(rec["u"] == 0.0 and rec["pair_scale"] == 1.0 for rec in t_homog["subjects"].values())
    us = [rec["u"] for rec in t_mixed["subjects"].values()]
    scales = [rec["pair_scale"] for rec in t_mixed["subjects"].values()]
    assert np.std(us) > 0 and np.std(scales) > 0 and min(scales) > 0


def test_gf_resolve_population_requires_every_pair_and_context():
    with pytest.raises(ValueError):
        gf.resolve_population({"beta_pair": {"not>apair": 1.0}})
    pop = gf.resolve_population({"beta_pair": {"rec>tech": 2.5}})
    assert pop["beta_pair"]["rec>tech"] == 2.5 and pop["beta_pair"]["tech>rec"] == gf.POPULATION_DEFAULTS["beta_pair"]["tech>rec"]
    B_ctx, B_pair = gf.context_tables(pop)
    assert B_ctx.shape == (2, 4) and B_pair.shape == (4, 4) and np.all(np.diag(B_pair) == 0)
    assert B_pair[0, 1] == 2.5
