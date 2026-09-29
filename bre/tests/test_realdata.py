"""Tests of the real-data pipeline: ``bre.realdata`` (assembly, subsampling, split (b), the A ->
B transfer protocol) and ``bre.phase4`` (the pre-registered evaluation driver, smoke settings).

Real public tables only (CLAUDE.md rule 3: nothing synthetic is loaded here); every fit uses
tiny settings and small subsamples so the whole file stays under four minutes. The CPC18 pairs
variant is built once per process by the loader (about 15 s) and cached by ``bre.realdata``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from bre import eval as E
from bre import phase4 as P
from bre import realdata as R
from bre import schema as S

warnings.filterwarnings("ignore")

SCRATCH_SUBJECTS = 20


@pytest.fixture(scope="module")
def cpc18_pairs_20():
    frame, data = R.load_real("cpc18_pairs", subjects=SCRATCH_SUBJECTS, seed=0)
    return frame, data


# ---------------------------------------------------------------------------------------------
# load_real
# ---------------------------------------------------------------------------------------------


def test_load_real_cpc18_pairs_subsample_validates_and_builds(cpc18_pairs_20):
    frame, data = cpc18_pairs_20
    # validated schema frame, real data only, consent flag kept as the loaders wrote it
    S.validate_frame(frame, expect_synthetic=False, strict=True)
    assert not frame["is_synthetic"].any() and not np.asarray(data.is_synthetic).any()
    assert data.n_subjects == SCRATCH_SUBJECTS and data.n == len(frame) > 0
    assert set(data.summary()["row_kinds"]) == {"sell"}  # lottery_choice rows enter the likelihood
    # Phase 4 mapping: feedback blocks only, exp:* contexts only, loss null where nothing was shown
    assert all("feedback:on" not in t and "block:2" not in t for t in frame["context_tags"])
    assert tuple(data.ctx_vocab[4:]) == ("exp:gain", "exp:loss")
    n_exp = R.exp_tag_count(data)
    assert set(np.unique(n_exp)) == {0, 1, 2}
    assert np.all(data.loss[n_exp == 0] == 0.0) and np.all(np.isfinite(data.loss))
    assert np.all(data.loss <= 1.0)
    # deterministic subsample: same (N, seed) -> same subjects; another seed -> other subjects
    _, again = R.load_real("cpc18_pairs", subjects=SCRATCH_SUBJECTS, seed=0, validate=False)
    _, other = R.load_real("cpc18_pairs", subjects=SCRATCH_SUBJECTS, seed=1, validate=False)
    assert np.array_equal(again.subject_ids, data.subject_ids)
    assert not np.array_equal(other.subject_ids, data.subject_ids)
    assert all(s.startswith("cpc18/") for s in data.subject_ids)  # subject_key = dataset/subject_id


def test_load_real_aggregate_and_psych201_tables():
    frame, data = R.load_real("choices13k", subjects=300, seed=0, single_subject=True)
    assert data.n == 300 and data.n_subjects == 1 and set(data.summary()["row_kinds"]) == {"rate"}
    assert np.all(data.w >= 15) and np.all((data.y >= 0) & (data.y <= 1))
    assert np.all((data.loss >= 0) & (data.loss <= 1)) and np.all(data.loss_signed <= 0)  # loss = |min(0, worst outcome)|
    assert {"feedback:on", "feedback:off", "block:1"} <= set(data.ctx_vocab)
    S.validate_frame(frame, expect_synthetic=False, strict=True)
    _, d15 = R.load_real("cpc15", ctx_vocab=data.ctx_vocab)
    assert d15.n == 750 and d15.ctx_vocab == data.ctx_vocab and np.all(d15.w == 1.0)
    fs, ds = R.load_real("psych201_spektor2024lossaversion", subjects=8, seed=0)
    assert ds.n_subjects == 8 and tuple(ds.ctx_vocab[4:]) == ("domain:gain", "domain:loss", "domain:mixed")
    assert np.all(ds.ctx_mask.sum(axis=1) == 1) and np.all((ds.loss >= 0) & (ds.loss <= 1))
    assert set(fs["session_id"]) <= {"session1", "session2"}
    with pytest.raises(KeyError):
        R.load_real("not-a-dataset")


def test_aggregate_cpc18_by_problem_is_computed_from_the_individual_rows():
    frame = R.aggregate_cpc18_by_problem(subjects=SCRATCH_SUBJECTS, seed=0)
    S.validate_frame(frame, expect_synthetic=False, strict=True)
    assert (frame["elicitation_type"] == "choice_rate").all()
    w = frame["covariates"].map(lambda c: json.loads(c)["weight"]).to_numpy()
    assert np.all(w >= 1) and w.sum() == len(R.subsample_subjects(R.read_frame("cpc18"), SCRATCH_SUBJECTS, 0))
    assert frame["response"].between(0, 1).all()
    assert all(len(t) == 2 and t[0].startswith("feedback:") and t[1].startswith("block:") for t in frame["context_tags"])
    assert frame["loss_pct"].between(-1, 0).all()
    assert frame["source_row_ref"].str.contains("aggregated-by-problem").all()


# ---------------------------------------------------------------------------------------------
# split (b)
# ---------------------------------------------------------------------------------------------


def test_split_b_has_only_pair_rows_in_test_and_only_single_rows_in_train(cpc18_pairs_20):
    _, data = cpc18_pairs_20
    train, test = R.split_pairs_by_exp_tags(data)
    n_exp = R.exp_tag_count(data)
    assert len(test) > 0 and np.all(n_exp[test] == 2) and np.all(data.ctx_mask[test].all(axis=1)) and np.all(data.sell_mask()[test])
    assert len(train) > 0 and np.all(n_exp[train] <= 1) and not np.any(data.ctx_mask[train].all(axis=1))
    assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == data.n
    # with exp:* as the only contexts this is the design's split (b) of bre.eval
    tr2, te2 = E.split_context_composition(data)
    assert np.array_equal(train, tr2) and np.array_equal(test, te2)
    splits = P.make_splits(data, ("a", "b", "c", "d"), seed=0)
    assert splits["b"]["applicable"] and np.array_equal(splits["b"]["test"], test)
    assert not splits["c"]["applicable"] and "not applicable" in splits["c"]["reason"]
    assert splits["a"]["applicable"] and splits["d"]["applicable"]
    # a table without pairs reports split (b) as not applicable
    _, ds = R.load_real("psych201_spektor2024lossaversion", subjects=4, seed=0, validate=False)
    assert not P.make_splits(ds, ("b",))["b"]["applicable"]


def test_auc_fast_equals_eval_auc_with_ties_and_weights(rng):
    p = np.round(rng.random(400), 2)  # ties
    y = (rng.random(400) < p).astype(float)
    w = rng.integers(1, 4, size=400).astype(float)
    assert abs(P.auc_fast(p, y, w) - E.auc(p, y, w)) < 1e-12
    assert abs(P.auc_fast(p, y) - E.auc(p, y)) < 1e-12
    assert np.isnan(P.auc_fast(p, np.ones(400)))


# ---------------------------------------------------------------------------------------------
# transfer protocol
# ---------------------------------------------------------------------------------------------


def test_transfer_protocol_runs_end_to_end_on_a_500_row_subsample(tmp_path):
    settings = {"steps": 20, "restarts": 1, "lbfgs_polish": False, "lbfgs_steps": 0, "eval_every": 5, "early_stopping_patience": 2}
    # one target keeps the test short; the CPC18 per-problem target is built and checked above
    results, json_path, md_path = R.run_transfer(("B1", "Q2"), rows_a=500, seed=0, settings=settings, n_boot=20, out_dir=tmp_path / "transfer", runs_dir=tmp_path / "runs", name="tiny", targets=("cpc15",), echo=False)
    assert json_path.exists() and md_path.exists()
    assert not results["is_synthetic"] and results["source"]["n_rows"] == 500
    assert set(results["targets"]) == {"cpc15"}
    for m in ("B1", "Q2"):
        e = results["models"][m]
        assert e["n_params"] == e["n_params_population"]  # random effect frozen: population count only
        if m == "Q2":
            assert e["population_summary"]["u_max_abs"] == 0.0 and len(e["population_summary"]["theta_ctx"]) == len(results["source"]["ctx_vocab"])
        for t in ("cpc15",):
            tt = e["targets"][t]
            for key in ("transfer_test", "transfer_all", "refit_test", "constant_test"):
                mt = tt[key]
                assert np.isfinite(mt["nll_per_respondent"]) and np.isfinite(mt["brier"]) and mt["n_rows"] > 0
                assert mt["nll_per_respondent_ci95"][0] <= mt["nll_per_respondent_ci95"][1]
            assert 0 < tt["constant_rate"] < 1
    md = md_path.read_text(encoding="utf-8")
    assert "Real data" in md and "| B1 |" in md and "| Q2 |" in md and "Target cpc15" in md
    assert R.transfer_report(json.loads(json_path.read_text(encoding="utf-8"))) == md
    with pytest.raises(TypeError):
        R.PopulationOnly(R.make_model("B5", R.load_real("cpc15", validate=False)[1]))


# ---------------------------------------------------------------------------------------------
# phase 4 smoke
# ---------------------------------------------------------------------------------------------


def test_phase4_smoke_b1_q2_on_30_subjects_writes_the_verdict(tmp_path):
    out, runs = tmp_path / "out", tmp_path / "runs"
    # splits (a), (b) and the not-applicable (c); (d) uses the same machinery and is left out for time
    argv = ["--dataset", "cpc18", "--smoke", "--subjects", "30", "--models", "B1", "Q2", "--splits", "a", "b", "c", "--steps", "60", "--n-boot", "200", "--n-samples", "4", "--no-artifacts", "--out", str(out), "--runs-dir", str(runs), "--quiet"]
    assert P.main(argv) == 0
    for f in ("metrics.json", "table.md", "structural.md", "verdict.json", "verdict.md"):
        assert (out / f).exists()
    v = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] in (E.VERDICT_SUPPORTED, E.VERDICT_NONE)
    for key in ("verdict", "best_quantum", "best_classical", "nll_diff", "ci95", "interference_ci", "n_params_q", "n_params_c", "data"):
        assert key in v
    assert v["data"] == "cpc18 pairs split (b)" and v["best_quantum"] == "Q2" and v["best_classical"] == "B1"
    assert v["smoke"] is True and v["is_real_data"] is True
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert not m["is_synthetic"] and m["subjects"] == 30 and m["failed"] == []
    assert set(m["splits"]) == {"a", "b", "c"} and not m["splits"]["c"]["applicable"]
    for s in ("a", "b"):
        sres = m["splits"][s]
        assert set(sres["models"]) == {"B1", "Q2"} and len(sres["table"]) == 2
        for e in sres["models"].values():
            assert np.isfinite(e["nll"]) and np.isfinite(e["nll_ci95"]).all() and e["nll_ci95"][0] <= e["nll_ci95"][1] and e["n_params"] > 0
        assert "nll_diff_vs_best_classical" in sres["models"]["Q2"]
    q2b = m["splits"]["b"]["models"]["Q2"]
    assert q2b["interference_test"]["n_samples"] == 4 and np.isfinite(q2b["interference_test"]["order_mean"])
    oe = m["structural"]["order_effects_by_problem"]
    assert oe["applicable"] and oe["n_problems"] > 0 and np.isfinite(oe["pooled"])
    assert not m["structural"]["ltp_violation_from_data"]["applicable"]
    assert "Decision rule (PLAN.md section 6, verbatim)" in (out / "verdict.md").read_text(encoding="utf-8")
    assert E.DECISION_RULE_TEXT in (out / "verdict.md").read_text(encoding="utf-8")
    # the job cache is reused by a second call (nothing refitted) and the outputs are rewritten
    assert (runs / "jobs" / "b__Q2.pkl").exists()
    assert P.main(argv) == 0
    log = (runs / "phase4.log").read_text(encoding="utf-8")
    assert "cached (b) Q2" in log
