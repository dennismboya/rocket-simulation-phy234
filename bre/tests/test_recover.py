"""End-to-end test of the Phase 2 recovery study driver (``bre.recover``) at N = 10, one seed,
two models, one worker, on temporary ``data/synthetic`` and report directories: every output
file of the module docstring is written, the selection table has one row per (generator, seed),
the N-target rule is stated, and the log holds one progress line per fit. A second test checks
that any registered model can be queued (jobs and the runtime estimate for all ten) and runs
one job each of B5 (external fit) and Q5 on a tiny table.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bre import recover as R


@pytest.mark.filterwarnings("ignore")
def test_recover_end_to_end_writes_every_output(tmp_path):
    data_dir = tmp_path / "data" / "synthetic"
    out_dir = tmp_path / "reports" / "recovery"
    runs_dir = tmp_path / "runs" / "recover"
    payload = R.run(
        generators=("gq", "gc", "gf"),
        ns=(10,),
        seeds=(0,),
        models=("B1", "Q2"),
        workers=1,
        data_dir=data_dir,
        out_dir=out_dir,
        runs_dir=runs_dir,
        model_settings={"classical": {"restarts": 1, "steps": 60, "lbfgs_steps": 5}, "quantum": {"restarts": 1, "steps": 60, "lbfgs_steps": 5}},
        n_boot=20,
        save_artifacts=True,
        tag="test",
    )
    # datasets and truth files
    for gen in ("gq", "gc", "gf"):
        assert (data_dir / f"{gen}_n10_seed0.parquet").exists() and (data_dir / f"{gen}_n10_seed0.truth.json").exists()
    # reports
    for name in ("confusion_N10.md", "N_target.json", "summary.md", "results.json", "recovery_scatter_gq_N10.png", "recovery_scatter_gc_N10.png"):
        assert (out_dir / name).exists(), name
    nt = json.loads((out_dir / "N_target.json").read_text())
    assert nt["rule"] == R.N_TARGET_RULE and nt["items_per_subject"] == 170 and nt["is_synthetic"] is True
    assert nt["grid"][0]["n"] == 10 and nt["grid"][0]["n_cells"] == 3
    assert (nt["N_target"] == 10 and nt["responses_needed_for_calibration"] == 1700) or (nt["N_target"] is None and "reason" in nt)
    # selection table and confusion matrix
    table = payload["table"]
    assert len(table) == 3 and {r["gen"] for r in table} == {"gq", "gc", "gf"}
    assert all(r["selected_model"] in ("B1", "Q2") for r in table)
    assert payload["n_jobs"] == 6 and payload["n_failed"] == 0 and payload["is_synthetic"] is True
    md = (out_dir / "confusion_N10.md").read_text()
    assert "True generator x selected model" in md and "| gq |" in md
    summary = (out_dir / "summary.md").read_text()
    assert "SYNTHETIC" in summary and "N = 10" in summary
    # parameter recovery entries exist for G_Q (Q2) and G_C (B1)
    rec = {(r["gen"], r["seed"]): r for r in payload["recovery"]}
    assert "Q2" in rec[("gq", 0)] and np.isfinite(rec[("gq", 0)]["Q2"]["theta_ctx_pearson"])
    assert "B1" in rec[("gc", 0)] and np.isfinite(rec[("gc", 0)]["B1"]["beta_ctx_pearson_both_positions"])
    # log with a progress line per fit, and saved artifacts
    log = (runs_dir / "test.log").read_text()
    assert log.count("[recover] done") == 6 and log.count("[fit] ") >= 6
    assert (runs_dir / "test" / "gq_n10_seed0" / "Q2" / "artifact.json").exists()
    # the selection-fold NLL reported equals a direct recomputation
    for f in payload["fits"]:
        assert abs(f["selection_nll"] - f["selection_nll_check"]) < 1e-9
        assert f["n_selection_rows"] == 10 * 34  # 20% of 170 items per subject


@pytest.mark.filterwarnings("ignore")
def test_recover_accepts_any_registered_model(tmp_path):
    from bre.registry import MODEL_REGISTRY

    data_dir = tmp_path / "data" / "synthetic"
    log_path = tmp_path / "runs" / "recover" / "t.log"
    log_path.parent.mkdir(parents=True)
    models = sorted(MODEL_REGISTRY)
    settings = {"classical": {"restarts": 1, "steps": 40, "lbfgs_steps": 3}, "quantum": {"restarts": 1, "steps": 40, "lbfgs_steps": 3}, "B2": {"restarts": 1, "svi_steps": 50, "svi_lr": 0.01}}
    jobs = R.build_jobs(("gq",), (6,), (0,), models, data_dir, settings, log_path, None, 20, log=lambda s: None)
    assert [j["model"] for j in jobs][:2] == ["Q4", "Q2"] and {j["model"] for j in jobs} == set(models)
    est = R.estimate_minutes(jobs, 1)
    assert np.isfinite(est) and est > 0
    assert R.step_cost("Q2") == (R.SECONDS_PER_STEP["Q2"], True) and R.step_cost("Q3") == (R.SECONDS_PER_STEP_ASSUMED["Q3"], False)
    assert R.step_cost("nonesuch") == (max(R.SECONDS_PER_STEP.values()), False)
    with pytest.raises(KeyError):
        R.run(models=("B1", "Z9"), ns=(6,), seeds=(0,), data_dir=data_dir, out_dir=tmp_path / "r", runs_dir=tmp_path / "runs")
    # one external-fit job (B5) and one Q-model without a defined delta_LTP (Q5)
    by_model = {j["model"]: j for j in jobs}
    b5 = R.run_job({**by_model["B5"], "artifact_dir": str(tmp_path / "art")})
    assert b5["family"] == "classical" and np.isfinite(b5["selection_nll"]) and b5["n_params"] > 0
    assert b5["restarts"][0]["method"] == "external" and "interference_train" not in b5 and b5["summary"]["ctx_vocab"]
    assert (tmp_path / "art" / "gq_n6_seed0" / "B5" / "params.npz").exists()
    q5 = R.run_job(by_model["Q5"])
    assert q5["family"] == "quantum" and np.isfinite(q5["selection_nll"])
    assert q5["interference_train"]["order_zero_by_construction"] and not q5["interference_train"]["ltp_defined"]
    assert q5["quarter_law_train"]["reference"] == 0.25 and "natural" in q5["summary"]
    assert log_path.read_text().count("[fit] ") >= 2
