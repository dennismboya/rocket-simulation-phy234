"""Model and data transparency (Phase 6, page 6).

Live model version and type (registry, selectable by an admin), the pre-registered decision
rule verbatim, the Phase 4 verdict and held-out metrics (shown only once
``reports/phase4/verdict.json`` exists: until then the page shows exactly
"Decision rule not yet run: no real-data numbers are shown" and no Phase 4 table), the served
model's training metrics and calibration plot (synthetic under the banner in demo mode),
parameter counts, "responses needed for calibration" from the Phase 2 recovery study, the data
provenance table (dataset, N, real or synthetic, licence), the model card (``reports/MODEL_CARD.md``
when written) and the Retrain button (``POST /retrain``, the same as ``make retrain``: consented
responses, refit, held-out evaluation, promotion only if not worse).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import api_client as api  # noqa: E402
from components import charts, chrome, market_bar  # noqa: E402

health = chrome.page_setup("Model and data transparency")
base = api.base_url()
try:
    model = api.model_info(base)
    tr = api.transparency(base)
    reg = api.registry(base)
except api.ApiError as exc:
    chrome.stop_on_error(exc)
market_bar.render(model)
settings = chrome.load_settings(base)
demo = chrome.demo_mode()

# ---------------------------------------------------------------------------------------------
# 1. Live model
# ---------------------------------------------------------------------------------------------

st.subheader("Live model")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Version", tr.get("model_version"))
m2.metric("Type", f"{tr.get('model_type')} ({tr.get('family')})")
m3.metric("Parameters", tr.get("n_params"), help="every free scalar; population-level count beside it")
m4.metric("Population-level parameters", tr.get("n_params_population") if tr.get("n_params_population") is not None else "n/a")
m5.metric("Trained on synthetic data", "yes" if tr.get("is_synthetic_training") else "no")
st.caption(f"Evaluation status: {model.get('evaluation_status')}. Intervals: {model.get('interval_source')} ({model.get('n_draws')} draws). Design {model.get('design_version')}.")

rows = reg.get("rows") or []
reg_df = pd.DataFrame([{"version": r["version"], "type": r["model_type"], "family": r.get("family"), "active": r["is_active"], "promoted": r.get("promoted_at"), "created": r.get("created_at"),
                        "synthetic training": r.get("is_synthetic_training"), "parameters": r.get("n_params"), "train NLL": r.get("train_nll"), "held-out NLL": r.get("held_out_nll"),
                        "artifact on disk": r.get("artifact_exists")} for r in rows])
st.dataframe(reg_df, hide_index=True, width="stretch", column_config={"train NLL": st.column_config.NumberColumn(format="%.4f"), "held-out NLL": st.column_config.NumberColumn(format="%.4f")})
if reg.get("admin"):
    a1, a2 = st.columns([3, 1])
    candidates = [r["version"] for r in rows if r.get("artifact_exists")]
    pick = a1.selectbox("Activate a registered version (admin)", candidates, index=candidates.index(reg["active_version"]) if reg.get("active_version") in candidates else 0, key="tr_version",
                        format_func=lambda v: f"{v} ({next(r['model_type'] for r in rows if r['version'] == v)})")
    if a2.button("Activate", key="tr_activate", disabled=pick == reg.get("active_version")):
        try:
            out = api.activate(base, pick)
            st.session_state["activate_result"] = out
            st.success(f"Activated {out['activated']} ({out['model_type']}); previous {out['previous']}.")
            st.rerun()
        except api.ApiError as exc:
            st.error(f"Not activated: {exc}")
    st.caption(f"Selectable model types for a refit: {', '.join(reg.get('selectable_model_types') or [])}. Admin rule: {reg.get('admin_rule')}.")
else:
    st.caption(f"Activation is an admin action: {reg.get('admin_rule')}.")

# ---------------------------------------------------------------------------------------------
# 2. Decision rule and verdict
# ---------------------------------------------------------------------------------------------

st.subheader("Pre-registered decision rule and verdict")
st.markdown("> " + str(tr.get("decision_rule")))
verdict = tr.get("verdict")
phase4 = tr.get("phase4_metrics")
if verdict is None:
    st.warning(tr.get("verdict_status"), icon="⏳")
    st.caption("Phase 4 (the pre-registered evaluation on real data) has not been run; no held-out real-data metric is shown on this page until reports/phase4/verdict.json exists.")
    st.session_state["transparency_phase4_rendered"] = False
else:
    st.success(f"Verdict: {tr.get('verdict_status')}")
    with st.expander("Verdict details (verbatim from reports/phase4/verdict.json)"):
        st.json(verdict)
    if phase4:
        st.markdown("**Held-out metrics (Phase 4)**")
        try:
            p4 = pd.DataFrame(phase4 if isinstance(phase4, list) else [{"metric": k, "value": v} for k, v in phase4.items()])
            st.dataframe(p4, hide_index=True, width="stretch")
        except ValueError:
            st.json(phase4)
        st.session_state["transparency_phase4_rendered"] = True
    else:
        st.session_state["transparency_phase4_rendered"] = False
st.session_state["transparency_real_numbers_shown"] = bool(tr.get("real_data_numbers_shown"))

# ---------------------------------------------------------------------------------------------
# 3. Training metrics and calibration plot of the served model
# ---------------------------------------------------------------------------------------------

st.subheader("Served model: training metrics and calibration")
tm = tr.get("training_metrics") or {}
if tm.get("withheld"):
    st.info(f"{tm.get('note')} (the served model was trained on real data; its numbers appear once the verdict exists).")
else:
    if tr.get("is_synthetic_training"):
        st.caption("SYNTHETIC: every number in this section comes from the synthetic demo book.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Training NLL per response", chrome.fmt_p(tm.get("train_nll_per_response", tm.get("train_nll")), 4))
    k2.metric("Training Brier", chrome.fmt_p(tm.get("train_brier"), 4))
    k3.metric("Training responses / subjects", f"{tm.get('n_responses')} / {tm.get('n_subjects')}")
    ho = tm.get("held_out") if isinstance(tm.get("held_out"), dict) else None
    k4.metric("Held-out NLL (recorded)", chrome.fmt_p(None if not ho else ho.get("nll"), 4), help=tm.get("held_out_note") or "none recorded")
    cal = tr.get("calibration_plot")
    if cal:
        c1, c2 = st.columns([1, 1])
        fig = charts.reliability_plot(cal.get("reliability") or [], cal.get("ece"), f"{'synthetic, ' if cal.get('is_synthetic') else ''}in-sample reliability")
        c1.pyplot(fig, width="stretch")
        plt.close(fig)
        c2.dataframe(pd.DataFrame(cal.get("reliability") or []), hide_index=True, width="stretch")
        c2.caption(f"{cal.get('scope')}; ECE {chrome.fmt_p(cal.get('ece'), 3)}, Brier {chrome.fmt_p(cal.get('brier'), 3)} on {cal.get('n_rows')} rows; model {cal.get('model_version')}.")
    else:
        st.caption("No calibration plot: no stored training rows the served model can score.")

# ---------------------------------------------------------------------------------------------
# 4. Responses needed for calibration (Phase 2)
# ---------------------------------------------------------------------------------------------

st.subheader("Responses needed for calibration (Phase 2 recovery study)")
nt = tr.get("n_target")
if nt:
    n1, n2, n3 = st.columns(3)
    n1.metric("N_target (subjects)", nt.get("N_target") if nt.get("N_target") is not None else "not reached")
    n2.metric("Responses needed", nt.get("responses_needed_for_calibration") if nt.get("responses_needed_for_calibration") is not None else "n/a", help="N_target x 170 items per subject on the shared design")
    n3.metric("Correct selection at N_target", chrome.fmt_p(next((g.get("correct_share") for g in nt.get("grid") or [] if g.get("n") == nt.get("N_target")), None)))
    st.caption(f"Rule: {nt.get('rule')} Source: {nt.get('source')}. Synthetic study: {nt.get('is_synthetic')}.")
    st.dataframe(pd.DataFrame(nt.get("grid") or []), hide_index=True, width="stretch")
else:
    st.info("The recovery study has not written reports/recovery/N_target.json yet.")

# ---------------------------------------------------------------------------------------------
# 5. Provenance
# ---------------------------------------------------------------------------------------------

st.subheader("Data provenance")
prov = pd.DataFrame(tr.get("provenance") or [])
if len(prov):
    st.dataframe(prov[["dataset", "catalog_entry", "catalog_name", "n_rows", "n_subjects", "real_or_synthetic", "license", "used_by_served_model", "catalog_status"]], hide_index=True, width="stretch")
st.caption(f"Training data of the served model: {'; '.join(model.get('training_data_refs') or [])}. {(model.get('provenance') or {}).get('note')}")

# ---------------------------------------------------------------------------------------------
# 6. Model card
# ---------------------------------------------------------------------------------------------

st.subheader("Model card")
if tr.get("model_card"):
    with st.expander(f"reports/MODEL_CARD.md", expanded=False):
        st.markdown(tr["model_card"])
else:
    st.info("No model card yet: reports/MODEL_CARD.md is written by Phase 8 (`make report`).")
with st.expander("Calibrated contexts of the served model"):
    cal_tbl = model.get("calibrated_contexts") or {}
    st.dataframe(pd.DataFrame([{"context": chrome.context_label(t), "tag": t, "N": e.get("n_responses"), "status": e.get("status"), "minimum N": e.get("min_responses")} for t, e in cal_tbl.items()]), hide_index=True, width="stretch")
    st.caption(f"Rule: {model.get('calibration_rule')}")

# ---------------------------------------------------------------------------------------------
# 7. Retrain
# ---------------------------------------------------------------------------------------------

st.subheader("Retrain")
st.caption(f"Rule: {tr.get('retrain_rule')}. Budget from the settings page (steps {settings.get('retrain', {}).get('steps')}, restarts {settings.get('retrain', {}).get('restarts')}, "
           f"draws {settings.get('retrain', {}).get('n_samples')}); prior strength {settings.get('prior_strength')}. Runs synchronously (minutes at the default budget).")
r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
mt = r1.selectbox("Model type", reg.get("selectable_model_types") or ["Q4"], index=(reg.get("selectable_model_types") or ["Q4"]).index(tr.get("model_type")) if tr.get("model_type") in (reg.get("selectable_model_types") or []) else 0, key="tr_model_type")
steps = r2.number_input("Steps", min_value=1, max_value=5000, value=int(settings.get("retrain", {}).get("steps", 1500)), step=50, key="tr_steps")
restarts = r3.number_input("Restarts", min_value=1, max_value=10, value=int(settings.get("retrain", {}).get("restarts", 3)), step=1, key="tr_restarts")
if r4.button("Retrain now", key="tr_retrain", type="primary"):
    with st.spinner("Pulling consented responses, refitting, evaluating the held-out subjects …"):
        try:
            out = api.retrain(base, {"model_type": mt, "steps": int(steps), "restarts": int(restarts)})
            st.session_state["retrain_result"] = out
        except api.ApiError as exc:
            st.session_state["retrain_result"] = {"error": str(exc)}
    st.rerun()
res = st.session_state.get("retrain_result")
if res:
    if "error" in res:
        st.error(f"Retrain failed: {res['error']}")
    else:
        (st.success if res.get("promoted") else st.warning)(f"{res.get('status')}: {res.get('reason')} — held-out NLL new {chrome.fmt_p(res.get('nll_new'), 4)} vs reference {chrome.fmt_p(res.get('nll_reference'), 4)}; active model now {res.get('active_version')}.")
        with st.expander("Retrain result"):
            st.json(res.get("result") or res)
