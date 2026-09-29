"""Client profile (Phase 6, page 3): baseline risk state, context sensitivities with intervals,
consistency score, the loss x context heatmap (``POST /predict`` per cell), a what-if panel with
live recompute, the behavioral drawdown capacity with an editable target (allocation
guardrail), the intervention ranking (predicted effect, not causally validated) with a session
log, the response-history timeline, the stated-vs-revealed panel (only when outcome behavior
exists) and the "Behavioral Risk Profile" PDF export.
"""

from __future__ import annotations

import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import api_client as api  # noqa: E402
from components import charts, chrome, exports, market_bar, store  # noqa: E402

LOSS_GRID_PCT: tuple[int, ...] = (5, 10, 15, 20, 30)
"""The shared design's loss grid (design 1.0.0, ``bre.design.LOSS_PCTS``); every ``/predict``
answer reports ``scenario.on_design_grid`` and the page says so if a level is off the grid."""

health = chrome.page_setup("Client profile")
base = api.base_url()
try:
    model = api.model_info(base)
    book = api.clients(base)
except api.ApiError as exc:
    chrome.stop_on_error(exc)
payload = market_bar.render(model)
demo = chrome.demo_mode()

# ---------------------------------------------------------------------------------------------
# Client selection
# ---------------------------------------------------------------------------------------------

ids = [c["client_id"] for c in book]
by_id = {c["client_id"]: c for c in book}
if not ids:
    st.info("The served book has no clients.")
    st.stop()
preselected = st.session_state.get("selected_client")
index = ids.index(preselected) if preselected in ids else 0
cid = st.selectbox("Client", ids, index=index, format_func=lambda i: f"{by_id[i].get('display_label') or i} ({i})", key="profile_client")
st.session_state["selected_client"] = cid
client = by_id[cid]
cov = client.get("covariates") or {}
cov_json = api.dumps(cov)
metrics = model.get("metrics") or {}

try:
    prof = api.profile(base, api.dumps({"client_id": cid, "covariates": cov, "include_book_r2": True}))
except api.ApiError as exc:
    chrome.stop_on_error(exc)

# ---------------------------------------------------------------------------------------------
# 1. Baseline risk state
# ---------------------------------------------------------------------------------------------

st.subheader("Baseline risk state")
b = prof.get("baseline") or {}
rs = prof.get("risk_score") or {}
cons = prof.get("consistency") or {}
q_score = cov.get("self_reported_risk_tolerance")
n_none = metrics.get("n_responses_no_context")
k1, k2, k3, k4, k5 = st.columns(5)
theta = b.get("bloch_theta")
k1.metric("State angle θ", "n/a" if theta is None else f"{theta:.3f} rad", help=b.get("note"))
k1.caption("n/a" if theta is None else f"{math.degrees(theta):.0f}° on the Bloch sphere; {'fitted subject effect' if prof.get('known_subject') else 'covariates only'}")
k2.metric(f"{chrome.WORDING}, no loss, no context", chrome.fmt_p(b.get("p_sell_base_model")))
k2.caption(f"80% {chrome.fmt_interval(b.get('ci80'))}; 95% {chrome.fmt_interval(b.get('ci95'))}; N = {n_none} training responses without context; this client's stored responses: {client.get('n_responses')}")
k3.metric("Classical-equivalent risk score", chrome.fmt_p(rs.get("value")), help=rs.get("definition"))
k3.caption(f"80% {chrome.fmt_interval(rs.get('ci80'))}; = logit of {chrome.fmt_p(rs.get('p_sell'))}; R² vs state angle across the book {chrome.fmt_p(rs.get('r2_vs_angle'))} (95% {chrome.fmt_interval(rs.get('r2_ci95'))})")
k4.metric("Client's own questionnaire score", "not recorded" if q_score is None else f"{q_score} / 7", help="self-reported risk tolerance from the intake covariates (1 = lowest, 7 = highest)")
k4.caption("shown beside the model score; both are inputs the advisor can edit in the intake record")
k5.metric("Consistency score (γ)", chrome.fmt_p(cons.get("gamma"), 3), help="dephasing rate of the decision state: higher γ means the state loses coherence faster, so context effects fade sooner")
k5.caption(f"80% {chrome.fmt_interval(cons.get('ci80'), 3)}; {cons.get('source') or cons.get('note')}")
with st.expander("Intake covariates (editable in the intake record; the only client data the model uses)"):
    st.dataframe(pd.DataFrame([{"field": k, "value": str(v)} for k, v in cov.items()]), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------
# 2. Sensitivities
# ---------------------------------------------------------------------------------------------

st.subheader("Context sensitivities")
sens = prof.get("sensitivities") or {}
m_loss = re.search(r"loss (-?\d+(?:\.\d+)?)", str(rs.get("definition") or ""))
loss_label = f"a {chrome.fmt_pct(abs(float(m_loss.group(1))))} loss" if m_loss else "the risk-score loss"
sens_rows = []
for tag in chrome.ordered_contexts(list(sens)):
    e = sens[tag]
    nc = e.get("n_calibration") or {}
    sens_rows.append({"label": chrome.context_label(tag), "tag": tag, "delta_p": e.get("delta_p_at_risk_loss"), "delta_ci80": e.get("delta_p_ci80"),
                      "theta_norm": e.get("theta_norm"), "theta_ci80": e.get("theta_norm_ci80"), "n": nc.get("n_responses"), "status": nc.get("status")})
if sens_rows:
    fig = charts.sensitivity_bars(sens_rows, loss_label)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.dataframe(pd.DataFrame([{"context": r["label"], "tag": r["tag"], f"change in P(sell) at {loss_label}": chrome.fmt_signed(r["delta_p"]), "80% interval": chrome.fmt_interval(r["delta_ci80"]),
                                "|θ_c| (rotation)": chrome.fmt_p(r["theta_norm"], 3), "|θ_c| 80% interval": chrome.fmt_interval(r["theta_ci80"], 3), "calibration N": r["n"], "status": r["status"]} for r in sens_rows]),
                 hide_index=True, width="stretch")
    st.caption(f"Change in the {chrome.WORDING} when the context is applied singly at {loss_label} (population parameter θ_c rotated onto this client's state); intervals: {model.get('interval_source')}.")

# ---------------------------------------------------------------------------------------------
# 3. Heatmap: loss x context (singles and ordered pairs) through POST /predict
# ---------------------------------------------------------------------------------------------

st.subheader(f"{chrome.WORDING.capitalize()} over loss × context")
contexts = chrome.ordered_contexts(list(model.get("calibrated_contexts") or {}))
conditions: list[tuple[str, ...]] = [()] + [(c,) for c in contexts] + [(a, b2) for a in contexts for b2 in contexts if a != b2]
progress = st.empty()
bar = progress.progress(0.0, text="scoring the grid through POST /predict …")
p_grid: dict[str, dict[str, float]] = {}
ci_grid: dict[str, dict[str, str]] = {}
grid_n_cal: dict[str, dict] = {}
off_grid: list[str] = []
n_draws = None
total = len(LOSS_GRID_PCT) * len(conditions)
done = 0
try:
    for L in LOSS_GRID_PCT:
        row_label = f"{L}% loss"
        p_grid[row_label] = {}
        ci_grid[row_label] = {}
        for cond in conditions:
            r = api.predict(base, api.dumps({"client_id": cid, "covariates": cov, "loss_pct": -L / 100.0, "context_tags": list(cond), "question_order_id": "scenario-first"}))
            col = chrome.condition_label(cond)
            p_grid[row_label][col] = float(r["p"])
            ci_grid[row_label][col] = f"{chrome.fmt_p(r['p'])} {chrome.fmt_interval(r.get('ci80'))}"
            grid_n_cal.update(r.get("n_calibration") or {})
            n_draws = r.get("n_draws")
            if not (r.get("scenario") or {}).get("on_design_grid", True):
                off_grid.append(row_label)
            done += 1
            bar.progress(done / total, text=f"scoring the grid through POST /predict … {done}/{total}")
except api.ApiError as exc:
    progress.empty()
    chrome.stop_on_error(exc)
progress.empty()
heat = pd.DataFrame(p_grid).T
heat.index.name = "loss"
st.session_state["profile_heatmap"] = heat
fig = charts.heatmap(heat, f"{client.get('display_label') or cid}: {chrome.WORDING} by loss and context sequence (point predictions; intervals in the table below)")
st.pyplot(fig, width="stretch")
heatmap_png = charts.png_bytes(fig)
plt.close(fig)
st.dataframe(heat, column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in heat.columns}, width="stretch")
with st.expander("The same grid with 80% intervals"):
    st.dataframe(pd.DataFrame(ci_grid).T, width="stretch")
heat_note = (f"{len(LOSS_GRID_PCT)} loss levels × {len(conditions)} context conditions (no context, {len(contexts)} singles, {len(conditions) - 1 - len(contexts)} ordered pairs), question order scenario-first; "
             f"intervals from {n_draws} parameter draws ({model.get('interval_source')}); calibration: {chrome.calibration_text(grid_n_cal)}"
             + (f"; OFF the design grid: {sorted(set(off_grid))}" if off_grid else "; every loss level is on the design grid"))
st.caption(heat_note)

# ---------------------------------------------------------------------------------------------
# 4. What-if panel (live recompute)
# ---------------------------------------------------------------------------------------------

st.subheader("What-if")
st.caption("A market state of your own for this client: the API maps it to a scenario (loss clipped to the design range, cause frame matched to the nearest calibrated context, contexts in order of arrival) and scores it.")
levels = api.enum_values(base, "MarketState", "social_cue_prevalence", market_bar.LEVELS_FALLBACK)
cause_choices = [market_bar.NONE_CHOICE] + market_bar.cause_options(model) + [market_bar.FREE_TEXT_CHOICE]
w1, w2, w3 = st.columns(3)
wi_loss = w1.slider("Portfolio loss (%)", 0.0, 50.0, 20.0, step=1.0, key="wi_loss", help="the design range is 5–30%; the API clips and says so")
wi_cause = w2.selectbox("Cause frame", cause_choices, key="wi_cause", format_func=lambda t: t if t in (market_bar.NONE_CHOICE, market_bar.FREE_TEXT_CHOICE) else chrome.context_label(t))
wi_cause_text = w2.text_input("Stated cause (free text)", key="wi_cause_text", max_chars=500) if wi_cause == market_bar.FREE_TEXT_CHOICE else ""
wi_media = w3.selectbox("Media intensity", levels, index=min(2, len(levels) - 1), key="wi_media", help="'none' drops the news context")
w4, w5, w6 = st.columns(3)
wi_social = w4.selectbox("Social cue prevalence", levels, index=0, key="wi_social", help="'friend sells' enters from medium upwards")
wi_recovery = w5.slider("Recovery since the trough (%)", 0.0, 30.0, 0.0, step=0.5, key="wi_recovery", help="the 5% recovery context enters from 5% upwards")
wi_days = w6.number_input("Time since the news (days)", min_value=0, max_value=3650, value=7, step=1, key="wi_days")
w6.caption("carried to the API only: elapsed time is not a context of the served design, so this field does not change the prediction")
wi_market = {"drawdown_pct": -wi_loss / 100.0, "duration_days": int(wi_days),
             "cause_frame": (wi_cause_text.strip() or None) if wi_cause == market_bar.FREE_TEXT_CHOICE else (None if wi_cause == market_bar.NONE_CHOICE else wi_cause),
             "recovery_pct": wi_recovery / 100.0, "vix_bucket": payload.get("vix_bucket"), "media_intensity": wi_media, "social_cue_prevalence": wi_social}
try:
    wi_scored = api.score_book(base, api.dumps(api.client_payload([client])), api.dumps(wi_market), float(st.session_state.get("capacity_target", 0.25)))
    wi_scen = wi_scored["scenario"]
    wi_pred = api.predict(base, api.dumps({"client_id": cid, "covariates": cov, "loss_pct": wi_scen["loss_pct"], "context_tags": list(wi_scen["context_tags"]), "question_order_id": wi_scen.get("question_order_id", "scenario-first")}))
except api.ApiError as exc:
    chrome.stop_on_error(exc)
wi_row = wi_scored["rows"][0]
market_bar.render_scenario(wi_scen, wi_pred.get("n_calibration"), heading="Scenario the API built from these inputs")
x1, x2, x3, x4 = st.columns(4)
x1.metric(chrome.WORDING, chrome.fmt_p(wi_pred["p"]))
x1.caption(f"80% {chrome.fmt_interval(wi_pred.get('ci80'))}; 95% {chrome.fmt_interval(wi_pred.get('ci95'))}; {wi_pred.get('n_draws')} draws")
x2.metric("Change vs baseline", chrome.fmt_signed(wi_row.get("change_vs_baseline")), help="minus the predicted probability with no loss and no context")
td = wi_pred.get("top_driver") or {}
x3.metric("Top driver", chrome.context_label(td["driver"]) if td.get("driver") and td["driver"] != "loss" else (td.get("driver") or "none"))
x3.caption(f"removing it changes the prediction by {chrome.fmt_signed(td.get('delta_p'))}")
intf = wi_pred.get("interference") or {}
x4.metric("Interference (LTP term)", chrome.fmt_signed(intf.get("ltp"), 3), help=intf.get("definitions") or intf.get("note"))
x4.caption("order effect " + (chrome.fmt_signed(intf.get("order_effect"), 3) if intf.get("order_effect") is not None else "n/a (needs an ordered pair)"))
st.caption(f"Calibration: {chrome.calibration_text(wi_pred.get('n_calibration'))}. On the design grid: {(wi_pred.get('scenario') or {}).get('on_design_grid')}; in the design range: {(wi_pred.get('scenario') or {}).get('in_design_range')}.")

# ---------------------------------------------------------------------------------------------
# 5. Behavioral drawdown capacity (allocation guardrail)
# ---------------------------------------------------------------------------------------------

st.subheader("Behavioral drawdown capacity — allocation guardrail")
cap_target = float(st.number_input(f"Target {chrome.WORDING} (guardrail)", min_value=0.05, max_value=0.95, value=float(st.session_state.get("capacity_target", 0.25)), step=0.05, format="%.2f", key="capacity_target_input"))
st.session_state["capacity_target"] = cap_target
try:
    cap_scored = api.score_book(base, api.dumps(api.client_payload([client])), api.dumps(payload), cap_target)
except api.ApiError as exc:
    chrome.stop_on_error(exc)
cap_row = cap_scored["rows"][0]
cap_meta = cap_scored["meta"]
g1, g2 = st.columns([1, 2])
g1.metric("Drawdown capacity", chrome.fmt_pct(cap_row.get("drawdown_capacity")), help=cap_meta.get("capacity_definition"))
g1.caption(f"{cap_row.get('capacity_status')}; {chrome.GUARDRAIL_LABEL}")
g1.caption(f"Definition: {cap_meta.get('capacity_definition')}.")
m_tags = re.search(r"\[(.*?)\]", str(cap_meta.get("capacity_definition") or ""))
crisis_tags = tuple(t.strip().strip("'\"") for t in m_tags.group(1).split(",")) if m_tags else ()
crisis_col = chrome.condition_label(crisis_tags) if crisis_tags else None
capacity_info = {"target": cap_target, "drawdown_capacity": cap_row.get("drawdown_capacity"), "capacity_status": cap_row.get("capacity_status"), "capacity_definition": cap_meta.get("capacity_definition")}
if crisis_col in heat.columns:
    fig = charts.capacity_curve(list(LOSS_GRID_PCT), heat[crisis_col].tolist(), cap_target, 100.0 * float(cap_row.get("drawdown_capacity") or 0.0), crisis_col)
    g2.pyplot(fig, width="stretch")
    plt.close(fig)
    g2.caption(f"the heatmap column for {crisis_col} against the target; the capacity is the API's value at this target")

# ---------------------------------------------------------------------------------------------
# 6. Intervention ranking
# ---------------------------------------------------------------------------------------------

st.subheader("Intervention ranking")
try:
    ranking = api.rank_interventions(base, api.dumps({"client_id": cid, "covariates": cov, "market_state": payload}))
except api.ApiError as exc:
    chrome.stop_on_error(exc)
iv_label = ranking.get("label") or chrome.INTERVENTION_LABEL_FALLBACK
st.warning(f"Every effect below is a {iv_label}: the transform each script applies to the context sequence is a modelling assumption.", icon="ℹ️")
st.write(f"Before any intervention under the current market state: {chrome.WORDING} {chrome.prob_text(ranking.get('p_before'), ranking.get('ci80_before'))}; calibration: {chrome.calibration_text(ranking.get('n_calibration'))}.")
ranked = ranking.get("ranked") or []
rank_df = pd.DataFrame([{"rank": r["rank"], "intervention": r["name"], "P(sell) before": r["p_before"], "P(sell) after": r["p_after"], "predicted change": r["delta_p"],
                         "80% interval of the change": chrome.fmt_interval(r.get("delta_ci80")), "contexts after": chrome.condition_label(r.get("contexts_after") or []),
                         "question order after": r.get("question_order_after"), "no change": r.get("no_change"), "label": r.get("label")} for r in ranked])
st.dataframe(rank_df, hide_index=True, width="stretch", column_config={"P(sell) before": st.column_config.NumberColumn(format="%.2f"), "P(sell) after": st.column_config.NumberColumn(format="%.2f"), "predicted change": st.column_config.NumberColumn(format="%+.2f")})
if ranking.get("inactive"):
    st.caption(f"Inactive (listed, not ranked): {', '.join(ranking['inactive'])}")
with st.expander("Scripts"):
    for r in ranked:
        st.markdown(f"**{r['rank']}. {r['name']}** — {r.get('label')}")
        st.write(r.get("script") or "")
l1, l2, l3 = st.columns([2, 3, 1])
log_choice = l1.selectbox("Intervention to log", [r["name"] for r in ranked], key="log_choice")
log_note = l2.text_input("Advisor note (optional, no client details)", key="log_note", max_chars=500)
if l3.button("Log intervention", key="log_button"):
    rec = next(r for r in ranked if r["name"] == log_choice)
    store.log_intervention(cid, log_choice, log_note, rec.get("p_before"), rec.get("delta_p"), iv_label)
    st.success(f"Logged {log_choice} for this client ({store.PERSISTENCE_NOTE}).")
session_log = store.intervention_log(cid)
if session_log:
    st.dataframe(pd.DataFrame(session_log), hide_index=True, width="stretch")
    st.caption(f"Contact status is now '{store.get_status(cid)}'. {store.PERSISTENCE_NOTE}.")

# ---------------------------------------------------------------------------------------------
# 7. Response history and 8. stated vs revealed
# ---------------------------------------------------------------------------------------------

st.subheader("Response history")
try:
    responses, resp_source = api.client_responses(base, cid)
except api.ApiError as exc:
    responses, resp_source = [], f"unavailable ({exc})"
responses_summary = None
if not responses:
    st.info(f"No stored responses for this client (source: {resp_source}).")
else:
    tl_rows = []
    table_rows = []
    for i, r in enumerate(responses):
        loss = r.get("loss_pct")
        typ = r.get("elicitation_type")
        resp = r.get("response")
        if typ == "binary_sell":
            kind = "sell" if resp == 1 else "hold"
        elif typ == "binary_yes_no":
            kind = "tolerance: yes" if resp == 1 else "tolerance: no"
        else:
            kind = f"{typ}: {resp}"
        tl_rows.append({"x": i + 1, "y": 0.0 if loss is None or (isinstance(loss, float) and math.isnan(loss)) else 100.0 * abs(float(loss)), "kind": kind})
        table_rows.append({"#": i + 1, "session": r.get("session_id"), "position": r.get("position_in_session"), "item": r.get("scenario_id"), "loss": "—" if loss is None or (isinstance(loss, float) and math.isnan(loss)) else chrome.fmt_pct(abs(float(loss))),
                           "contexts": chrome.condition_label(r.get("context_tags") or []), "question order": r.get("question_order_id"), "type": typ, "answer": kind,
                           "timestamp": r.get("timestamp") or "—", "outcome behavior": "—" if not r.get("outcome_behavior") else str(r["outcome_behavior"]), "synthetic": r.get("is_synthetic")})
    fig = charts.response_timeline(tl_rows)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
    n_sell_items = sum(1 for r in responses if r.get("elicitation_type") == "binary_sell")
    n_sold = sum(1 for r in responses if r.get("elicitation_type") == "binary_sell" and r.get("response") == 1)
    st.caption(f"{len(responses)} stored items ({n_sell_items} sell/hold items, {n_sold} sold); stated intake choices, not predictions; source: {resp_source}; synthetic rows: {all(r.get('is_synthetic') for r in responses)}.")
    responses_summary = {"n_items": len(responses), "n_sell_items": n_sell_items, "n_sell": n_sold, "source": resp_source, "synthetic": all(r.get("is_synthetic") for r in responses)}
    revealed = [r for r in responses if r.get("outcome_behavior")]
    if revealed:
        st.subheader("Stated vs revealed")
        sv = pd.DataFrame([{"item": r.get("scenario_id"), "stated": ("sell" if r.get("response") == 1 else "hold") if r.get("elicitation_type") == "binary_sell" else r.get("response"),
                            "revealed action": (r["outcome_behavior"] or {}).get("action"), "lag (days)": (r["outcome_behavior"] or {}).get("lag_days")} for r in revealed])
        st.dataframe(sv, hide_index=True, width="stretch")
        agree = sum(1 for r in revealed if r.get("elicitation_type") == "binary_sell" and (("sell" if r.get("response") == 1 else "hold") == str((r["outcome_behavior"] or {}).get("action", "")).lower()))
        st.caption(f"{len(revealed)} items with observed behavior; stated choice matched the revealed action in {agree} of them (observational, no causal claim).")
        responses_summary["stated_vs_revealed"] = f"{agree} of {len(revealed)} stated choices matched the revealed action"

# ---------------------------------------------------------------------------------------------
# 9. PDF export
# ---------------------------------------------------------------------------------------------

st.subheader("Export")
pdf_bytes = exports.profile_pdf(
    demo=demo, client=client, health=health, model_info=model, market_state=payload, scenario=cap_scored.get("scenario"), profile=prof,
    heatmap_png=heatmap_png, heatmap_note=heat_note, capacity=capacity_info, ranking=ranking, responses_summary=responses_summary, session_log=session_log,
)
st.session_state["profile_pdf"] = pdf_bytes
st.download_button("Download Behavioral Risk Profile (PDF)", pdf_bytes, file_name=f"bre_profile_{cid}_{exports.stamp().replace(':', '')}.pdf", mime="application/pdf", key="profile_pdf_download",
                   help="header on every page: the SYNTHETIC DATA banner in demo mode; every probability is labelled as a predicted probability of selling")
