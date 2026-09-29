"""Book triage (Phase 6, page 2): one row per client from ``POST /score_book`` under the
current market state — predicted probability of selling with its 80% interval and the
calibration N, change vs baseline, top driver, suggested intervention (predicted effect, not
causally validated), behavioral drawdown capacity, last contact and an editable contact status.
Sortable, filterable, CSV export (first line: the SYNTHETIC DATA banner in demo mode) and the
"N clients above threshold" alert with an editable threshold.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import api_client as api  # noqa: E402
from components import chrome, exports, market_bar, store  # noqa: E402

health = chrome.page_setup("Book triage")
base = api.base_url()
try:
    model = api.model_info(base)
    book = api.clients(base)
except api.ApiError as exc:
    chrome.stop_on_error(exc)

payload = market_bar.render(model)
target = float(st.session_state.setdefault("capacity_target", 0.25))
try:
    scored = api.score_book(base, api.dumps(api.client_payload(book)), api.dumps(payload), target)
except api.ApiError as exc:
    chrome.stop_on_error(exc)

market_bar.render_scenario(scored["scenario"], scored["n_calibration"])
meta = scored["meta"]
n_cal = scored["n_calibration"]
n_min = chrome.calibration_min_n(n_cal)
iv_label = meta.get("intervention_label") or chrome.INTERVENTION_LABEL_FALLBACK

# ---------------------------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------------------------

frame = pd.DataFrame(scored["rows"])
frame["interval80"] = [chrome.fmt_interval([lo, hi]) for lo, hi in zip(frame["ci80_low"], frame["ci80_high"])]
frame["calibration_n"] = n_min
frame["top_driver_label"] = [chrome.context_label(d) if d and d != "loss" else ("loss" if d == "loss" else "none") for d in frame["top_driver"]]
frame["capacity_pct"] = 100.0 * frame["drawdown_capacity"].astype(float)
frame["contact_status"] = [store.get_status(c) for c in frame["client_id"]]
frame["last_contact"] = [store.last_contact(c) or "—" for c in frame["client_id"]]
frame = frame.rename(columns={"status": "model_status"})

COLUMNS = ["display_label", "client_id", "p_sell", "interval80", "calibration_n", "change_vs_baseline", "top_driver_label", "top_driver_delta",
           "suggested_intervention", "suggested_delta", "capacity_pct", "capacity_status", "model_status", "contact_status", "last_contact", "known_client"]
COLUMN_CONFIG = {
    "display_label": st.column_config.TextColumn("client", help="display label (the only identifying text stored)"),
    "client_id": st.column_config.TextColumn("client id"),
    "p_sell": st.column_config.NumberColumn(f"P(sell)", format="%.2f", help=f"{chrome.WORDING} under the current market state"),
    "interval80": st.column_config.TextColumn("80% interval", help=meta.get("interval")),
    "calibration_n": st.column_config.NumberColumn("calibration N", help="smallest number of training responses behind the scenario's contexts: " + chrome.calibration_text(n_cal)),
    "change_vs_baseline": st.column_config.NumberColumn("change vs baseline", format="%+.2f", help="minus the predicted probability with no loss and no context"),
    "top_driver_label": st.column_config.TextColumn("top driver", help="the context (or the loss) whose removal changes the prediction the most"),
    "top_driver_delta": st.column_config.NumberColumn("driver effect", format="%+.2f"),
    "suggested_intervention": st.column_config.TextColumn("suggested intervention", help=iv_label),
    "suggested_delta": st.column_config.NumberColumn("predicted change", format="%+.2f", help=iv_label),
    "capacity_pct": st.column_config.NumberColumn("drawdown capacity (%)", format="%.0f%%", help=f"{meta.get('capacity_definition')} — {chrome.GUARDRAIL_LABEL}"),
    "capacity_status": st.column_config.TextColumn("capacity status"),
    "model_status": st.column_config.TextColumn("model status", help=f"thresholds {meta.get('status_thresholds')}"),
    "contact_status": st.column_config.TextColumn("contact status", help=store.PERSISTENCE_NOTE),
    "last_contact": st.column_config.TextColumn("last contact", help="last status change or logged intervention in this session"),
    "known_client": st.column_config.CheckboxColumn("known to model", help="the client's fitted random effects were used"),
}

# ---------------------------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------------------------

if "alert_threshold" not in st.session_state:
    st.session_state["alert_threshold"] = float(meta["status_thresholds"]["high"])
threshold = float(st.number_input(f"Alert threshold ({chrome.WORDING})", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="alert_threshold"))
n_above = int((frame["p_sell"] >= threshold).sum())
alert = f"{n_above} clients above threshold {threshold:.2f} ({chrome.WORDING} ≥ {threshold:.2f}, point prediction; intervals in the table)"
if n_above:
    st.error(alert, icon="🚨")
else:
    st.success(alert)

# ---------------------------------------------------------------------------------------------
# Filters and sort
# ---------------------------------------------------------------------------------------------

f1, f2, f3, f4, f5, f6 = st.columns([2, 2, 2, 2, 2, 1])
search = f1.text_input("Search label / id", key="triage_search")
model_statuses = sorted(frame["model_status"].unique())
sel_model = f2.multiselect("Model status", model_statuses, default=model_statuses, key="triage_model_status")
sel_contact = f3.multiselect("Contact status", list(store.STATUS_OPTIONS), default=list(store.STATUS_OPTIONS), key="triage_contact_status")
min_p = f4.slider("Minimum P(sell)", 0.0, 1.0, 0.0, step=0.05, key="triage_min_p")
sort_by = f5.selectbox("Sort by", COLUMNS, index=COLUMNS.index("p_sell"), format_func=lambda c: COLUMN_CONFIG[c].get("label") if isinstance(COLUMN_CONFIG.get(c), dict) else c, key="triage_sort")
descending = f6.checkbox("Descending", value=True, key="triage_desc")

view = frame
if search.strip():
    s = search.strip().lower()
    view = view[view["display_label"].fillna("").str.lower().str.contains(s, regex=False) | view["client_id"].str.lower().str.contains(s, regex=False)]
view = view[view["model_status"].isin(sel_model) & view["contact_status"].isin(sel_contact) & (view["p_sell"] >= min_p)]
view = view.sort_values(sort_by, ascending=not descending, kind="stable").reset_index(drop=True)

st.subheader(f"{len(view)} of {len(frame)} clients")
st.dataframe(view[COLUMNS], column_config=COLUMN_CONFIG, hide_index=True, width="stretch")
st.caption(
    f"P(sell) is the {chrome.WORDING} under the scenario above; the 80% interval is {meta.get('interval')}. "
    f"Calibration: {chrome.calibration_text(n_cal)}. Suggested interventions: {iv_label}. "
    f"Drawdown capacity: {meta.get('capacity_definition')} (target {target:.2f}, editable on the client profile) — {chrome.GUARDRAIL_LABEL}. "
    f"Contact status and last contact: {store.PERSISTENCE_NOTE}."
)

# ---------------------------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------------------------

csv_text = exports.triage_csv(view[COLUMNS], demo=chrome.demo_mode(), model_version=str(health.get("model_version")), market_state=payload)
st.session_state["triage_csv"] = csv_text
st.download_button("Download triage CSV", csv_text, file_name=f"bre_triage_{exports.stamp().replace(':', '')}.csv", mime="text/csv", key="triage_csv_download",
                   help="first line: the SYNTHETIC DATA banner in demo mode; second line: model version, wording and market state")

# ---------------------------------------------------------------------------------------------
# Contact status editor (session store) and profile link
# ---------------------------------------------------------------------------------------------


def _apply_status_edits() -> None:
    state = st.session_state.get("status_editor") or {}
    edits = state.get("edited_rows") or {}
    ids = st.session_state.get("status_editor_ids") or []
    applied = st.session_state.setdefault("status_editor_applied", {})
    for idx, change in edits.items():
        key = str(idx)
        if applied.get(key) == change:
            continue
        if "contact_status" in change and int(idx) < len(ids):
            store.set_status(ids[int(idx)], change["contact_status"])
        applied[key] = dict(change)


with st.expander("Edit contact status (kept in this session)", expanded=False):
    st.caption(store.PERSISTENCE_NOTE)
    edit_df = frame.sort_values("client_id")[["display_label", "client_id", "contact_status", "last_contact"]].reset_index(drop=True)
    st.session_state["status_editor_ids"] = edit_df["client_id"].tolist()
    if not ((st.session_state.get("status_editor") or {}).get("edited_rows")):
        st.session_state["status_editor_applied"] = {}
    st.data_editor(
        edit_df,
        key="status_editor",
        on_change=_apply_status_edits,
        hide_index=True,
        disabled=["display_label", "client_id", "last_contact"],
        column_config={
            "display_label": st.column_config.TextColumn("client"),
            "client_id": st.column_config.TextColumn("client id"),
            "contact_status": st.column_config.SelectboxColumn("contact status", options=list(store.STATUS_OPTIONS), required=True),
            "last_contact": st.column_config.TextColumn("last contact"),
        },
        width="stretch",
    )

ids = view["client_id"].tolist() if len(view) else frame["client_id"].tolist()
label_of = dict(zip(frame["client_id"], frame["display_label"]))
p1, p2 = st.columns([3, 1])
chosen = p1.selectbox("Open a client profile", ids, format_func=lambda i: f"{label_of.get(i) or i} ({i})", key="triage_open_client")
if p2.button("Open profile", key="triage_open_button"):
    st.session_state["selected_client"] = chosen
    st.switch_page("pages/2_Client_profile.py")
