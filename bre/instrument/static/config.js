// config.js -- deployment settings for the BRE intake battery. Plain script, no build step.
// Edit this file in the deployed copy; nothing else needs to change.
window.BRE_CONFIG = {
  // Where the responses JSON is POSTed on completion. Leave "" for download-only operation.
  // See README.md ("Collecting responses") for the free form-endpoint option.
  ENDPOINT_URL: "",

  // "" posts the bare responses array (byte-identical to the downloaded file).
  // Most form endpoints require a JSON object, e.g. "responses" posts {"responses": [...]}.
  ENDPOINT_WRAP_KEY: "",

  // Extra top-level fields merged into the wrapper object when ENDPOINT_WRAP_KEY is set
  // (for example an access key that a form endpoint requires). Ignored when unwrapped.
  ENDPOINT_EXTRA_FIELDS: {},

  // When wrapped, also include the session log (assignment, page timings) under "session_log".
  ENDPOINT_INCLUDE_SESSION_LOG: false,

  // Shown on the completion page: where volunteers send the downloaded file (e.g. an email address).
  STUDY_CONTACT: "",

  // Testing only. null = use battery.json design.delay_page.display_seconds. Whatever value is
  // actually used is written to the session log as delay_seconds_used / delay_override_active.
  DELAY_SECONDS_OVERRIDE: null
};
