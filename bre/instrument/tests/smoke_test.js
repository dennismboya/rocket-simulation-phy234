// End-to-end smoke test of the static app under jsdom: loads index.html from file://, clicks through
// every page, then checks the produced DecisionEvent rows, the session log and the POST body.
//
// Needs jsdom (not vendored; dev-only). From a temporary directory:
//   npm install jsdom && NODE_PATH=$PWD/node_modules /opt/node22/bin/node <repo>/bre/instrument/tests/smoke_test.js full
// usage: node smoke_test.js <full|short> [consent:yes|no] [endpoint:yes|no] [wrap:yes|no]
const fs = require('fs');
const os = require('os');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const FORM = process.argv[2] || 'full';
const CONSENT = (process.argv[3] || 'consent:yes').split(':')[1] === 'yes';
const ENDPOINT = (process.argv[4] || 'endpoint:yes').split(':')[1] === 'yes';
const WRAP = (process.argv[5] || 'wrap:no').split(':')[1] === 'yes';
const ROOT = path.resolve(__dirname, '..');
const SRC = path.join(ROOT, 'static');
const DST = fs.mkdtempSync(path.join(os.tmpdir(), 'bre-smoke-' + FORM + '-'));
fs.cpSync(SRC, DST, { recursive: true });
fs.writeFileSync(path.join(DST, 'config.js'), `window.BRE_CONFIG = {
  ENDPOINT_URL: ${ENDPOINT ? '"https://example.invalid/collect"' : '""'},
  ENDPOINT_WRAP_KEY: ${WRAP ? '"responses"' : '""'},
  ENDPOINT_EXTRA_FIELDS: ${WRAP ? '{"access_key":"k"}' : '{}'},
  ENDPOINT_INCLUDE_SESSION_LOG: ${WRAP ? 'true' : 'false'},
  STUDY_CONTACT: "pilot@example.invalid",
  DELAY_SECONDS_OVERRIDE: 0.25
};`);

const battery = JSON.parse(fs.readFileSync(path.join(ROOT, 'battery.json'), 'utf8'));
const SCHEMA = battery.output.schema_fields;
const universe = new Map(battery.design.scenario_universe.map(u => [u.scenario_id, u]));

function assert(c, m) { if (!c) { throw new Error('ASSERT: ' + m); } }
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const posts = [];
  const errors = [];
  const vc = new VirtualConsole();
  vc.on('jsdomError', e => errors.push(String(e.message || e)));
  vc.on('error', m => errors.push(String(m)));
  const dom = await JSDOM.fromFile(path.join(DST, 'index.html'), {
    url: 'file://' + path.join(DST, 'index.html') + '?form=' + FORM,
    runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(window) {
      window.fetch = (url, opts) => { posts.push({ url, opts }); return Promise.resolve({ ok: true, status: 200 }); };
    }
  });
  const w = dom.window, d = w.document;

  for (let i = 0; i < 400 && !w.BRE_STATE; i++) { await sleep(25); }
  assert(w.BRE_STATE, 'BRE_STATE never appeared; errors: ' + errors.join(' | '));
  const S = w.BRE_STATE;
  console.log('form', S.form, 'seed', S.seed, 'presentations', S.assignment.n_presentations, 'pages', S.totalPages);

  let steps = 0;
  const t0 = Date.now();
  while (!S.finished && steps < 1000) {
    steps++;
    const form = d.querySelector('#jspsych-survey-html-form');
    const slider = d.querySelector('#jspsych-html-slider-response-response');
    const btns = Array.from(d.querySelectorAll('#jspsych-html-button-response-btngroup .jspsych-btn'));
    if (form) {
      const names = new Set(Array.from(form.querySelectorAll('input')).map(i => i.name).filter(Boolean));
      for (const name of names) {
        const inputs = Array.from(form.querySelectorAll('input[name="' + name + '"]'));
        const t = inputs[0].type;
        if (t === 'checkbox') {
          inputs[0].checked = (name === 'consent_participate') ? true : CONSENT;
        } else if (t === 'radio') {
          let pickEl = inputs.filter(i => i.value !== '__null__');
          pickEl = name === 'self_reported_risk_tolerance' ? inputs.find(i => i.value === '5') : pickEl[pickEl.length - 1];
          pickEl.checked = true;
        } else if (t === 'number') {
          inputs[0].value = '7';
        }
      }
      form.querySelector('#jspsych-survey-html-form-next').click();
    } else if (slider) {
      const next = d.querySelector('#jspsych-html-slider-response-next');
      assert(next.disabled === true, 'slider continue must start disabled (require_movement)');
      slider.value = '35';
      slider.dispatchEvent(new w.Event('change', { bubbles: true }));
      await sleep(5);
      assert(next.disabled === false, 'slider continue enabled after movement');
      next.click();
    } else if (btns.length) {
      const enabled = btns.filter(b => !b.disabled);
      if (!enabled.length) { await sleep(30); continue; } // delay page: buttons disabled until enable_button_after
      const kick = d.querySelector('.bre-kicker') ? d.querySelector('.bre-kicker').textContent : '';
      const q = d.querySelector('.bre-question') ? d.querySelector('.bre-question').textContent : '';
      let idx = 0;
      if (q === battery.design.questions.tolerance.text) { idx = 1; }        // "No" -> 0
      else if (q === battery.design.questions.sell_hold.text) { idx = 0; }   // "Sell" -> 1
      else if (kick.startsWith('Knowledge question')) {
        const n = parseInt(kick.replace(/\D+/g, ' ').trim().split(' ')[0], 10);
        const item = battery.intake.financial_literacy_quiz.items[n - 1];
        idx = n <= 3 ? item.answer_index : item.options.length - 1;  // first 3 right, last 2 "I don't know"
      }
      if (d.querySelector('#bre-post-status')) { break; }
      btns[idx].click();
    } else {
      await sleep(20);
    }
    await sleep(8);
  }
  for (let i = 0; i < 200 && !S.finished; i++) { await sleep(25); }
  assert(S.finished, 'did not finish after ' + steps + ' steps');
  console.log('finished in', steps, 'steps,', ((Date.now() - t0) / 1000).toFixed(1), 's; page errors:', errors.length);
  assert(errors.length === 0, 'errors: ' + errors.join(' | '));

  // ---------------- rows ----------------
  const rows = S.records;
  const nPres = S.assignment.n_presentations;
  assert(rows.length === nPres * 3, 'rows = 3 per presentation, got ' + rows.length);
  rows.forEach((r, i) => {
    assert(JSON.stringify(Object.keys(r)) === JSON.stringify(SCHEMA), 'row keys/order mismatch: ' + Object.keys(r).join(','));
    assert(r.dataset === 'intake_battery' && r.session_id === S.seed && r.subject_id === S.subject_id, 'ids');
    assert(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(r.subject_id), 'uuid v4 format ' + r.subject_id);
    assert(r.position_in_session === i, 'position sequential');
    assert(!isNaN(Date.parse(r.timestamp)) && r.timestamp.endsWith('Z'), 'timestamp ISO UTC');
    const u = universe.get(r.scenario_id);
    assert(u, 'scenario_id in universe: ' + r.scenario_id);
    assert(r.loss_pct === u.loss_pct && r.horizon_days === 365, 'loss/horizon match universe');
    assert(JSON.stringify(r.context_tags) === JSON.stringify(u.context_tags), 'context_tags ordered list');
    assert(r.question_order_id === u.question_order_id, 'order id');
    assert(['binary_sell', 'allocation_pct', 'likert', 'lottery_choice', 'binary_yes_no', 'choice_rate'].includes(r.elicitation_type), 'elicitation enum');
    if (r.elicitation_type === 'allocation_pct') { assert(r.response === 0.35, 'allocation coded as fraction, got ' + r.response); }
    if (r.elicitation_type === 'binary_sell') { assert(r.response === 1, 'Sell -> 1'); }
    if (r.elicitation_type === 'binary_yes_no') { assert(r.response === 0, 'tolerance No -> 0'); }
    assert(!(r.source_row_ref.endsWith(':tolerance')) || r.elicitation_type === 'binary_yes_no', 'tolerance rows are binary_yes_no');
    assert(Number.isInteger(r.response_time_ms) && r.response_time_ms >= 0, 'rt integer');
    assert(r.outcome_behavior === null && r.incentivized === false && r.is_synthetic === false, 'constants');
    assert(r.consent_training === CONSENT, 'consent_training == ' + CONSENT);
    assert(r.battery_version === battery.battery_version, 'battery_version');
    assert(JSON.stringify(Object.keys(r.covariates)) === JSON.stringify(['age_band', 'wealth_band', 'invest_experience_yrs', 'self_reported_risk_tolerance', 'financial_literacy_score', 'education']), 'covariate keys');
    assert(r.covariates.age_band === '60+' && r.covariates.wealth_band === '>1M' && r.covariates.education === 'graduate', 'covariate bands as chosen');
    assert(r.covariates.invest_experience_yrs === 7 && r.covariates.self_reported_risk_tolerance === 5, 'numeric covariates');
    assert(r.covariates.financial_literacy_score === 3, 'financial literacy score = 3 correct, got ' + r.covariates.financial_literacy_score);
    assert(/^(full|short):i\d{2}(:rep_of_i\d{2})?:(tolerance|sell_hold|allocation_share)$/.test(r.source_row_ref), 'source_row_ref format: ' + r.source_row_ref);
  });
  for (let i = 0; i < rows.length; i += 3) {
    const grp = rows.slice(i, i + 3);
    const qids = grp.map(r => r.source_row_ref.split(':').pop());
    const priors = grp.map(r => r.prior_question_ids);
    if (grp[0].question_order_id === 'tolerance-first') {
      assert(qids.join() === 'tolerance,sell_hold,allocation_share', 'TF question sequence ' + qids.join());
      assert(JSON.stringify(priors) === JSON.stringify([[], ['tolerance'], ['tolerance', 'sell_hold']]), 'TF priors ' + JSON.stringify(priors));
    } else {
      assert(qids.join() === 'sell_hold,allocation_share,tolerance', 'SF question sequence ' + qids.join());
      assert(JSON.stringify(priors) === JSON.stringify([[], ['sell_hold'], ['sell_hold', 'allocation_share']]), 'SF priors ' + JSON.stringify(priors));
    }
    assert(new Set(grp.map(r => r.scenario_id)).size === 1, 'same scenario within item');
  }
  const repRows = rows.filter(r => r.source_row_ref.includes('rep_of'));
  assert(repRows.length === S.assignment.n_repeats * 3, 'repeat rows = 3 x n_repeats');
  repRows.forEach(r => {
    const m = /:i(\d{2}):rep_of_i(\d{2}):/.exec(r.source_row_ref);
    const orig = rows.find(x => x.source_row_ref.startsWith(S.form + ':i' + m[2] + ':'));
    assert(orig && orig.scenario_id === r.scenario_id, 'repeat verbatim scenario id');
  });

  // ---------------- session log ----------------
  const log = S.log;
  assert(log.items.length === nPres, 'log items');
  log.items.forEach(it => {
    if (it.has_news_delay) { assert(typeof it.delay_display_ms === 'number' && it.delay_display_ms >= 250, 'delay_display_ms >= 250 for news item, got ' + it.delay_display_ms); }
    else { assert(it.delay_display_ms === null, 'no delay for non-news item'); }
  });
  assert(log.delay_seconds_used === 0.25 && log.delay_override_active === true, 'delay override recorded');
  assert(log.ended_at && log.n_rows === rows.length && log.consent_training === CONSENT, 'log summary');

  // ---------------- completion page & POST ----------------
  assert(d.querySelector('#bre-dl-responses') && d.querySelector('#bre-dl-log'), 'download buttons present');
  assert(d.body.textContent.includes('pilot@example.invalid'), 'study contact shown');
  d.querySelector('#bre-show-json').click();
  const ta = d.querySelector('#bre-json');
  assert(ta.style.display === 'block' && JSON.parse(ta.value).length === rows.length, 'Show JSON works');
  d.querySelector('#bre-dl-responses').click(); // must not throw (data: URI fallback under jsdom)
  if (ENDPOINT) {
    await sleep(20);
    assert(posts.length === 1, 'exactly one POST, got ' + posts.length);
    assert(posts[0].url === 'https://example.invalid/collect' && posts[0].opts.method === 'POST', 'POST target');
    const body = JSON.parse(posts[0].opts.body);
    if (WRAP) {
      assert(body.access_key === 'k' && JSON.stringify(body.responses) === JSON.stringify(rows) && body.session_log && body.session_log.seed === S.seed, 'wrapped POST body');
    } else {
      assert(JSON.stringify(body) === JSON.stringify(rows), 'POST body identical to the responses array');
    }
    assert(d.querySelector('#bre-post-status').textContent.includes('sent'), 'status shows sent');
    d.querySelector('#bre-resend').click(); await sleep(20);
    assert(posts.length === 2, 'Send again re-posts');
  } else {
    assert(posts.length === 0, 'no POST without endpoint');
    assert(d.body.textContent.includes('Nothing has been sent'), 'download-only notice');
  }
  // localStorage backup: jsdom treats file:// as an opaque origin and throws; real browsers allow it.
  try {
    const ls = JSON.parse(w.localStorage.getItem('bre_intake_' + S.seed));
    assert(ls && ls.responses.length === rows.length, 'localStorage backup');
    console.log('localStorage backup: present');
  } catch (e) {
    console.log('localStorage backup: not testable under jsdom (' + e.name + '); app-side write is try/catch-guarded');
  }
  // assignment reproducible from the seed alone
  const BRE = require(path.join(SRC, 'battery.js'));
  const again = BRE.buildAssignment(battery, S.form, S.seed);
  assert(JSON.stringify(again.slots.map(s => [s.item_index, s.is_repeat, s.repeat_of, s.item.scenario_id])) ===
         JSON.stringify(S.assignment.slots.map(s => [s.item_index, s.is_repeat, s.repeat_of, s.item.scenario_id])), 'assignment regenerated offline from seed');

  console.log('rows:', rows.length, '| repeat rows:', repRows.length, '| POSTs:', posts.length, '| fl score:', rows[0].covariates.financial_literacy_score);
  console.log('sample row:', JSON.stringify(rows[0]));
  console.log('SMOKE TEST PASSED (' + FORM + ', consent=' + CONSENT + ', endpoint=' + ENDPOINT + ', wrap=' + WRAP + ')');
  w.close();
  fs.rmSync(DST, { recursive: true, force: true });
})().catch(e => { console.error(e.stack || e); process.exit(1); });
