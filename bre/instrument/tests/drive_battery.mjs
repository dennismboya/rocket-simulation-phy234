#!/usr/bin/env node
// drive_battery.mjs -- end-to-end drive of the BRE intake battery in a real (headless) Chromium.
//
// Serves ../static with `python -m http.server` on a free port, opens the app, clicks through the
// full form (training consent checked) and the short form (training consent unchecked) choosing
// every answer at random (but validly), then drives the short form again in a 375 px phone viewport.
// It captures the JSON that the "Download responses" and "Download session log" buttons produce and
// checks the rows against the DecisionEvent schema (PLAN.md section 2, incl. the 2026-09-17
// amendment: tolerance rows are binary_yes_no), battery.json, the clicks that were made, and the
// page layout (no horizontal overflow, every control inside the viewport) on every page.
//
// Usage (from the repository root; needs the `playwright` package somewhere Node can find it: next
// to this file, in $BRE_PW_DIR, in the global node_modules of the running Node, or in
// /tmp/claude-0/pw-scratch; never `playwright install` here, the browser is under
// PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers and /opt/pw-browsers/chromium is the fallback binary):
//
//   /opt/node22/bin/node bre/instrument/tests/drive_battery.mjs
//       [--forms full,short] [--consent alternate|yes|no|random] [--phone short|full|both|none]
//       [--phone-width 375] [--fast] [--seed N] [--battery-seed <32 hex>] [--out DIR] [--shots DIR]
//       [--url http://host:port]
//
//   --consent       training-consent checkbox per session: alternate (default: checked on the 1st
//                   session, unchecked on the 2nd, ...), yes, no, or random.
//   --phone         which form(s) to drive again in the phone viewport (default short; none skips it).
//   --shots DIR     save one full-page screenshot per page kind and session into DIR (for eyeballing
//                   the phone layout).
//   --fast          route config.js with DELAY_SECONDS_OVERRIDE = 0.3 so the timed delay page does not
//                   wait 10 s per news item (the served files on disk are never modified).
//   --seed N        seed of the driver's own PRNG (which buttons it clicks); printed at start so a run
//                   can be replayed. Independent of the app's assignment seed.
//   --battery-seed  32-hex seed passed to the app as ?seed= (replays an assignment).
//   --url           use an already running server instead of spawning http.server.
//
// Prints PASS/FAIL per check and a summary. Exit status 0 when every check passes, 1 otherwise, 2 on
// a driver error. Nothing under instrument/ is modified; outputs go to --out or a temp directory.
import fs from 'node:fs';
import os from 'node:os';
import net from 'node:net';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const INSTRUMENT = path.resolve(__dirname, '..');
const STATIC = path.join(INSTRUMENT, 'static');
const BATTERY_PATH = path.join(INSTRUMENT, 'battery.json');
const PYTHON = process.env.BRE_PYTHON || '/home/user/bre-venv/bin/python';
const CHROMIUM_FALLBACK = process.env.BRE_CHROMIUM || '/opt/pw-browsers/chromium';
const FAST_DELAY_SECONDS = 0.3;
const DESKTOP = { width: 1000, height: 900, mobile: false };

// The 21 DecisionEvent columns, in order (PLAN.md section 2 / db/models.py SCHEMA_COLUMNS).
const SCHEMA = [
  'subject_id', 'dataset', 'session_id', 'timestamp', 'position_in_session', 'scenario_id', 'loss_pct',
  'horizon_days', 'context_tags', 'question_order_id', 'prior_question_ids', 'elicitation_type', 'response',
  'response_time_ms', 'covariates', 'outcome_behavior', 'incentivized', 'consent_training', 'battery_version',
  'is_synthetic', 'source_row_ref',
];
const COVARIATE_KEYS = ['age_band', 'wealth_band', 'invest_experience_yrs', 'self_reported_risk_tolerance', 'financial_literacy_score', 'education'];
// Schema enum incl. the PLAN.md section 2 amendment of 2026-09-17 (binary_yes_no, choice_rate).
const ELICITATION_TYPES = ['binary_sell', 'allocation_pct', 'likert', 'lottery_choice', 'binary_yes_no', 'choice_rate'];
const BINARY_TYPES = ['binary_sell', 'binary_yes_no', 'lottery_choice'];
const UNIT_INTERVAL_TYPES = ['allocation_pct', 'choice_rate'];
// The contract this battery must honour: question id -> elicitation type, and label -> response code.
const QUESTION_TYPES = { tolerance: 'binary_yes_no', sell_hold: 'binary_sell', allocation_share: 'allocation_pct' };
const LABEL_CODES = { tolerance: { Yes: 1, No: 0 }, sell_hold: { Sell: 1, Hold: 0 } };
const ORDER_IDS = ['tolerance-first', 'scenario-first'];
const CONTEXT_TAG_RE = /^[a-z_]+:[a-z0-9_]+$/;
const ISO_UTC_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?Z$/;
const UUID4_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const SEED_RE = /^[0-9a-f]{32}$/;
const SRC_REF_RE = /^(full|short):i(\d{2})(?::rep_of_i(\d{2}))?:([a-z_]+)$/;

// ------------------------------------------------------------------ args ---------------------
function parseArgs(argv) {
  const a = { forms: ['full', 'short'], consent: 'alternate', phone: 'short', phoneWidth: 375, fast: false, seed: null, batterySeed: null, out: null, shots: null, url: null, headed: false };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i], v = argv[i + 1];
    if (k === '--forms') { a.forms = v.split(',').map(s => s.trim()).filter(Boolean); i++; }
    else if (k === '--consent') { a.consent = v; i++; }
    else if (k === '--phone') { a.phone = v; i++; }
    else if (k === '--phone-width') { a.phoneWidth = parseInt(v, 10); i++; }
    else if (k === '--fast') { a.fast = true; }
    else if (k === '--seed') { a.seed = v; i++; }
    else if (k === '--battery-seed') { a.batterySeed = v; i++; }
    else if (k === '--out') { a.out = v; i++; }
    else if (k === '--shots') { a.shots = v; i++; }
    else if (k === '--url') { a.url = v.replace(/\/$/, ''); i++; }
    else if (k === '--headed') { a.headed = true; }
    else if (k === '--help' || k === '-h') { console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(1, 32).join('\n')); process.exit(0); }
    else { throw new Error('unknown argument: ' + k); }
  }
  for (const f of a.forms) { if (!['full', 'short'].includes(f)) { throw new Error('unknown form: ' + f); } }
  if (!['alternate', 'yes', 'no', 'random'].includes(a.consent)) { throw new Error('--consent must be alternate|yes|no|random'); }
  if (!['none', 'short', 'full', 'both'].includes(a.phone)) { throw new Error('--phone must be none|short|full|both'); }
  if (!(Number.isInteger(a.phoneWidth) && a.phoneWidth >= 240)) { throw new Error('--phone-width must be an integer >= 240'); }
  if (a.batterySeed !== null && !SEED_RE.test(a.batterySeed)) { throw new Error('--battery-seed must be 32 lowercase hex characters'); }
  return a;
}

// ------------------------------------------------------------------ driver PRNG --------------
function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6D2B79F5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function makeDriverRng(seedArg) {
  const seed = seedArg === null ? crypto.randomInt(0, 2 ** 31) : (parseInt(seedArg, 10) >>> 0);
  // Mix the seed (splitmix-style) and discard warm-up draws: mulberry32's first output is biased
  // high for small seeds (seeds 1..6 all start above 0.5), which showed up as --consent random
  // never drawing "no".
  let z = (seed + 0x9E3779B9) >>> 0;
  z = Math.imul(z ^ (z >>> 16), 0x85EBCA6B) >>> 0;
  z = Math.imul(z ^ (z >>> 13), 0xC2B2AE35) >>> 0;
  const f = mulberry32((z ^ (z >>> 16)) >>> 0);
  for (let k = 0; k < 8; k++) { f(); }
  return { seed, float: f, int: n => Math.floor(f() * n), pick: arr => arr[Math.floor(f() * arr.length)] };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ------------------------------------------------------------------ playwright ---------------
async function loadPlaywright() {
  const require = createRequire(import.meta.url);
  const globalLib = path.resolve(path.dirname(process.execPath), '..', 'lib', 'node_modules');
  const candidates = [process.env.BRE_PW_DIR, process.cwd(), __dirname, globalLib, '/tmp/claude-0/pw-scratch'].filter(Boolean);
  let resolved = null;
  for (const dir of candidates) {
    try { resolved = require.resolve('playwright', { paths: [dir] }); break; } catch (e) { /* try next */ }
  }
  if (!resolved) { throw new Error('playwright not found in ' + candidates.join(', ') + '. `npm install playwright` in a scratch directory and set BRE_PW_DIR to it.'); }
  const mod = await import(pathToFileURL(resolved).href);
  return { pw: mod.chromium ? mod : mod.default, from: resolved };
}

async function launchChromium(pw, headed) {
  const opts = { headless: !headed };
  try {
    return { browser: await pw.chromium.launch(opts), how: 'default browser lookup (PLAYWRIGHT_BROWSERS_PATH=' + (process.env.PLAYWRIGHT_BROWSERS_PATH || 'unset') + ')' };
  } catch (e) {
    if (!fs.existsSync(CHROMIUM_FALLBACK)) { throw e; }
    return { browser: await pw.chromium.launch({ ...opts, executablePath: CHROMIUM_FALLBACK }), how: 'executablePath ' + CHROMIUM_FALLBACK };
  }
}

// ------------------------------------------------------------------ static server ------------
function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.on('error', reject);
    s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); });
  });
}
async function startServer(port) {
  const proc = spawn(PYTHON, ['-m', 'http.server', String(port), '--bind', '127.0.0.1', '--directory', STATIC], { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  proc.stderr.on('data', d => { stderr += d; });
  for (let i = 0; i < 100; i++) {
    if (proc.exitCode !== null) { throw new Error('http.server exited: ' + stderr); }
    try { const r = await fetch(`http://127.0.0.1:${port}/index.html`); if (r.ok) { return proc; } } catch (e) { /* not up yet */ }
    await sleep(100);
  }
  proc.kill();
  throw new Error('http.server did not come up on port ' + port + ': ' + stderr);
}

// ------------------------------------------------------------------ checks -------------------
class Checks {
  constructor(name) { this.name = name; this.results = []; }
  check(cond, label, detail) {
    const ok = !!cond;
    this.results.push({ ok, label, detail: ok ? '' : String(detail || '') });
    console.log((ok ? '  PASS  ' : '  FAIL  ') + label + (ok || !detail ? '' : ' -- ' + String(detail).slice(0, 400)));
    return ok;
  }
  get failed() { return this.results.filter(r => !r.ok); }
  get passed() { return this.results.filter(r => r.ok); }
}

// Runs in the page: describe what is on screen right now.
function snapshotFn() {
  const S = window.BRE_STATE;
  const q = s => document.querySelector(s);
  const form = q('#jspsych-survey-html-form');
  const slider = q('#jspsych-html-slider-response-response');
  const btns = Array.from(document.querySelectorAll('#jspsych-html-button-response-btngroup .jspsych-btn'));
  const txt = s => (q(s) ? q(s).textContent.trim() : '');
  return {
    finished: !!S.finished,
    pagesDone: S.pagesDone,
    hasForm: !!form,
    hasSlider: !!slider,
    sliderValue: slider ? Number(slider.value) : null,
    sliderNextDisabled: q('#jspsych-html-slider-response-next') ? q('#jspsych-html-slider-response-next').disabled : null,
    buttons: btns.map(b => ({ label: b.textContent.trim(), disabled: b.disabled })),
    kicker: txt('.bre-kicker'),
    question: txt('.bre-question'),
    title: txt('.bre-title'),
    fields: form ? Array.from(form.querySelectorAll('input')).map(i => ({ name: i.name, type: i.type, value: i.value })) : [],
  };
}

// Runs in the page: horizontal overflow of the document and of #jspsych-target (which is the
// scroll container jsPsych sets overflow on), plus every visible element / control that sticks
// out of the viewport horizontally.
function layoutProbeFn() {
  const de = document.documentElement;
  const vw = de.clientWidth;
  const target = document.getElementById('jspsych-target');
  const docOverflow = Math.max(0, Math.max(de.scrollWidth, document.body ? document.body.scrollWidth : 0) - vw);
  const targetOverflow = target ? Math.max(0, target.scrollWidth - target.clientWidth) : 0;
  const offenders = [], controlsOutside = [];
  const name = el => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (typeof el.className === 'string' && el.className.trim() ? '.' + el.className.trim().split(/\s+/).join('.') : '');
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') { continue; }
    const b = el.getBoundingClientRect();
    if (b.width === 0 && b.height === 0) { continue; }
    const outside = b.right > vw + 1 || b.left < -1;
    if (outside) { offenders.push(name(el) + ' [' + Math.round(b.left) + ',' + Math.round(b.right) + ']'); }
    if (outside && /^(button|input|textarea|select)$/i.test(el.tagName)) { controlsOutside.push(name(el)); }
  }
  const txt = s => (document.querySelector(s) ? document.querySelector(s).textContent.trim() : '');
  const label = (txt('.bre-kicker') + ' ' + (txt('.bre-title') || txt('.bre-question').slice(0, 40))).trim() || document.title;
  return { vw, overflow: Math.max(docOverflow, targetOverflow), offenders: offenders.slice(0, 6), controlsOutside: controlsOutside.slice(0, 6), label };
}

function itemIndexFromKicker(kicker) {
  const m = /Scenario (\d+) of (\d+)/.exec(kicker);
  return m ? parseInt(m[1], 10) : null;
}

// Which kind of page a snapshot shows (for --shots: one screenshot per kind and session).
function pageKind(st, Q, delayTitle) {
  if (st.hasForm) { return st.fields.some(f => f.name === 'consent_participate') ? 'consent' : (st.title === 'About you' ? 'about_you' : 'investing'); }
  if (st.hasSlider) { return 'allocation'; }
  if (st.question === Q.tolerance.text) { return 'tolerance'; }
  if (st.question === Q.sell_hold.text) { return 'sell_hold'; }
  if (st.kicker.startsWith('Knowledge question')) { return 'quiz'; }
  if (st.title === delayTitle) { return 'delay'; }
  if (st.kicker.startsWith('Scenario')) { return 'scenario'; }
  return st.title === 'The scenarios' ? 'transition' : 'instructions';
}
async function screenshotOnce(page, dir, prefix, kind, taken) {
  if (taken.has(kind)) { return; }
  taken.add(kind);
  fs.mkdirSync(dir, { recursive: true });
  await page.screenshot({ path: path.join(dir, `${prefix}-${kind}.png`), fullPage: true });
}

// ------------------------------------------------------------------ one session --------------
async function driveSession(browser, battery, opts) {
  const { form, consent, fast, url, batterySeed, out, rng, C, viewport, shots } = opts;
  const shotsTaken = new Set();
  const Q = battery.design.questions;
  const quiz = battery.intake.financial_literacy_quiz;
  const covDefs = Object.fromEntries(battery.intake.covariates.map(cv => [cv.key, cv]));
  const delayTitle = battery.design.delay_page.title;

  const ctxOpts = { acceptDownloads: true, viewport: { width: viewport.width, height: viewport.height } };
  if (viewport.mobile) { Object.assign(ctxOpts, { isMobile: true, hasTouch: true, deviceScaleFactor: 2 }); }
  const context = await browser.newContext(ctxOpts);
  const page = await context.newPage();
  const pageErrors = [], consoleErrors = [], badResponses = [];
  page.on('pageerror', e => pageErrors.push(String(e && e.message ? e.message : e)));
  // Chromium's console text for a failed resource does not name the URL; m.location().url does.
  page.on('console', m => { if (m.type() === 'error') { consoleErrors.push(((m.location() || {}).url || '') + ' :: ' + m.text()); } });
  page.on('response', r => { if (r.status() >= 400) { badResponses.push(r.status() + ' ' + r.url()); } });
  page.on('requestfailed', r => { badResponses.push('FAILED ' + r.url() + ' ' + ((r.failure() || {}).errorText || '')); });
  if (fast) {
    // Serve a modified config.js from memory; the file on disk is untouched.
    const cfg = fs.readFileSync(path.join(STATIC, 'config.js'), 'utf8');
    await context.route('**/config.js', route => route.fulfill({
      status: 200, contentType: 'application/javascript',
      body: cfg + `\nwindow.BRE_CONFIG.DELAY_SECONDS_OVERRIDE = ${FAST_DELAY_SECONDS}; // injected by drive_battery.mjs --fast\n`,
    }));
  }

  const target = `${url}/index.html?form=${form}` + (batterySeed ? `&seed=${batterySeed}` : '');
  console.log(`\n=== ${form} form, consent_training=${consent}, fast=${fast}, viewport=${viewport.width}x${viewport.height}${viewport.mobile ? ' (mobile emulation)' : ''}: ${target}`);
  const t0 = Date.now();
  await page.goto(target, { waitUntil: 'load' });
  await page.waitForFunction(() => window.BRE_STATE && window.BRE_STATE.assignment && window.BRE_STATE.totalPages > 0, null, { timeout: 20000 });
  const info = await page.evaluate(() => ({
    form: BRE_STATE.form, seed: BRE_STATE.seed, subject_id: BRE_STATE.subject_id,
    n_presentations: BRE_STATE.assignment.n_presentations, totalPages: BRE_STATE.totalPages, delaySeconds: BRE_STATE.delaySecondsUsed,
    innerWidth: window.innerWidth,
  }));
  console.log(`  session_id=${info.seed} subject_id=${info.subject_id} presentations=${info.n_presentations} pages=${info.totalPages} delay=${info.delaySeconds}s innerWidth=${info.innerWidth}px`);
  const clickTimeout = Math.round(info.delaySeconds * 1000) + 20000;

  // What the driver chose, keyed the way the rows will report it.
  const expected = { covariates: {}, quizCorrect: 0, responses: {}, labels: {}, delayPagesSeen: 0, delayStartedDisabled: 0, sliderStarts: [], sliderNextStartedDisabled: 0 };
  const layout = { pagesProbed: 0, overflowPages: [], controlsOutsidePages: [], maxOverflow: 0 };
  const probeLayout = async (labelOverride) => {
    const L = await page.evaluate(layoutProbeFn);
    layout.pagesProbed++;
    const label = labelOverride || L.label;
    if (L.overflow > 0 || L.offenders.length) { layout.overflowPages.push(`${label}: overflow ${L.overflow}px, ${L.offenders.join(' ; ')}`); }
    if (L.controlsOutside.length) { layout.controlsOutsidePages.push(`${label}: ${L.controlsOutside.join(' ; ')}`); }
    layout.maxOverflow = Math.max(layout.maxOverflow, L.overflow);
    return L;
  };
  let steps = 0, idle = 0, probedPage = -1;
  while (steps < 3000) {
    steps++;
    const st = await page.evaluate(snapshotFn);
    if (st.finished) { break; }
    if (!st.hasForm && !st.hasSlider && !st.buttons.length) {
      if (++idle > 400) { throw new Error('nothing to act on for too long; kicker=' + st.kicker + ' title=' + st.title); }
      await sleep(25); continue;
    }
    idle = 0;
    if (probedPage !== st.pagesDone) {
      await probeLayout();
      probedPage = st.pagesDone;
      if (shots) { await screenshotOnce(page, shots, `${form}-${viewport.width}px`, pageKind(st, Q, delayTitle), shotsTaken); }
    }
    await sleep(10 + rng.int(60)); // a little "thinking" time so clicks are not back to back
    const itemIdx = itemIndexFromKicker(st.kicker);
    const iKey = itemIdx === null ? null : 'i' + String(itemIdx).padStart(2, '0');

    if (st.hasForm) {
      const names = [...new Set(st.fields.map(f => f.name).filter(Boolean))];
      if (names.includes('consent_participate')) {
        await page.check('#jspsych-survey-html-form input[name="consent_participate"]');
        const ct = page.locator('#jspsych-survey-html-form input[name="consent_training"]');
        if (consent) { await ct.check(); } else { await ct.uncheck(); }
      } else {
        for (const name of names) {
          const cv = covDefs[name];
          if (!cv) { throw new Error('form field without a covariate definition: ' + name); }
          const inputs = st.fields.filter(f => f.name === name);
          if (cv.input === 'number') {
            const n = cv.min + rng.int(cv.max - cv.min + 1);
            await page.fill(`#jspsych-survey-html-form input[name="${name}"]`, String(n));
            expected.covariates[name] = n;
          } else {
            const choice = rng.pick(inputs); // radios: any option, including "prefer not to say" (__null__)
            await page.check(`#jspsych-survey-html-form input[name="${name}"][value="${choice.value}"]`);
            expected.covariates[name] = choice.value === '__null__' ? null : (cv.input === 'likert' ? parseInt(choice.value, 10) : choice.value);
          }
        }
      }
      await page.click('#jspsych-survey-html-form-next');
    } else if (st.hasSlider) {
      expected.sliderStarts.push(st.sliderValue);
      if (st.sliderNextDisabled === true) { expected.sliderNextStartedDisabled++; }
      const track = page.locator('#jspsych-html-slider-response-response');
      const box = await track.boundingBox();
      const pad = 8; // half the thumb width, roughly
      await page.mouse.click(box.x + pad + rng.float() * (box.width - 2 * pad), box.y + box.height / 2);
      const value = await track.evaluate(el => Number(el.value));
      expected.responses[`${iKey}:${Q.allocation_share.question_id}`] = value / Q.allocation_share.slider.max;
      expected.labels[`${iKey}:${Q.allocation_share.question_id}`] = String(value);
      await page.click('#jspsych-html-slider-response-next');
    } else {
      const buttons = page.locator('#jspsych-html-button-response-btngroup .jspsych-btn');
      let idx = 0;
      if (st.question === Q.tolerance.text) {
        idx = rng.int(Q.tolerance.choices.length);
        expected.responses[`${iKey}:${Q.tolerance.question_id}`] = Q.tolerance.choices[idx].response;
        expected.labels[`${iKey}:${Q.tolerance.question_id}`] = st.buttons[idx].label;
      } else if (st.question === Q.sell_hold.text) {
        idx = rng.int(Q.sell_hold.choices.length);
        expected.responses[`${iKey}:${Q.sell_hold.question_id}`] = Q.sell_hold.choices[idx].response;
        expected.labels[`${iKey}:${Q.sell_hold.question_id}`] = st.buttons[idx].label;
      } else if (st.kicker.startsWith('Knowledge question')) {
        const n = parseInt(/Knowledge question (\d+)/.exec(st.kicker)[1], 10);
        const item = quiz.items[n - 1];
        idx = rng.int(item.options.length);
        if (idx === item.answer_index) { expected.quizCorrect++; }
      } else if (st.title === delayTitle) {
        expected.delayPagesSeen++;
        if (st.buttons.every(b => b.disabled)) { expected.delayStartedDisabled++; }
        idx = 0; // click() below waits until enable_button_after re-enables the button
      } else {
        idx = 0; // instructions / scenario / transition: a single Continue-type button
      }
      await buttons.nth(idx).click({ timeout: clickTimeout });
    }
    await page.waitForFunction(n => window.BRE_STATE.finished || window.BRE_STATE.pagesDone > n, st.pagesDone, { timeout: clickTimeout });
  }
  const elapsed = (Date.now() - t0) / 1000;
  console.log(`  finished in ${steps} steps, ${elapsed.toFixed(1)} s`);

  // ---- completion page: capture both downloads, the textarea and the in-page objects ----
  await page.waitForSelector('#bre-dl-responses', { timeout: 10000 });
  await probeLayout('completion page');
  if (shots) { await screenshotOnce(page, shots, `${form}-${viewport.width}px`, 'done', shotsTaken); }
  const [dl1] = await Promise.all([page.waitForEvent('download', { timeout: 15000 }), page.click('#bre-dl-responses')]);
  const respName = dl1.suggestedFilename();
  const respPath = path.join(out, respName);
  await dl1.saveAs(respPath);
  const [dl2] = await Promise.all([page.waitForEvent('download', { timeout: 15000 }), page.click('#bre-dl-log')]);
  const logName = dl2.suggestedFilename();
  const logPath = path.join(out, logName);
  await dl2.saveAs(logPath);
  await page.click('#bre-show-json');
  const shownText = await page.inputValue('#bre-json');
  await probeLayout('completion page with JSON shown');
  const inPage = await page.evaluate(() => JSON.parse(JSON.stringify({ records: BRE_STATE.records, log: BRE_STATE.log })));
  const ls = await page.evaluate(seed => { try { return JSON.parse(localStorage.getItem('bre_intake_' + seed)); } catch (e) { return { error: String(e) }; } }, info.seed);
  const bodyText = await page.textContent('body');
  await context.close();

  const rows = JSON.parse(fs.readFileSync(respPath, 'utf8'));
  const log = JSON.parse(fs.readFileSync(logPath, 'utf8'));
  console.log(`  saved ${respPath} (${rows.length} rows) and ${logPath}`);
  verify(C, { battery, form, consent, fast, batterySeed, viewport, info, expected, layout, rows, log, respName, logName, shownText, inPage, ls, pageErrors, consoleErrors, badResponses, bodyText, elapsed });
  return { info, rows, log, respPath, logPath, elapsed };
}

// ------------------------------------------------------------------ verification -------------
function verify(C, x) {
  const { battery, form, consent, fast, batterySeed, viewport, info, expected, layout, rows, log, respName, logName, shownText, inPage, ls, pageErrors, consoleErrors, badResponses, bodyText } = x;
  const F = battery.forms[form];
  const D = battery.design;
  const Q = D.questions;
  const universe = new Map(D.scenario_universe.map(u => [u.scenario_id, u]));
  const orderIds = D.question_orders.map(o => o.question_order_id);
  const qSeq = Object.fromEntries(D.question_orders.map(o => [o.question_order_id, o.page_sequence.filter(s => Q[s]).map(s => Q[s].question_id)]));
  const nQ = qSeq[orderIds[0]].length;
  const covOpts = Object.fromEntries(battery.intake.covariates.map(cv => [cv.key, cv]));
  const lossSet = new Set(D.loss_pcts);
  const bandSet = key => new Set(covOpts[key].options.filter(o => o.value !== null).map(o => o.value));
  const j = v => JSON.stringify(v);
  const qidOf = r => { const m = SRC_REF_RE.exec(r.source_row_ref); return m ? m[4] : null; };

  C.check(pageErrors.length === 0, 'no uncaught page errors', pageErrors.join(' | '));
  const isFavicon = m => /\/favicon\.ico( |$|\s|::)/.test(m);
  const realConsoleErrors = consoleErrors.filter(m => !isFavicon(m));
  C.check(realConsoleErrors.length === 0, 'no console errors (the browser\'s own /favicon.ico probe is ignored)', realConsoleErrors.join(' | '));
  const realBad = badResponses.filter(m => !isFavicon(m));
  C.check(realBad.length === 0, 'every app resource loaded (no HTTP >= 400 or failed request)', realBad.join(' | '));
  C.check(j(SCHEMA) === j(battery.output.schema_fields), 'battery.output.schema_fields == the 21-column schema list');
  C.check(Array.isArray(rows), 'downloaded responses file is a JSON array');
  C.check(j(rows) === j(inPage.records), 'downloaded responses identical to in-page BRE_STATE.records');
  C.check(j(JSON.parse(shownText)) === j(rows), '"Show JSON" textarea identical to the download');
  C.check(j(log) === j(inPage.log), 'downloaded session log identical to in-page BRE_STATE.log');
  C.check(respName === `intake_${info.seed}.json`, 'responses file name follows output.file_names', respName);
  C.check(logName === `intake_${info.seed}.session.json`, 'session log file name follows output.file_names', logName);
  C.check(SEED_RE.test(info.seed), 'session_id is a 32-hex seed', info.seed);
  if (batterySeed) { C.check(info.seed === batterySeed, 'session_id equals the ?seed= replay parameter'); }
  C.check(rows.length === F.n_presentations * nQ, `row count == forms.${form}.n_presentations (${F.n_presentations}) x ${nQ} questions per presentation`, 'got ' + rows.length);

  // ---- per-row schema and type checks ----
  const problems = [];
  const P = (i, msg) => problems.push(`row ${i}: ${msg}`);
  const consentValues = new Set(), datasets = new Set(), synth = new Set(), rtBad = [], badScenario = [], badOrder = [], badTags = [], badTs = [];
  let lastTs = null;
  rows.forEach((r, i) => {
    const keys = Object.keys(r);
    if (j(keys) !== j(SCHEMA)) { P(i, 'keys differ from schema: ' + keys.join(',')); }
    if (typeof r.subject_id !== 'string' || !UUID4_RE.test(r.subject_id)) { P(i, 'subject_id not a uuid4: ' + r.subject_id); }
    if (r.subject_id !== info.subject_id) { P(i, 'subject_id differs from the session subject_id'); }
    if (r.dataset !== 'intake_battery') { P(i, 'dataset != intake_battery: ' + r.dataset); }
    datasets.add(r.dataset);
    if (r.session_id !== info.seed) { P(i, 'session_id != seed: ' + r.session_id); }
    if (typeof r.timestamp !== 'string' || !ISO_UTC_RE.test(r.timestamp) || isNaN(Date.parse(r.timestamp))) { P(i, 'timestamp not ISO 8601 UTC: ' + r.timestamp); badTs.push(r.timestamp); }
    if (lastTs !== null && Date.parse(r.timestamp) < Date.parse(lastTs)) { P(i, 'timestamp goes backwards'); }
    lastTs = r.timestamp;
    if (!Number.isInteger(r.position_in_session) || r.position_in_session !== i) { P(i, 'position_in_session not sequential 0-based: ' + r.position_in_session); }
    const u = universe.get(r.scenario_id);
    if (typeof r.scenario_id !== 'string' || !u) { P(i, 'scenario_id not in battery.json scenario_universe: ' + r.scenario_id); badScenario.push(r.scenario_id); }
    if (typeof r.loss_pct !== 'number' || !lossSet.has(r.loss_pct)) { P(i, 'loss_pct not a design loss level: ' + r.loss_pct); }
    if (!Number.isInteger(r.horizon_days) || r.horizon_days !== D.horizon_days) { P(i, 'horizon_days != ' + D.horizon_days + ': ' + r.horizon_days); }
    if (!Array.isArray(r.context_tags) || !r.context_tags.every(t => typeof t === 'string' && D.contexts[t] && t !== 'none')) { P(i, 'context_tags not a list of known non-null tags: ' + j(r.context_tags)); }
    if (!Array.isArray(r.context_tags) || !r.context_tags.every(t => CONTEXT_TAG_RE.test(t))) { badTags.push(j(r.context_tags)); }
    if (typeof r.question_order_id !== 'string' || !orderIds.includes(r.question_order_id)) { P(i, 'question_order_id unknown: ' + r.question_order_id); }
    if (!ORDER_IDS.includes(r.question_order_id)) { badOrder.push(r.question_order_id); }
    if (!Array.isArray(r.prior_question_ids) || !r.prior_question_ids.every(t => typeof t === 'string')) { P(i, 'prior_question_ids not a list of strings'); }
    if (!ELICITATION_TYPES.includes(r.elicitation_type)) { P(i, 'elicitation_type not in enum: ' + r.elicitation_type); }
    if (typeof r.response !== 'number' || !isFinite(r.response)) { P(i, 'response not a finite number: ' + r.response); }
    else if (BINARY_TYPES.includes(r.elicitation_type) && !(r.response === 0 || r.response === 1)) { P(i, r.elicitation_type + ' response not 0/1: ' + r.response); }
    else if (UNIT_INTERVAL_TYPES.includes(r.elicitation_type) && !(r.response >= 0 && r.response <= 1)) { P(i, r.elicitation_type + ' response outside [0,1]: ' + r.response); }
    else if (r.elicitation_type === 'likert' && !(Number.isInteger(r.response) && r.response >= 1 && r.response <= 7)) { P(i, 'likert response not an integer 1-7: ' + r.response); }
    if (!Number.isInteger(r.response_time_ms) || r.response_time_ms <= 0) { P(i, 'response_time_ms not a positive integer: ' + r.response_time_ms); rtBad.push(r.response_time_ms); }
    if (r.covariates === null || typeof r.covariates !== 'object' || Array.isArray(r.covariates)) { P(i, 'covariates not an object'); }
    else {
      const c = r.covariates;
      if (j(Object.keys(c)) !== j(COVARIATE_KEYS)) { P(i, 'covariate keys differ: ' + Object.keys(c).join(',')); }
      if (!(c.age_band === null || bandSet('age_band').has(c.age_band))) { P(i, 'age_band invalid: ' + c.age_band); }
      if (!(c.wealth_band === null || bandSet('wealth_band').has(c.wealth_band))) { P(i, 'wealth_band invalid: ' + c.wealth_band); }
      if (!(c.education === null || bandSet('education').has(c.education))) { P(i, 'education invalid: ' + c.education); }
      if (!(Number.isInteger(c.invest_experience_yrs) && c.invest_experience_yrs >= covOpts.invest_experience_yrs.min && c.invest_experience_yrs <= covOpts.invest_experience_yrs.max)) { P(i, 'invest_experience_yrs invalid: ' + c.invest_experience_yrs); }
      if (!(Number.isInteger(c.self_reported_risk_tolerance) && c.self_reported_risk_tolerance >= 1 && c.self_reported_risk_tolerance <= 7)) { P(i, 'self_reported_risk_tolerance invalid: ' + c.self_reported_risk_tolerance); }
      if (!(Number.isInteger(c.financial_literacy_score) && c.financial_literacy_score >= 0 && c.financial_literacy_score <= 5)) { P(i, 'financial_literacy_score invalid: ' + c.financial_literacy_score); }
    }
    if (r.outcome_behavior !== null) { P(i, 'outcome_behavior not null'); }
    if (r.incentivized !== false) { P(i, 'incentivized not false'); }
    if (typeof r.consent_training !== 'boolean') { P(i, 'consent_training not boolean'); }
    consentValues.add(r.consent_training);
    if (r.battery_version !== battery.battery_version) { P(i, 'battery_version != ' + battery.battery_version + ': ' + r.battery_version); }
    if (r.is_synthetic !== false) { P(i, 'is_synthetic not false'); }
    synth.add(r.is_synthetic);
    if (typeof r.source_row_ref !== 'string' || !SRC_REF_RE.test(r.source_row_ref) || !r.source_row_ref.startsWith(form + ':')) { P(i, 'source_row_ref format: ' + r.source_row_ref); }
    if (u) {
      if (r.loss_pct !== u.loss_pct || r.horizon_days !== u.horizon_days) { P(i, 'loss/horizon disagree with the universe entry'); }
      if (j(r.context_tags) !== j(u.context_tags)) { P(i, 'context_tags disagree with the universe entry'); }
      if (r.question_order_id !== u.question_order_id) { P(i, 'question_order_id disagrees with the universe entry'); }
    }
  });
  C.check(rows.every(r => j(Object.keys(r)) === j(SCHEMA)), 'every row has exactly the 21 schema fields, in order');
  C.check(problems.length === 0, 'every row passes the per-field type/range checks', problems.slice(0, 5).join(' ; ') + (problems.length > 5 ? ` ... (${problems.length} total)` : ''));
  C.check(synth.size === 1 && synth.has(false), 'is_synthetic is false on every row');
  C.check(datasets.size === 1 && datasets.has('intake_battery'), 'dataset is "intake_battery" on every row');
  C.check(rows.every(r => r.incentivized === false), 'incentivized is false on every row');
  C.check(consentValues.size === 1 && consentValues.has(consent), `consent_training == ${consent} (the checkbox state) on every row`, [...consentValues].join(','));
  C.check(log.consent_training === consent, 'session log consent_training matches the checkbox');
  C.check(badScenario.length === 0, 'every scenario_id exists in battery.json', badScenario.slice(0, 3).join(','));
  C.check(badOrder.length === 0, 'question_order_id is exactly "tolerance-first" or "scenario-first" on every row', badOrder.slice(0, 3).join(','));
  C.check(badTags.length === 0, 'every context tag is namespace:value', badTags.slice(0, 3).join(','));
  C.check(badTs.length === 0, 'timestamps are ISO 8601 UTC (YYYY-MM-DDTHH:MM:SS[.mmm]Z) on every row', badTs.slice(0, 3).join(','));
  C.check(rtBad.length === 0, 'response_time_ms is a positive integer on every row', rtBad.slice(0, 5).join(','));
  C.check(rows.every(r => j(r.covariates) === j(rows[0].covariates)), 'covariates identical on every row');

  // ---- elicitation contract per question (PLAN.md section 2 amendment: tolerance -> binary_yes_no) ----
  C.check(j(Object.fromEntries(Object.keys(QUESTION_TYPES).map(k => [k, Q[k].elicitation_type]))) === j(QUESTION_TYPES), 'battery.json question -> elicitation_type map is tolerance:binary_yes_no, sell_hold:binary_sell, allocation_share:allocation_pct', j(Object.fromEntries(Object.keys(Q).map(k => [k, Q[k].elicitation_type]))));
  const typeProblems = [];
  rows.forEach((r, i) => {
    const qid = qidOf(r);
    if (!qid || !QUESTION_TYPES[qid]) { typeProblems.push(`row ${i}: unknown question ${qid}`); return; }
    if (r.elicitation_type !== QUESTION_TYPES[qid]) { typeProblems.push(`row ${i}: ${qid} is ${r.elicitation_type}, expected ${QUESTION_TYPES[qid]}`); }
    if (BINARY_TYPES.includes(QUESTION_TYPES[qid]) && !(r.response === 0 || r.response === 1)) { typeProblems.push(`row ${i}: ${qid} response ${r.response} not in {0,1}`); }
  });
  C.check(typeProblems.length === 0, 'every row carries the contracted elicitation_type for its question (tolerance rows are binary_yes_no with response in {0,1})', typeProblems.slice(0, 4).join(' ; '));
  C.check(rows.filter(r => qidOf(r) === 'tolerance').length === F.n_presentations && rows.filter(r => r.elicitation_type === 'binary_yes_no').length === F.n_presentations, `exactly one binary_yes_no (tolerance) row per presentation (${F.n_presentations})`, `${rows.filter(r => r.elicitation_type === 'binary_yes_no').length} binary_yes_no rows`);
  C.check(!rows.some(r => r.elicitation_type === 'lottery_choice'), 'no row uses lottery_choice any more');
  const labelProblems = [];
  rows.forEach(r => {
    const m = SRC_REF_RE.exec(r.source_row_ref);
    if (!m || !LABEL_CODES[m[4]]) { return; }
    const label = expected.labels[`i${m[2]}:${m[4]}`];
    if (!(label in LABEL_CODES[m[4]])) { labelProblems.push(`${m[4]} clicked "${label}" (not a contracted label)`); }
    else if (LABEL_CODES[m[4]][label] !== r.response) { labelProblems.push(`${m[4]} clicked "${label}" -> recorded ${r.response}, contract says ${LABEL_CODES[m[4]][label]}`); }
  });
  C.check(labelProblems.length === 0, 'clicked "Yes" -> 1 / "No" -> 0 (tolerance) and "Sell" -> 1 / "Hold" -> 0 (sell_hold) on every row', labelProblems.slice(0, 4).join(' ; '));

  // ---- covariates and responses equal what was clicked ----
  const expCov = { ...expected.covariates, financial_literacy_score: expected.quizCorrect };
  const covOrdered = Object.fromEntries(COVARIATE_KEYS.map(k => [k, expCov[k]]));
  C.check(j(rows[0].covariates) === j(covOrdered), 'covariates equal the options chosen in the form (incl. financial_literacy_score = correct quiz answers)', `got ${j(rows[0].covariates)} expected ${j(covOrdered)}`);
  const respMismatch = [];
  rows.forEach(r => {
    const m = SRC_REF_RE.exec(r.source_row_ref);
    if (!m) { return; }
    const key = `i${m[2]}:${m[4]}`;
    if (!(key in expected.responses)) { respMismatch.push(key + ' never clicked'); }
    else if (expected.responses[key] !== r.response) { respMismatch.push(`${key}: clicked ${expected.responses[key]} recorded ${r.response}`); }
  });
  C.check(Object.keys(expected.responses).length === rows.length, 'one clicked answer per row', `${Object.keys(expected.responses).length} clicks vs ${rows.length} rows`);
  C.check(respMismatch.length === 0, 'every response equals the answer that was clicked / the slider position', respMismatch.slice(0, 5).join(' ; '));
  const allocRows = rows.filter(r => r.elicitation_type === 'allocation_pct');
  C.check(expected.sliderStarts.length === allocRows.length && expected.sliderStarts.every(v => v === Q.allocation_share.slider.start), `slider starts at ${Q.allocation_share.slider.start} on every allocation page`);
  C.check(expected.sliderNextStartedDisabled === allocRows.length, 'slider Continue button starts disabled (require_movement)');

  // ---- presentations, question sequence, prior_question_ids ----
  const groups = new Map();
  rows.forEach(r => { const m = SRC_REF_RE.exec(r.source_row_ref); if (m) { const k = parseInt(m[2], 10); if (!groups.has(k)) { groups.set(k, []); } groups.get(k).push({ r, rep: m[3] ? parseInt(m[3], 10) : null, qid: m[4] }); } });
  const idxs = [...groups.keys()];
  C.check(idxs.length === F.n_presentations && j(idxs) === j(idxs.map((_, k) => k + 1)), `rows cover presentations 1..${F.n_presentations} in order`, idxs.join(','));
  const seqProblems = [];
  for (const [k, g] of groups) {
    const order = g[0].r.question_order_id;
    const qids = g.map(e => e.qid);
    if (j(qids) !== j(qSeq[order])) { seqProblems.push(`i${k}: ${qids.join('>')} != ${qSeq[order].join('>')}`); }
    g.forEach((e, n) => { if (j(e.r.prior_question_ids) !== j(qids.slice(0, n))) { seqProblems.push(`i${k}:${e.qid} prior ${j(e.r.prior_question_ids)}`); } });
    if (new Set(g.map(e => e.r.scenario_id)).size !== 1 || new Set(g.map(e => e.r.question_order_id)).size !== 1) { seqProblems.push(`i${k}: mixed scenario/order`); }
    if (g.length !== nQ) { seqProblems.push(`i${k}: ${g.length} rows`); }
  }
  C.check(seqProblems.length === 0, 'within each presentation the question sequence follows its question order and prior_question_ids accumulate', seqProblems.slice(0, 4).join(' ; '));
  const elicSet = new Set(rows.map(r => r.elicitation_type));
  C.check(j([...elicSet].sort()) === j([Q.tolerance.elicitation_type, Q.sell_hold.elicitation_type, Q.allocation_share.elicitation_type].sort()), 'the three question types appear with their battery.json elicitation types', [...elicSet].join(','));

  // ---- items, repeats and balance ----
  const firstRows = idxs.map(k => groups.get(k)[0]);
  const uniqueScen = new Set(firstRows.map(e => e.r.scenario_id));
  C.check(uniqueScen.size === F.n_items, `distinct scenario_ids == forms.${form}.n_items (${F.n_items})`, 'got ' + uniqueScen.size);
  const reps = firstRows.filter(e => e.rep !== null);
  C.check(reps.length === F.n_repeats, `repeated presentations == forms.${form}.n_repeats (${F.n_repeats})`, 'got ' + reps.length);
  const repProblems = [];
  reps.forEach(e => {
    const k = parseInt(SRC_REF_RE.exec(e.r.source_row_ref)[2], 10);
    const orig = groups.get(e.rep);
    if (!orig) { repProblems.push(`i${k}: original i${e.rep} missing`); return; }
    if (orig[0].rep !== null) { repProblems.push(`i${k}: original is itself a repeat`); }
    if (orig[0].r.scenario_id !== e.r.scenario_id) { repProblems.push(`i${k}: scenario differs from original`); }
    if (k - e.rep < D.repeated_scenario_rule.min_gap_items) { repProblems.push(`i${k}: gap ${k - e.rep} < ${D.repeated_scenario_rule.min_gap_items}`); }
    if (e.rep > 6) { repProblems.push(`i${k}: original position ${e.rep} > 6`); }
  });
  C.check(repProblems.length === 0, 'repeats are verbatim, drawn from positions 1-6, gap >= min_gap_items', repProblems.join(' ; '));
  const items = firstRows.filter(e => e.rep === null).map(e => universe.get(e.r.scenario_id)).filter(Boolean);
  const count = (arr, f) => arr.reduce((m, v) => { const k = f(v); m[k] = (m[k] || 0) + 1; return m; }, {});
  const condCounts = count(items, u => u.condition_type);
  C.check(j(Object.fromEntries(Object.keys(F.balance.condition_type_counts).map(k => [k, condCounts[k] || 0]))) === j(F.balance.condition_type_counts), `condition-type counts match forms.${form}.balance`, j(condCounts));
  const lossCounts = count(items, u => u.loss_pct);
  if (form === 'full') {
    C.check(D.loss_pcts.every(l => (lossCounts[l] || 0) >= 2), 'full form: every loss level appears at least twice', j(lossCounts));
    const singles = items.filter(u => u.condition_type === 'single').map(u => u.context_tags[0]);
    C.check(j([...singles].sort()) === j([...D.context_tag_order].sort()), 'full form: each non-null context once as a single', singles.join(','));
    const pairs = items.filter(u => u.condition_type === 'pair').map(u => u.context_tags);
    const firsts = pairs.map(p => p[0]).sort(), seconds = pairs.map(p => p[1]).sort();
    C.check(j(firsts) === j([...D.context_tag_order].sort()) && j(seconds) === j([...D.context_tag_order].sort()), 'full form: every context once first and once second in the pairs', j(pairs));
    const ordersPerLoss = {};
    items.forEach(u => { (ordersPerLoss[u.loss_pct] = ordersPerLoss[u.loss_pct] || new Set()).add(u.question_order_id); });
    const both = D.loss_pcts.every(l => ordersPerLoss[l] && ordersPerLoss[l].size === 2);
    C.check(both === !!log.assignment.meta.order_balance_satisfied, 'full form: every loss level with both question orders iff the log says order_balance_satisfied', j(Object.fromEntries(Object.entries(ordersPerLoss).map(([k, v]) => [k, [...v]]))));
    const ordersPerBlock = count(items, u => u.condition_type + '/' + u.question_order_id);
    C.check(Object.values(ordersPerBlock).every(n => n === 2), 'full form: 2 tolerance-first + 2 scenario-first per condition-type block', j(ordersPerBlock));
  } else {
    C.check(D.loss_pcts.every(l => (lossCounts[l] || 0) === 1), 'short form: every loss level exactly once', j(lossCounts));
    const orderCounts = count(items, u => u.question_order_id);
    C.check(j(Object.values(orderCounts).sort()) === j([2, 3]), 'short form: 3/2 split of question orders', j(orderCounts));
    const pairs = items.filter(u => u.condition_type === 'pair').map(u => u.context_tags);
    C.check(pairs.length === 2 && new Set(pairs.flat()).size === 4, 'short form: the two pairs are disjoint and cover the 4 contexts', j(pairs));
  }

  // ---- assignment record in the session log ----
  const A = log.assignment;
  C.check(A && typeof A === 'object' && A.seed === info.seed && A.form === form && A.battery_version === battery.battery_version, 'session log carries the assignment record with seed, form and battery_version');
  C.check(A && Array.isArray(A.slots) && A.slots.length === F.n_presentations && A.n_presentations === F.n_presentations && A.n_items === F.n_items && A.n_repeats === F.n_repeats, 'assignment slots count == n_presentations');
  const slotProblems = [];
  (A && A.slots || []).forEach((s, k) => {
    if (s.item_index !== k + 1) { slotProblems.push(`slot ${k}: item_index ${s.item_index}`); }
    if (typeof s.is_repeat !== 'boolean' || (s.is_repeat ? !Number.isInteger(s.repeat_of) : s.repeat_of !== null)) { slotProblems.push(`slot ${k}: repeat flags`); }
    const it = s.item || {};
    if (!universe.has(it.scenario_id) || !Array.isArray(it.context_tags) || !ORDER_IDS.includes(it.question_order_id)) { slotProblems.push(`slot ${k}: item fields`); }
    const g = groups.get(k + 1);
    if (!g || g[0].r.scenario_id !== it.scenario_id || g[0].r.question_order_id !== it.question_order_id || j(g[0].r.context_tags) !== j(it.context_tags)) { slotProblems.push(`slot ${k}: rows disagree with assignment (item order / question order)`); }
    if (g && ((g[0].rep !== null) !== s.is_repeat || (s.is_repeat && g[0].rep !== s.repeat_of))) { slotProblems.push(`slot ${k}: repeat marking disagrees with rows`); }
  });
  C.check(slotProblems.length === 0, 'assignment slots (item order, question orders, repeats) agree with the rows', slotProblems.slice(0, 4).join(' ; '));
  try {
    const require = createRequire(import.meta.url);
    const BRE = require(path.join(STATIC, 'battery.js'));
    const again = BRE.buildAssignment(battery, form, info.seed);
    const view = a => a.slots.map(s => [s.item_index, s.is_repeat, s.repeat_of, s.item.scenario_id]);
    C.check(j(view(again)) === j(view(A)), 'assignment regenerates offline from battery.json + form + session_id (buildAssignment is pure)');
  } catch (e) { C.check(false, 'assignment regenerates offline', e.message); }

  // ---- session log: delay page, timings, identity ----
  const delayUsed = fast ? FAST_DELAY_SECONDS : D.delay_page.display_seconds;
  C.check(log.delay_seconds_used === delayUsed && log.delay_override_active === fast, `session log records delay_seconds_used=${delayUsed}, delay_override_active=${fast}`, `${log.delay_seconds_used} / ${log.delay_override_active}`);
  C.check(Array.isArray(log.items) && log.items.length === F.n_presentations, 'session log has one item entry per presentation');
  const newsItems = (log.items || []).filter(it => it.has_news_delay);
  const newsExpected = (A && A.slots || []).filter(s => s.item.has_news_delay).length;
  C.check(newsItems.length === newsExpected && newsItems.length === expected.delayPagesSeen, `delay page shown exactly once per news item (${newsItems.length} items)`, `${newsItems.length} / ${newsExpected} / seen ${expected.delayPagesSeen}`);
  C.check(expected.delayStartedDisabled === expected.delayPagesSeen, 'delay page button starts disabled every time');
  const delayProblems = (log.items || []).filter(it => it.has_news_delay ? !(typeof it.delay_display_ms === 'number' && it.delay_display_ms >= delayUsed * 1000 * 0.98) : it.delay_display_ms !== null).map(it => `i${it.item_index}: ${it.delay_display_ms}`);
  C.check(delayProblems.length === 0, `delay_display_ms >= ${delayUsed * 1000} ms on news items and null elsewhere`, delayProblems.join(','));
  C.check(log.session_id === info.seed && log.subject_id === info.subject_id && log.n_rows === rows.length && !!log.started_at && !!log.ended_at, 'session log identity/summary fields');
  C.check(log.post_status === null && bodyText.includes('Nothing has been sent'), 'download-only mode: no POST attempted and the notice is shown');
  C.check(ls && Array.isArray(ls.responses) && ls.responses.length === rows.length && j(ls.responses) === j(rows), 'localStorage backup holds the same rows', j(ls).slice(0, 120));

  // ---- layout at this viewport ----
  const vw = info.innerWidth;
  C.check(vw === viewport.width, `page laid out at the requested viewport width (${viewport.width}px)`, `innerWidth ${vw}`);
  C.check(layout.pagesProbed >= info.totalPages, `layout probed on every page (${layout.pagesProbed} probes for ${info.totalPages} pages + completion)`);
  C.check(layout.overflowPages.length === 0, `no horizontal overflow and no element outside the ${vw}px viewport on any page`, layout.overflowPages.slice(0, 4).join(' | '));
  C.check(layout.controlsOutsidePages.length === 0, `every button / input lies inside the ${vw}px viewport on every page (reachable without horizontal scrolling)`, layout.controlsOutsidePages.slice(0, 4).join(' | '));
}

// ------------------------------------------------------------------ main ---------------------
async function main() {
  const args = parseArgs(process.argv.slice(2));
  const battery = JSON.parse(fs.readFileSync(BATTERY_PATH, 'utf8'));
  const rng = makeDriverRng(args.seed);
  const out = args.out || fs.mkdtempSync(path.join(os.tmpdir(), 'bre-drive-'));
  fs.mkdirSync(out, { recursive: true });
  console.log(`drive_battery: driver seed=${rng.seed} (replay with --seed ${rng.seed}); output dir ${out}`);

  const { pw, from } = await loadPlaywright();
  console.log(`playwright from ${from}`);
  let server = null, url = args.url;
  if (!url) { const port = await freePort(); server = await startServer(port); url = `http://127.0.0.1:${port}`; console.log(`serving ${STATIC} at ${url} (python -m http.server)`); }
  const { browser, how } = await launchChromium(pw, args.headed);
  console.log(`chromium ${browser.version()} via ${how}`);

  const phone = { width: args.phoneWidth, height: 667, mobile: true };
  const sessions = args.forms.map(form => ({ form, viewport: DESKTOP }));
  const phoneForms = args.phone === 'none' ? [] : args.phone === 'both' ? ['short', 'full'] : [args.phone];
  phoneForms.forEach(form => sessions.push({ form, viewport: phone }));

  const all = [];
  let exitCode = 0;
  try {
    for (let n = 0; n < sessions.length; n++) {
      const { form, viewport } = sessions[n];
      const consent = args.consent === 'yes' ? true : args.consent === 'no' ? false : args.consent === 'random' ? rng.int(2) === 1 : (n % 2 === 0);
      const C = new Checks(`${form}/consent=${consent}/fast=${args.fast}/viewport=${viewport.width}px${viewport.mobile ? '-mobile' : ''}`);
      try {
        await driveSession(browser, battery, { form, consent, fast: args.fast, url, batterySeed: args.batterySeed, out, rng, C, viewport, shots: args.shots });
      } catch (e) {
        C.check(false, 'session completed without a driver error', e.stack || e.message);
      }
      all.push(C);
    }
  } finally {
    await browser.close();
    if (server) { server.kill(); }
  }
  console.log('\n=== summary ===');
  for (const C of all) {
    const f = C.failed;
    console.log(`${f.length ? 'FAIL' : 'PASS'}  ${C.name}: ${C.passed.length} passed, ${f.length} failed`);
    f.forEach(r => console.log('        - ' + r.label + (r.detail ? ' -- ' + r.detail.slice(0, 300) : '')));
    if (f.length) { exitCode = 1; }
  }
  console.log(`outputs in ${out}`);
  process.exit(exitCode);
}

main().catch(e => { console.error(e.stack || e); process.exit(2); });
