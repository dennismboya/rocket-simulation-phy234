/* battery.js -- BRE intake battery runner (jsPsych 7.x, plain script, no build step).
 *
 * Reads window.BRE_BATTERY (generated from ../battery.json) and window.BRE_CONFIG (config.js),
 * draws a seeded random subset of the 170-item design per the form's balance rule, runs the
 * intake + scenarios, and on completion produces one DecisionEvent row per elicited decision
 * (PLAN.md section 2 field list, exactly, in order). Elicitation types produced: tolerance ->
 * binary_yes_no (1 = Yes), sell_hold -> binary_sell (1 = Sell), allocation_share -> allocation_pct
 * (fraction sold); the values come from battery.json and are checked against the schema enum at
 * start-up (checkQuestions), so a mis-edited battery.json fails loudly instead of producing rows
 * the loader would reject.
 *
 * The pure functions (PRNG, shuffle, buildAssignment, checkQuestions) are exported on BRE and,
 * under Node, on module.exports so the balance rule can be tested offline.
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------ schema contract -----------
  // elicitation_type enum of the DecisionEvent schema (PLAN.md section 2 with the 2026-09-17
  // amendment; mirrored in db/models.py ELICITATION_TYPES). This battery uses three of them.
  var ELICITATION_TYPES = ['binary_sell', 'allocation_pct', 'likert', 'lottery_choice', 'binary_yes_no', 'choice_rate'];
  var QUESTION_TYPES = { tolerance: 'binary_yes_no', sell_hold: 'binary_sell', allocation_share: 'allocation_pct' };

  /** Pure: throws when battery.json's questions do not carry the contracted elicitation types / codings. */
  function checkQuestions(battery) {
    var Q = battery.design.questions;
    Object.keys(QUESTION_TYPES).forEach(function (qid) {
      var q = Q[qid];
      if (!q) { throw new Error('battery.json: question ' + qid + ' is missing'); }
      if (ELICITATION_TYPES.indexOf(q.elicitation_type) < 0) { throw new Error('battery.json: ' + qid + ' has elicitation_type "' + q.elicitation_type + '", not in the schema enum'); }
      if (q.elicitation_type !== QUESTION_TYPES[qid]) { throw new Error('battery.json: ' + qid + ' must be ' + QUESTION_TYPES[qid] + ', got "' + q.elicitation_type + '"'); }
      if (q.choices) {
        var codes = q.choices.map(function (c) { return c.response; }).sort().join();
        if (codes !== '0,1') { throw new Error('battery.json: ' + qid + ' choices must code exactly 0 and 1, got ' + codes); }
      }
    });
    var yes = Q.tolerance.choices.filter(function (c) { return c.response === 1; })[0];
    if (!yes || String(yes.label).toLowerCase() !== 'yes') { throw new Error('battery.json: tolerance response 1 must be the "Yes" choice'); }
    var sell = Q.sell_hold.choices.filter(function (c) { return c.response === 1; })[0];
    if (!sell || String(sell.label).toLowerCase() !== 'sell') { throw new Error('battery.json: sell_hold response 1 must be the "Sell" choice'); }
    return true;
  }

  // ------------------------------------------------------------------ seeded PRNG ---------------
  // sfc32: small fast counter PRNG, 128-bit state, from the PractRand suite (public domain).
  function sfc32(a, b, c, d) {
    return function () {
      a |= 0; b |= 0; c |= 0; d |= 0;
      var t = (((a + b) | 0) + d) | 0;
      d = (d + 1) | 0;
      a = b ^ (b >>> 9);
      b = (c + (c << 3)) | 0;
      c = (c << 21) | (c >>> 11);
      c = (c + t) | 0;
      return (t >>> 0) / 4294967296;
    };
  }

  function isValidSeed(s) {
    return typeof s === 'string' && /^[0-9a-f]{32}$/.test(s);
  }

  function makeRng(seedHex) {
    if (!isValidSeed(seedHex)) { throw new Error('seed must be 32 lowercase hex characters'); }
    var w = [];
    for (var i = 0; i < 4; i++) { w.push(parseInt(seedHex.slice(i * 8, i * 8 + 8), 16) >>> 0); }
    var rng = sfc32(w[0], w[1], w[2], w[3]);
    for (var k = 0; k < 12; k++) { rng(); } // warm-up draws
    return rng;
  }

  function randomWords(n) {
    var arr = new Uint32Array(n);
    var cr = (typeof root.crypto !== 'undefined' && root.crypto) || (typeof crypto !== 'undefined' ? crypto : null);
    if (!cr || typeof cr.getRandomValues !== 'function') { throw new Error('crypto.getRandomValues is not available in this browser'); }
    cr.getRandomValues(arr);
    return Array.prototype.slice.call(arr);
  }

  function drawSeedHex() {
    return randomWords(4).map(function (x) { return ('00000000' + (x >>> 0).toString(16)).slice(-8); }).join('');
  }

  function uuid4() {
    var cr = (typeof root.crypto !== 'undefined' && root.crypto) || null;
    if (cr && typeof cr.randomUUID === 'function') {
      try { return cr.randomUUID(); } catch (e) { /* fall through */ }
    }
    var b = randomWords(4);
    var hex = b.map(function (x) { return ('00000000' + (x >>> 0).toString(16)).slice(-8); }).join('');
    var chars = hex.split('');
    chars[12] = '4';
    chars[16] = '89ab'.charAt(b[0] & 3);
    hex = chars.join('');
    return hex.slice(0, 8) + '-' + hex.slice(8, 12) + '-' + hex.slice(12, 16) + '-' + hex.slice(16, 20) + '-' + hex.slice(20, 32);
  }

  function shuffle(arr, rng) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  function pick(arr, rng) { return arr[Math.floor(rng() * arr.length)]; }

  // ------------------------------------------------------------------ assignment ----------------
  function fmtLoss(l) { return Number(l).toFixed(2); }

  function scenarioId(loss, tags, order) {
    return 'L' + fmtLoss(loss) + '|' + (tags.length ? tags.join('>') : 'none') + '|' + order;
  }

  function makeItem(battery, loss, tags, order, conditionType) {
    var ctx = battery.design.contexts;
    var hasNews = tags.some(function (t) { return ctx[t] && ctx[t].kind === 'news'; });
    return {
      scenario_id: scenarioId(loss, tags, order),
      loss_pct: loss,
      horizon_days: battery.design.horizon_days,
      context_tags: tags.slice(),
      condition_type: conditionType,
      question_order_id: order,
      has_news_delay: hasNews
    };
  }

  function orderIds(battery) {
    return battery.design.question_orders.map(function (o) { return o.question_order_id; });
  }

  function buildFullItems(battery, rng, meta) {
    var D = battery.design;
    var losses = D.loss_pcts.slice();
    var ctx = D.context_tag_order.slice();
    var orders = orderIds(battery); // [tolerance-first, scenario-first]
    var blocks = ['none', 'single', 'pair'];

    // Loss rule: each block of 4 omits one loss level; the three omitted levels are distinct.
    var dropped = shuffle(losses, rng).slice(0, 3);
    var blockLosses = blocks.map(function (b, i) {
      return shuffle(losses.filter(function (l) { return l !== dropped[i]; }), rng);
    });
    // Single rule: each non-null context once.
    var singles = shuffle(ctx, rng).map(function (c) { return [c]; });
    // Pair rule: random 4-cycle -> 4 ordered pairs, each context once first and once second.
    var cyc = shuffle(ctx, rng);
    var pairs = shuffle([[cyc[0], cyc[1]], [cyc[1], cyc[2]], [cyc[2], cyc[3]], [cyc[3], cyc[0]]], rng);
    var conds = { none: [[], [], [], []], single: singles, pair: pairs };

    // Order rule: 2+2 per block; redraw until every loss level has both orders (bounded).
    var MAX_ATTEMPTS = 200;
    var items = null, attempts = 0, satisfied = false;
    while (attempts < MAX_ATTEMPTS) {
      attempts++;
      var cand = [];
      for (var bi = 0; bi < blocks.length; bi++) {
        var ord = shuffle([orders[0], orders[0], orders[1], orders[1]], rng);
        for (var k = 0; k < 4; k++) {
          cand.push(makeItem(battery, blockLosses[bi][k], conds[blocks[bi]][k], ord[k], blocks[bi]));
        }
      }
      var byLoss = {};
      cand.forEach(function (it) {
        byLoss[it.loss_pct] = byLoss[it.loss_pct] || {};
        byLoss[it.loss_pct][it.question_order_id] = true;
      });
      items = cand;
      var ok = losses.every(function (l) { return byLoss[l] && Object.keys(byLoss[l]).length === 2; });
      if (ok) { satisfied = true; break; }
    }
    meta.dropped_loss_per_block = { none: dropped[0], single: dropped[1], pair: dropped[2] };
    meta.context_cycle = cyc;
    meta.order_balance_attempts = attempts;
    meta.order_balance_satisfied = satisfied;
    return items;
  }

  function buildShortItems(battery, rng, meta) {
    var D = battery.design;
    var losses5 = shuffle(D.loss_pcts.slice(), rng);
    var ctx = D.context_tag_order.slice();
    var orders = orderIds(battery);
    var singles = shuffle(ctx, rng).slice(0, 2);
    var cyc = shuffle(ctx, rng);
    var conds = [
      { type: 'none', tags: [] },
      { type: 'single', tags: [singles[0]] },
      { type: 'single', tags: [singles[1]] },
      { type: 'pair', tags: [cyc[0], cyc[1]] },
      { type: 'pair', tags: [cyc[2], cyc[3]] }
    ];
    var maj = pick(orders, rng);
    var min = orders.filter(function (o) { return o !== maj; })[0];
    var ord = shuffle([maj, maj, maj, min, min], rng);
    meta.context_cycle = cyc;
    meta.majority_order = maj;
    meta.order_balance_satisfied = true;
    return conds.map(function (c, i) { return makeItem(battery, losses5[i], c.tags, ord[i], c.type); });
  }

  /** Pure: (battery, 'full'|'short', seedHex) -> assignment. Same inputs always give the same output. */
  function buildAssignment(battery, formId, seedHex) {
    var form = battery.forms[formId];
    if (!form) { throw new Error('unknown form: ' + formId); }
    var rng = makeRng(seedHex);
    var meta = {};
    var items = formId === 'full' ? buildFullItems(battery, rng, meta) : buildShortItems(battery, rng, meta);
    if (items.length !== form.n_items) { throw new Error('internal: expected ' + form.n_items + ' items, built ' + items.length); }

    var presented = shuffle(items, rng);
    var slots = presented.map(function (it) { return { item_index: 0, is_repeat: false, repeat_of: null, source_slot: null, item: it }; });

    // Repeated-scenario rule.
    var rule = battery.design.repeated_scenario_rule;
    var nRep = form.n_repeats;
    if (nRep > 0) {
      var minGap = rule.min_gap_items;
      var pool = slots.slice(0, Math.min(6, slots.length));
      var chosen = shuffle(pool, rng).slice(0, nRep);
      chosen.forEach(function (src) {
        var origPos = slots.indexOf(src);
        var lo = origPos + minGap;
        var hi = slots.length; // insertion index may equal length (append)
        if (lo > hi) { lo = hi; }
        var at = lo + Math.floor(rng() * (hi - lo + 1));
        slots.splice(at, 0, { item_index: 0, is_repeat: true, repeat_of: null, source_slot: src, item: src.item });
      });
    }
    slots.forEach(function (s, i) { s.item_index = i + 1; });
    slots.forEach(function (s) { if (s.is_repeat) { s.repeat_of = s.source_slot.item_index; } delete s.source_slot; });
    if (slots.length !== form.n_presentations) { throw new Error('internal: expected ' + form.n_presentations + ' presentations, built ' + slots.length); }

    return {
      battery_version: battery.battery_version,
      form: formId,
      seed: seedHex,
      n_items: form.n_items,
      n_repeats: nRep,
      n_presentations: slots.length,
      meta: meta,
      slots: slots
    };
  }

  // ------------------------------------------------------------------ helpers ------------------
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }

  function fill(template, vars) {
    return template.replace(/\{([a-z_]+)\}/g, function (m, k) { return Object.prototype.hasOwnProperty.call(vars, k) ? vars[k] : m; });
  }

  function getParam(name) {
    try {
      var qs = root.location && root.location.search ? root.location.search : '';
      var m = new RegExp('[?&]' + name + '=([^&#]*)').exec(qs);
      return m ? decodeURIComponent(m[1].replace(/\+/g, ' ')) : null;
    } catch (e) { return null; }
  }

  function nowIso() { return new Date().toISOString(); }

  // ------------------------------------------------------------------ page builders -----------
  function lossDisplay(battery, loss) { return battery.design.loss_pct_display[fmtLoss(loss)]; }

  function kicker(S, slot) {
    return '<div class="bre-kicker">Scenario ' + slot.item_index + ' of ' + S.assignment.n_presentations + '</div>';
  }

  function contextBlock(battery, item, compact) {
    var ctx = battery.design.contexts;
    var tpl = battery.design.scenario.template;
    if (item.context_tags.length === 0) { return ''; }
    var lead = item.context_tags.length === 1 ? tpl.context_lead_in.single : tpl.context_lead_in.pair;
    var html = '<p class="bre-lead">' + esc(lead) + '</p>';
    if (item.context_tags.length === 1) {
      html += '<p class="bre-ctx">' + esc(ctx[item.context_tags[0]].display) + '</p>';
    } else {
      html += '<ol class="bre-ctx-list">' + item.context_tags.map(function (t) { return '<li>' + esc(ctx[t].display) + '</li>'; }).join('') + '</ol>';
    }
    return html;
  }

  function scenarioPageHtml(S, slot) {
    var B = S.battery, tpl = B.design.scenario.template, item = slot.item;
    var vars = { loss_pct_display: lossDisplay(B, item.loss_pct) };
    return '<div class="bre-card">' + kicker(S, slot) +
      '<p>' + esc(tpl.intro) + '</p>' +
      '<p class="bre-loss">' + esc(fill(tpl.loss_sentence, vars)) + '</p>' +
      contextBlock(B, item, false) +
      '<p class="bre-hint">Take your time reading. The questions come next.</p></div>';
  }

  function summaryHtml(S, slot) {
    var B = S.battery, tpl = B.design.scenario.template, item = slot.item;
    var vars = { loss_pct_display: lossDisplay(B, item.loss_pct) };
    return '<div class="bre-summary"><span class="bre-loss">' + esc(fill(tpl.reminder, vars)) + '</span>' +
      contextBlock(B, item, true) + '</div>';
  }

  function reminderHtml(S, slot) {
    var B = S.battery, tpl = B.design.scenario.template, item = slot.item;
    var vars = { loss_pct_display: lossDisplay(B, item.loss_pct) };
    return '<div class="bre-summary"><span class="bre-loss">' + esc(fill(tpl.reminder, vars)) + '</span></div>';
  }

  // ------------------------------------------------------------------ recording ---------------
  function covariatesSnapshot(S) {
    return {
      age_band: S.covariates.age_band,
      wealth_band: S.covariates.wealth_band,
      invest_experience_yrs: S.covariates.invest_experience_yrs,
      self_reported_risk_tolerance: S.covariates.self_reported_risk_tolerance,
      financial_literacy_score: S.covariates.financial_literacy_score,
      education: S.covariates.education
    };
  }

  function sourceRowRef(S, slot, qid) {
    var idx = ('0' + slot.item_index).slice(-2);
    var ref = S.form + ':i' + idx;
    if (slot.is_repeat) { ref += ':rep_of_i' + ('0' + slot.repeat_of).slice(-2); }
    return ref + ':' + qid;
  }

  function record(S, slot, ps, qid, elicitationType, response, rt) {
    var item = slot.item;
    var row = {
      subject_id: S.subject_id,
      dataset: S.battery.dataset,
      session_id: S.seed,
      timestamp: nowIso(),
      position_in_session: S.position,
      scenario_id: item.scenario_id,
      loss_pct: item.loss_pct,
      horizon_days: item.horizon_days,
      context_tags: item.context_tags.slice(),
      question_order_id: item.question_order_id,
      prior_question_ids: ps.prior.slice(),
      elicitation_type: elicitationType,
      response: response,
      response_time_ms: (typeof rt === 'number' && isFinite(rt)) ? Math.round(rt) : null,
      covariates: covariatesSnapshot(S),
      outcome_behavior: null,
      incentivized: false,
      consent_training: !!S.consent_training,
      battery_version: S.battery.battery_version,
      is_synthetic: false,
      source_row_ref: sourceRowRef(S, slot, qid)
    };
    S.position += 1;
    S.records.push(row);
    ps.prior.push(qid);
    return row;
  }

  // ------------------------------------------------------------------ timeline ----------------
  function buildTimeline(S, jsPsych) {
    var B = S.battery, I = B.intake, Q = B.design.questions;
    var timeline = [];

    function page(trial) {
      var orig = trial.on_finish;
      trial.on_finish = function (data) {
        if (orig) { orig(data); }
        S.pagesDone += 1;
        try { jsPsych.setProgressBar(Math.min(1, S.pagesDone / S.totalPages)); } catch (e) { /* ignore */ }
      };
      timeline.push(trial);
      return trial;
    }

    // --- consent ---
    var consentHtml = '<div class="bre-card"><div class="bre-title">' + esc(I.consent.title) + '</div>' +
      I.consent.text.map(function (p) { return '<p>' + esc(p) + '</p>'; }).join('') +
      '<div class="bre-consent">' + I.consent.checkboxes.map(function (cb) {
        return '<label><input type="checkbox" name="' + esc(cb.name) + '" value="yes"' + (cb.required ? ' required' : '') + '>' + esc(cb.label) + '</label>';
      }).join('') + '</div></div>';
    page({
      type: jsPsychSurveyHtmlForm,
      html: consentHtml,
      button_label: I.consent.button_label,
      data: { bre_page: 'consent' },
      on_finish: function (d) {
        var r = d.response || {};
        S.consent_training = r.consent_training === 'yes';
        S.log.consent_training = S.consent_training;
        S.log.pages.push({ page: 'consent', rt: d.rt });
      }
    });

    // --- instructions ---
    var instrVars = { n_scenarios: String(S.assignment.n_presentations) };
    page({
      type: jsPsychHtmlButtonResponse,
      stimulus: '<div class="bre-card"><div class="bre-title">' + esc(I.instructions.title) + '</div>' +
        I.instructions.text.map(function (p) { return '<p>' + esc(fill(p, instrVars)) + '</p>'; }).join('') + '</div>',
      choices: [I.instructions.button_label],
      data: { bre_page: 'instructions' },
      on_finish: function (d) { S.log.pages.push({ page: 'instructions', rt: d.rt }); }
    });

    // --- covariates (two pages, built from battery.json) ---
    function covariateFieldHtml(cv) {
      var html = '<fieldset class="bre-field"><legend>' + esc(cv.text) + '</legend>';
      if (cv.input === 'radio') {
        cv.options.forEach(function (o, i) {
          var val = o.value === null ? '__null__' : o.value;
          html += '<label class="bre-opt"><input type="radio" name="' + esc(cv.key) + '" value="' + esc(val) + '"' + (cv.required && i === 0 ? ' required' : '') + '> ' + esc(o.label) + '</label>';
        });
      } else if (cv.input === 'number') {
        html += '<input type="number" name="' + esc(cv.key) + '" min="' + cv.min + '" max="' + cv.max + '" step="' + cv.step + '" inputmode="numeric"' + (cv.required ? ' required' : '') + '>';
      } else if (cv.input === 'likert') {
        html += '<div class="bre-likert">';
        for (var v = cv.min; v <= cv.max; v++) {
          html += '<label><input type="radio" name="' + esc(cv.key) + '" value="' + v + '"' + (cv.required && v === cv.min ? ' required' : '') + '><span>' + v + '</span></label>';
        }
        html += '<span class="bre-anchor">' + esc(cv.min + ' = ' + cv.anchors[String(cv.min)] + ', ' + cv.max + ' = ' + cv.anchors[String(cv.max)]) + '</span></div>';
      }
      return html + '</fieldset>';
    }

    function covariatePage(pageKey, title) {
      var fields = I.covariates.filter(function (cv) { return cv.page === pageKey; });
      page({
        type: jsPsychSurveyHtmlForm,
        html: '<div class="bre-card"><div class="bre-title">' + esc(title) + '</div>' + fields.map(covariateFieldHtml).join('') + '</div>',
        button_label: 'Continue',
        data: { bre_page: 'covariates_' + pageKey },
        on_finish: function (d) {
          var r = d.response || {};
          fields.forEach(function (cv) {
            var raw = r[cv.key];
            var val = null;
            if (raw === undefined || raw === null || raw === '' || raw === '__null__') {
              val = null;
            } else if (cv.input === 'number' || cv.input === 'likert') {
              var n = parseInt(raw, 10);
              if (!isNaN(n)) { val = Math.max(cv.min, Math.min(cv.max, n)); }
            } else {
              val = String(raw);
            }
            S.covariates[cv.key] = val;
          });
          S.log.pages.push({ page: 'covariates_' + pageKey, rt: d.rt });
        }
      });
    }
    covariatePage('about_you', 'About you');
    covariatePage('investing', 'Your investing');

    // --- financial literacy quiz ---
    var quiz = I.financial_literacy_quiz;
    quiz.items.forEach(function (qi, qn) {
      page({
        type: jsPsychHtmlButtonResponse,
        stimulus: '<div class="bre-card"><div class="bre-kicker">Knowledge question ' + (qn + 1) + ' of ' + quiz.items.length + '</div>' +
          (qn === 0 ? '<p class="bre-hint">' + esc(quiz.intro) + '</p>' : '') +
          '<p class="bre-question">' + esc(qi.text) + '</p></div>',
        choices: qi.options,
        button_html: '<div class="bre-btn-stack"><button class="jspsych-btn">%choice%</button></div>',
        data: { bre_page: 'quiz', item_id: qi.item_id },
        on_finish: function (d) {
          var idx = (typeof d.response === 'number') ? d.response : null;
          var correct = idx !== null && idx === qi.answer_index;
          S.quiz.push({ item_id: qi.item_id, response_index: idx, correct: correct, rt: d.rt });
          S.covariates.financial_literacy_score = S.quiz.reduce(function (n, a) { return n + (a.correct ? 1 : 0); }, 0);
          S.log.pages.push({ page: 'quiz:' + qi.item_id, rt: d.rt });
        }
      });
    });

    // --- transition ---
    page({
      type: jsPsychHtmlButtonResponse,
      stimulus: '<div class="bre-card"><div class="bre-title">The scenarios</div><p>Thank you. The ' + S.assignment.n_presentations +
        ' scenarios start now. Each one stands on its own: treat every scenario as a fresh situation.</p></div>',
      choices: ['Start'],
      data: { bre_page: 'transition' },
      on_finish: function (d) { S.log.pages.push({ page: 'transition', rt: d.rt }); }
    });

    // --- scenario items ---
    var delayCfg = B.design.delay_page;
    var delaySeconds = S.delaySecondsUsed;
    var orderSeq = {};
    B.design.question_orders.forEach(function (o) { orderSeq[o.question_order_id] = o.page_sequence; });

    S.assignment.slots.forEach(function (slot) {
      var item = slot.item;
      var ps = { prior: [] };
      var logEntry = {
        item_index: slot.item_index, is_repeat: slot.is_repeat, repeat_of: slot.repeat_of,
        scenario_id: item.scenario_id, has_news_delay: item.has_news_delay, page_rt_ms: {}, delay_display_ms: null
      };
      S.log.items.push(logEntry);

      orderSeq[item.question_order_id].forEach(function (step) {
        if (step === 'tolerance') {
          page({
            type: jsPsychHtmlButtonResponse,
            stimulus: '<div class="bre-card">' + kicker(S, slot) + '<p class="bre-question">' + esc(Q.tolerance.text) + '</p></div>',
            choices: Q.tolerance.choices.map(function (c) { return c.label; }),
            data: { bre_page: 'tolerance', item_index: slot.item_index },
            on_finish: function (d) {
              var choice = Q.tolerance.choices[d.response];
              logEntry.page_rt_ms.tolerance = d.rt;
              record(S, slot, ps, Q.tolerance.question_id, Q.tolerance.elicitation_type, choice ? choice.response : null, d.rt);
            }
          });
        } else if (step === 'scenario') {
          page({
            type: jsPsychHtmlButtonResponse,
            stimulus: scenarioPageHtml(S, slot),
            choices: ['Continue'],
            data: { bre_page: 'scenario', item_index: slot.item_index },
            on_finish: function (d) { logEntry.page_rt_ms.scenario = d.rt; }
          });
        } else if (step === 'delay_if_news') {
          if (item.has_news_delay) {
            page({
              type: jsPsychHtmlButtonResponse,
              stimulus: '<div class="bre-card">' + kicker(S, slot) + '<div class="bre-title">' + esc(delayCfg.title) + '</div><p>' + esc(delayCfg.text) + '</p></div>',
              choices: [delayCfg.button_label],
              enable_button_after: Math.round(delaySeconds * 1000),
              data: { bre_page: 'delay', item_index: slot.item_index },
              on_finish: function (d) { logEntry.page_rt_ms.delay = d.rt; logEntry.delay_display_ms = d.rt; }
            });
          }
        } else if (step === 'sell_hold') {
          page({
            type: jsPsychHtmlButtonResponse,
            stimulus: '<div class="bre-card">' + kicker(S, slot) + summaryHtml(S, slot) + '<p class="bre-question">' + esc(Q.sell_hold.text) + '</p></div>',
            choices: Q.sell_hold.choices.map(function (c) { return c.label; }),
            data: { bre_page: 'sell_hold', item_index: slot.item_index },
            on_finish: function (d) {
              var choice = Q.sell_hold.choices[d.response];
              logEntry.page_rt_ms.sell_hold = d.rt;
              record(S, slot, ps, Q.sell_hold.question_id, Q.sell_hold.elicitation_type, choice ? choice.response : null, d.rt);
            }
          });
        } else if (step === 'allocation_share') {
          var sl = Q.allocation_share.slider;
          page({
            type: jsPsychHtmlSliderResponse,
            stimulus: '<div class="bre-card">' + kicker(S, slot) + reminderHtml(S, slot) + '<p class="bre-question">' + esc(Q.allocation_share.text) + '</p></div>',
            min: sl.min, max: sl.max, step: sl.step, slider_start: sl.start,
            labels: sl.labels, require_movement: !!sl.require_movement,
            button_label: 'Continue',
            data: { bre_page: 'allocation_share', item_index: slot.item_index },
            on_finish: function (d) {
              var v = (typeof d.response === 'number') ? d.response : parseFloat(d.response);
              var resp = isNaN(v) ? null : Math.max(0, Math.min(1, v / sl.max));
              logEntry.page_rt_ms.allocation_share = d.rt;
              record(S, slot, ps, Q.allocation_share.question_id, Q.allocation_share.elicitation_type, resp, d.rt);
            }
          });
        }
      });
    });

    return timeline;
  }

  // ------------------------------------------------------------------ completion --------------
  function downloadJson(filename, obj) {
    var text = JSON.stringify(obj, null, 2);
    var useBlob = (typeof Blob === 'function' && typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function');
    var url = useBlob ? URL.createObjectURL(new Blob([text], { type: 'application/json' }))
                      : 'data:application/json;charset=utf-8,' + encodeURIComponent(text);
    var a = document.createElement('a');
    a.href = url; a.download = filename; a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { if (useBlob) { URL.revokeObjectURL(url); } a.remove(); }, 1500);
  }

  function postResults(S, statusEl) {
    var cfg = S.config;
    var body;
    if (cfg.ENDPOINT_WRAP_KEY) {
      body = {};
      var extra = cfg.ENDPOINT_EXTRA_FIELDS || {};
      Object.keys(extra).forEach(function (k) { body[k] = extra[k]; });
      body[cfg.ENDPOINT_WRAP_KEY] = S.records;
      if (cfg.ENDPOINT_INCLUDE_SESSION_LOG) { body.session_log = S.log; }
    } else {
      body = S.records;
    }
    function setStatus(cls, msg) { statusEl.className = 'bre-status ' + cls; statusEl.textContent = msg; S.log.post_status = msg; }
    if (typeof fetch !== 'function') { setStatus('err', 'Sending failed: this browser has no fetch(). Please download the file instead.'); return; }
    setStatus('', 'Sending your responses...');
    fetch(cfg.ENDPOINT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (r) {
      if (r.ok) { setStatus('ok', 'Your responses were sent (HTTP ' + r.status + '). You can still download a copy below.'); }
      else { setStatus('err', 'Sending failed (HTTP ' + r.status + '). Please download the file and send it to the study contact.'); }
    }).catch(function (e) {
      setStatus('err', 'Sending failed (' + (e && e.message ? e.message : 'network error') + '). Please download the file and send it to the study contact.');
    });
  }

  function finish(S, jsPsych) {
    S.log.ended_at = nowIso();
    S.log.n_rows = S.records.length;
    var el = jsPsych.getDisplayElement();
    var cfg = S.config;
    var fileNames = S.battery.output.file_names;
    var respName = fill(fileNames.responses, { session_id: S.seed });
    var logName = fill(fileNames.session_log, { session_id: S.seed });
    var contact = cfg.STUDY_CONTACT ? '<p>Please send the downloaded file to <strong>' + esc(cfg.STUDY_CONTACT) + '</strong>.</p>' : '';
    var hasEndpoint = !!(cfg.ENDPOINT_URL && String(cfg.ENDPOINT_URL).trim());

    el.innerHTML =
      '<div class="jspsych-content-wrapper"><div class="jspsych-content bre-done"><div class="bre-card">' +
      '<div class="bre-title">All done. Thank you.</div>' +
      '<p>You answered ' + S.records.length + ' questions across ' + S.assignment.n_presentations + ' scenarios.</p>' +
      (hasEndpoint ? '' : '<p>Nothing has been sent anywhere. Your responses exist only in this browser until you download them.</p>') +
      contact +
      '<div id="bre-post-status" class="bre-status"></div>' +
      '<div class="bre-actions">' +
      '<button id="bre-dl-responses" class="jspsych-btn">Download responses (JSON)</button>' +
      '<button id="bre-dl-log" class="jspsych-btn">Download session log (JSON)</button>' +
      (hasEndpoint ? '<button id="bre-resend" class="jspsych-btn">Send again</button>' : '') +
      '<button id="bre-show-json" class="jspsych-btn">Show JSON</button>' +
      '</div>' +
      '<p class="bre-hint">Session id: <span class="bre-mono">' + esc(S.seed) + '</span><br>Subject id: <span class="bre-mono">' + esc(S.subject_id) + '</span></p>' +
      '<textarea id="bre-json" class="bre-json" readonly style="display:none"></textarea>' +
      '</div></div></div>';

    var statusEl = el.querySelector('#bre-post-status');
    el.querySelector('#bre-dl-responses').addEventListener('click', function () { downloadJson(respName, S.records); });
    el.querySelector('#bre-dl-log').addEventListener('click', function () { downloadJson(logName, S.log); });
    el.querySelector('#bre-show-json').addEventListener('click', function () {
      var ta = el.querySelector('#bre-json');
      ta.value = JSON.stringify(S.records, null, 2);
      ta.style.display = ta.style.display === 'none' ? 'block' : 'none';
    });
    if (hasEndpoint) {
      el.querySelector('#bre-resend').addEventListener('click', function () { postResults(S, statusEl); });
      postResults(S, statusEl);
    }
    try { root.localStorage.setItem('bre_intake_' + S.seed, JSON.stringify({ responses: S.records, session_log: S.log })); } catch (e) { /* private mode etc. */ }
    S.finished = true;
    if (typeof S.onFinished === 'function') { S.onFinished(S); }
  }

  // ------------------------------------------------------------------ entry point -------------
  function run(options) {
    options = options || {};
    var battery = root.BRE_BATTERY;
    var config = root.BRE_CONFIG || {};
    if (!battery) { throw new Error('battery.data.js did not load'); }
    checkQuestions(battery);

    var formParam = getParam('form');
    var formId = (formParam === 'short') ? 'short' : 'full';
    var seedParam = getParam('seed');
    var seed = isValidSeed(seedParam) ? seedParam : drawSeedHex();
    var assignment = buildAssignment(battery, formId, seed);

    var override = config.DELAY_SECONDS_OVERRIDE;
    var overrideActive = (typeof override === 'number' && isFinite(override) && override >= 0);

    var S = {
      battery: battery,
      config: config,
      form: formId,
      seed: seed,
      subject_id: uuid4(),
      assignment: assignment,
      consent_training: false,
      covariates: { age_band: null, wealth_band: null, invest_experience_yrs: null, self_reported_risk_tolerance: null, financial_literacy_score: null, education: null },
      quiz: [],
      records: [],
      position: 0,
      pagesDone: 0,
      totalPages: 0,
      delaySecondsUsed: overrideActive ? override : battery.design.delay_page.display_seconds,
      finished: false,
      onFinished: options.onFinished || null,
      log: {
        dataset: battery.dataset,
        battery_version: battery.battery_version,
        form: formId,
        seed: seed,
        session_id: seed,
        subject_id: null,
        started_at: nowIso(),
        ended_at: null,
        seed_from_url: isValidSeed(seedParam),
        delay_seconds_used: overrideActive ? override : battery.design.delay_page.display_seconds,
        delay_override_active: overrideActive,
        user_agent: (root.navigator && root.navigator.userAgent) || null,
        assignment: assignment,
        consent_training: null,
        pages: [],
        items: [],
        post_status: null,
        n_rows: null
      }
    };
    S.log.subject_id = S.subject_id;

    var jsPsych = initJsPsych({
      display_element: 'jspsych-target',
      show_progress_bar: true,
      auto_update_progress_bar: false,
      message_progress_bar: 'Progress',
      on_finish: function () { finish(S, jsPsych); }
    });
    var timeline = buildTimeline(S, jsPsych);
    S.totalPages = timeline.length;
    root.BRE_STATE = S; // for debugging / tests
    jsPsych.run(timeline);
    return S;
  }

  var BRE = {
    run: run,
    buildAssignment: buildAssignment,
    checkQuestions: checkQuestions,
    ELICITATION_TYPES: ELICITATION_TYPES.slice(),
    QUESTION_TYPES: QUESTION_TYPES,
    makeRng: makeRng,
    drawSeedHex: drawSeedHex,
    isValidSeed: isValidSeed,
    shuffle: shuffle,
    scenarioId: scenarioId,
    uuid4: uuid4
  };
  root.BRE = BRE;
  if (typeof module !== 'undefined' && module.exports) { module.exports = BRE; }
})(typeof window !== 'undefined' ? window : globalThis);
