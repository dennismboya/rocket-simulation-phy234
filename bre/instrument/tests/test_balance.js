// Offline test of buildAssignment's balance rule over many random seeds. Pure Node, no packages.
//   /opt/node22/bin/node bre/instrument/tests/test_balance.js [n_seeds]
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const ROOT = path.resolve(__dirname, '..');
const BRE = require(path.join(ROOT, 'static', 'battery.js'));
const battery = JSON.parse(fs.readFileSync(path.join(ROOT, 'battery.json'), 'utf8'));

const NONNULL = battery.design.context_tag_order;
const LOSSES = battery.design.loss_pcts;
const ORDERS = battery.design.question_orders.map(o => o.question_order_id);
const universe = new Set(battery.design.scenario_universe.map(u => u.scenario_id));

function count(arr, f) { const m = {}; arr.forEach(x => { const k = f(x); m[k] = (m[k] || 0) + 1; }); return m; }
function assert(cond, msg) { if (!cond) throw new Error(msg); }

const N = parseInt(process.argv[2] || '3000', 10);
let unsatisfied = 0, maxAttempts = 0;
const gapHist = {}, repeatPosHist = {};

for (let s = 0; s < N; s++) {
  const seed = crypto.randomBytes(16).toString('hex');

  // ---------------- full ----------------
  const A = BRE.buildAssignment(battery, 'full', seed);
  const A2 = BRE.buildAssignment(battery, 'full', seed);
  assert(JSON.stringify(A) === JSON.stringify(A2), 'not deterministic for seed ' + seed);
  assert(A.slots.length === battery.forms.full.n_presentations, 'full: n_presentations');
  const originals = A.slots.filter(x => !x.is_repeat), repeats = A.slots.filter(x => x.is_repeat);
  assert(originals.length === 12 && repeats.length === 2, 'full: 12 originals + 2 repeats');
  const items = originals.map(x => x.item);
  items.forEach(it => assert(universe.has(it.scenario_id), 'scenario id not in universe: ' + it.scenario_id));
  assert(new Set(items.map(i => i.scenario_id)).size === 12, 'full: 12 unique scenario ids');
  const byType = count(items, i => i.condition_type);
  assert(byType.none === 4 && byType.single === 4 && byType.pair === 4, 'full: 4/4/4 types ' + JSON.stringify(byType));
  const byOrder = count(items, i => i.question_order_id);
  assert(byOrder[ORDERS[0]] === 6 && byOrder[ORDERS[1]] === 6, 'full: 6/6 orders');
  ['none', 'single', 'pair'].forEach(t => {
    const blk = items.filter(i => i.condition_type === t);
    const bo = count(blk, i => i.question_order_id);
    assert(bo[ORDERS[0]] === 2 && bo[ORDERS[1]] === 2, 'full: 2/2 orders within block ' + t);
    assert(new Set(blk.map(i => i.loss_pct)).size === 4, 'full: 4 distinct losses within block ' + t);
  });
  const byLoss = count(items, i => i.loss_pct);
  LOSSES.forEach(l => assert(byLoss[l] >= 2, 'full: loss ' + l + ' appears < 2'));
  assert(Object.values(byLoss).filter(v => v === 2).length === 3 && Object.values(byLoss).filter(v => v === 3).length === 2, 'full: 3 losses x2, 2 losses x3');
  if (A.meta.order_balance_satisfied) {
    LOSSES.forEach(l => {
      const os = new Set(items.filter(i => i.loss_pct === l).map(i => i.question_order_id));
      assert(os.size === 2, 'full: loss ' + l + ' lacks both orders');
    });
  } else { unsatisfied++; }
  maxAttempts = Math.max(maxAttempts, A.meta.order_balance_attempts);
  const singles = items.filter(i => i.condition_type === 'single').map(i => i.context_tags[0]);
  assert(new Set(singles).size === 4 && singles.every(c => NONNULL.includes(c)), 'full: singles cover all 4 contexts');
  const pairs = items.filter(i => i.condition_type === 'pair').map(i => i.context_tags);
  assert(new Set(pairs.map(p => p[0])).size === 4 && new Set(pairs.map(p => p[1])).size === 4, 'full: each context once first and once second');
  pairs.forEach(p => assert(p[0] !== p[1], 'full: pair with identical contexts'));
  assert(new Set(pairs.map(p => [p[0], p[1]].sort().join('|'))).size === 4, 'full: unordered pair repeated');
  assert(items.every(i => i.has_news_delay === i.context_tags.some(t => battery.design.contexts[t].kind === 'news')), 'full: delay flag');
  const idx = {}; A.slots.forEach(x => { idx[x.item_index] = x; });
  A.slots.forEach((x, i) => assert(x.item_index === i + 1, 'full: item_index is 1-based position'));
  repeats.forEach(r => {
    const src = idx[r.repeat_of];
    assert(src && !src.is_repeat, 'full: repeat_of points to an original');
    assert(src.item.scenario_id === r.item.scenario_id, 'full: repeat is verbatim');
    assert(src.item_index <= 6, 'full: repeat source within positions 1-6, got ' + src.item_index);
    const gap = r.item_index - src.item_index;
    assert(gap >= battery.design.repeated_scenario_rule.min_gap_items, 'full: repeat gap ' + gap + ' < min_gap');
    gapHist[gap] = (gapHist[gap] || 0) + 1;
    repeatPosHist[r.item_index] = (repeatPosHist[r.item_index] || 0) + 1;
  });
  assert(repeats[0].repeat_of !== repeats[1].repeat_of, 'full: two repeats of the same item');

  // ---------------- short ----------------
  const B = BRE.buildAssignment(battery, 'short', seed);
  assert(B.slots.length === 5 && B.slots.every(x => !x.is_repeat), 'short: 5 presentations, no repeats');
  const sit = B.slots.map(x => x.item);
  sit.forEach(it => assert(universe.has(it.scenario_id), 'short: id not in universe'));
  assert(new Set(sit.map(i => i.loss_pct)).size === 5, 'short: 5 distinct losses');
  const st = count(sit, i => i.condition_type);
  assert(st.none === 1 && st.single === 2 && st.pair === 2, 'short: 1/2/2 types');
  const so = count(sit, i => i.question_order_id);
  assert(Object.values(so).sort().join() === '2,3', 'short: 3/2 orders');
  const ss = sit.filter(i => i.condition_type === 'single').map(i => i.context_tags[0]);
  assert(new Set(ss).size === 2, 'short: 2 distinct singles');
  const sp = sit.filter(i => i.condition_type === 'pair').map(i => i.context_tags);
  assert(new Set(sp.flat()).size === 4, 'short: pairs disjoint and cover all 4 contexts');
}

console.log('seeds tested:', N);
console.log('order balance unsatisfied after bounded attempts:', unsatisfied, '(max attempts used:', maxAttempts + ')');
console.log('repeat gap histogram:', JSON.stringify(gapHist));
console.log('repeat position histogram:', JSON.stringify(repeatPosHist));
const exSeed = '0123456789abcdef0123456789abcdef';
console.log('example full assignment (seed ' + exSeed + '):');
BRE.buildAssignment(battery, 'full', exSeed).slots.forEach(s => console.log('  ', s.item_index, s.is_repeat ? '(repeat of ' + s.repeat_of + ')' : '', s.item.scenario_id));
console.log('BALANCE TEST PASSED');
