'use strict';
const T = Traffic;
const $ = (id) => document.getElementById(id);
const labels = ['Phase 1 · Exploring', 'Phase 2 · Learning', 'Phase 3 · Trained'];
const dirNames = ['North', 'South', 'East', 'West'];
const carColors = ['#65b8ff', '#f7c65a', '#f49f88', '#b6a1ff'];

let H = 300;
let history = [], validation = [], bestEpisode = 0, completed = 0;
let episodes = {}, checkpoints = [], trained = [], saved = [], labIndex = 0, trainedLimit = 0;
let trainingToken = 0, training = false;
let runs = [], phase = 0, pos = 0, playing = false, last = 0;

const f = (v) => Number(v).toFixed(2);
const isTrained = () => checkpoints.length > 0;
// The three tabs are checkpoints of one agent; before training all three are the random, untrained agent.
const phases = () => (isTrained() ? checkpoints : [0, 1, 2].map(() => ({ episode: 0, Q: {}, eps: 1 })));
const queueAt = (r, e) => (pos === 0 ? r.initial : e.q);

function table(headers, rows) {
  return '<div class="scroll"><table><thead><tr>' + headers.map((x) => `<th>${x}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r) => '<tr>' + r.map((x) => `<td>${x}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
}

function showError(message) {
  $('message').textContent = message;
  $('message').classList.add('error');
}

function setBusy(busy) {
  for (const id of ['train', 'run']) $(id).disabled = busy;
}

// ---------- Inputs ----------

$('counts').innerHTML = dirNames
  .map((n, i) => `<label>${n}<input id="n${i}" type="number" min="0" max="2000" value="${[40, 25, 60, 15][i]}"></label>`)
  .join('');

function config() {
  const val = (id) => ($(id).value.trim() === '' ? NaN : Number($(id).value));
  const counts = dirNames.map((_, i) => val('n' + i));
  const left = val('left'), right = val('right'), arrival = val('arrival'), distance = val('distance');
  if (
    [left, right, arrival, distance].some((v) => !Number.isFinite(v)) ||
    counts.some((n) => !Number.isInteger(n) || n < 0 || n > 2000) ||
    left < 0 || right < 0 || left + right > 100 ||
    arrival < 0 || arrival > 0.5 ||
    distance < 0 || distance > 3000
  ) throw Error('Check vehicle counts, percentages, arrival rate and distance.');
  const duration = val('duration');
  if (!Number.isInteger(duration) || duration < 60 || duration > 3600) throw Error('Duration must be 60–3600 seconds.');
  // "Rush hour peak" is rush headways plus a mid-run demand peak with a busier N/S axis.
  const peak = $('condition').value === 'peak';
  return {
    duration, counts, left, right, arrival, distance,
    amb: $('amb').checked, ambDir: val('ambDir'), ambMove: val('ambMove'),
    condition: peak ? 'rush' : $('condition').value,
    ...(peak && { peak: 2, weights: [1.6, 1.6, 0.6, 0.6] }),
  };
}

function readSeed() {
  const seed = Number($('seed').value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw Error('Use a non-negative integer seed.');
  return seed;
}

// ---------- Training ----------

function model() {
  const names = ['Exploring', 'Learning', 'Trained'];
  document.querySelectorAll('#tabs button').forEach((b, i) => (b.textContent = `0${i + 1} ${names[i]} (ep ${phases()[i].episode})`));
  $('model').textContent = isTrained() ? `Best checkpoint: episode ${bestEpisode} of ${completed} trained · ${T.FEATURES.length} learned weights` : 'Untrained — no learned data';
}

function train() {
  if (training) return;
  try {
    config();
    readSeed();
  } catch (e) {
    return showError(e.message);
  }
  const n = Number($('episodes').value);
  if (!Number.isInteger(n) || n < 10 || n > 3000) return showError('Enter 10–3000 episodes.');
  clearLearning();
  run();
  $('message').classList.remove('error');
  training = true;
  playing = false;
  const token = ++trainingToken;
  setBusy(true);
  const session = Training.session(n);

  function finish(result) {
    trainedLimit = n;
    episodes = result.traces;
    checkpoints = trained = result.checkpoints;
    saved = result.saved;
    training = false;
    setBusy(false);
    model();
    refreshLab();
    run();
    $('training').value = 1;
    $('message').textContent =
      `${result.stopped ? `Stopped early at ${completed} of ${n}: no lower validation wait in the last 400 episodes` : 'Training complete'}; ` +
      `${bestEpisode ? 'restored best checkpoint ' + bestEpisode : 'no checkpoint beat the random start'}. ` +
      `Validation wait: random ${result.start.wait.toFixed(2)}s → trained ${result.best.wait.toFixed(2)}s; reward ${result.start.reward.toFixed(0)} → ${result.best.reward.toFixed(0)}.`;
  }

  // Train in small batches so the page stays responsive; a newer token cancels this session.
  function batch() {
    if (token !== trainingToken) return;
    try {
      let more = true;
      for (let j = 0; j < 5 && more; j++) more = session.step();
      const result = session.result();
      ({ history, points: validation, bestEpisode, completed } = result);
      $('training').value = completed / n;
      $('message').textContent = `Training ${completed}/${n} · best validation wait ${result.best.wait.toFixed(2)}s`;
      chart();
      if (more) setTimeout(batch, 0);
      else finish(result);
    } catch (e) {
      training = false;
      setBusy(false);
      showError(e.message);
    }
  }
  batch();
}

function clearLearning() {
  trainingToken++;
  training = false;
  history = [];
  validation = [];
  bestEpisode = 0;
  completed = 0;
  episodes = {};
  checkpoints = trained = saved = [];
  labIndex = 0;
  phase = 0;
  pos = 0;
  playing = false;
  runs = [];
  setBusy(false);
  $('validationNote').textContent = '';
  $('trainingNote').textContent = '';
  $('training').value = 0;
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', Number(b.dataset.i) === 0));
  model();
  chart();
  refreshLab();
  $('checkpointResults').textContent = 'No learned checkpoints. Press Learn Model.';
  for (const id of ['metrics', 'bars', 'queues', 'status', 'priority', 'roundCompare', 'math', 'decision', 'roundStatus', 'timeline'])
    $(id).textContent = '';
  $('clock').textContent = '0 / ' + H + ' s';
  $('seek').value = 0;
  document.body.dataset.phase = 0;
  $('play').textContent = '▶ Play';
  $('phaseBanner').textContent = 'Untrained agent — reset to beginning';
  for (let j = 0; j < 3; j++) $('miniStats' + j).textContent = '';
}

// ---------- Comparison run ----------

// Early checkpoints swing a lot, so a middle one picked on the validation average can do worse than random here.
// Phase 2 keeps the validation pick when it sits between Phases 1 and 3 here; otherwise it is the earlier
// checkpoint closest to halfway on this scenario, preferring ones that sit between.
function pickMiddle(sc, seed) {
  const [start, validMid, top] = trained;
  const wait = (cp) => T.run(sc, cp.Q, seed + 1, false, cp.eps).metrics.avgWait;
  const hi = wait(start), lo = wait(top), half = (hi + lo) / 2;
  const between = (w) => w < hi && w > lo;
  checkpoints = trained;
  if (between(wait(validMid))) return model();
  const cost = ([, w]) => (between(w) ? 0 : 1e6) + Math.abs(w - half);
  const [mid] = saved
    .filter((x) => x.episode < top.episode)
    .map((x) => [x, wait(x)])
    .sort((a, b) => cost(a) - cost(b))[0] ?? [top];
  checkpoints = [start, { ...mid, ...Training.evaluate(mid.Q, Training.suite()) }, top];
  model();
}

function run() {
  try {
    $('message').classList.remove('error');
    const cfg = config(), seed = readSeed();
    const sc = T.scenario(cfg, seed);
    H = cfg.duration;
    $('seek').max = H;
    if (isTrained()) pickMiddle(sc, seed);
    runs = phases().map((cp) => T.run(sc, cp.Q, seed + 1, true, cp.eps));
    pos = 0;
    playing = false;
    metrics();
    update();
    showCheckpoints(sc, seed);
    $('message').textContent = isTrained() ? `Phases ready. Training ran ${completed} episodes; the best was episode ${bestEpisode}.` : 'All three tabs show the untrained agent. Press Learn Model.';
  } catch (e) {
    showError(e.message);
  }
}

function showCheckpoints(sc, seed) {
  if (!checkpoints.length) {
    $('checkpointResults').textContent = 'No learned checkpoints. Press Learn Model.';
    return;
  }
  const rows = checkpoints.map((cp, i) => {
    const m = T.run(sc, cp.Q, seed + 1, false, cp.eps).metrics;
    return [labels[i], cp.episode, f(cp.reward), f(cp.wait), f(m.avgWait), m.served, m.remaining, m.waste];
  });
  $('checkpointResults').innerHTML = table(
    ['Learning stage', 'Episode', 'Validation reward', 'Validation wait (s)', 'Current scenario wait (s)', 'Cleared', 'Remaining', 'Wasted green (s)'],
    rows,
  );
}

// ---------- Charts ----------

function clearCanvas(canvas, background) {
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (background) {
    g.fillStyle = background;
    g.fillRect(0, 0, canvas.width, canvas.height);
  }
  return g;
}

function polyline(g, points, color) {
  g.strokeStyle = color;
  g.lineWidth = 2;
  g.beginPath();
  points.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.stroke();
}

function chart() {
  const c = $('chart'), w = c.width, h = c.height;
  const g = clearCanvas(c, '#f3f7fa');
  if (!validation.length) {
    clearCanvas($('trainingChart'));
    g.fillStyle = '#65788b';
    g.fillText('Train to compare validation checkpoints.', 15, 30);
    return;
  }
  const values = validation.flatMap((v) => [v.wait, v.best]);
  const lo = Math.min(...values) - 1, hi = Math.max(...values) + 1, max = Math.max(1, completed);
  const x = (ep) => 45 + (ep / max) * (w - 55);
  const y = (v) => 12 + ((hi - v) / (hi - lo || 1)) * (h - 30);
  for (const [key, color] of [['wait', '#087f83'], ['best', '#15803d']])
    polyline(g, validation.map((v) => [x(v.episode), y(v[key])]), color);
  g.fillStyle = '#506174';
  g.fillText(hi.toFixed(0), 4, 15);
  g.fillText(lo.toFixed(0), 4, h - 14);
  g.fillText(`${completed} episodes`, w - 100, h - 3);
  $('validationNote').textContent = `Best episode ${bestEpisode} · best validation wait ${validation.at(-1).best.toFixed(2)}s`;
  rewardChart();
}

// Training reward, averaged over blocks of 10 episodes (training scenarios vary, so this is noisy).
function rewardChart() {
  const rewards = history.filter(Number.isFinite);
  const c = $('trainingChart'), w = c.width, h = c.height;
  const g = clearCanvas(c, '#f3f7fa');
  const means = [];
  for (let i = 0; i < rewards.length; i += 10) {
    const b = rewards.slice(i, i + 10);
    means.push(b.reduce((s, v) => s + v, 0) / b.length);
  }
  if (means.length) {
    const lo = Math.min(...means), hi = Math.max(...means);
    polyline(g, means.map((v, i) => [45 + (i / Math.max(1, means.length - 1)) * (w - 55), 12 + ((hi - v) / (hi - lo || 1)) * (h - 30)]), '#087f83');
    g.fillStyle = '#506174';
    g.fillText(hi.toFixed(0), 4, 15);
    g.fillText(lo.toFixed(0), 4, h - 14);
  }
  const recent = rewards.slice(-10);
  $('trainingNote').textContent = rewards.length
    ? `Training reward (last 10 mean): ${(recent.reduce((s, x) => s + x, 0) / recent.length).toFixed(2)}. Training scenarios vary; validation is comparable.`
    : '';
}

// ---------- Metrics ----------

const ambulance = (m) => (m.ambDone === null ? 'Not cleared' : `${m.ambDone}s`);

function metrics() {
  for (const r of runs) r.metrics.throughput = (60 * r.metrics.served) / H;
  const rows = [
    ['Arrived', 'total'], ['Cleared', 'served'], ['Remaining', 'remaining'],
    ['Observed avg wait (s)', 'avgWait'], ['Max observed wait (s)', 'maxWait'], ['Wasted green (s)', 'waste'],
    ['Throughput (vehicles/min)', 'throughput'],
    ['Signal changes', 'switches'], ['Reward', 'reward'],
  ];
  const cell = (v) => `<td>${Number.isInteger(v) ? v : v.toFixed(2)}</td>`;
  $('metrics').innerHTML =
    '<thead><tr><th>Metric</th>' + labels.map((x) => `<th>${x}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(([n, k]) => `<tr><td>${n}</td>${runs.map((r) => cell(r.metrics[k])).join('')}</tr>`).join('') +
    `<tr><td>Ambulance crossing (elapsed)</td>${runs.map((r) => `<td>${$('amb').checked ? ambulance(r.metrics) : 'Disabled'}</td>`).join('')}</tr></tbody>`;

  const max = Math.max(1, ...runs.map((r) => r.metrics.avgWait));
  $('bars').innerHTML =
    '<h2>Observed average waiting time · lower is better</h2><p>The same agent at three points in its training, on this scenario. If the halfway validation checkpoint is out of order here, Phase 2 is the saved checkpoint closest to halfway on this scenario.</p>' +
    runs.map((r, i) =>
      `<div class="barrow"><span>${labels[i]}</span><div class="track"><div class="fill" style="width:${(100 * r.metrics.avgWait) / max}%;background:${['#b91c1c', '#b8860b', '#15803d'][i]}"></div></div><b>${r.metrics.avgWait.toFixed(1)} s</b></div>`,
    ).join('');
}

// ---------- Live view ----------

function signalLeft(r, i) {
  const e = r.events[i];
  let j = i + 1;
  while (j < r.events.length && r.events[j].signal === e.signal && r.events[j].p === e.p) j++;
  return j - i;
}

function update() {
  if (!runs.length) return;
  const r = runs[phase], i = Math.min(H - 1, Math.floor(pos)), e = r.events[i];
  const d = r.decisions.find((x) => x.start <= i && i < x.end);
  const q = queueAt(r, e);
  $('seek').value = pos;
  document.body.dataset.phase = phase;
  const cp = phases()[phase];
  const about = !isTrained() ? 'Untrained agent: every choice is random — press Learn Model'
    : cp.eps >= 1 ? 'Episode 0: no experience yet, every choice is random'
    : phase === 1 ? `Episode ${cp.episode}: partly learned weights, still makes mistakes`
    : `Episode ${cp.episode}: best learned weights, always picks the highest Q-value`;
  $('phaseBanner').innerHTML = `<strong>${labels[phase]}</strong> · ${about}`;
  $('clock').textContent = `${Math.floor(pos)} / ${H} s`;
  $('play').textContent = playing ? 'Ⅱ Pause' : '▶ Play';
  $('queues').innerHTML = dirNames
    .map((n, j) => `<tr><td>${n}</td>${q.slice(j * 3, j * 3 + 3).map((v) => `<td>${v}</td>`).join('')}</tr>`)
    .join('');
  const amb = e.ambDistance === null ? 'Disabled' : e.ambDone !== null ? 'Cleared' : `${e.ambDistance} m`;
  $('status').innerHTML = [
    ['Signal', e.signal.toUpperCase()],
    ['Stage', T.names[e.p]],
    ['Signal countdown', `${signalLeft(r, i)}s`],
    ['Vehicles waiting', q.reduce((s, v) => s + v, 0)],
    ['Ambulance', amb],
  ].map(([a, b]) => `<div class="stat"><span>${a}</span><b>${b}</b></div>`).join('');
  renderExtras(d, i);
  if (!d) {
    $('decision').innerHTML = '';
    return;
  }
  $('decision').innerHTML = `<b>${d.reason}</b><br>Action: ${T.names[d.p]}, ${d.d}s green<br>` +
    `Q-values: ${d.current.map((v, p) => `${T.names[p]} ${f(v)}`).join(' · ')}<br>` +
    'Reward available after round completion. Bellman target shown in completed round details.<br>Evaluation is frozen (no training updates).';
}

function renderExtras(d, i) {
  const r = runs[phase];
  const done = r.rounds.filter((x) => x.end <= pos);
  const round = done.at(-1), next = r.rounds[done.length];
  renderPriority(next?.decisions[0] || round?.decisions[0] || d);
  $('roundStatus').textContent = `Round ${Math.min(done.length + 1, r.rounds.length)} / ${r.rounds.length} · ${done.filter((x) => x.complete).length} complete`;
  renderRoundMath(round);
  $('roundCompare').innerHTML = table(
    ['Phase', 'Round interval', 'Cleared', 'Wait (veh-s)', 'Waste', 'Reward'],
    runs.map((rr, j) => {
      const x = rr.rounds[done.length - 1];
      return [labels[j], x ? `${x.start}–${x.end}s${x.complete ? '' : ' (partial)'}` : '—', x?.cleared ?? '—', x?.wait ?? '—', x?.waste ?? '—', x ? f(x.reward) : '—'];
    }),
  );
  for (let j = 0; j < 3; j++) {
    const e = runs[j].events[i];
    const waiting = queueAt(runs[j], e).reduce((a, b) => a + b, 0);
    $('miniStats' + j).textContent = `${waiting} waiting · ${pos === 0 ? 0 : e.served} cleared · ${pos === 0 ? 0 : e.waste}s idle`;
  }
  const stageColors = ['#3579a8', '#a86b25', '#28836b', '#8464ac'];
  $('timeline').innerHTML = runs.map((rr, j) =>
    `<div class="tl"><b>${labels[j]}</b><div class="segments">${rr.decisions.map((x) =>
      `<span title="${T.names[x.p]} ${x.start}–${x.end}s" style="width:${(100 * (x.end - x.start)) / H}%;background:${stageColors[x.p]}"></span>`,
    ).join('')}<i style="left:${(100 * pos) / H}%"></i></div></div>`,
  ).join('');
}

// Traffic snapshot at the start of the current round, with the agent's learned value for each stage.
function renderPriority(first) {
  const rows = first.priorities.map((p, j) => [T.names[j], p.demand, p.age + 's', p.required + 's', f(first.current[j])]);
  const byDirection = dirNames.map((n, j) => {
    const q = first.before.slice(j * 3, j * 3 + 3);
    return [n, ...q, Math.max(...first.laneAges.slice(j * 3, j * 3 + 3)) + 's', q.reduce((x, y) => x + y, 0)];
  });
  $('priority').innerHTML =
    '<p>Snapshot at ' + first.start + 's (round start). There is no hand-written priority rule: the agent turns these numbers into features and picks the stage with the highest learned Q-value.</p>' +
    table(['Stage', 'Vehicles', 'Oldest wait', 'Clearance need', 'Learned Q'], rows) +
    table(['Direction', 'Left', 'Straight', 'Right', 'Oldest wait', 'Total'], byDirection);
}

function renderRoundMath(round) {
  if (!round) {
    $('math').innerHTML = '<p>Press Next round or play to finish the first round. No future outcomes are shown here.</p>';
    return;
  }
  $('math').innerHTML =
    '<p>Last completed interval: ' + round.start + '–' + round.end + 's ' + (round.complete ? '(full round)' : '(partial at horizon)') + '</p>' +
    table(['Metric', 'Value'], [
      ['Cleared', round.cleared], ['Queue waiting (vehicle-seconds)', round.wait], ['Wasted green (seconds)', round.waste],
      ['Total reward', f(round.reward)], ['Signal decisions', round.decisions.length],
    ]) +
    table(['Stage', 'Planned green', 'Cleared', 'Reward'], round.decisions.map((x) => [T.names[x.p], x.d + 's', x.cleared, f(x.reward)])) +
    '<details><summary>Individual signal calculations and Q-values</summary>' +
    round.decisions.map((x) => '<h3>' + T.names[x.p] + '</h3>' + mathView(x)).join('') + '</details>';
}

function mathView(d, learning = false) {
  if (!d) return 'No decision yet.';
  let html = table(['Quantity', 'Actual calculation'], [
    ['Stage', T.names[d.p]],
    ['Planned green', d.d + ' s'],
    ['Decision interval', d.end - d.start + ' s (includes transition)'],
    ['Cleared reward', `2 × ${d.cleared} = ${f(2 * d.cleared)}`],
    ['Waiting penalty', `−0.08 × ${d.wait} = ${f(-0.08 * d.wait)}`],
    ['Idle penalty', `−0.5 × ${d.waste} = ${f(-0.5 * d.waste)}`],
    ['Emergency penalty', `−${T.AMB_PENALTY} × ${d.delay} = ${f(-T.AMB_PENALTY * d.delay)}`],
    ['Reward', f(d.reward)],
  ]);
  html += table(['Stage', 'Queue', 'Oldest wait (s)', 'Clearance need (s)'], d.priorities.map((p, i) => [T.names[i], p.demand, p.age, p.required]));
  html += table(['Stage', 'Q(s, stage)', "Q(s′, stage)"], d.current.map((v, p) => [
    T.names[p] + (p === d.p ? ' ← chosen' : ''), f(v), d.terminal ? 'terminal' : f(d.nextValues[p]),
  ]));
  html += '<p>Q = sum of feature × weight for the chosen stage.</p>' +
    table(['Feature', 'Value', 'Weight', 'Feature × weight'], T.FEATURES.map((n, i) => [n, f(d.features[i]), f(d.weights[i]), f(d.features[i] * d.weights[i])]));
  return html + table(['Bellman term', 'Value'], [
    ['Step size α', learning ? d.alpha.toFixed(6) : 'none (frozen evaluation)'],
    ['Old Q', f(d.old)],
    ['Reward', f(d.reward)],
    ['Future (0.9 × best next Q)', f(d.target - d.reward)],
    ['Target', f(d.target)],
    ['TD error', f(d.target - d.old)],
    ['New Q', learning ? f(d.updated) + ' (each weight += α × TD error × its feature)' : 'Frozen evaluation; no update'],
  ]);
}

// ---------- Intersection drawing ----------

const roadCtx = $('road').getContext('2d');
const C = 310, R = 100;
const rot = [0, Math.PI, Math.PI / 2, -Math.PI / 2];

// Rotate a point from approach d's local frame (approach from the top) into canvas space.
function point(d, x, y) {
  const a = rot[d];
  return [C + x * Math.cos(a) - y * Math.sin(a), C + x * Math.sin(a) + y * Math.cos(a)];
}

function line(g, x, y, X, Y, color, width = 1) {
  g.strokeStyle = color;
  g.lineWidth = width;
  g.beginPath();
  g.moveTo(x, y);
  g.lineTo(X, Y);
  g.stroke();
}

function car(g, x, y, a, color, amb = false) {
  g.save();
  g.translate(x, y);
  g.rotate(a);
  g.fillStyle = color;
  g.fillRect(-7, -12, 14, 24);
  g.fillStyle = '#e5eef4';
  g.fillRect(-5, -7, 10, 6);
  if (amb) {
    g.fillStyle = '#ef4444';
    g.fillRect(-2, 1, 4, 8);
    g.fillRect(-5, 4, 10, 3);
  }
  g.restore();
}

// Position and heading of a departing car (quadratic Bezier through the junction), u in [0, 1].
function path(d, m, u) {
  const x = 28 + 24 * (2 - m), y = -R - 15;
  const end = m === 0 ? [180, -76] : m === 2 ? [-180, 28] : [x, 150];
  const control = m === 0 ? [x, -76] : m === 2 ? [x, 28] : [x, 30];
  const px = (1 - u) * (1 - u) * x + 2 * (1 - u) * u * control[0] + u * u * end[0];
  const py = (1 - u) * (1 - u) * y + 2 * (1 - u) * u * control[1] + u * u * end[1];
  const vx = 2 * (1 - u) * (control[0] - x) + 2 * u * (end[0] - control[0]);
  const vy = 2 * (1 - u) * (control[1] - y) + 2 * u * (end[1] - control[1]);
  return [...point(d, px, py), Math.atan2(vy, vx) - Math.PI / 2 + rot[d]];
}

function drawRoad(g) {
  g.fillStyle = '#dce9d8';
  g.fillRect(0, 0, 620, 620);
  g.fillStyle = '#344152';
  g.fillRect(C - R, 0, 2 * R, 620);
  g.fillRect(0, C - R, 620, 2 * R);
  g.fillStyle = '#485565';
  g.fillRect(C - R, C - R, R * 2, R * 2);
  for (let d = 0; d < 4; d++) {
    g.save();
    g.translate(C, C);
    g.rotate(rot[d]);
    line(g, 0, -310, 0, -R, '#f5c75a', 2);
    g.setLineDash([8, 8]);
    for (const x of [16, 40, 64, -16, -40, -64]) line(g, x, -310, x, -R, '#8693a0');
    g.setLineDash([]);
    line(g, 3, -R - 2, 96, -R - 2, 'white', 3);
    g.fillStyle = '#fff';
    g.font = 'bold 12px system-ui';
    g.fillText('R    S    L', 16, -R - 9);
    g.restore();
  }
}

function draw(g = roadCtx, which = phase) {
  drawRoad(g);
  const r = runs[which];
  const e = r ? r.events[Math.min(H - 1, Math.floor(pos))] : null;
  if (!e) return;
  const q = queueAt(r, e);
  // Signal heads and queued cars per lane.
  for (let d = 0; d < 4; d++) {
    for (let m = 0; m < 3; m++) {
      const l = d * 3 + m, x = 28 + 24 * (2 - m), count = q[l], green = T.stages[e.p].includes(l);
      g.fillStyle = green && e.signal === 'green' ? '#35e58a' : green && e.signal === 'yellow' ? '#f9bf35' : '#f45b59';
      g.beginPath();
      g.arc(...point(d, x, -R + 7), 7, 0, Math.PI * 2);
      g.fill();
      for (let j = 0; j < Math.min(count, 6); j++) car(g, ...point(d, x, -R - 40 - j * 27), rot[d], carColors[d]);
      const label = point(d, x, -292);
      g.fillStyle = 'white';
      g.font = 'bold 11px system-ui';
      g.fillText(count, label[0] - 5, label[1] + 4);
    }
  }
  // Cars that departed in the last few seconds, animated across the junction.
  for (let k = Math.max(0, Math.floor(pos) - 3); k <= Math.min(H - 1, Math.floor(pos)); k++) {
    for (const v of r.events[k].depart) {
      const u = (pos - k) / 1.8;
      if (pos > 0 && u >= 0 && u <= 1) {
        const dir = Math.floor(v.l / 3);
        car(g, ...path(dir, v.l % 3, u), v.amb ? '#fff' : carColors[dir], v.amb);
      }
    }
  }
  g.fillStyle = '#10243b';
  g.font = 'bold 12px system-ui';
  g.fillText('N', 15, 22);
  g.fillText('S', 15, 605);
  g.fillText('W', 15, 304);
  g.fillText('E', 600, 304);
  if (e.ambDistance !== null && e.ambDone === null && e.ambDistance > 0) {
    g.fillStyle = '#fff';
    g.fillRect(10, 35, 157, 23);
    g.fillStyle = '#b91c1c';
    g.fillText(`AMB approaching: ${e.ambDistance}m`, 15, 51);
  }
}

function frame(now) {
  if (playing && runs.length) {
    const old = Math.floor(pos);
    pos = Math.min(H, pos + Math.min(0.1, (now - last) / 1000) * Number($('speed').value));
    if (pos >= H) playing = false;
    if (Math.floor(pos) !== old || !playing) update();
  }
  last = now;
  draw();
  if (runs.length && $('compareDetails').open) for (let j = 0; j < 3; j++) draw($('mini' + j).getContext('2d'), j);
  requestAnimationFrame(frame);
}

// Jump to the previous/next round boundary of the current controller.
function decisionNav(dir) {
  if (!runs.length) return 0;
  const ends = [0, ...runs[phase].rounds.map((r) => r.end)];
  return dir > 0 ? ends.find((t) => t > pos) ?? H : [...ends].reverse().find((t) => t < pos) ?? 0;
}

// ---------- Learning laboratory ----------

function refreshLab() {
  $('episodePick').innerHTML = Object.keys(episodes)
    .map((k) => `<option value="${k}">Episode ${k}${Number(k) === bestEpisode ? ' — best checkpoint' : ''}</option>`)
    .join('');
  labIndex = 0;
  labRender();
}

function labRender() {
  const ep = episodes[$('episodePick').value];
  if (!ep) {
    $('labMath').textContent = 'Press Learn Model to capture real episode traces.';
    $('labStep').textContent = '';
    $('labPrev').disabled = true;
    $('labNext').disabled = true;
    return;
  }
  const d = ep.decisions[labIndex];
  // Traces are keyed by episode number (1-based); training used index episode - 1.
  const eps = Math.max(0.05, 1 - (Number($('episodePick').value) - 1) / Math.max(1, trainedLimit - 1));
  $('labStep').textContent = `Decision ${labIndex + 1}/${ep.decisions.length} · Episode reward ${f(ep.metrics.reward)}`;
  $('labMath').innerHTML =
    '<p><b>Exploration ε:</b> ' + eps.toFixed(3) + ' · ' + d.reason + '</p>' +
    table(['Approach', 'Left', 'Straight', 'Right'], dirNames.map((n, j) => [n, ...d.before.slice(j * 3, j * 3 + 3)])) +
    mathView(d, true);
  $('labPrev').disabled = labIndex === 0;
  $('labNext').disabled = labIndex === ep.decisions.length - 1;
}

// ---------- Wiring ----------

function exportCsv() {
  if (!runs.length) return;
  const keys = Object.keys(runs[0].metrics);
  const csv = 'phase,' + keys.join(',') + '\n' + runs.map((r, i) => labels[i] + ',' + keys.map((k) => r.metrics[k] ?? '').join(',')).join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'traffic-comparison.csv';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function seekTo(value) {
  playing = false;
  pos = value;
  update();
}

const FORMULAS = [
  ['RL action', 'Pick one of 4 stages for 10 s of green; same stage again = extend with no switch cost'],
  ['Q-value', 'Q(s, stage) = Σ feature × weight; 11 features: ' + T.FEATURES.join(', ')],
  ['Policy', 'Training: random stage with probability ε (1 → 0.05), else highest Q. Evaluation: highest Q'],
  ['Headway', '2s normal/rush, 3s rain, 4s roadwork'],
  ['Queue', 'previous + arrivals − departures'],
  ['Waiting', 'sum of queued vehicles each second (vehicle-seconds)'],
  ['Average observed wait', 'total queue waiting / arrivals (including ambulance if queued)'],
  ['Wasted green', 'one second when no eligible vehicle is queued'],
  ['Reward', `2C − 0.08W − 0.5I − ${T.AMB_PENALTY}E`],
  ['Q update', 'TD error = reward + 0.9 × max next Q − old Q (terminal future = 0); each weight += α × TD error × feature; α 0.0005 → 0.00005'],
  ['Ambulance', `12 m/s approach; no override — the agent must learn to give it green (feature + ${T.AMB_PENALTY} penalty per second below 500 m)`],
  ['Clearance', '3s yellow + 2s all-red when stage changes'],
  ['Throughput', '60 × cleared / elapsed seconds'],
];

function init() {
  $('mini').innerHTML = labels
    .map((n, j) => `<div class="miniCard"><b>${n}</b><canvas id="mini${j}" width="620" height="620"></canvas><small id="miniStats${j}"></small></div>`)
    .join('');
  $('formulas').innerHTML = table(['Metric', 'Implementation'], FORMULAS);

  $('train').onclick = train;
  $('run').onclick = run;
  $('reset').onclick = () => {
    clearLearning();
    run();
    if (runs.length) $('message').textContent = 'All learning data erased. Phase 1 restarted. Press Learn Model to start from zero.';
  };
  $('tabs').onclick = (e) => {
    if (e.target.dataset.i === undefined) return;
    phase = Number(e.target.dataset.i);
    document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', Number(b.dataset.i) === phase));
    seekTo(0);
  };
  $('play').onclick = () => {
    if (!runs.length || training) return;
    if (pos >= H) pos = 0;
    playing = !playing;
    update();
  };
  $('back').onclick = () => seekTo(decisionNav(-1));
  $('next').onclick = () => seekTo(decisionNav(1));
  $('seek').oninput = () => seekTo(Number($('seek').value));
  $('restart').onclick = () => seekTo(0);
  $('end').onclick = () => seekTo(H);
  $('export').onclick = exportCsv;
  $('episodePick').onchange = () => {
    labIndex = 0;
    labRender();
  };
  $('labPrev').onclick = () => {
    labIndex = Math.max(0, labIndex - 1);
    labRender();
  };
  $('labNext').onclick = () => {
    const ep = episodes[$('episodePick').value];
    if (ep) labIndex = Math.min(ep.decisions.length - 1, labIndex + 1);
    labRender();
  };

  refreshLab();
  model();
  chart();
  run();
  requestAnimationFrame(frame);
}

init();
