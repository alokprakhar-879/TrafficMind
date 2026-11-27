const assert = require('node:assert/strict');
const T = require('./engine');
const V = require('./training');

// 1. Bellman arithmetic and the linear update on a real training trace.
const Q = {};
let trace;
for (let i = 0; i < 300; i++) {
  if (i === 299) trace = T.trainEpisode(Q, i, true, 300);
  else T.trainEpisode(Q, i, false, 300);
}
assert(Q.w.some((v) => v !== 0) && Q.w.every(Number.isFinite));
for (const d of trace.decisions) {
  // Moving each weight by alpha * TD error * feature changes Q by alpha * TD error * sum(feature²).
  const norm = d.features.reduce((s, x) => s + x * x, 0);
  assert(Math.abs(d.updated - (d.old + d.alpha * (d.target - d.old) * norm)) < 1e-6);
  const future = d.terminal ? 0 : 0.9 * Math.max(...d.nextValues);
  assert(Math.abs(d.target - (d.reward + future)) < 1e-8);
}

// 2. Simulation invariants for the agent, greedy and random.
let tests = 0;
for (const counts of [[40, 25, 60, 15], [654, 25, 456, 115], [0, 0, 0, 0], [0, 40, 0, 0]])
  for (const condition of ['normal', 'rain', 'roadwork', 'rush'])
    for (const amb of [false, true])
      for (const duration of [60, 300, 900]) {
        const sc = T.scenario({ counts, left: 20, right: 20, arrival: 0.05, condition, amb, distance: 650, ambDir: 0, ambMove: 2, duration }, 42);
        for (const eps of [0, 1]) {
          const r = T.run(sc, Q, 43, true, eps);
          assert.equal(r.events.length, duration);
          assert.equal(r.metrics.served + r.metrics.remaining, r.metrics.total);
          const ids = new Set();
          for (const e of r.events) {
            assert(e.q.every((n) => n >= 0));
            for (const c of e.depart) {
              assert.equal(e.signal, 'green');
              assert(T.stages[e.p].includes(c.l));
              assert(!ids.has(c.id));
              ids.add(c.id);
            }
          }
          for (const d of r.decisions) assert(Math.abs(d.reward - (2 * d.cleared - 0.08 * d.wait - 0.5 * d.waste - T.AMB_PENALTY * d.delay)) < 1e-8);
          tests++;
        }
      }
console.log(`Passed ${tests} runs and training Bellman checks.`);

// 3. Checkpointing, early stopping and the three learning phases.
const session = V.session(1000);
while (session.step());
const result = session.result();
assert(result.completed <= 1000);
assert.equal(result.best.wait, Math.min(...result.points.map((p) => p.wait)));
assert(Math.abs(V.score(result.Q, V.suite()) - result.best.wait) < 1e-8);
assert(result.points.every((p, i) => !i || p.best <= result.points[i - 1].best));
const [p1, p2, p3] = result.checkpoints;
assert(p1.episode === 0 && p1.eps === 1 && p2.episode <= p3.episode);
assert(p3.wait < p1.wait, 'trained agent must wait less than the random start');
assert(p2.wait <= p1.wait && p2.wait >= p3.wait);
assert(result.saved.every((x, i) => x.episode === 5 * (i + 1)));
console.log('Checkpoint and phase checks passed', result.completed, result.checkpoints.map((c) => `ep ${c.episode}: ${c.wait.toFixed(2)}s`).join(' → '));
