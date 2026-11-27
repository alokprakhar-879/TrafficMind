const T = require('./engine');
const V = require('./training');

const session = V.session(1000);
while (session.step());
const r = session.result();
console.log('Training completed:', r.completed, 'Best episode:', r.bestEpisode, 'Validation wait:', r.start.wait.toFixed(2), '→', r.best.wait.toFixed(2), 'reward:', r.start.reward.toFixed(0), '→', r.best.reward.toFixed(0));
for (const [name, set] of [['Held-out', V.suite(18100, 16)], ['Unseen', V.suite(27100, 16)], ['Rush-hour', V.rushSuite(36100, 16)]])
  console.log(`${name} mean wait by phase:`, r.checkpoints.map((c) => `ep ${c.episode} ${V.score(c.Q, set, c.eps).toFixed(2)}`).join(' → '));

// Ambulance set: 24 held-out cases, each with an ambulance at 300–1050 m.
const amb = Array.from({ length: 24 }, (_, i) => T.scenario({
  counts: [[40, 25, 60, 15], [60, 5, 10, 0], [100, 100, 100, 100], [20, 15, 5, 70]][i % 4], left: 20, right: 20, arrival: 0.04 + (i % 3) * 0.04,
  condition: ['normal', 'rain', 'rush', 'roadwork'][i % 4], amb: true, distance: 300 + (i % 6) * 150, ambDir: i % 4, ambMove: i % 3, duration: 300,
}, 55100 + i));
for (const c of r.checkpoints) {
  const waits = amb.map((sc, i) => [T.run(sc, c.Q, 100 + i, false, c.eps).metrics.ambDone, Math.ceil(sc.cfg.distance / 12)])
    .filter(([t]) => t !== null).map(([t, arrive]) => t - arrive);
  console.log(`Ambulance ep ${c.episode}: cleared ${waits.length}/24, mean wait at the light ${(waits.reduce((s, x) => s + x, 0) / waits.length).toFixed(1)}s`);
}
