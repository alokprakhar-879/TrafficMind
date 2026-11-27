(function (root) {
  const T = typeof module !== 'undefined' ? require('./engine') : root.Traffic;
  const VALIDATE_EVERY = 25;
  const PATIENCE = 16; // stop after this many validations (400 episodes) with no improvement; best checkpoints came late in tests

  const PRESETS = [[40, 25, 60, 15], [60, 5, 10, 0], [100, 100, 100, 100], [20, 15, 5, 70]];
  const CONDITIONS = ['normal', 'rain', 'rush', 'roadwork'];

  // Fixed validation scenarios. Different `base` seeds give unseen test sets of the same families.
  function suite(base = 9100, n = 8) {
    return Array.from({ length: n }, (_, i) =>
      T.scenario(
        {
          counts: PRESETS[i % 4],
          left: 15 + (i % 3) * 5,
          right: 20,
          arrival: 0.04 + (i % 3) * 0.04,
          condition: CONDITIONS[Math.floor(i / 4) % 4],
          amb: i % 3 === 0,
          distance: 650,
          ambDir: i % 4,
          ambMove: i % 3,
          duration: 300,
        },
        base + i,
      ),
    );
  }

  // Rush-hour test set: 10 minutes, demand peaks mid-run, one road axis much busier (alternates N/S and E/W).
  function rushSuite(base = 36100, n = 8, arrival = 0.06) {
    return Array.from({ length: n }, (_, i) =>
      T.scenario(
        {
          counts: [10, 10, 5, 5],
          left: 20,
          right: 20,
          arrival,
          condition: 'rush',
          peak: 2,
          weights: i % 2 ? [0.6, 0.6, 1.6, 1.6] : [1.6, 1.6, 0.6, 0.6],
          amb: i % 4 === 0,
          distance: 900,
          ambDir: i % 4,
          ambMove: 1,
          duration: 600,
        },
        base + i,
      ),
    );
  }

  // Mean average wait (lower is better) and mean reward (higher is better). eps = 1 scores the untrained, random agent.
  function evaluate(Q, scenarios, eps = 0) {
    const ms = scenarios.map((sc, i) => T.run(sc, Q, 100 + i, false, eps).metrics);
    const mean = (k) => ms.reduce((t, m) => t + m[k], 0) / ms.length;
    return { wait: mean('avgWait'), reward: mean('reward') };
  }
  const score = (Q, scenarios, eps = 0) => evaluate(Q, scenarios, eps).wait;

  // No input model is reused; all learning and checkpoints exist only in this session.
  function session(limit = 1000) {
    const cases = suite();
    const working = {};
    const history = [];
    const traces = {};
    const start = { episode: 0, Q: {}, eps: 1, ...evaluate({}, cases, 1) };
    // Checkpoints are ranked by validation wait (lower is better).
    let top = start;
    let stale = 0;
    let i = 0;
    const points = [{ episode: 0, wait: start.wait, best: start.wait }];
    const snapshots = [start];
    const saved = []; // every 5th episode, so the app can pick Phase 2 per scenario
    const more = () => i < limit && stale < PATIENCE;

    function step() {
      if (!more()) return false;
      const capture = i === 0 || i === 19 || i === 99 || (i + 1) % 50 === 0 || i + 1 === limit;
      const trace = T.trainEpisode(working, i, capture, limit);
      i++;
      if (typeof trace === 'object') {
        traces[i] = trace;
        history.push(trace.metrics.reward);
      } else history.push(trace);
      if (i % 5 === 0) saved.push({ episode: i, Q: structuredClone(working), eps: 0 });
      if (i % VALIDATE_EVERY === 0 || i === limit) {
        const cp = { episode: i, Q: structuredClone(working), eps: 0, ...evaluate(working, cases) };
        snapshots.push(cp);
        if (cp.wait < top.wait - 1e-6) {
          top = cp;
          stale = 0;
        } else stale++;
        points.push({ episode: i, wait: cp.wait, best: top.wait });
      }
      return more();
    }

    function result() {
      // Three learning phases of the same agent: random start, halfway point, best checkpoint.
      // If nothing beat the random start, all three phases stay random.
      const half = (start.wait + top.wait) / 2;
      const mid = snapshots
        .filter((x) => x.episode > 0 && x.episode < top.episode && x.wait < start.wait && x.wait > top.wait)
        .sort((x, y) => Math.abs(x.wait - half) - Math.abs(y.wait - half))[0];
      return {
        Q: top.Q, history, points, start, best: top, bestEpisode: top.episode, completed: i, traces,
        checkpoints: [start, mid || top, top], saved, stopped: i < limit,
      };
    }

    return { step, result };
  }

  const api = { suite, rushSuite, evaluate, score, session };
  if (typeof module !== 'undefined') module.exports = api;
  else root.Training = api;
})(typeof window !== 'undefined' ? window : globalThis);
