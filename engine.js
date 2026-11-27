/* Shared discrete-time traffic environment and one reinforcement-learning signal agent. */
(function (root) {
  'use strict';

  const dirs = ['N', 'S', 'E', 'W'];
  const moves = ['L', 'S', 'R'];
  // Lane index = direction * 3 + movement (0 left, 1 straight, 2 right).
  const stages = [[0, 1, 3, 4], [2, 5], [6, 7, 9, 10], [8, 11]];
  const names = ['N/S straight + left', 'N/S protected right', 'E/W straight + left', 'E/W protected right'];
  const H = 300;
  const rates = { normal: 2, rain: 3, roadwork: 4, rush: 2 }; // seconds per departure (headway)
  const GREEN = 10; // RL decides every 10 s green; picking the same stage again extends it with no switch cost
  // Larger values cleared more test ambulances (23/24 at 50) but raised normal-traffic waits.
  const AMB_PENALTY = 0.8;
  const GAMMA = 0.9;
  const FEATURES = [
    'Bias', 'Stage queue / 20', 'Stage clearance time / 60', 'Stage oldest wait / 100',
    'Stage is current green', 'Current green × clearance / 60', 'Other queues / 50', 'Other oldest wait / 100',
    'Ambulance needs this stage', 'Ambulance needs another stage', 'Stage is empty',
  ];

  const sum = (a) => a.reduce((s, x) => s + x, 0);

  function rng(seed) {
    let x = seed >>> 0;
    return () => {
      x = (Math.imul(1664525, x) + 1013904223) >>> 0;
      return x / 4294967296;
    };
  }

  function scenario(cfg, seed) {
    const H = cfg.duration || 300;
    const r = rng(seed);
    const arrivals = Array.from({ length: H }, () => []);
    const initial = [];
    let id = 0;
    const car = (l, t) => ({ id: ++id, l, t });
    const lane = (d) => {
      const u = r() * 100;
      return d * 3 + (u < cfg.left ? 0 : u < 100 - cfg.right ? 1 : 2);
    };
    for (let d = 0; d < 4; d++) {
      for (let i = 0; i < cfg.counts[d]; i++) initial.push(car(lane(d), 0));
    }
    const rate = cfg.arrival * (cfg.condition === 'rush' ? 1.7 : 1);
    // Optional rush-hour profile: arrivals rise to (1 + peak)x mid-run; weights make some approaches busier.
    const peak = cfg.peak || 0, weights = cfg.weights || [1, 1, 1, 1];
    for (let t = 1; t < H; t++) {
      const surge = 1 + peak * Math.sin((Math.PI * t) / H);
      for (let d = 0; d < 4; d++) if (r() < rate * surge * weights[d]) arrivals[t].push(car(lane(d), t));
    }
    return { initial, arrivals, cfg };
  }

  function env(sc, record = false) {
    const q = Array.from({ length: 12 }, () => []);
    sc.initial.forEach((c) => q[c.l].push(c));
    return {
      sc, q, t: 0, prev: -1, served: 0, wait: 0, waste: 0, maxWait: 0,
      ambDone: null,
      ambDistance: sc.cfg.amb ? sc.cfg.distance : null,
      ambLane: sc.cfg.ambDir * 3 + sc.cfg.ambMove,
      ambReady: false,
      events: [], record, reward: 0, switches: 0,
    };
  }

  const demand = (e, p) => sum(stages[p].map((l) => e.q[l].length));
  const age = (e, p) => Math.max(0, ...stages[p].map((l) => (e.q[l].length ? e.t - e.q[l][0].t : 0)));
  const required = (e, p) => Math.max(...stages[p].map((l) => e.q[l].length)) * rates[e.sc.cfg.condition];
  const emergency = (e) => e.ambDistance !== null && e.ambDone === null && e.ambDistance < 500;
  const ambStage = (e) => stages.findIndex((x) => x.includes(e.ambLane));

  // What the agent observes about giving green to stage p. Raw measurements only; no hand-made priority.
  function features(e, p) {
    const others = [0, 1, 2, 3].filter((x) => x !== p);
    const keep = p === e.prev ? 1 : 0;
    const em = emergency(e);
    return [
      1, demand(e, p) / 20, required(e, p) / 60, age(e, p) / 100,
      keep, (keep * required(e, p)) / 60,
      sum(others.map((x) => demand(e, x))) / 50, Math.max(...others.map((x) => age(e, x))) / 100,
      em && ambStage(e) === p ? 1 : 0, em && ambStage(e) !== p ? 1 : 0, demand(e, p) === 0 ? 1 : 0,
    ];
  }

  // Linear Q-function: Q(s, p) = w · features(s, p). One weight per feature, shared by all stages.
  const weights = (Q) => Q.w || (Q.w = Array(FEATURES.length).fill(0));
  const dot = (Q, f) => sum(f.map((x, i) => x * weights(Q)[i]));
  const values = (Q, e) => [0, 1, 2, 3].map((p) => dot(Q, features(e, p)));

  function tick(e, p, signal, depart) {
    const t = e.t;
    if (t >= e.sc.arrivals.length) return;
    if (t > 0) e.sc.arrivals[t].forEach((c) => e.q[c.l].push(c));
    if (e.ambDistance !== null && e.ambDone === null) {
      e.ambDistance = Math.max(0, e.ambDistance - 12);
      if (e.ambDistance === 0 && !e.ambReady) {
        e.q[e.ambLane].push({ id: 0, l: e.ambLane, t, amb: true });
        e.ambReady = true;
      }
    }
    if (emergency(e)) e.emergencyWait = (e.emergencyWait || 0) + 1;
    e.wait += sum(e.q.map((a) => a.length));
    const active = signal === 'green' ? stages[p] : [];
    if (active.length && !active.some((l) => e.q[l].length)) e.waste++;
    for (const l of active) {
      if (t % rates[e.sc.cfg.condition] === 0 && e.q[l].length) {
        const c = e.q[l].shift();
        e.maxWait = Math.max(e.maxWait, t - c.t);
        if (c.amb) e.ambDone = t;
        else e.served++;
        depart.push(c);
      }
    }
    if (e.record) {
      e.events.push({
        t, p, signal,
        q: e.q.map((a) => a.length),
        depart: depart.slice(),
        ambDistance: e.ambDistance, ambDone: e.ambDone,
        served: e.served, wait: e.wait, waste: e.waste,
      });
    }
    e.t++;
  }

  function step(e, p, d) {
    const start = e.t;
    const w = e.wait, c = e.served, z = e.waste, ed = e.emergencyWait || 0;
    const live = () => e.t < e.sc.arrivals.length;
    if (e.prev !== -1 && e.prev !== p) {
      e.switches++;
      for (let i = 0; i < 3 && live(); i++) tick(e, e.prev, 'yellow', []);
      for (let i = 0; i < 2 && live(); i++) tick(e, p, 'red', []);
    }
    e.prev = p;
    for (let i = 0; i < d && live(); i++) tick(e, p, 'green', []);
    const delay = (e.emergencyWait || 0) - ed;
    const reward = 2 * (e.served - c) - 0.08 * (e.wait - w) - 0.5 * (e.waste - z) - AMB_PENALTY * delay;
    e.reward += reward;
    return { reward, start, end: e.t, p, d, cleared: e.served - c, wait: e.wait - w, waste: e.waste - z, delay };
  }

  // Epsilon-greedy over learned Q; ties go to the lowest stage.
  function policy(e, Q, r, eps = 0) {
    if (r() < eps) return { p: Math.floor(r() * 4), d: GREEN, reason: eps >= 1 ? 'Random exploration' : 'Exploration (random)' };
    const v = values(Q, e);
    return { p: v.indexOf(Math.max(...v)), d: GREEN, reason: 'Highest learned Q-value' };
  }

  // Group decisions into rounds: a round ends once every stage that had demand at its start has been served.
  function buildRounds(decisions, H) {
    const rounds = [];
    let start = 0, seen = new Set(), need = new Set();
    for (const d of decisions) {
      if (!need.size) need = new Set([0, 1, 2, 3].filter((p) => stages[p].some((l) => d.before[l] > 0)));
      if (!need.size) need.add(d.p);
      seen.add(d.p);
      const complete = [...need].every((p) => seen.has(p));
      if (complete || d.end === H) {
        const ds = decisions.filter((x) => x.start >= start && x.end <= d.end);
        rounds.push({
          start, end: d.end, complete, decisions: ds,
          cleared: sum(ds.map((x) => x.cleared)),
          wait: sum(ds.map((x) => x.wait)),
          waste: sum(ds.map((x) => x.waste)),
          reward: sum(ds.map((x) => x.reward)),
        });
        start = d.end;
        seen.clear();
        need.clear();
      }
    }
    return rounds;
  }

  function run(sc, Q, seed = 1, record = true, eps = 0, learn = false, alpha = 0.0001) {
    if (!learn) Q = { w: weights(Q).slice() };
    const H = sc.arrivals.length, e = env(sc, record), r = rng(seed), decisions = [];
    while (e.t < H) {
      const before = e.q.map((a) => a.length);
      const laneAges = e.q.map((a) => (a.length ? e.t - a[0].t : 0));
      const priorities = names.map((_, p) => ({ demand: demand(e, p), age: age(e, p), required: required(e, p) }));
      const current = values(Q, e);
      const choice = policy(e, Q, r, eps);
      const f = features(e, choice.p);
      const old = current[choice.p];
      const out = step(e, choice.p, choice.d);
      const nextValues = values(Q, e);
      // Bellman target: r + gamma * max over next stages; terminal future = 0.
      const target = out.reward + (e.t < H ? GAMMA * Math.max(...nextValues) : 0);
      // Semi-gradient update: each weight moves by alpha * TD error * its feature value.
      if (learn) f.forEach((x, i) => (Q.w[i] += alpha * (target - old) * x));
      if (record) {
        decisions.push({
          ...out, ...choice, features: f, old, target, before, laneAges, priorities,
          current, nextValues, alpha, terminal: e.t >= H,
          updated: dot(Q, f), weights: weights(Q).slice(),
        });
      }
    }
    const remaining = sum(e.q.map((a) => a.filter((c) => !c.amb).length));
    const total = sc.initial.length + sum(sc.arrivals.map((a) => a.length));
    const maxWait = Math.max(e.maxWait, ...e.q.flat().map((c) => H - c.t));
    return {
      rounds: buildRounds(decisions, H),
      events: e.events,
      decisions,
      initial: sc.initial.reduce((a, c) => (a[c.l]++, a), Array(12).fill(0)),
      cfg: sc.cfg,
      metrics: {
        total, served: e.served, remaining,
        avgWait: e.wait / (total + (e.ambReady ? 1 : 0) || 1),
        maxWait, waste: e.waste, ambDone: e.ambDone, reward: e.reward, switches: e.switches,
      },
    };
  }

  // One training episode on a random scenario; exploration and step size decay over the budget.
  function trainEpisode(Q, i, record = false, limit = 1000) {
    const r = rng(i + 8931);
    const cfg = {
      counts: Array.from({ length: 4 }, () => Math.floor(r() * (i % 5 === 0 ? 250 : 65))),
      left: 10 + Math.floor(r() * 25),
      right: 10 + Math.floor(r() * 30),
      condition: ['normal', 'rain', 'roadwork', 'rush'][Math.floor(r() * 4)],
      arrival: r() * 0.18,
      amb: r() < 0.25,
      distance: Math.floor(r() * 900),
      ambDir: Math.floor(r() * 4),
      ambMove: Math.floor(r() * 3),
    };
    if (r() < 0.3) {
      // Rush-hour peak with one busier axis, so the agent also meets changing demand.
      cfg.peak = 2;
      cfg.weights = r() < 0.5 ? [1.6, 1.6, 0.6, 0.6] : [0.6, 0.6, 1.6, 1.6];
    }
    const progress = i / Math.max(1, limit - 1);
    const eps = Math.max(0.05, 1 - progress);
    const alpha = Math.max(0.00005, 0.0005 - 0.00045 * progress); // larger steps made learning jumpy or diverge in tests
    const result = run(scenario(cfg, i + 731), Q, i + 91, record, eps, true, alpha);
    return record ? result : result.metrics.reward;
  }

  const api = { dirs, moves, stages, names, H, rates, GREEN, AMB_PENALTY, FEATURES, scenario, run, trainEpisode };
  if (typeof module !== 'undefined') module.exports = api;
  else root.Traffic = api;
})(typeof window !== 'undefined' ? window : globalThis);
