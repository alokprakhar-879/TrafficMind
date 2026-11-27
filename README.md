# TrafficMind — Traffic AI Lab

## Links

- [Open the local app](http://127.0.0.1:3000/)
- [Run the app on GitHub Pages](https://alokprakhar-879.github.io/TrafficMind/)
- [View the source on GitHub](https://github.com/alokprakhar-879/TrafficMind)

Open `index.html`. You need no install, internet, API or backend.
1. Choose the traffic settings.
2. Click **Learn Model**.
3. Use the three tabs.

All three tabs show the **same reinforcement-learning agent** at different points in its training:

1. **Exploring (episode 0)**: the agent has no experience yet. Every choice is random (ε = 1).
2. **Learning (episode N)**: the validation checkpoint whose wait is about halfway between the random start and the best. If it is out of order on the scenario you are viewing, the app uses the checkpoint (saved every 5 episodes) closest to halfway on that scenario.
3. **Trained (episode M)**: the checkpoint with the lowest validation wait. It always picks the highest Q-value.

The phases are not separate algorithms. They are copies of the agent's weights, saved while it learns.

## The agent
- **Choices:** every 10 s it picks one of 4 protected stages.
  - Picking the same stage again extends the green, with no switch cost.
  - A change costs 3 s yellow and 2 s all-red.
- **Model:** linear Q-learning, Q(s, stage) = w · φ(s, stage). It uses 11 raw features:
  - stage queue, clearance time and oldest wait
  - current green
  - other queues and other oldest wait
  - ambulance on this stage or on another stage
  - empty stage
  - bias
- **Reward:** 2 × cleared − 0.08 × waiting − 0.5 × wasted green − 0.8 × ambulance delay.
- **Update:** w ← w + α [r + 0.9 max Q(s′) − Q(s)] φ.
  - Exploration ε falls from 1 to 0.05 over the training budget.
  - The step size α falls from 0.0005 to 0.00005.
- **No hand-written rules:** there is no ambulance override, starvation guard or duration cap. The agent learns everything from the reward.
- **Training data:** scenarios vary at random.
  - A quarter of the training episodes have an ambulance.
  - 30% have a rush-hour peak, with one axis busier than the others.

## Learning lifecycle
- Training starts from zero every time. Nothing is saved, not even in browser storage.
- **Budget:** 1000 episodes by default; you can choose 10–3000.
- **Validation:** every 25 episodes, on 8 fixed cases.
- **Early stop:** training stops after 400 episodes with no improvement.
- **Reset Learned Model** clears the weights, history and checkpoints. Refreshing the page also starts from zero.
- **Learning lab:** it replays real training decisions. It shows the features, weights, Q-values and the exact update.

## Reproducible tests
Run `node test.js` and `node benchmark.js`. They need Node, but the app does not.

**`test.js` checks:**
- 192 simulation runs. Each run checks vehicle conservation, non-negative queues, permitted movements and the reward.
- The Bellman target and the linear update, on a real training trace.
- That the best checkpoint really has the lowest validation wait.
- That the phases are in order: phase 3 waits less than phase 1, and phase 2 sits between them.

**`benchmark.js` results.** Training stopped at episode 550. The best checkpoint was episode 150 and the middle checkpoint was episode 125.

Mean wait, in seconds (lower is better):

| Test set (16 cases each) | Phase 1 (ep 0) | Phase 2 (ep 125) | Phase 3 (ep 150) |
|---|---|---|---|
| Held-out | 139.56 | 130.83 | 102.67 |
| Unseen | 141.41 | 132.15 | 105.33 |
| Rush hour (10 min peak, never used in validation) | 130.05 | 62.48 | 59.49 |

Default UI scenario (seed 42): 94.7 → 87.3 → 69.7 s.

Ambulance cases (24 held-out cases, each with an ambulance): the trained agent clears 18 of 24. This is the weak point. Raising the ambulance penalty to 50 cleared 23 of 24 in tests, but it raised normal-traffic waits (default scenario 84.0 s instead of 69.7 s), so it is not used.

This is an educational queue model, not a certified traffic controller. No result is forced.
