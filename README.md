# Underwater Neuroevolution (Rust, 3D)

A 3D tank of fish, jellyfish and sharks, each steered by its own small neural network.
Nothing is scripted and there is no gradient training. Behavior emerges through **neuroevolution**:
creatures that eat well live long enough to reproduce, and their children inherit a mutated
(and sometimes crossed-over) copy of the parents' brains.

## Run

```bash
cargo run --release
```

The first build downloads and compiles macroquad, which takes about a minute.

## The ecosystem

| Species | Eats | Danger | Notes |
|---|---|---|---|
| Plankton (green specks) | — | everything | Spawns near the sunlit surface, drifts with the current |
| Fish (orange) | plankton | sharks, jellyfish | Fast and agile, breeds quickly |
| Jellyfish (pink, translucent) | plankton | — | Slow drifters whose tentacles sting fish and sharks |
| Shark (grey) | fish | jellyfish | Fast apex predator, digests between kills |

Dead creatures that weren't eaten sink back into the food web as plankton ("marine snow").

## The brains

Each creature has a feed-forward network: **19 inputs, 14 hidden (tanh), 5 outputs**.

- **Inputs**: direction and closeness of the nearest food, threat and same-species neighbor;
  its own velocity, position in the tank and energy; an internal clock; a memory value; a bias.
- **Outputs**: a 3D thrust direction, a throttle, and a memory value that is fed back next tick.

**Evolution:**
- A creature with enough energy, age and cooldown reproduces. If another of its species is
  nearby, the child gets a neuron-wise crossover of both brains; otherwise a clone. Then ~12%
  of its weights are mutated with gaussian noise.
- Every species keeps a **hall of fame** of its fittest genomes (meals, children, lifespan).
  When a population falls below its minimum, it's re-seeded from those elites instead of
  going extinct.
- Lineages carry a heritable color tint, so successful families become visible as color clusters.

Starting from random brains, the creatures at first mostly drift and starve. Within roughly 10–20
simulated minutes, fish learn to chase plankton and the populations take off. Press **T** for
turbo to get there in seconds.

## Controls

| Input | Action |
|---|---|
| Left-drag / wheel / WASD QE | Orbit / zoom / pan the camera |
| Click | Select a creature: shows its stats, live neural activity, and lines to what it sees (green food, red threat) |
| F | Follow the selected creature |
| B / N / Esc | Select the fittest / a random creature / deselect |
| Space | Pause |
| + / - | Simulation speed (0.25x to 16x) |
| T | Turbo: run as many steps as possible |
| [ / ] | Lower / raise the plankton spawn rate |
| G / H | Toggle the population graph / help |
| R | Restart with fresh random brains |

## Code layout

- `src/brain.rs`: neural network, mutation and crossover
- `src/creature.rs`: species biology (all tunable numbers live in `PARAMS`)
- `src/world.rs`: the simulation step (sensing, thinking, physics, eating, birth/death, hall of fame) and a spatial grid
- `src/render.rs`: CPU-lit, fogged meshes for creatures, sand, rocks, seaweed, bubbles, water surface
- `src/hud.rs`: stats, population graph, brain visualizer
- `src/main.rs`: window, camera, input, main loop

## Test

```bash
cargo test --release -- --nocapture
```

This runs 10 headless simulated minutes and prints population and generation numbers.
Set `SIM_MINUTES=60` for a longer run.
