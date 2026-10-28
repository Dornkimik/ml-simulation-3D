# Learning to Walk (Rust, 3D)

A physics-simulated person on a running track learns to walk from scratch.
Its muscles are joint motors, its brain is a small neural network, and nothing about walking is
programmed in. A trainer in the background improves the brain with **evolution strategies**.
The window shows one person: the best brain found so far, trying to walk as far as it can in 15 seconds.

## Run

```bash
cargo run --release
```

The first build downloads and compiles macroquad (graphics) and rapier3d (physics), which takes a minute or two.

## What you'll see

- **First minute:** the person stands, wobbles, and shuffles a little.
- **About 1 minute in (~60 generations):** it starts taking real steps.
- **About 3 minutes in (~250 generations):** it walks roughly 20 m in 15 s, about 1.5 m/s,
  with alternating steps (measured in the test below).

Training keeps going for as long as the window is open. The best brain is saved to `walker_brain.txt`,
and the next run picks up from it. Press **R** to forget everything and start over.

## How it works

**Body** (`src/walker.rs`): a ~75 kg ragdoll made of 9 rigid parts (torso with head, thighs,
shins, feet, arms) connected by 8 motorized joints. The hips can swing forward/back and out to the side;
knees, ankles and shoulders swing forward/back. Physics runs at 240 Hz in [rapier3d](https://rapier.rs).

**Brain** (`src/brain.rs`): a network with 18 inputs, 16 hidden neurons and 5 outputs that runs 60 times a second.
- **Inputs:** a gait-rhythm clock; torso lean and its rate of change; height; speed; this leg's and the
  other leg's joint angles; which feet are touching the ground.
- **Outputs:** target angles for the hip (forward and sideways), knee, ankle and shoulder.

The **same network drives both legs**. The right leg gets mirrored inputs and runs half a stride behind
the left leg, so the gaits it finds come out symmetric.

**Training** (`src/trainer.rs`): every generation, the trainer makes 96 slightly jittered copies of the
current brain, in +/- pairs. Each copy gets one 15-second walking attempt, simulated headless on all CPU cores.
The trainer then moves the brain's weights toward the jitters that scored better (OpenAI-style ES with the Adam optimizer).

**Score:** distance walked, plus a small bonus for staying up and a small penalty for veering sideways.
An attempt ends early if the person falls (torso too low or tilted, or head too low).

## Controls

| Input | Action |
|---|---|
| Left-drag / wheel | Orbit / zoom the camera (it follows the person) |
| N | Restart the attempt now with the latest champion |
| Space / + / - | Pause the view / change view speed (0.25x to 8x) |
| P | Pause or resume training |
| R | Forget all training (deletes `walker_brain.txt`) and start over |
| B / G / H | Toggle the live brain view / learning-curve graph / help |

## Test

```bash
cargo test --release -- --nocapture
```

This runs 15 generations of headless training, then checks the champion's gait (time airborne,
single- vs double-leg stance, step rate). Set `GENERATIONS=250` for a longer run.
