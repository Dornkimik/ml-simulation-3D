# Learning to Play Soccer (Rust, 3D)

Two players, **Blue** and **Red**, learn to play one-on-one soccer against each other from scratch.
Each player has its own neural-network brain, and nothing about soccer is programmed in. A trainer in the
background improves both brains with **evolution strategies**, each by playing against the other.
The window shows the latest brains playing a 45-second match.

Brains can be **exported** to a file and **imported** back into either player, so you can keep training a
good brain, pit an old brain against a new one, or swap brains between players.

## Run

```bash
cargo run --release
```

The first build downloads and compiles macroquad (graphics) and rfd (file dialogs), which takes a minute or two.

## What you'll see

- **First seconds:** both players wander around at random.
- **After ~100 generations (a few seconds):** they chase the ball and push it around.
- **After ~1000 generations (under a minute on a multi-core CPU):** they dribble, turn to aim, and shoot.
  Between them they score about 6 goals per 45-second match (measured in the test below).

Training keeps going for as long as the window is open. Each player's brain is saved to `blue_brain.txt` /
`red_brain.txt` every few seconds and on exit, and the next run picks up from there.

## Importing and exporting brains

Each player's panel (top left for Blue, top right for Red) has four buttons:

| Button | Action |
|---|---|
| **Import** | Load a brain file into this player. Training continues from the imported brain. |
| **Export** | Save this player's current brain to a file of your choice. |
| **Freeze** / **Train** | Stop this player learning (it keeps playing), or let it learn again. |
| **Reset** | Replace this player's brain with a random one and start over. |

The keys **I**, **E**, **F** and **R** do the same for the selected player (switch with **Tab** or by clicking a panel).

Brains don't depend on which side they play on, so a brain exported from Blue can be imported into Red.
Some things to try:

- Export a good brain, keep training, then import the old one into the other player to see who wins.
- Freeze one player and let the other learn to beat it.
- Import the same brain into both players.

A brain file is plain text: a header, how many generations the brain has trained, the network shape, and its weights.

## How it works

**Match** (`src/soccer.rs`): a 28 x 18 m pitch with low boards all around, so the ball bounces off the walls
instead of going out. The goals are 5 m wide and 2 m high. Players run (up to 6 m/s), turn, and kick in the
direction they face; running into the ball pushes it along, so they can dribble. The ball rolls, bounces and
flies in 3D. After a goal, or if the ball stays in one spot for 5 seconds (say, both players are stuck in a corner),
play restarts from the middle.

**Brain** (`src/brain.rs`): a network with 18 inputs, 24 hidden neurons and 5 outputs that runs 60 times a second.
- **Inputs**, all measured relative to the player itself (ahead of it, to its left): where the ball is and how it
  moves, the ball's height, where the opponent is, where both goals are, the player's own speed and position on
  the pitch, and whether the ball is close enough to kick.
- **Outputs:** run forward/back, strafe, turn, kick (and how hard), and how high to lift the ball.

**Look** (`src/pose.rs`, `src/render.rs`): the players are drawn as people whose running and kicking are
animated from how they move in the match.

**Training** (`src/trainer.rs`): every generation, the trainer makes 64 slightly jittered copies of each
player's brain, in +/- pairs. Each copy plays three 20-second matches: two against the other player's latest
brain and one against an older version of it, simulated headless on all CPU cores. Then each brain moves toward
the jitters that scored better (OpenAI-style ES with the Adam optimizer). As one player gets better, the other
has to keep up.

**Score:** goals scored minus goals conceded, plus small bonuses for moving the ball toward the opponent's goal
and for kicking it, and a small penalty for staying far from the ball (so beginners learn to chase it).

## Controls

| Input | Action |
|---|---|
| Left-drag / wheel | Orbit / zoom the camera (it follows the ball) |
| Click a panel / Tab | Select Blue or Red |
| I / E | Import / export the selected player's brain |
| F | Freeze or unfreeze the selected player's training |
| R | Reset the selected player to a random brain |
| N | Start a new match with the latest brains |
| Space | Pause the view |
| + / - or Up / Down | Change view speed (0.25x to 8x) |
| P | Pause or resume all training |
| B / G / H | Toggle the live brain view / learning-curve graph / help |

## Test

```bash
cargo test --release -- --nocapture
```

This checks that both teams play by the same rules (a mirror-image start gives a mirror-image game), that
goals count, and that brain files round-trip. It also trains both players for 20 generations, then prints
how they do against untrained brains and against each other. Set `GENERATIONS=1500` for a longer run.
