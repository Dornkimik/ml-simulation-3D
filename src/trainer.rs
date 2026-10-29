//! Trains both players' brains in a background thread, each against the
//! other, with evolution strategies (ES).
//!
//! Each player has its own "parent" set of network weights. Every generation
//! the trainer makes a population of slightly jittered copies of each parent
//! (in mirrored pairs: +noise and -noise). Every copy plays a few short
//! matches against the other player's current brain and one of its older
//! brains (simulated headless, in parallel on all CPU cores), and each
//! parent is nudged toward the jitters that played better. As one player
//! improves, the other has to keep up.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::brain::{gaussian, Brain};
use crate::soccer::{play, Outcome, Rng};

/// Candidates per player per generation (must be even: they come in pairs).
pub const POPULATION: usize = 64;
/// Size of the random jitter applied to the weights.
const NOISE: f32 = 0.06;
const LEARNING_RATE: f32 = 0.03;
/// Gently pulls weights toward zero so none grow without bound.
const WEIGHT_DECAY: f32 = 0.005;
/// Every candidate plays this many matches against the opponent's current
/// brain, plus `MATCHES_VS_PAST` against an older one, so it can't win by
/// exploiting one quirk of the current opponent.
const MATCHES_VS_CURRENT: usize = 2;
const MATCHES_VS_PAST: usize = 1;
/// Older brains are kept every this many generations, up to `PAST_BRAINS`.
const SNAPSHOT_EVERY: u32 = 10;
const PAST_BRAINS: usize = 30;
const SAVE_EVERY: u32 = 10;
pub const SAVE_FILES: [&str; 2] = ["blue_brain.txt", "red_brain.txt"];

/// One player's results in one generation, against the other's current brain.
#[derive(Clone, Copy, Default)]
pub struct GenStats {
    pub goals_for: f32,
    pub goals_against: f32,
}

#[derive(Clone)]
pub struct PlayerProgress {
    /// The latest trained brain.
    pub brain: Brain,
    /// Generations this brain has trained for.
    pub generation: u32,
    pub learning: bool,
    pub resumed_from_file: bool,
}

pub struct Progress {
    pub generation: u32,
    pub players: [PlayerProgress; 2],
    pub history: Vec<[GenStats; 2]>,
    pub matches_per_sec: f32,
    pending: Vec<Command>,
}

enum Command {
    Replace { player: usize, brain: Brain, generation: u32 },
    SetLearning { player: usize, learning: bool },
}

pub struct Trainer {
    pub progress: Arc<Mutex<Progress>>,
    pub paused: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Trainer {
    /// Starts training, resuming each player from its save file if there is one.
    pub fn start() -> Self {
        let players = [0, 1].map(|i| match Brain::load(Path::new(SAVE_FILES[i])) {
            Ok((brain, generation)) => PlayerProgress { brain, generation, learning: true, resumed_from_file: true },
            Err(_) => PlayerProgress { brain: Brain::random(), generation: 0, learning: true, resumed_from_file: false },
        });
        let progress = Arc::new(Mutex::new(Progress {
            generation: 0,
            players,
            history: Vec::new(),
            matches_per_sec: 0.0,
            pending: Vec::new(),
        }));
        let paused = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = {
            let (progress, paused, stop) = (progress.clone(), paused.clone(), stop.clone());
            std::thread::spawn(move || train(progress, paused, stop))
        };
        Self { progress, paused, stop, handle: Some(handle) }
    }

    /// Replaces a player's brain (and what it learned so far) with `brain`.
    /// Training continues from there.
    pub fn replace_brain(&self, player: usize, brain: Brain, generation: u32) {
        let mut p = self.progress.lock().unwrap();
        p.players[player].brain = brain.clone();
        p.players[player].generation = generation;
        p.pending.push(Command::Replace { player, brain, generation });
    }

    /// Freezes a player's brain (it keeps playing but stops changing), or
    /// lets it learn again.
    pub fn set_learning(&self, player: usize, learning: bool) {
        let mut p = self.progress.lock().unwrap();
        p.players[player].learning = learning;
        p.pending.push(Command::SetLearning { player, learning });
    }
}

impl Drop for Trainer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Everything the trainer knows about one player.
struct Learner {
    es: Es,
    generation: u32,
    learning: bool,
    /// Older versions of this brain, used as extra opponents for the other player.
    past: Vec<Brain>,
}

impl Learner {
    fn new(p: &PlayerProgress) -> Self {
        Self { es: Es::new(p.brain.weights().collect()), generation: p.generation, learning: p.learning, past: Vec::new() }
    }

    fn save(&self, player: usize) {
        let _ = self.es.brain().save(Path::new(SAVE_FILES[player]), self.generation);
    }
}

/// Applies commands from the window. Returns which players got a new brain,
/// so the generation that was running for them can be thrown away.
fn apply_commands(progress: &mut Progress, learners: &mut [Learner; 2]) -> [bool; 2] {
    let mut replaced = [false; 2];
    for cmd in progress.pending.drain(..) {
        match cmd {
            Command::Replace { player, brain, generation } => {
                let l = &mut learners[player];
                l.es = Es::new(brain.weights().collect());
                l.generation = generation;
                l.save(player);
                replaced[player] = true;
            }
            Command::SetLearning { player, learning } => learners[player].learning = learning,
        }
    }
    replaced
}

fn train(progress: Arc<Mutex<Progress>>, paused: Arc<AtomicBool>, stop: Arc<AtomicBool>) {
    let mut rng = Rng::new(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64));
    let mut learners = {
        let p = progress.lock().unwrap();
        [Learner::new(&p.players[0]), Learner::new(&p.players[1])]
    };

    while !stop.load(Ordering::Relaxed) {
        apply_commands(&mut progress.lock().unwrap(), &mut learners);
        if paused.load(Ordering::Relaxed) || !learners.iter().any(|l| l.learning) {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }

        let started = Instant::now();
        let round = play_round(&mut learners, &mut rng);
        let elapsed = started.elapsed().as_secs_f32().max(1e-3);

        let mut p = progress.lock().unwrap();
        let replaced = apply_commands(&mut p, &mut learners);
        finish_round(&mut learners, &round, replaced, &mut rng);
        p.generation += 1;
        p.matches_per_sec = round.matches as f32 / elapsed;
        p.history.push(round.stats);
        for (pp, l) in p.players.iter_mut().zip(&learners) {
            pp.brain = l.es.brain();
            pp.generation = l.generation;
        }
        if p.generation % SAVE_EVERY == 0 {
            learners[0].save(0);
            learners[1].save(1);
        }
    }
    learners[0].save(0);
    learners[1].save(1);
}

/// The matches of one generation, played but not yet learned from.
struct Round {
    /// Total score of each candidate, per player (empty if not learning).
    fitness: [Vec<f32>; 2],
    stats: [GenStats; 2],
    matches: usize,
}

struct Job<'a> {
    player: usize,
    candidate: usize,
    brain: &'a Brain,
    opponent: &'a Brain,
    seed: u64,
    team: usize,
    vs_current: bool,
}

fn play_round(learners: &mut [Learner; 2], rng: &mut Rng) -> Round {
    let current = [learners[0].es.brain(), learners[1].es.brain()];
    let past = [0, 1].map(|i| {
        let past = &learners[i].past;
        if past.is_empty() {
            current[i].clone()
        } else {
            past[(rng.next_u64() % past.len() as u64) as usize].clone()
        }
    });
    // Every candidate plays the same kickoffs, so they are compared fairly.
    let seeds: Vec<u64> = (0..MATCHES_VS_CURRENT + MATCHES_VS_PAST).map(|_| rng.next_u64()).collect();
    let populations = [0, 1].map(|i| if learners[i].learning { learners[i].es.ask() } else { Vec::new() });

    let mut jobs = Vec::new();
    for (player, population) in populations.iter().enumerate() {
        for (candidate, brain) in population.iter().enumerate() {
            for (m, &seed) in seeds.iter().enumerate() {
                let vs_current = m < MATCHES_VS_CURRENT;
                let opponent = if vs_current { &current[1 - player] } else { &past[1 - player] };
                // Swap ends between matches, in case either end has an edge.
                jobs.push(Job { player, candidate, brain, opponent, seed, team: m % 2, vs_current });
            }
        }
    }
    let outcomes = play_all(&jobs);

    let mut fitness = [0, 1].map(|i| vec![0.0; populations[i].len()]);
    let mut totals = [[0.0f32; 3]; 2];
    for (job, o) in jobs.iter().zip(&outcomes) {
        fitness[job.player][job.candidate] += o.fitness;
        if job.vs_current {
            let t = &mut totals[job.player];
            t[0] += o.goals_for as f32;
            t[1] += o.goals_against as f32;
            t[2] += 1.0;
        }
    }
    let own = totals.map(|[f, a, n]| (n > 0.0).then(|| GenStats { goals_for: f / n, goals_against: a / n }));
    // A frozen player's stats are the mirror image of its opponent's.
    let stats = [0, 1].map(|i| {
        own[i].or(own[1 - i].map(|o| GenStats { goals_for: o.goals_against, goals_against: o.goals_for })).unwrap_or_default()
    });
    Round { fitness, stats, matches: jobs.len() }
}

fn finish_round(learners: &mut [Learner; 2], round: &Round, skip: [bool; 2], rng: &mut Rng) {
    for (i, l) in learners.iter_mut().enumerate() {
        if skip[i] || !l.learning || round.fitness[i].is_empty() {
            continue;
        }
        l.es.tell(&round.fitness[i]);
        l.generation += 1;
        if l.generation % SNAPSHOT_EVERY == 0 {
            if l.past.len() >= PAST_BRAINS {
                // Forget a random old brain, but never the newest one.
                let k = (rng.next_u64() % (l.past.len() - 1) as u64) as usize;
                l.past.remove(k);
            }
            l.past.push(l.es.brain());
        }
    }
}

/// Plays every match, spreading the work over all but one CPU core so the
/// window stays smooth.
fn play_all(jobs: &[Job]) -> Vec<Outcome> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(1).max(1);
    let chunk = jobs.len().div_ceil(threads).max(1);
    let mut outcomes = vec![Outcome::default(); jobs.len()];
    std::thread::scope(|s| {
        for (jobs, out) in jobs.chunks(chunk).zip(outcomes.chunks_mut(chunk)) {
            s.spawn(move || {
                for (j, o) in jobs.iter().zip(out) {
                    *o = play(j.brain, j.opponent, j.seed, j.team);
                }
            });
        }
    });
    outcomes
}

/// OpenAI-style evolution strategies: estimates the gradient of fitness from
/// mirrored random perturbations and climbs it with the Adam optimizer.
struct Es {
    theta: Vec<f32>,
    noise: Vec<Vec<f32>>,
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Es {
    fn new(theta: Vec<f32>) -> Self {
        let d = theta.len();
        Self { theta, noise: Vec::new(), m: vec![0.0; d], v: vec![0.0; d], t: 0 }
    }

    fn brain(&self) -> Brain {
        Brain::from_weights(&self.theta).unwrap()
    }

    /// This generation's candidates: theta + noise and theta - noise, pairwise.
    fn ask(&mut self) -> Vec<Brain> {
        let d = self.theta.len();
        self.noise = (0..POPULATION / 2).map(|_| (0..d).map(|_| gaussian()).collect()).collect();
        let mut out = Vec::with_capacity(POPULATION);
        for eps in &self.noise {
            for sign in [1.0, -1.0] {
                let w: Vec<f32> = self.theta.iter().zip(eps).map(|(t, e)| t + sign * NOISE * e).collect();
                out.push(Brain::from_weights(&w).unwrap());
            }
        }
        out
    }

    fn tell(&mut self, fitness: &[f32]) {
        // Use fitness ranks rather than raw values, so one lucky outlier
        // can't dominate the update.
        let n = fitness.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]));
        let mut rank = vec![0.0f32; n];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r as f32 / (n - 1) as f32 - 0.5;
        }

        let d = self.theta.len();
        let mut grad = vec![0.0f32; d];
        for (k, eps) in self.noise.iter().enumerate() {
            let w = rank[2 * k] - rank[2 * k + 1];
            for (g, e) in grad.iter_mut().zip(eps) {
                *g += w * e;
            }
        }

        self.t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        for i in 0..d {
            let g = grad[i] / (n as f32 * NOISE) - WEIGHT_DECAY * self.theta[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / (1.0 - b1.powi(self.t));
            let v_hat = self.v[i] / (1.0 - b2.powi(self.t));
            self.theta[i] += LEARNING_RATE * m_hat / (v_hat.sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Average goal difference of `brain` against untrained random brains.
    fn vs_random(brain: &Brain, rng: &mut Rng) -> f32 {
        let mut diff = 0.0;
        for k in 0..20 {
            let o = play(brain, &Brain::random(), rng.next_u64(), k % 2);
            diff += o.goals_for as f32 - o.goals_against as f32;
        }
        diff / 20.0
    }

    /// Trains both players headless and prints progress. Set GENERATIONS
    /// for longer runs.
    #[test]
    fn both_players_learn() {
        macroquad::rand::srand(7);
        let generations: u32 = std::env::var("GENERATIONS").ok().and_then(|g| g.parse().ok()).unwrap_or(20);
        let mut rng = Rng::new(7);
        let fresh = PlayerProgress { brain: Brain::random(), generation: 0, learning: true, resumed_from_file: false };
        let mut learners = [Learner::new(&fresh), Learner::new(&PlayerProgress { brain: Brain::random(), ..fresh.clone() })];
        let before = [vs_random(&learners[0].es.brain(), &mut rng), vs_random(&learners[1].es.brain(), &mut rng)];
        for generation in 1..=generations {
            let t = Instant::now();
            let round = play_round(&mut learners, &mut rng);
            finish_round(&mut learners, &round, [false; 2], &mut rng);
            if generation % 10 == 0 || generation == 1 {
                let [b, r] = round.stats;
                println!(
                    "gen {generation:>4}: blue {:.2}:{:.2} red per match   {:.2}s/gen ({:.0} matches/s)",
                    b.goals_for,
                    r.goals_for,
                    t.elapsed().as_secs_f32(),
                    round.matches as f32 / t.elapsed().as_secs_f32()
                );
            }
        }
        let after = [vs_random(&learners[0].es.brain(), &mut rng), vs_random(&learners[1].es.brain(), &mut rng)];
        println!("goal difference per match vs random brains: blue {:.2} -> {:.2}, red {:.2} -> {:.2}", before[0], after[0], before[1], after[1]);

        // How the trained brains play each other in a full-length match.
        let (mut goals, mut restarts) = ([0, 0], 0);
        let mut brains = [learners[0].es.brain(), learners[1].es.brain()];
        for k in 0..20 {
            let mut game = crate::soccer::Match::new(k, 45.0, 1.5);
            while !game.over() {
                game.step(&mut brains);
                crate::soccer::tests::assert_ball_legal(&game);
            }
            goals[0] += game.score[0];
            goals[1] += game.score[1];
            restarts += game.restarts;
        }
        println!("20 matches of 45 s: blue {} : {} red, {restarts} restarts after stalls", goals[0], goals[1]);
        assert!(learners.iter().all(|l| l.es.theta.iter().all(|w| w.is_finite())));
    }
}
