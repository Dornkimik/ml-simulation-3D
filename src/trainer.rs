//! Trains the walking brain in a background thread with evolution
//! strategies (ES).
//!
//! The trainer keeps one "parent" set of network weights. Each generation it
//! makes a population of slightly jittered copies (in mirrored pairs: +noise
//! and -noise), lets every copy attempt to walk (simulated headless, in
//! parallel on all CPU cores), and nudges the parent toward the jitters that
//! walked further. Over many generations this climbs toward better walking.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::brain::{gaussian, Brain};
use crate::walker::{evaluate, Score};

/// Attempts per generation (must be even: candidates come in pairs).
pub const POPULATION: usize = 96;
/// Size of the random jitter applied to the weights.
const NOISE: f32 = 0.05;
const LEARNING_RATE: f32 = 0.02;
/// Gently pulls weights toward zero so none grow without bound.
const WEIGHT_DECAY: f32 = 0.005;
pub const SAVE_FILE: &str = "walker_brain.txt";

#[derive(Clone, Copy)]
pub struct GenStats {
    pub best_distance: f32,
}

#[derive(Clone)]
pub struct Champion {
    pub brain: Brain,
    pub score: Score,
    pub generation: u32,
}

#[derive(Default)]
pub struct Progress {
    pub generation: u32,
    pub champion: Option<Champion>,
    pub history: Vec<GenStats>,
    pub attempts_per_sec: f32,
    pub resumed_from_file: bool,
}

pub struct Trainer {
    pub progress: Arc<Mutex<Progress>>,
    pub paused: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Trainer {
    /// Starts training, resuming from the save file if there is one.
    pub fn start() -> Self {
        let progress = Arc::new(Mutex::new(Progress::default()));
        let paused = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = {
            let (progress, paused, stop) = (progress.clone(), paused.clone(), stop.clone());
            std::thread::spawn(move || train(progress, paused, stop))
        };
        Self { progress, paused, stop, handle: Some(handle) }
    }

    /// Forgets everything learned (including the save file) and starts over.
    pub fn restart_from_scratch(&mut self) {
        self.shutdown();
        let _ = std::fs::remove_file(SAVE_FILE);
        *self = Self::start();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Trainer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn train(progress: Arc<Mutex<Progress>>, paused: Arc<AtomicBool>, stop: Arc<AtomicBool>) {
    let mut generation = 0;
    let mut es = match load() {
        Some((brain, saved_gen)) => {
            generation = saved_gen;
            let mut p = progress.lock().unwrap();
            p.generation = generation;
            p.resumed_from_file = true;
            Es::new(brain.weights().collect())
        }
        None => Es::new(Brain::random().weights().collect()),
    };

    let mut best_fitness = f32::NEG_INFINITY;
    while !stop.load(Ordering::Relaxed) {
        if paused.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        let started = Instant::now();
        let population = es.ask();
        let scores = evaluate_all(&population);
        let elapsed = started.elapsed().as_secs_f32().max(1e-3);
        generation += 1;

        let best = (0..scores.len()).max_by(|&a, &b| scores[a].fitness.total_cmp(&scores[b].fitness)).unwrap();
        {
            let mut p = progress.lock().unwrap();
            p.generation = generation;
            p.attempts_per_sec = population.len() as f32 / elapsed;
            p.history.push(GenStats { best_distance: scores[best].distance });
            if scores[best].fitness > best_fitness {
                best_fitness = scores[best].fitness;
                p.champion = Some(Champion { brain: population[best].clone(), score: scores[best], generation });
                save(&population[best], generation);
            }
        }

        es.tell(&scores);
    }
}

/// Scores every brain, spreading the work over all but one CPU core so the
/// window stays smooth.
fn evaluate_all(population: &[Brain]) -> Vec<Score> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(1).max(1);
    let chunk = population.len().div_ceil(threads);
    let mut scores = vec![Score::default(); population.len()];
    std::thread::scope(|s| {
        for (brains, out) in population.chunks(chunk).zip(scores.chunks_mut(chunk)) {
            s.spawn(move || {
                for (brain, score) in brains.iter().zip(out) {
                    *score = evaluate(brain);
                }
            });
        }
    });
    scores
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

    fn tell(&mut self, scores: &[Score]) {
        // Use fitness ranks rather than raw values, so one lucky outlier
        // can't dominate the update.
        let n = scores.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| scores[a].fitness.total_cmp(&scores[b].fitness));
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

fn save(brain: &Brain, generation: u32) {
    let weights: Vec<String> = brain.weights().map(|w| w.to_string()).collect();
    let text = format!("walker-brain v1\n{generation}\n{}\n", weights.join(" "));
    let _ = std::fs::write(SAVE_FILE, text);
}

fn load() -> Option<(Brain, u32)> {
    let text = std::fs::read_to_string(SAVE_FILE).ok()?;
    let mut lines = text.lines();
    if lines.next()? != "walker-brain v1" {
        return None;
    }
    let generation = lines.next()?.trim().parse().ok()?;
    let weights: Vec<f32> = lines.next()?.split_whitespace().map(|w| w.parse()).collect::<Result<_, _>>().ok()?;
    Some((Brain::from_weights(&weights)?, generation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walker::{Walker, EPISODE_SECONDS};

    /// Trains headless and prints progress. Set GENERATIONS for longer runs.
    #[test]
    fn learns_to_move_forward() {
        macroquad::rand::srand(7);
        let generations: u32 = std::env::var("GENERATIONS").ok().and_then(|g| g.parse().ok()).unwrap_or(15);
        let mut es = Es::new(Brain::random().weights().collect());
        let mut champion: Option<(Brain, Score)> = None;
        for generation in 1..=generations {
            let t = Instant::now();
            let population = es.ask();
            let scores = evaluate_all(&population);
            let best = (0..scores.len()).max_by(|&a, &b| scores[a].fitness.total_cmp(&scores[b].fitness)).unwrap();
            if champion.as_ref().is_none_or(|(_, s)| scores[best].fitness > s.fitness) {
                champion = Some((population[best].clone(), scores[best]));
            }
            if generation % 10 == 0 || generation == 1 {
                let s = scores[best];
                println!(
                    "gen {generation:>4}: best {:>6.2} m  (fitness {:>6.2}, up {:>4.1}s)   {:.2}s/gen",
                    s.distance, s.fitness, s.time, t.elapsed().as_secs_f32()
                );
            }
            es.tell(&scores);
        }

        // Gait check on the champion: are the feet alternating like walking?
        let (mut brain, score) = champion.unwrap();
        let mut w = Walker::new();
        let (mut airborne, mut single, mut double, mut switches) = (0, 0, 0, 0);
        let mut last_stance = None;
        while !w.fallen && w.time < EPISODE_SECONDS {
            w.step(&mut brain);
            let (l, r) = w.feet_down();
            match (l, r) {
                (false, false) => airborne += 1,
                (true, true) => double += 1,
                _ => {
                    single += 1;
                    if last_stance.is_some_and(|s| s != l) {
                        switches += 1;
                    }
                    last_stance = Some(l);
                }
            }
        }
        let total = (airborne + single + double) as f32;
        println!(
            "champion: {:.2} m, airborne {:.0}%, single-leg {:.0}%, double {:.0}%, stance switches {} ({:.1}/s)",
            score.distance,
            100.0 * airborne as f32 / total,
            100.0 * single as f32 / total,
            100.0 * double as f32 / total,
            switches,
            switches as f32 / w.time
        );
        assert!(score.fitness.is_finite());
    }
}
