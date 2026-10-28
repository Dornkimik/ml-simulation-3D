//! The ecosystem: tank, plankton, creatures, and the evolutionary loop.

use macroquad::prelude::*;
use macroquad::rand::gen_range;

use crate::brain::{Brain, INPUTS};
use crate::creature::{mutate_tint, Creature, Species};

pub const TANK_MIN: Vec3 = Vec3::new(-30.0, 0.0, -20.0);
pub const TANK_MAX: Vec3 = Vec3::new(30.0, 30.0, 20.0);
/// Creatures stay this far above the tank floor (which has dunes and rocks).
const FLOOR_CLEARANCE: f32 = 1.2;

/// Fixed simulation timestep.
pub const DT: f32 = 1.0 / 60.0;

const MAX_PLANKTON: usize = 900;
const PLANKTON_LIFETIME: f32 = 80.0;
const HALL_OF_FAME_SIZE: usize = 20;
const MUTATION_RATE: f32 = 0.12;
const MUTATION_STRENGTH: f32 = 0.3;
/// Two parents within this distance reproduce sexually (crossover).
const MATE_RADIUS: f32 = 6.0;
/// Energy drained per second while touching a jellyfish.
const STING_FISH: f32 = 35.0;
const STING_SHARK: f32 = 20.0;
/// Energy drained per second while pressed against the tank walls.
const WALL_PENALTY: f32 = 3.0;

pub struct Plankton {
    pub pos: Vec3,
    pub age: f32,
    pub alive: bool,
}

/// A remembered high-fitness genome, used to re-seed a species that is
/// close to extinction.
pub struct Elite {
    pub fitness: f32,
    pub brain: Brain,
    pub tint: Vec3,
    pub generation: u32,
}

#[derive(Default, Clone, Copy)]
pub struct SpeciesStats {
    pub count: usize,
    pub max_generation: u32,
    pub births: u64,
    pub deaths: u64,
    pub reseeds: u64,
    pub best_fitness: f32,
}

/// Population sample taken once per simulated second.
#[derive(Clone, Copy)]
pub struct Sample {
    pub counts: [usize; 3],
    pub plankton: usize,
}

pub struct World {
    pub creatures: Vec<Creature>,
    pub plankton: Vec<Plankton>,
    pub time: f32,
    pub plankton_rate: f32,
    pub stats: [SpeciesStats; 3],
    pub history: Vec<Sample>,
    pub kills: u64,
    pub hall: [Vec<Elite>; 3],
    next_id: u64,
    plankton_acc: f32,
    sample_timer: f32,
    plankton_grid: Grid,
    creature_grid: Grid,
}

#[derive(Clone, Copy)]
struct Snap {
    species: Species,
    pos: Vec3,
    alive: bool,
}

#[derive(Clone, Copy)]
struct Nearest {
    idx: usize,
    d2: f32,
}

impl Nearest {
    const NONE: Nearest = Nearest { idx: usize::MAX, d2: f32::INFINITY };

    fn offer(&mut self, idx: usize, d2: f32) {
        if d2 < self.d2 {
            *self = Nearest { idx, d2 };
        }
    }

    fn found(&self) -> bool {
        self.idx != usize::MAX
    }
}

impl World {
    pub fn new() -> Self {
        let mut world = Self {
            creatures: Vec::new(),
            plankton: Vec::new(),
            time: 0.0,
            plankton_rate: 36.0,
            stats: Default::default(),
            history: Vec::new(),
            kills: 0,
            hall: Default::default(),
            next_id: 0,
            plankton_acc: 0.0,
            sample_timer: 0.0,
            plankton_grid: Grid::new(5.0),
            creature_grid: Grid::new(6.0),
        };
        for species in Species::ALL {
            for _ in 0..species.params().initial_pop {
                world.spawn_random(species);
            }
        }
        for _ in 0..450 {
            let pos = random_plankton_pos();
            world.plankton.push(Plankton { pos, age: gen_range(0.0, PLANKTON_LIFETIME * 0.5), alive: true });
        }
        world.refresh_stats();
        world
    }

    fn alloc_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn spawn_random(&mut self, species: Species) {
        let id = self.alloc_id();
        let pos = random_pos_in_tank(species.params().size);
        self.creatures.push(Creature::new(id, species, pos, Brain::random(), 0, Vec3::ZERO));
    }

    /// Re-seeds a struggling species from its best remembered genomes.
    fn spawn_from_elites(&mut self, species: Species) {
        let hall = &self.hall[species.index()];
        let (brain, tint, generation) = if hall.len() >= 2 && gen_range(0.0, 1.0) < 0.85 {
            let a = &hall[tournament(hall.len())];
            let b = &hall[tournament(hall.len())];
            let brain = Brain::crossover(&a.brain, &b.brain).mutated(MUTATION_RATE, MUTATION_STRENGTH);
            (brain, mutate_tint(a.tint), a.generation.max(b.generation) + 1)
        } else if let Some(a) = hall.first() {
            (a.brain.mutated(MUTATION_RATE * 2.0, MUTATION_STRENGTH), mutate_tint(a.tint), a.generation + 1)
        } else {
            (Brain::random(), Vec3::ZERO, 0)
        };
        let id = self.alloc_id();
        let pos = random_pos_in_tank(species.params().size);
        self.creatures.push(Creature::new(id, species, pos, brain, generation, tint));
        self.stats[species.index()].reseeds += 1;
    }

    fn record_elite(&mut self, c: &Creature) {
        let hall = &mut self.hall[c.species.index()];
        let fitness = c.fitness();
        if hall.len() >= HALL_OF_FAME_SIZE && hall.last().is_some_and(|e| e.fitness >= fitness) {
            return;
        }
        let at = hall.partition_point(|e| e.fitness >= fitness);
        hall.insert(at, Elite { fitness, brain: c.brain.clone(), tint: c.tint, generation: c.generation });
        hall.truncate(HALL_OF_FAME_SIZE);
    }

    pub fn find(&self, id: u64) -> Option<&Creature> {
        self.creatures.iter().find(|c| c.id == id)
    }

    pub fn step(&mut self) {
        let dt = DT;
        self.time += dt;
        self.update_plankton(dt);

        // Spatial grids for neighbor queries.
        self.plankton_grid.clear();
        for (i, p) in self.plankton.iter().enumerate() {
            self.plankton_grid.insert(p.pos, i);
        }
        self.creature_grid.clear();
        for (i, c) in self.creatures.iter().enumerate() {
            self.creature_grid.insert(c.pos, i);
        }

        let mut snap: Vec<Snap> = self
            .creatures
            .iter()
            .map(|c| Snap { species: c.species, pos: c.pos, alive: c.alive })
            .collect();
        let n = self.creatures.len();
        let mut mates: Vec<Option<usize>> = vec![None; n];
        let mut kills: Vec<(usize, usize)> = Vec::new();
        let time = self.time;

        let World { creatures, plankton, plankton_grid, creature_grid, .. } = self;

        for i in 0..n {
            if !snap[i].alive {
                continue;
            }
            let me = snap[i];
            let p = me.species.params();
            let r = p.sense_radius;
            let r2 = r * r;

            // --- Perception ---
            let mut kin = Nearest::NONE;
            let mut fish = Nearest::NONE;
            let mut shark = Nearest::NONE;
            let mut jelly = Nearest::NONE;
            creature_grid.for_each_near(me.pos, r, |j| {
                let other = &snap[j];
                if j == i || !other.alive {
                    return;
                }
                let d2 = other.pos.distance_squared(me.pos);
                if d2 > r2 {
                    return;
                }
                if other.species == me.species {
                    kin.offer(j, d2);
                }
                match other.species {
                    Species::Fish => fish.offer(j, d2),
                    Species::Shark => shark.offer(j, d2),
                    Species::Jellyfish => jelly.offer(j, d2),
                }
            });

            let mut food_plankton = Nearest::NONE;
            if me.species != Species::Shark {
                plankton_grid.for_each_near(me.pos, r, |j| {
                    let pl = &plankton[j];
                    if pl.alive {
                        let d2 = pl.pos.distance_squared(me.pos);
                        if d2 <= r2 {
                            food_plankton.offer(j, d2);
                        }
                    }
                });
            }

            let food_pos = match me.species {
                Species::Shark => fish.found().then(|| snap[fish.idx].pos),
                _ => food_plankton.found().then(|| plankton[food_plankton.idx].pos),
            };
            let threat = match me.species {
                Species::Fish => {
                    if shark.d2 <= jelly.d2 {
                        shark
                    } else {
                        jelly
                    }
                }
                Species::Shark => jelly,
                Species::Jellyfish => Nearest::NONE,
            };
            let threat_pos = threat.found().then(|| snap[threat.idx].pos);
            let kin_pos = kin.found().then(|| snap[kin.idx].pos);

            let c = &mut creatures[i];
            c.seen_food = food_pos;
            c.seen_threat = threat_pos;
            let sense = |target: Option<Vec3>| -> Vec3 {
                match target {
                    Some(t) => {
                        let d = t - me.pos;
                        let dist = d.length().max(1e-4);
                        d / dist * (1.0 - dist / r).max(0.0)
                    }
                    None => Vec3::ZERO,
                }
            };
            let food_v = sense(food_pos);
            let threat_v = sense(threat_pos);
            let kin_v = sense(kin_pos);
            let vel_v = c.vel / p.max_speed;
            let half = (TANK_MAX - TANK_MIN) * 0.5;
            let pos_v = (c.pos - (TANK_MIN + half)) / half;
            let input: [f32; INPUTS] = [
                food_v.x, food_v.y, food_v.z,
                threat_v.x, threat_v.y, threat_v.z,
                kin_v.x, kin_v.y, kin_v.z,
                vel_v.x, vel_v.y, vel_v.z,
                pos_v.x, pos_v.y, pos_v.z,
                c.energy / p.max_energy * 2.0 - 1.0,
                (time * 2.0 + c.phase).sin(),
                c.memory,
                1.0,
            ];

            // --- Decision ---
            let out = c.brain.think(&input);
            let mut thrust = vec3(out[0], out[1], out[2]);
            if thrust.length_squared() > 1.0 {
                thrust = thrust.normalize();
            }
            c.throttle = (out[3] + 1.0) * 0.5;
            c.memory = out[4];

            // --- Physics ---
            let effort = c.throttle * thrust.length();
            c.vel += thrust * p.accel * c.throttle * dt;
            c.vel -= c.vel * p.drag * dt;
            let speed = c.vel.length();
            if speed > p.max_speed {
                c.vel *= p.max_speed / speed;
            }
            c.pos += (c.vel + water_current(c.pos, time) * 0.35) * dt;
            if keep_in_tank(&mut c.pos, &mut c.vel, p.size) {
                c.energy -= WALL_PENALTY * dt;
            }
            if c.vel.length_squared() > 0.01 {
                let target = c.vel.normalize();
                c.heading = (c.heading + (target - c.heading) * (dt * 6.0).min(1.0)).normalize_or_zero();
                if c.heading == Vec3::ZERO {
                    c.heading = target;
                }
            }
            let beat = match c.species {
                Species::Jellyfish => 1.5 + 3.0 * effort,
                _ => 3.0 + 14.0 * effort,
            };
            c.phase += beat * dt;

            // --- Metabolism ---
            c.energy -= (p.metabolism + p.move_cost * effort * effort) * dt;
            c.age += dt;
            c.repro_timer -= dt;
            c.digest_timer -= dt;
            c.stung = (c.stung - dt).max(0.0);

            // --- Interactions ---
            match me.species {
                Species::Fish | Species::Jellyfish => {
                    if food_plankton.found() && food_plankton.d2.sqrt() < p.eat_radius {
                        plankton[food_plankton.idx].alive = false;
                        c.energy = (c.energy + p.meal_energy).min(p.max_energy);
                        c.meals += 1;
                    }
                }
                Species::Shark => {
                    let reach = p.eat_radius + Species::Fish.params().size;
                    if c.digest_timer <= 0.0 && fish.found() && fish.d2.sqrt() < reach {
                        snap[fish.idx].alive = false;
                        kills.push((i, fish.idx));
                        c.digest_timer = 2.5;
                    }
                }
            }
            if me.species != Species::Jellyfish && jelly.found() {
                let touch = Species::Jellyfish.params().size * 1.15 + p.size * 0.5;
                if jelly.d2.sqrt() < touch {
                    let sting = if me.species == Species::Fish { STING_FISH } else { STING_SHARK };
                    c.energy -= sting * dt;
                    c.stung = 0.3;
                }
            }
            if kin.found() && kin.d2 < MATE_RADIUS * MATE_RADIUS {
                mates[i] = Some(kin.idx);
            }
        }

        // Resolve predation.
        let mut eaten = vec![false; n];
        for &(hunter, prey) in &kills {
            let prey_energy = creatures[prey].energy.max(0.0);
            creatures[prey].alive = false;
            eaten[prey] = true;
            let h = &mut creatures[hunter];
            let p = h.species.params();
            h.energy = (h.energy + p.meal_energy + prey_energy * 0.4).min(p.max_energy);
            h.meals += 1;
        }
        self.kills += kills.len() as u64;

        // Deaths: starvation, old age, predation.
        for i in 0..n {
            let c = &mut self.creatures[i];
            let p = c.species.params();
            if c.alive && (c.energy <= 0.0 || c.age > p.max_age) {
                c.alive = false;
            }
        }
        for i in 0..n {
            if self.creatures[i].alive {
                continue;
            }
            let (species, pos, size) = {
                let c = &self.creatures[i];
                (c.species, c.pos, c.species.params().size)
            };
            let c = std::mem::replace(&mut self.creatures[i], placeholder(species));
            self.record_elite(&c);
            self.stats[species.index()].deaths += 1;
            if !eaten[i] {
                // Remains sink back into the food web as "marine snow".
                for _ in 0..(size * 4.0) as usize {
                    if self.plankton.len() < MAX_PLANKTON {
                        let jitter = vec3(gen_range(-1.0, 1.0), gen_range(-1.0, 1.0), gen_range(-1.0, 1.0));
                        self.plankton.push(Plankton { pos: pos + jitter, age: 0.0, alive: true });
                    }
                }
            }
        }

        // Births.
        let mut counts = [0usize; 3];
        for c in &self.creatures {
            if c.alive {
                counts[c.species.index()] += 1;
            }
        }
        let mut newborns = Vec::new();
        for i in 0..n {
            let c = &self.creatures[i];
            let p = c.species.params();
            let si = c.species.index();
            if !c.alive
                || c.energy < p.repro_threshold
                || c.age < p.maturity_age
                || c.repro_timer > 0.0
                || counts[si] >= p.max_pop
            {
                continue;
            }
            let mate = mates[i].filter(|&m| self.creatures[m].alive && self.creatures[m].species == c.species);
            let (brain, generation) = match mate {
                Some(m) => {
                    let other = &self.creatures[m];
                    (
                        Brain::crossover(&c.brain, &other.brain),
                        c.generation.max(other.generation) + 1,
                    )
                }
                None => (c.brain.clone(), c.generation + 1),
            };
            let brain = brain.mutated(MUTATION_RATE, MUTATION_STRENGTH);
            let pos = c.pos - c.heading * p.size * 2.0;
            let tint = mutate_tint(c.tint);
            newborns.push((c.species, pos, brain, generation, tint, p.repro_cost * 0.9));
            counts[si] += 1;
            if let Some(m) = mate {
                self.creatures[m].children += 1;
            }
            let c = &mut self.creatures[i];
            c.energy -= p.repro_cost;
            c.repro_timer = p.repro_cooldown;
            c.children += 1;
        }
        for (species, pos, brain, generation, tint, energy) in newborns {
            let id = self.alloc_id();
            let mut child = Creature::new(id, species, pos, brain, generation, tint);
            child.energy = energy;
            self.creatures.push(child);
            self.stats[species.index()].births += 1;
        }

        self.creatures.retain(|c| c.alive);
        self.plankton.retain(|p| p.alive);

        // Keep every species from going extinct.
        for species in Species::ALL {
            let si = species.index();
            if counts[si] < species.params().min_pop {
                self.spawn_from_elites(species);
            }
        }

        self.sample_timer += dt;
        if self.sample_timer >= 1.0 {
            self.sample_timer -= 1.0;
            for hall in &mut self.hall {
                // Old records slowly fade so the hall keeps tracking current conditions.
                for e in hall.iter_mut() {
                    e.fitness *= 0.995;
                }
            }
            self.refresh_stats();
            let s = Sample {
                counts: [self.stats[0].count, self.stats[1].count, self.stats[2].count],
                plankton: self.plankton.len(),
            };
            self.history.push(s);
            if self.history.len() > 600 {
                self.history.remove(0);
            }
        }
    }

    pub fn refresh_stats(&mut self) {
        for s in &mut self.stats {
            s.count = 0;
            s.max_generation = 0;
            s.best_fitness = 0.0;
        }
        for c in &self.creatures {
            let s = &mut self.stats[c.species.index()];
            s.count += 1;
            s.max_generation = s.max_generation.max(c.generation);
            s.best_fitness = s.best_fitness.max(c.fitness());
        }
    }

    fn update_plankton(&mut self, dt: f32) {
        self.plankton_acc += self.plankton_rate * dt;
        while self.plankton_acc >= 1.0 {
            self.plankton_acc -= 1.0;
            if self.plankton.len() < MAX_PLANKTON {
                self.plankton.push(Plankton { pos: random_plankton_pos(), age: 0.0, alive: true });
            }
        }
        let time = self.time;
        for p in &mut self.plankton {
            p.age += dt;
            p.pos += (water_current(p.pos, time) + vec3(0.0, -0.08, 0.0)) * dt;
            let mut v = Vec3::ZERO;
            keep_in_tank(&mut p.pos, &mut v, 0.2);
            if p.age > PLANKTON_LIFETIME {
                p.alive = false;
            }
        }
    }
}

/// Dead body left behind while `step` is still iterating by index.
fn placeholder(species: Species) -> Creature {
    let mut c = Creature::new(0, species, Vec3::ZERO, Brain::random(), 0, Vec3::ZERO);
    c.alive = false;
    c
}

/// Picks an index biased toward the front of a fitness-sorted list.
fn tournament(len: usize) -> usize {
    let a = gen_range(0, len);
    let b = gen_range(0, len);
    a.min(b)
}

/// A slow, swirling current that carries plankton and nudges swimmers.
pub fn water_current(p: Vec3, t: f32) -> Vec3 {
    vec3(
        (p.z * 0.11 + t * 0.21).sin() + 0.3,
        (p.x * 0.13 + t * 0.15).sin() * 0.25,
        (p.x * 0.09 - p.y * 0.07 + t * 0.17).cos(),
    ) * 0.7
}

/// Clamps a point to the tank, bouncing velocity off the walls.
/// Returns true if a wall was hit.
fn keep_in_tank(pos: &mut Vec3, vel: &mut Vec3, margin: f32) -> bool {
    let lo = TANK_MIN + vec3(margin, margin + FLOOR_CLEARANCE, margin);
    let hi = TANK_MAX - Vec3::splat(margin);
    let mut hit = false;
    for axis in 0..3 {
        if pos[axis] < lo[axis] {
            pos[axis] = lo[axis];
            vel[axis] = vel[axis].abs() * 0.3;
            hit = true;
        } else if pos[axis] > hi[axis] {
            pos[axis] = hi[axis];
            vel[axis] = -vel[axis].abs() * 0.3;
            hit = true;
        }
    }
    hit
}

fn random_pos_in_tank(margin: f32) -> Vec3 {
    let m = margin + 1.0;
    vec3(
        gen_range(TANK_MIN.x + m, TANK_MAX.x - m),
        gen_range(TANK_MIN.y + m + FLOOR_CLEARANCE, TANK_MAX.y - m),
        gen_range(TANK_MIN.z + m, TANK_MAX.z - m),
    )
}

/// Plankton grows where there is light, so it spawns mostly near the surface.
fn random_plankton_pos() -> Vec3 {
    let u: f32 = gen_range(0.0, 1.0);
    vec3(
        gen_range(TANK_MIN.x + 0.5, TANK_MAX.x - 0.5),
        TANK_MIN.y + FLOOR_CLEARANCE + (TANK_MAX.y - TANK_MIN.y - FLOOR_CLEARANCE - 0.5) * (1.0 - u * u),
        gen_range(TANK_MIN.z + 0.5, TANK_MAX.z - 0.5),
    )
}

/// Uniform grid over the tank for fast radius queries.
struct Grid {
    cell: f32,
    dims: [i32; 3],
    cells: Vec<Vec<usize>>,
}

impl Grid {
    fn new(cell: f32) -> Self {
        let size = TANK_MAX - TANK_MIN;
        let dims = [
            (size.x / cell).ceil() as i32,
            (size.y / cell).ceil() as i32,
            (size.z / cell).ceil() as i32,
        ];
        let cells = vec![Vec::new(); (dims[0] * dims[1] * dims[2]) as usize];
        Self { cell, dims, cells }
    }

    fn coord(&self, p: Vec3) -> [i32; 3] {
        let rel = (p - TANK_MIN) / self.cell;
        [
            (rel.x.floor() as i32).clamp(0, self.dims[0] - 1),
            (rel.y.floor() as i32).clamp(0, self.dims[1] - 1),
            (rel.z.floor() as i32).clamp(0, self.dims[2] - 1),
        ]
    }

    fn slot(&self, c: [i32; 3]) -> usize {
        ((c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]) as usize
    }

    fn clear(&mut self) {
        for c in &mut self.cells {
            c.clear();
        }
    }

    fn insert(&mut self, p: Vec3, idx: usize) {
        let s = self.slot(self.coord(p));
        self.cells[s].push(idx);
    }

    fn for_each_near(&self, p: Vec3, r: f32, mut f: impl FnMut(usize)) {
        let lo = self.coord(p - Vec3::splat(r));
        let hi = self.coord(p + Vec3::splat(r));
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    for &idx in &self.cells[self.slot([x, y, z])] {
                        f(idx);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs ten simulated minutes headless and checks the ecosystem stays sane.
    #[test]
    fn ecosystem_survives_ten_minutes() {
        macroquad::rand::srand(42);
        let mut world = World::new();
        let minutes: u32 = std::env::var("SIM_MINUTES").ok().and_then(|m| m.parse().ok()).unwrap_or(10);
        for minute in 1..=minutes {
            for _ in 0..(60.0 / DT) as usize {
                world.step();
            }
            world.refresh_stats();
            let s = &world.stats;
            println!(
                "min {minute:>2}: fish {:>3} (gen {:>3}, births {:>4})  jelly {:>2} (gen {:>2})  shark {:>2} (gen {:>2}, reseeds {:>3})  plankton {:>3}  kills {}",
                s[0].count, s[0].max_generation, s[0].births, s[1].count, s[1].max_generation,
                s[2].count, s[2].max_generation, s[2].reseeds, world.plankton.len(), world.kills
            );
        }
        for c in &world.creatures {
            assert!(c.pos.is_finite() && c.vel.is_finite() && c.energy.is_finite());
        }
        for species in Species::ALL {
            assert!(world.stats[species.index()].count >= species.params().min_pop.min(1));
        }
    }
}
