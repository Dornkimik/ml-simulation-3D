use macroquad::prelude::*;
use macroquad::rand::gen_range;

use crate::brain::{gaussian, Brain};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Species {
    /// Small, agile grazers that eat plankton and are hunted by sharks.
    Fish,
    /// Slow drifting filter feeders whose tentacles sting anything that touches them.
    Jellyfish,
    /// Fast apex predators that hunt fish.
    Shark,
}

impl Species {
    pub const ALL: [Species; 3] = [Species::Fish, Species::Jellyfish, Species::Shark];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Species::Fish => "Fish",
            Species::Jellyfish => "Jellyfish",
            Species::Shark => "Shark",
        }
    }

    pub fn params(self) -> &'static SpeciesParams {
        &PARAMS[self.index()]
    }
}

/// Tunable biology for one species. Distances are in world units (the tank
/// is 60 x 30 x 40), times in seconds, energy in arbitrary units.
pub struct SpeciesParams {
    pub size: f32,
    pub max_speed: f32,
    pub accel: f32,
    pub drag: f32,
    pub sense_radius: f32,
    pub eat_radius: f32,
    pub max_energy: f32,
    pub start_energy: f32,
    /// Energy burned per second just by being alive.
    pub metabolism: f32,
    /// Extra energy per second at full throttle.
    pub move_cost: f32,
    /// Energy gained per meal (plus a share of the prey's energy for sharks).
    pub meal_energy: f32,
    pub repro_threshold: f32,
    pub repro_cost: f32,
    pub repro_cooldown: f32,
    pub maturity_age: f32,
    pub max_age: f32,
    pub max_pop: usize,
    /// Below this the species is re-seeded from its hall of fame.
    pub min_pop: usize,
    pub initial_pop: usize,
    pub color: Color,
}

pub static PARAMS: [SpeciesParams; 3] = [
    SpeciesParams {
        size: 0.55,
        max_speed: 7.5,
        accel: 16.0,
        drag: 1.2,
        sense_radius: 11.0,
        eat_radius: 1.5,
        max_energy: 100.0,
        start_energy: 55.0,
        metabolism: 1.3,
        move_cost: 2.4,
        meal_energy: 11.0,
        repro_threshold: 85.0,
        repro_cost: 45.0,
        repro_cooldown: 6.0,
        maturity_age: 8.0,
        max_age: 140.0,
        max_pop: 170,
        min_pop: 24,
        initial_pop: 70,
        color: Color::new(1.0, 0.62, 0.15, 1.0),
    },
    SpeciesParams {
        size: 1.15,
        max_speed: 2.2,
        accel: 3.5,
        drag: 0.9,
        sense_radius: 9.0,
        eat_radius: 2.3,
        max_energy: 100.0,
        start_energy: 60.0,
        metabolism: 0.55,
        move_cost: 1.0,
        meal_energy: 12.0,
        repro_threshold: 88.0,
        repro_cost: 50.0,
        repro_cooldown: 14.0,
        maturity_age: 18.0,
        max_age: 220.0,
        max_pop: 32,
        min_pop: 6,
        initial_pop: 14,
        color: Color::new(0.95, 0.45, 0.9, 0.75),
    },
    SpeciesParams {
        size: 1.5,
        max_speed: 9.0,
        accel: 8.0,
        drag: 0.8,
        sense_radius: 17.0,
        eat_radius: 2.0,
        max_energy: 220.0,
        start_energy: 140.0,
        metabolism: 2.0,
        move_cost: 3.6,
        meal_energy: 55.0,
        repro_threshold: 190.0,
        repro_cost: 100.0,
        repro_cooldown: 20.0,
        maturity_age: 25.0,
        max_age: 260.0,
        max_pop: 18,
        min_pop: 4,
        initial_pop: 6,
        color: Color::new(0.55, 0.62, 0.72, 1.0),
    },
];

pub struct Creature {
    pub id: u64,
    pub species: Species,
    pub pos: Vec3,
    pub vel: Vec3,
    /// Smoothed facing direction, used for rendering.
    pub heading: Vec3,
    pub energy: f32,
    pub age: f32,
    pub generation: u32,
    pub children: u32,
    pub meals: u32,
    pub alive: bool,
    pub brain: Brain,
    /// Last value of the brain's memory output, fed back next tick.
    pub memory: f32,
    /// Last throttle output, for animation and energy cost.
    pub throttle: f32,
    pub repro_timer: f32,
    /// Sharks digest after a kill before they can eat again.
    pub digest_timer: f32,
    /// > 0 while being stung; makes the creature flash.
    pub stung: f32,
    /// Animation phase (tail beat / bell pulse).
    pub phase: f32,
    /// Heritable color offset so lineages are visible in the tank.
    pub tint: Vec3,
    /// What the creature perceived last tick, drawn when it is selected.
    pub seen_food: Option<Vec3>,
    pub seen_threat: Option<Vec3>,
}

impl Creature {
    pub fn new(id: u64, species: Species, pos: Vec3, brain: Brain, generation: u32, tint: Vec3) -> Self {
        let dir = random_unit();
        Self {
            id,
            species,
            pos,
            vel: dir * 0.5,
            heading: dir,
            energy: species.params().start_energy,
            age: 0.0,
            generation,
            children: 0,
            meals: 0,
            alive: true,
            brain,
            memory: 0.0,
            throttle: 0.0,
            repro_timer: species.params().repro_cooldown,
            digest_timer: 0.0,
            stung: 0.0,
            phase: gen_range(0.0, std::f32::consts::TAU),
            tint,
            seen_food: None,
            seen_threat: None,
        }
    }

    /// Fitness used to rank genomes for the hall of fame.
    pub fn fitness(&self) -> f32 {
        self.meals as f32 + self.children as f32 * 4.0 + self.age / 20.0
    }

    pub fn color(&self) -> Color {
        let base = self.species.params().color;
        let mut c = Color::new(
            (base.r + self.tint.x).clamp(0.05, 1.0),
            (base.g + self.tint.y).clamp(0.05, 1.0),
            (base.b + self.tint.z).clamp(0.05, 1.0),
            base.a,
        );
        if self.stung > 0.0 {
            let k = (self.stung * 4.0).min(1.0);
            c.r = c.r + (1.0 - c.r) * k;
            c.g *= 1.0 - 0.6 * k;
            c.b *= 1.0 - 0.6 * k;
        }
        c
    }
}

pub fn mutate_tint(t: Vec3) -> Vec3 {
    (t + vec3(gaussian(), gaussian(), gaussian()) * 0.03).clamp(Vec3::splat(-0.35), Vec3::splat(0.35))
}

pub fn random_unit() -> Vec3 {
    loop {
        let v = vec3(gen_range(-1.0, 1.0), gen_range(-1.0, 1.0), gen_range(-1.0, 1.0));
        let l = v.length_squared();
        if l > 0.01 && l <= 1.0 {
            return v / l.sqrt();
        }
    }
}
