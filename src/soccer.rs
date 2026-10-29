//! The soccer match: a small walled pitch, a ball, and two players that are
//! each controlled by their own brain.
//!
//! Coordinates: +x points from the blue goal to the red goal, +y is up, and
//! +z is blue's left when it attacks. Blue (team 0) attacks +x, red (team 1)
//! attacks -x. Units are meters and seconds. Players move on the ground; the
//! ball flies, bounces and rolls in 3D.

use std::f32::consts::{PI, TAU};

use macroquad::prelude::{vec2, vec3, Vec2, Vec3};

use crate::brain::{Brain, INPUTS, OUTPUTS};

/// Control step: the brains decide 60 times per simulated second.
pub const DT: f32 = 1.0 / 60.0;
/// Length of one training match.
pub const TRAINING_MATCH_SECONDS: f32 = 20.0;

/// Pitch size: the walls are at x = +/-HALF_LENGTH and z = +/-HALF_WIDTH.
pub const HALF_LENGTH: f32 = 14.0;
pub const HALF_WIDTH: f32 = 9.0;
pub const GOAL_HALF_WIDTH: f32 = 2.5;
pub const GOAL_HEIGHT: f32 = 2.0;
pub const GOAL_DEPTH: f32 = 1.2;
pub const POST_RADIUS: f32 = 0.06;
pub const BALL_RADIUS: f32 = 0.11;
pub const PLAYER_RADIUS: f32 = 0.3;
const PLAYER_HEIGHT: f32 = 1.8;

pub const MAX_SPEED: f32 = 6.0;
const ACCELERATION: f32 = 14.0;
const TURN_RATE: f32 = 6.0;
/// The ball can be kicked when its center is this close and in front.
const KICK_REACH: f32 = 0.8;
const KICK_COOLDOWN: f32 = 0.35;
const KICK_MIN_SPEED: f32 = 4.0;
const KICK_MAX_SPEED: f32 = 16.0;
const GRAVITY: f32 = 9.81;
const WALL_BOUNCE: f32 = 0.6;
const POST_BOUNCE: f32 = 0.6;
const NET_BOUNCE: f32 = 0.15;
/// The ball moves in smaller steps than the players, so even the hardest
/// shot can't skip through a post or a wall between two steps.
const BALL_SUBSTEPS: usize = 4;
/// How bouncy the ball is when a player runs into it (low, so it can be dribbled).
const BODY_BOUNCE: f32 = 0.3;
/// If the ball stays within `STALL_DISTANCE` of one spot for this long (say,
/// both players are stuck in a corner), the referee restarts with a kickoff.
const STALL_SECONDS: f32 = 5.0;
const STALL_DISTANCE: f32 = 1.0;

/// Training score: what each part of a match is worth.
const GOAL_REWARD: f32 = 4.0;
/// Per meter the ball moves toward the opponent's goal (negative when away).
const PROGRESS_REWARD: f32 = 0.15;
/// Per meter of average distance from the ball, so beginners learn to chase it.
const DISTANCE_PENALTY: f32 = 0.05;
const KICK_REWARD: f32 = 0.05;
const MAX_REWARDED_KICKS: u32 = 20;

/// +1 for blue, which attacks +x, and -1 for red.
pub fn team_sign(team: usize) -> f32 {
    if team == 0 {
        1.0
    } else {
        -1.0
    }
}

pub struct Player {
    /// Position on the ground as (x, z).
    pub pos: Vec2,
    pub vel: Vec2,
    /// Heading angle: the player faces (cos, sin) in (x, z).
    pub facing: f32,
    cooldown: f32,
    /// Number of kicks so far, so the renderer can spot new ones.
    pub kicks: u32,
}

impl Player {
    pub fn dir(&self) -> Vec2 {
        vec2(self.facing.cos(), self.facing.sin())
    }

    fn left(&self) -> Vec2 {
        let d = self.dir();
        vec2(-d.y, d.x)
    }

    /// A ground vector in the player's own frame: (ahead, to the left).
    pub fn to_body(&self, v: Vec2) -> Vec2 {
        vec2(v.dot(self.dir()), v.dot(self.left()))
    }
}

pub struct Ball {
    pub pos: Vec3,
    pub vel: Vec3,
    /// Behind a goal line, having crossed it through the goal mouth. Only
    /// then is the ball held by the net instead of bouncing off the wall.
    in_goal: bool,
}

impl Ball {
    fn at_rest(pos: Vec3) -> Self {
        Self { pos, vel: Vec3::ZERO, in_goal: false }
    }

    fn ground_pos(&self) -> Vec2 {
        vec2(self.pos.x, self.pos.z)
    }

    /// Pushes the ball out of a round bar (a post or the crossbar) and
    /// bounces it off. `offset` goes from the bar's axis to the ball's
    /// center, at right angles to the bar.
    fn bounce_off_bar(&mut self, offset: Vec3) {
        let dist = offset.length();
        let min = BALL_RADIUS + POST_RADIUS;
        if dist >= min || dist < 1e-6 {
            return;
        }
        let n = offset / dist;
        self.pos += n * (min - dist);
        let into = self.vel.dot(n);
        if into < 0.0 {
            self.vel -= n * into * (1.0 + POST_BOUNCE);
        }
    }
}

fn clamp_to_pitch(p: &mut Player) {
    let (lx, lz) = (HALF_LENGTH - PLAYER_RADIUS, HALF_WIDTH - PLAYER_RADIUS);
    if p.pos.x.abs() > lx {
        p.pos.x = lx.copysign(p.pos.x);
        p.vel.x = 0.0;
    }
    if p.pos.y.abs() > lz {
        p.pos.y = lz.copysign(p.pos.y);
        p.vel.y = 0.0;
    }
}

/// What one team did during a match, for the training score.
#[derive(Clone, Copy, Default)]
struct Tally {
    goals: u32,
    ball_progress: f32,
    ball_distance_sum: f32,
    kicks: u32,
}

/// A match result from one team's point of view.
#[derive(Clone, Copy, Default)]
pub struct Outcome {
    pub fitness: f32,
    pub goals_for: u32,
    pub goals_against: u32,
}

pub struct Match {
    pub players: [Player; 2],
    pub ball: Ball,
    pub time: f32,
    pub duration: f32,
    pub score: [u32; 2],
    /// Seconds left of the pause after a goal, before the next kickoff.
    pub celebrating: f32,
    pub last_scorer: Option<usize>,
    /// Kickoffs after a stall, without a goal.
    pub restarts: u32,
    celebration: f32,
    /// Where the ball has stayed close to, and for how long.
    stall_spot: Vec2,
    stall_time: f32,
    tally: [Tally; 2],
    steps: u32,
    rng: Rng,
}

impl Match {
    /// A new match whose kickoffs are drawn from `seed`. After a goal the
    /// game pauses for `celebration` seconds (0 for training).
    pub fn new(seed: u64, duration: f32, celebration: f32) -> Self {
        let idle = || Player { pos: Vec2::ZERO, vel: Vec2::ZERO, facing: 0.0, cooldown: 0.0, kicks: 0 };
        let mut m = Self {
            players: [idle(), idle()],
            ball: Ball::at_rest(Vec3::ZERO),
            time: 0.0,
            duration,
            score: [0, 0],
            celebrating: 0.0,
            last_scorer: None,
            restarts: 0,
            celebration,
            stall_spot: Vec2::ZERO,
            stall_time: 0.0,
            tally: [Tally::default(); 2],
            steps: 0,
            rng: Rng::new(seed),
        };
        m.kickoff();
        m
    }

    pub fn over(&self) -> bool {
        self.time >= self.duration && self.celebrating <= 0.0
    }

    /// Ball near the middle, players somewhere in their own half facing
    /// roughly toward the opponent's goal.
    fn kickoff(&mut self) {
        let r = &mut self.rng;
        self.ball = Ball::at_rest(vec3(0.0, BALL_RADIUS, r.range(-3.0, 3.0)));
        for (team, p) in self.players.iter_mut().enumerate() {
            let s = team_sign(team);
            p.pos = vec2(-s * r.range(4.0, 7.0), r.range(-5.0, 5.0));
            p.vel = Vec2::ZERO;
            p.facing = team as f32 * PI + r.range(-0.6, 0.6);
            p.cooldown = 0.0;
        }
        self.stall_spot = self.ball.ground_pos();
        self.stall_time = 0.0;
    }

    fn ball_in_reach(&self, i: usize) -> bool {
        let p = &self.players[i];
        let rel = self.ball.ground_pos() - p.pos;
        let dist = rel.length();
        dist < KICK_REACH && rel.dot(p.dir()) > 0.3 * dist && self.ball.pos.y < 0.8
    }

    /// Sensor readings for player `i`, all relative to the player itself.
    fn sense(&self, i: usize) -> [f32; INPUTS] {
        let me = &self.players[i];
        let opp = &self.players[1 - i];
        let s = team_sign(i);
        let ball = me.to_body(self.ball.ground_pos() - me.pos);
        let ball_vel = me.to_body(vec2(self.ball.vel.x, self.ball.vel.z));
        let opp_rel = me.to_body(opp.pos - me.pos);
        let goal = me.to_body(vec2(s * HALF_LENGTH, 0.0) - me.pos);
        let own_goal = me.to_body(vec2(-s * HALF_LENGTH, 0.0) - me.pos);
        let vel = me.to_body(me.vel);
        [
            ball.x / 8.0,
            ball.y / 8.0,
            ball.length() / 8.0,
            ball_vel.x / 8.0,
            ball_vel.y / 8.0,
            self.ball.pos.y - BALL_RADIUS,
            opp_rel.x / 8.0,
            opp_rel.y / 8.0,
            goal.x / 15.0,
            goal.y / 15.0,
            own_goal.x / 15.0,
            own_goal.y / 15.0,
            vel.x / MAX_SPEED,
            vel.y / MAX_SPEED,
            s * me.pos.x / HALF_LENGTH,
            s * me.pos.y / HALF_WIDTH,
            if self.ball_in_reach(i) { 1.0 } else { -1.0 },
            1.0,
        ]
    }

    /// Advances the match by one control step.
    pub fn step(&mut self, brains: &mut [Brain; 2]) {
        if self.over() {
            return;
        }
        if self.celebrating > 0.0 {
            // The players jog to a stop while the ball settles in the net.
            for p in &mut self.players {
                p.vel *= 0.92;
                p.pos += p.vel * DT;
            }
            self.update_ball();
            self.celebrating -= DT;
            if self.celebrating <= 0.0 {
                self.kickoff();
            }
            return;
        }

        let inputs = [self.sense(0), self.sense(1)];
        let outputs = [brains[0].think(&inputs[0]), brains[1].think(&inputs[1])];
        for (i, out) in outputs.iter().enumerate() {
            self.run(i, out);
        }
        // Alternate who gets to kick first, so neither side has an edge.
        let first = (self.steps % 2) as usize;
        for i in [first, 1 - first] {
            self.kick(i, &outputs[i]);
        }
        self.collide_players();

        let before = self.ball.pos.x;
        self.update_ball();
        let moved = self.ball.pos.x - before;
        self.tally[0].ball_progress += moved;
        self.tally[1].ball_progress -= moved;
        for i in 0..2 {
            self.tally[i].ball_distance_sum += self.ball.ground_pos().distance(self.players[i].pos);
        }

        self.time += DT;
        self.steps += 1;
        self.check_goal();
        if self.celebrating > 0.0 {
            return;
        }

        if self.ball.ground_pos().distance(self.stall_spot) > STALL_DISTANCE {
            self.stall_spot = self.ball.ground_pos();
            self.stall_time = 0.0;
        }
        self.stall_time += DT;
        if self.stall_time > STALL_SECONDS {
            self.restarts += 1;
            self.kickoff();
        }
    }

    fn run(&mut self, i: usize, out: &[f32; OUTPUTS]) {
        let p = &mut self.players[i];
        p.facing = (p.facing + out[2] * TURN_RATE * DT).rem_euclid(TAU);
        // Running backwards is slower than running forwards.
        let ahead = out[0] * if out[0] > 0.0 { MAX_SPEED } else { MAX_SPEED * 0.5 };
        let desired = p.dir() * ahead + p.left() * (out[1] * MAX_SPEED * 0.6);
        p.vel += (desired - p.vel).clamp_length_max(ACCELERATION * DT);
        p.pos += p.vel * DT;
        p.cooldown = (p.cooldown - DT).max(0.0);
        clamp_to_pitch(p);
    }

    /// Kicks the ball where the player is facing, if it wants to and can.
    fn kick(&mut self, i: usize, out: &[f32; OUTPUTS]) {
        if out[3] <= 0.0 || self.players[i].cooldown > 0.0 || !self.ball_in_reach(i) {
            return;
        }
        let p = &mut self.players[i];
        let power = KICK_MIN_SPEED + (KICK_MAX_SPEED - KICK_MIN_SPEED) * out[3];
        let ground = p.dir() * power + p.vel * 0.3;
        self.ball.vel = vec3(ground.x, out[4].max(0.0) * 0.4 * power, ground.y);
        p.cooldown = KICK_COOLDOWN;
        p.kicks += 1;
        self.tally[i].kicks += 1;
    }

    /// Players can't overlap: push them apart and share their speed along
    /// the line between them, like two equal bodies bumping.
    fn collide_players(&mut self) {
        let [a, b] = &mut self.players;
        let d = b.pos - a.pos;
        let dist = d.length();
        let min = 2.0 * PLAYER_RADIUS;
        if dist >= min {
            return;
        }
        let n = if dist > 1e-6 { d / dist } else { Vec2::X };
        let push = n * (min - dist) * 0.5;
        a.pos -= push;
        b.pos += push;
        let closing = (b.vel - a.vel).dot(n);
        if closing < 0.0 {
            a.vel += n * closing * 0.5;
            b.vel -= n * closing * 0.5;
        }
        // Being bumped mustn't push anyone through a wall.
        clamp_to_pitch(a);
        clamp_to_pitch(b);
    }

    /// Moves the ball through one control step, in substeps.
    fn update_ball(&mut self) {
        let h = DT / BALL_SUBSTEPS as f32;
        for _ in 0..BALL_SUBSTEPS {
            let b = &mut self.ball;
            let r = BALL_RADIUS;
            let prev = b.pos;
            b.vel.y -= GRAVITY * h;
            b.pos += b.vel * h;

            if b.pos.y < r {
                b.pos.y = r;
                b.vel.y = if b.vel.y < -1.0 { -b.vel.y * 0.55 } else { 0.0 };
            }
            // Rolling friction on the grass, light air drag in flight.
            let on_ground = b.pos.y <= r + 1e-3;
            let ground = vec2(b.vel.x, b.vel.z);
            let speed = ground.length();
            if speed > 0.0 {
                let slowed = if on_ground { speed * (-0.6 * h).exp() - 0.25 * h } else { speed * (-0.05 * h).exp() };
                let ground = ground * (slowed.max(0.0) / speed);
                b.vel.x = ground.x;
                b.vel.z = ground.y;
            }

            self.collide_ball_with_pitch(prev);
            self.collide_ball_with_players();
        }
    }

    /// Keeps the ball inside the walls, bounces it off the goal frames, and
    /// lets it into a goal only through the goal mouth. `prev` is where the
    /// ball was before it last moved.
    fn collide_ball_with_pitch(&mut self, prev: Vec3) {
        let b = &mut self.ball;
        let r = BALL_RADIUS;

        // Posts and crossbars.
        for x in [-HALF_LENGTH, HALF_LENGTH] {
            if b.pos.y < GOAL_HEIGHT + POST_RADIUS {
                for z in [-GOAL_HALF_WIDTH - POST_RADIUS, GOAL_HALF_WIDTH + POST_RADIUS] {
                    b.bounce_off_bar(vec3(b.pos.x - x, 0.0, b.pos.z - z));
                }
            }
            if b.pos.z.abs() < GOAL_HALF_WIDTH + POST_RADIUS {
                b.bounce_off_bar(vec3(b.pos.x - x, b.pos.y - (GOAL_HEIGHT + POST_RADIUS), 0.0));
            }
        }

        // Side walls.
        if b.pos.z.abs() > HALF_WIDTH - r {
            b.pos.z = (HALF_WIDTH - r).copysign(b.pos.z);
            if b.vel.z * b.pos.z > 0.0 {
                b.vel.z *= -WALL_BOUNCE;
            }
        }

        let side = b.pos.x.signum();
        let x = b.pos.x.abs();
        if !b.in_goal && x > HALF_LENGTH && prev.x.abs() <= HALF_LENGTH {
            // The ball's center crossed a goal line: where?
            let t = (HALF_LENGTH - prev.x.abs()) / (x - prev.x.abs());
            let at = prev + (b.pos - prev) * t;
            b.in_goal = at.z.abs() < GOAL_HALF_WIDTH && at.y < GOAL_HEIGHT;
        } else if b.in_goal && x < HALF_LENGTH {
            // Back out onto the pitch, through the mouth.
            b.in_goal = false;
        }

        if b.in_goal {
            // The net: back, sides and roof.
            let back = HALF_LENGTH + GOAL_DEPTH - r;
            if x > back {
                b.pos.x = side * back;
                if b.vel.x * side > 0.0 {
                    b.vel.x *= -NET_BOUNCE;
                }
            }
            if b.pos.z.abs() > GOAL_HALF_WIDTH - r {
                b.pos.z = (GOAL_HALF_WIDTH - r).copysign(b.pos.z);
                if b.vel.z * b.pos.z > 0.0 {
                    b.vel.z *= -NET_BOUNCE;
                }
            }
            if b.pos.y > GOAL_HEIGHT - r {
                b.pos.y = GOAL_HEIGHT - r;
                b.vel.y = b.vel.y.min(0.0);
            }
        } else if x > HALF_LENGTH - r {
            // The end wall, open only at the goal mouth (the posts and
            // crossbar take care of its edges).
            let in_mouth = b.pos.z.abs() < GOAL_HALF_WIDTH && b.pos.y < GOAL_HEIGHT;
            if !in_mouth || x > HALF_LENGTH {
                b.pos.x = side * (HALF_LENGTH - r);
                if b.vel.x * side > 0.0 {
                    b.vel.x *= -WALL_BOUNCE;
                }
            }
        }
    }

    /// Players are solid upright cylinders: running into the ball pushes it
    /// along, and a ball dropping onto a head bounces up off it. If the ball
    /// is pinned against a wall, the player stops instead of pushing it
    /// through.
    fn collide_ball_with_players(&mut self) {
        let min = PLAYER_RADIUS + BALL_RADIUS;
        for i in 0..2 {
            let p = &self.players[i];
            let d = self.ball.ground_pos() - p.pos;
            let dist = d.length();
            let below_top = PLAYER_HEIGHT - (self.ball.pos.y - BALL_RADIUS);
            if dist >= min || below_top <= 0.0 {
                continue;
            }
            // Leave the player the shorter way: up over the top, or out sideways.
            if below_top < min - dist {
                let b = &mut self.ball;
                b.pos.y += below_top;
                if b.vel.y < 0.0 {
                    b.vel.y *= -BODY_BOUNCE;
                }
                continue;
            }
            let n = if dist > 1e-6 { d / dist } else { p.dir() };
            let (p_pos, p_vel) = (p.pos, p.vel);
            let b = &mut self.ball;
            let before = b.pos;
            b.pos.x = p_pos.x + n.x * min;
            b.pos.z = p_pos.y + n.y * min;
            let mut v = vec2(b.vel.x, b.vel.z);
            let closing = (v - p_vel).dot(n);
            if closing < 0.0 {
                v -= n * closing * (1.0 + BODY_BOUNCE);
                b.vel.x = v.x;
                b.vel.z = v.y;
            }

            self.collide_ball_with_pitch(before);
            let ball = self.ball.ground_pos();
            let p = &mut self.players[i];
            let d = ball - p.pos;
            if d.length() < min - 1e-4 {
                let n = d.try_normalize().unwrap_or(n);
                p.pos = ball - n * min;
                let into = p.vel.dot(n);
                if into > 0.0 {
                    p.vel -= n * into;
                }
                clamp_to_pitch(p);
            }
        }
    }

    fn check_goal(&mut self) {
        let b = &self.ball;
        if !b.in_goal || b.pos.x.abs() < HALF_LENGTH + BALL_RADIUS {
            return;
        }
        let scorer = if b.pos.x > 0.0 { 0 } else { 1 };
        self.score[scorer] += 1;
        self.tally[scorer].goals += 1;
        self.last_scorer = Some(scorer);
        if self.celebration > 0.0 {
            self.celebrating = self.celebration;
        } else {
            self.kickoff();
        }
    }

    /// The match so far from `team`'s point of view.
    pub fn outcome(&self, team: usize) -> Outcome {
        let me = &self.tally[team];
        let them = &self.tally[1 - team];
        let fitness = GOAL_REWARD * (me.goals as f32 - them.goals as f32) + PROGRESS_REWARD * me.ball_progress
            - DISTANCE_PENALTY * me.ball_distance_sum / self.steps.max(1) as f32
            + KICK_REWARD * me.kicks.min(MAX_REWARDED_KICKS) as f32;
        Outcome { fitness, goals_for: me.goals, goals_against: them.goals }
    }
}

/// Plays a full training match with `brain` on `team` against `opponent`
/// and scores it from `brain`'s point of view.
pub fn play(brain: &Brain, opponent: &Brain, seed: u64, team: usize) -> Outcome {
    let mut brains = [brain.clone(), opponent.clone()];
    if team == 1 {
        brains.swap(0, 1);
    }
    let mut game = Match::new(seed, TRAINING_MATCH_SECONDS, 0.0);
    while !game.over() {
        game.step(&mut brains);
    }
    game.outcome(team)
}

/// A small, fast random number generator (xorshift64*), so matches can be
/// replayed exactly from a seed on any thread.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The same brain on both teams, started from mirror-image positions,
    /// must play a mirror-image game. This catches sign mistakes in the
    /// sensors or controls of either team.
    #[test]
    fn teams_are_mirror_images() {
        macroquad::rand::srand(3);
        let brain = Brain::random();
        let mut brains = [brain.clone(), brain];
        let mut game = Match::new(1, 10.0, 0.0);
        game.ball.pos = vec3(0.0, BALL_RADIUS, 0.0);
        game.players[0].pos = vec2(-3.0, 1.5);
        game.players[0].facing = 0.4;
        game.players[1].pos = vec2(3.0, -1.5);
        game.players[1].facing = 0.4 + PI;
        // Short enough that nobody reaches the ball (two kicks on the same
        // step can't be mirrored).
        for _ in 0..40 {
            game.step(&mut brains);
            let [a, b] = &game.players;
            assert!((a.pos + b.pos).length() < 1e-2, "players {:?} and {:?} are not mirrored", a.pos, b.pos);
            assert!(game.ball.ground_pos().length() < 1e-2);
        }
    }

    /// A brain that ignores its senses: it always runs at `run` (-1..1) and
    /// never kicks.
    fn runner(run: f32) -> Brain {
        let mut brain = Brain::from_weights(&vec![0.0; Brain::random().weights().count()]).unwrap();
        let out = &mut brain.layers[1];
        let stride = out.inputs + 1;
        out.weights[stride - 1] = run.clamp(-0.999, 0.999).atanh();
        brain
    }

    /// Players standing still, far from where the ball is.
    fn quiet_match() -> (Match, [Brain; 2]) {
        let mut game = Match::new(1, 60.0, 0.0);
        game.players[0].pos = vec2(-10.0, 5.0);
        game.players[1].pos = vec2(-10.0, -5.0);
        (game, [runner(0.0), runner(0.0)])
    }

    /// The ball is on the pitch, or inside one of the goals.
    pub(crate) fn assert_ball_legal(game: &Match) {
        let (b, r, eps) = (game.ball.pos, BALL_RADIUS, 1e-3);
        assert!(b.is_finite(), "ball at {b}");
        assert!(b.y >= r - eps, "ball below the ground at {b}");
        assert!(b.z.abs() <= HALF_WIDTH - r + eps, "ball through a side wall at {b}");
        if b.x.abs() > HALF_LENGTH - r + eps {
            assert!(
                b.x.abs() <= HALF_LENGTH + GOAL_DEPTH - r + eps && b.z.abs() <= GOAL_HALF_WIDTH && b.y <= GOAL_HEIGHT,
                "ball through an end wall at {b}"
            );
        }
    }

    /// Steps the match, checking the ball after every step.
    fn run_checked(game: &mut Match, brains: &mut [Brain; 2], steps: usize) {
        for _ in 0..steps {
            game.step(brains);
            assert_ball_legal(game);
        }
    }

    #[test]
    fn ball_in_the_net_is_a_goal() {
        let (mut game, mut brains) = quiet_match();
        game.ball.pos = vec3(HALF_LENGTH - 1.0, BALL_RADIUS, 0.5);
        game.ball.vel = vec3(10.0, 0.0, 0.0);
        run_checked(&mut game, &mut brains, 30);
        assert_eq!(game.score, [1, 0]);
    }

    #[test]
    fn hard_shot_at_the_wall_beside_the_goal_bounces_back() {
        for z in [GOAL_HALF_WIDTH + 0.5, 4.0, HALF_WIDTH - 0.5] {
            let (mut game, mut brains) = quiet_match();
            game.ball.pos = vec3(HALF_LENGTH - 0.5, BALL_RADIUS, z);
            game.ball.vel = vec3(KICK_MAX_SPEED + 2.0, 0.0, 0.0);
            run_checked(&mut game, &mut brains, 30);
            assert_eq!(game.score, [0, 0], "a shot at z = {z} went in");
            assert!(game.ball.vel.x < 0.0, "a shot at z = {z} didn't bounce back");
        }
    }

    #[test]
    fn shot_over_the_bar_is_no_goal() {
        let (mut game, mut brains) = quiet_match();
        game.ball.pos = vec3(HALF_LENGTH - 2.0, GOAL_HEIGHT + 0.4, 0.0);
        game.ball.vel = vec3(KICK_MAX_SPEED, 1.0, 0.0);
        run_checked(&mut game, &mut brains, 60);
        assert_eq!(game.score, [0, 0]);
    }

    #[test]
    fn shot_at_the_post_bounces_off() {
        let (mut game, mut brains) = quiet_match();
        game.ball.pos = vec3(HALF_LENGTH - 2.0, BALL_RADIUS, GOAL_HALF_WIDTH + POST_RADIUS);
        game.ball.vel = vec3(12.0, 0.0, 0.0);
        run_checked(&mut game, &mut brains, 60);
        assert_eq!(game.score, [0, 0]);
        assert!(game.ball.pos.x < HALF_LENGTH - 0.5, "the ball didn't bounce off the post");
    }

    /// A player running the ball into a wall must not push it through.
    #[test]
    fn ball_pinned_against_a_wall_stays_on_the_pitch() {
        // Into the end wall beside the goal, into a goal post, and into a
        // side wall.
        for (pos, facing) in [(vec2(12.0, 3.2), 0.0), (vec2(12.0, GOAL_HALF_WIDTH + 2.0 * POST_RADIUS), 0.0), (vec2(0.0, HALF_WIDTH - 1.5), PI / 2.0)] {
            let (mut game, _) = quiet_match();
            let mut brains = [runner(1.0), runner(0.0)];
            game.players[0].pos = pos;
            game.players[0].facing = facing;
            let dir = game.players[0].dir();
            game.ball.pos = vec3(pos.x + dir.x * 0.5, BALL_RADIUS, pos.y + dir.y * 0.5);
            run_checked(&mut game, &mut brains, 180);
            assert_eq!(game.score, [0, 0], "pushing from {pos} scored");
            let gap = game.ball.ground_pos().distance(game.players[0].pos);
            assert!(gap >= PLAYER_RADIUS + BALL_RADIUS - 1e-3, "pushing from {pos}: player and ball overlap ({gap})");
        }
    }

    #[test]
    fn ball_dropping_on_a_head_bounces_up() {
        let (mut game, mut brains) = quiet_match();
        let head = game.players[0].pos;
        game.ball.pos = vec3(head.x + 0.15, PLAYER_HEIGHT + 1.0, head.y);
        for _ in 0..60 {
            let prev = game.ball.pos;
            game.step(&mut brains);
            assert!(game.ball.ground_pos().distance(vec2(prev.x, prev.z)) < 0.05, "the ball jumped sideways off the head");
            if game.ball.vel.y > 0.0 {
                return;
            }
        }
        panic!("the ball never bounced off the head");
    }

    /// Lots of chaotic matches between random brains (which kick a lot):
    /// the ball must stay on the pitch, and every goal must have gone in
    /// through the goal mouth.
    #[test]
    fn random_matches_obey_the_walls() {
        macroquad::rand::srand(11);
        for seed in 0..60 {
            let mut brains = [Brain::random(), Brain::random()];
            let mut game = Match::new(seed, TRAINING_MATCH_SECONDS, 0.0);
            let mut prev = game.ball.pos;
            while !game.over() {
                let goals = game.score;
                game.step(&mut brains);
                assert_ball_legal(&game);
                if game.score != goals {
                    assert!(
                        prev.z.abs() < GOAL_HALF_WIDTH && prev.y < GOAL_HEIGHT && prev.x.abs() > HALF_LENGTH - 1.0,
                        "seed {seed}: goal from a ball that was at {prev}"
                    );
                }
                prev = game.ball.pos;
            }
        }
    }
}
