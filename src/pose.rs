//! What a player looks like: body part sizes, and a running and kicking
//! animation driven by how the player moves in the match.
//!
//! Body-local axes: +x is where the player faces, +y is up, +z is its left.

use std::f32::consts::{PI, TAU};

use macroquad::prelude::*;

use crate::soccer::{Player, DT, MAX_SPEED};

pub const TRUNK_HALF: [f32; 3] = [0.09, 0.30, 0.16];
pub const HEAD_RADIUS: f32 = 0.11;
pub const HEAD_OFFSET: f32 = 0.47;
pub const THIGH_HALF_LEN: f32 = 0.215;
pub const THIGH_RADIUS: f32 = 0.065;
pub const SHIN_HALF_LEN: f32 = 0.215;
pub const SHIN_RADIUS: f32 = 0.05;
pub const FOOT_HALF: [f32; 3] = [0.13, 0.04, 0.05];
pub const ARM_HALF_LEN: f32 = 0.3;
pub const ARM_RADIUS: f32 = 0.045;
const HIP_WIDTH: f32 = 0.1;
const SHOULDER_WIDTH: f32 = 0.21;
const TRUNK_Y: f32 = 1.25;
/// Where the foot's center sits relative to the ankle, in the foot's frame.
const FOOT_OFFSET: Vec3 = vec3(0.04, -0.05, 0.0);
const KICK_SECONDS: f32 = 0.4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PartKind {
    Trunk,
    Thigh,
    Shin,
    Foot,
    Arm,
}

#[derive(Clone, Copy)]
pub struct PartPose {
    pub kind: PartKind,
    pub pos: Vec3,
    pub rot: Quat,
}

/// Animation state of one player.
pub struct Anim {
    /// Running cycle, 0..TAU per stride.
    phase: f32,
    /// Seconds since the last kick started.
    kick: f32,
    kicks_seen: u32,
    lean: f32,
}

impl Anim {
    pub fn new(p: &Player) -> Self {
        Self { phase: 0.0, kick: KICK_SECONDS, kicks_seen: p.kicks, lean: 0.0 }
    }

    /// Advances the animation by one match step.
    pub fn update(&mut self, p: &Player) {
        let speed = p.vel.length();
        let stride = 1.0 + 0.3 * speed;
        self.phase = (self.phase + speed * DT / stride * TAU) % TAU;
        if p.kicks != self.kicks_seen {
            self.kicks_seen = p.kicks;
            self.kick = 0.0;
        }
        self.kick = (self.kick + DT).min(KICK_SECONDS);
        let target_lean = 0.25 * p.to_body(p.vel).x / MAX_SPEED;
        self.lean += (target_lean - self.lean) * (8.0 * DT);
    }

    /// World poses of all body parts, with the lower foot on the ground.
    pub fn parts(&self, p: &Player) -> Vec<PartPose> {
        let speed = p.vel.length();
        let amount = (speed / MAX_SPEED).min(1.0);
        let moving = if speed > 0.05 { p.to_body(p.vel) / speed } else { Vec2::X };

        let trunk_rot = Quat::from_rotation_y(-p.facing) * Quat::from_rotation_z(-self.lean);
        let trunk_pos = vec3(p.pos.x, TRUNK_Y, p.pos.y);
        let mut parts = vec![PartPose { kind: PartKind::Trunk, pos: trunk_pos, rot: trunk_rot }];
        let kick = (self.kick < KICK_SECONDS).then(|| kick_angles(self.kick / KICK_SECONDS));

        let mut lowest = f32::INFINITY;
        for side in 0..2 {
            let z = if side == 0 { 1.0 } else { -1.0 };
            let phase = self.phase + side as f32 * PI;
            // Legs swing along the direction of travel; the knee bends most
            // while the leg swings forward.
            let swing = 0.75 * amount * phase.sin();
            let mut hip = swing * moving.x;
            let roll = 0.6 * swing * moving.y;
            let mut knee = -(0.08 + amount * (0.25 + 1.2 * phase.cos().max(0.0)));
            let ankle = 0.15 * amount * phase.sin();
            let mut arm_out = 0.08;
            // The right foot kicks.
            if let Some((kick_hip, kick_knee, weight)) = kick {
                if side == 1 {
                    hip += (kick_hip - hip) * weight;
                    knee += (kick_knee - knee) * weight;
                }
                arm_out += 0.5 * weight;
            }

            let hip_pos = trunk_pos + trunk_rot * vec3(0.0, -TRUNK_HALF[1], z * HIP_WIDTH);
            let thigh_rot = trunk_rot * Quat::from_rotation_z(hip) * Quat::from_rotation_x(-roll);
            let knee_pos = hip_pos + thigh_rot * vec3(0.0, -2.0 * THIGH_HALF_LEN, 0.0);
            let shin_rot = thigh_rot * Quat::from_rotation_z(knee);
            let ankle_pos = knee_pos + shin_rot * vec3(0.0, -2.0 * SHIN_HALF_LEN, 0.0);
            let foot_rot = shin_rot * Quat::from_rotation_z(ankle);
            let foot_pos = ankle_pos + foot_rot * FOOT_OFFSET;
            for (sx, sz) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                let corner = vec3(sx * FOOT_HALF[0], -FOOT_HALF[1], sz * FOOT_HALF[2]);
                lowest = lowest.min((foot_pos + foot_rot * corner).y);
            }
            parts.push(PartPose { kind: PartKind::Thigh, pos: hip_pos + thigh_rot * vec3(0.0, -THIGH_HALF_LEN, 0.0), rot: thigh_rot });
            parts.push(PartPose { kind: PartKind::Shin, pos: knee_pos + shin_rot * vec3(0.0, -SHIN_HALF_LEN, 0.0), rot: shin_rot });
            parts.push(PartPose { kind: PartKind::Foot, pos: foot_pos, rot: foot_rot });

            // Arms swing against the leg on the same side.
            let shoulder = trunk_pos + trunk_rot * vec3(0.0, 0.25, z * SHOULDER_WIDTH);
            let arm_rot = trunk_rot * Quat::from_rotation_z(-0.9 * swing * moving.x) * Quat::from_rotation_x(-z * arm_out);
            parts.push(PartPose { kind: PartKind::Arm, pos: shoulder + arm_rot * vec3(0.0, -ARM_HALF_LEN, 0.0), rot: arm_rot });
        }

        for part in &mut parts {
            part.pos.y -= lowest;
        }
        parts
    }
}

/// Hip and knee angles of the kicking leg at time `t` (0..1) of a kick, and
/// how much they override the running animation: a whip-like swing through
/// the ball, then back.
fn kick_angles(t: f32) -> (f32, f32, f32) {
    let (hip, knee) = if t < 0.3 {
        let s = t / 0.3;
        (-0.5 + 1.7 * s, -1.3 * (1.0 - s))
    } else {
        let s = (t - 0.3) / 0.7;
        (1.2 * (1.0 - s), -0.3 * s)
    };
    let weight = (t / 0.06).min(1.0) * ((1.0 - t) / 0.25).min(1.0);
    (hip, knee, weight)
}
