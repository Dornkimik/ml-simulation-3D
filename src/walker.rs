//! The physics body of the person and the loop that lets a brain control it.
//!
//! Coordinates: +x is forward (the walking direction), +y is up, and +z is
//! the person's left. Units are meters, kilograms and seconds.

use rapier3d::prelude::*;

use crate::brain::{Brain, INPUTS};

/// Control step: the brain decides 60 times per simulated second.
pub const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: usize = 4;
/// Length of one training attempt.
pub const EPISODE_SECONDS: f32 = 15.0;
/// One full stride (left step + right step) per this many seconds.
const GAIT_PERIOD: f32 = 1.0;

const DENSITY: f32 = 1000.0;
const BODY_GROUP: Group = Group::GROUP_1;
const GROUND_GROUP: Group = Group::GROUP_2;

/// Body dimensions.
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PartKind {
    Trunk,
    Thigh,
    Shin,
    Foot,
    Arm,
}

/// Pose of one body part, in plain arrays so the renderer doesn't depend on
/// the physics engine's math types.
#[derive(Clone, Copy)]
pub struct PartPose {
    pub kind: PartKind,
    pub pos: [f32; 3],
    /// Quaternion as (x, y, z, w).
    pub rot: [f32; 4],
}

/// Target joint angles a network output of 0 maps to (a relaxed stance),
/// and how far the output can move each one.
struct JointRange {
    center: f32,
    span: f32,
}

const HIP_PITCH: JointRange = JointRange { center: 0.05, span: 0.9 };
const HIP_ROLL: JointRange = JointRange { center: 0.0, span: 0.3 };
const KNEE: JointRange = JointRange { center: -0.12, span: 0.7 };
const ANKLE: JointRange = JointRange { center: 0.07, span: 0.5 };
const SHOULDER: JointRange = JointRange { center: 0.0, span: 0.8 };

struct Side {
    thigh: RigidBodyHandle,
    shin: RigidBodyHandle,
    foot: RigidBodyHandle,
    arm: RigidBodyHandle,
    hip: ImpulseJointHandle,
    knee: ImpulseJointHandle,
    ankle: ImpulseJointHandle,
    shoulder: ImpulseJointHandle,
}

pub struct Walker {
    world: PhysicsWorld,
    trunk: RigidBodyHandle,
    /// Index 0 is the left side (+z), 1 the right side (-z).
    sides: [Side; 2],
    pub time: f32,
    pub fallen: bool,
    /// Distance of the trunk along +x from the start line.
    pub distance: f32,
    pub sideways_drift: f32,
}

/// Outcome of a full training attempt.
#[derive(Clone, Copy, Default)]
pub struct Score {
    pub fitness: f32,
    pub distance: f32,
    pub time: f32,
}

fn parts_groups() -> InteractionGroups {
    InteractionGroups::new(BODY_GROUP, GROUND_GROUP, InteractionTestMode::And)
}

fn add_part(world: &mut PhysicsWorld, pos: Vec3, collider: ColliderBuilder) -> RigidBodyHandle {
    let body = RigidBodyBuilder::dynamic().translation(pos).can_sleep(false).angular_damping(0.5);
    let h = world.bodies.insert(body);
    let collider = collider.density(DENSITY).friction(1.0).collision_groups(parts_groups());
    world.colliders.insert_with_parent(collider, h, &mut world.bodies);
    h
}

/// A joint with the given free angular axes, a PD motor and limits on each.
fn motor_joint(anchor1: Vec3, anchor2: Vec3, axes: &[(JointAxis, [f32; 2], f32, f32)]) -> GenericJoint {
    let mut locked = JointAxesMask::LOCKED_FIXED_AXES;
    for (axis, ..) in axes {
        locked.remove(match axis {
            JointAxis::AngX => JointAxesMask::ANG_X,
            JointAxis::AngY => JointAxesMask::ANG_Y,
            _ => JointAxesMask::ANG_Z,
        });
    }
    let mut b = GenericJointBuilder::new(locked)
        .local_anchor1(anchor1)
        .local_anchor2(anchor2)
        .contacts_enabled(false);
    for &(axis, limits, stiffness, max_torque) in axes {
        b = b
            .limits(axis, limits)
            .motor_model(axis, MotorModel::ForceBased)
            .motor_position(axis, 0.0, stiffness, stiffness * 0.1)
            .motor_max_force(axis, max_torque);
    }
    b.build()
}

impl Walker {
    pub fn new() -> Self {
        let mut world = PhysicsWorld::new();
        // Stiff joint motors on light feet jitter at 60 Hz, so the physics
        // runs several substeps per control step.
        world.integration_parameters.dt = DT / SUBSTEPS as f32;
        world.integration_parameters.num_solver_iterations = 6;

        let ground = world.bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
        world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(1000.0, 0.5, 1000.0)
                .friction(1.0)
                .collision_groups(InteractionGroups::new(GROUND_GROUP, BODY_GROUP, InteractionTestMode::And)),
            ground,
            &mut world.bodies,
        );

        let trunk_collider = ColliderBuilder::cuboid(TRUNK_HALF[0], TRUNK_HALF[1], TRUNK_HALF[2]);
        let trunk = add_part(&mut world, Vec3::new(0.0, TRUNK_Y, 0.0), trunk_collider);
        world.colliders.insert_with_parent(
            ColliderBuilder::ball(HEAD_RADIUS)
                .translation(Vec3::new(0.0, HEAD_OFFSET, 0.0))
                .density(DENSITY)
                .collision_groups(parts_groups()),
            trunk,
            &mut world.bodies,
        );

        let hip_y = TRUNK_Y - TRUNK_HALF[1];
        let knee_y = hip_y - 2.0 * THIGH_HALF_LEN;
        let ankle_y = knee_y - 2.0 * SHIN_HALF_LEN;
        let foot_y = FOOT_HALF[1];
        let foot_x = 0.04;
        let shoulder_y = TRUNK_Y + 0.25;

        let make_side = |world: &mut PhysicsWorld, z: f32| -> Side {
            let thigh = add_part(
                world,
                Vec3::new(0.0, hip_y - THIGH_HALF_LEN, z),
                ColliderBuilder::capsule_y(THIGH_HALF_LEN, THIGH_RADIUS),
            );
            let shin = add_part(
                world,
                Vec3::new(0.0, knee_y - SHIN_HALF_LEN, z),
                ColliderBuilder::capsule_y(SHIN_HALF_LEN, SHIN_RADIUS),
            );
            let foot = add_part(
                world,
                Vec3::new(foot_x, foot_y, z),
                ColliderBuilder::cuboid(FOOT_HALF[0], FOOT_HALF[1], FOOT_HALF[2]),
            );
            let arm = add_part(
                world,
                Vec3::new(0.0, shoulder_y - ARM_HALF_LEN, z.signum() * SHOULDER_WIDTH),
                ColliderBuilder::capsule_y(ARM_HALF_LEN, ARM_RADIUS),
            );

            let hip = world.impulse_joints.insert(
                trunk,
                thigh,
                motor_joint(
                    Vec3::new(0.0, -TRUNK_HALF[1], z),
                    Vec3::new(0.0, THIGH_HALF_LEN, 0.0),
                    &[
                        (JointAxis::AngZ, [-0.9, 1.3], 3000.0, 400.0),
                        (JointAxis::AngX, [-0.45, 0.45], 3000.0, 400.0),
                    ],
                ),
                true,
            );
            let knee = world.impulse_joints.insert(
                thigh,
                shin,
                motor_joint(
                    Vec3::new(0.0, -THIGH_HALF_LEN, 0.0),
                    Vec3::new(0.0, SHIN_HALF_LEN, 0.0),
                    &[(JointAxis::AngZ, [-1.6, 0.0], 3000.0, 400.0)],
                ),
                true,
            );
            let ankle = world.impulse_joints.insert(
                shin,
                foot,
                motor_joint(
                    Vec3::new(0.0, -SHIN_HALF_LEN, 0.0),
                    Vec3::new(-foot_x, ankle_y - foot_y, 0.0),
                    &[(JointAxis::AngZ, [-0.6, 0.8], 2500.0, 250.0)],
                ),
                true,
            );
            let shoulder = world.impulse_joints.insert(
                trunk,
                arm,
                motor_joint(
                    Vec3::new(0.0, 0.25, z.signum() * SHOULDER_WIDTH),
                    Vec3::new(0.0, ARM_HALF_LEN, 0.0),
                    &[(JointAxis::AngZ, [-1.0, 1.0], 150.0, 60.0)],
                ),
                true,
            );
            Side { thigh, shin, foot, arm, hip, knee, ankle, shoulder }
        };
        let left = make_side(&mut world, HIP_WIDTH);
        let right = make_side(&mut world, -HIP_WIDTH);

        Self { world, trunk, sides: [left, right], time: 0.0, fallen: false, distance: 0.0, sideways_drift: 0.0 }
    }

    fn body(&self, h: RigidBodyHandle) -> &RigidBody {
        &self.world.bodies[h]
    }

    /// Angle of `child` relative to `parent` in the sagittal (pitch) and
    /// frontal (roll) planes, measured on the child's downward axis.
    fn limb_angles(&self, parent: RigidBodyHandle, child: RigidBodyHandle) -> (f32, f32) {
        let rel = self.body(parent).rotation().inverse() * *self.body(child).rotation();
        let down = rel * Vec3::NEG_Y;
        (down.x.atan2(-down.y), (-down.z).atan2(-down.y))
    }

    fn ankle_angle(&self, side: &Side) -> f32 {
        let rel = self.body(side.shin).rotation().inverse() * *self.body(side.foot).rotation();
        let fwd = rel * Vec3::X;
        fwd.y.atan2(fwd.x)
    }

    fn foot_down(&self, side: &Side) -> bool {
        // The lowest corner of the foot box is touching the ground.
        let b = self.body(side.foot);
        let mut lowest = f32::INFINITY;
        for sx in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let corner = Vec3::new(sx * FOOT_HALF[0], -FOOT_HALF[1], sz * FOOT_HALF[2]);
                lowest = lowest.min(b.position().transform_point(corner).y);
            }
        }
        lowest < 0.02
    }

    /// Sensor readings for one side. Lateral quantities are mirrored for the
    /// right side so one network can drive both legs.
    fn sense(&self, side_index: usize) -> [f32; INPUTS] {
        let mirror = if side_index == 0 { 1.0 } else { -1.0 };
        let phase = (self.time / GAIT_PERIOD + side_index as f32 * 0.5) * std::f32::consts::TAU;
        let trunk = self.body(self.trunk);
        let up = *trunk.rotation() * Vec3::Y;
        let angvel = trunk.angvel();
        let vel = trunk.linvel();
        let me = &self.sides[side_index];
        let other = &self.sides[1 - side_index];
        let (hip_pitch, hip_roll) = self.limb_angles(self.trunk, me.thigh);
        let (knee, _) = self.limb_angles(me.thigh, me.shin);
        let (other_hip, _) = self.limb_angles(self.trunk, other.thigh);
        let (other_knee, _) = self.limb_angles(other.thigh, other.shin);
        [
            phase.sin(),
            phase.cos(),
            up.x.asin() * 2.0,
            up.z.asin() * 2.0 * mirror,
            -angvel.z * 0.3,
            angvel.x * 0.3 * mirror,
            (trunk.translation().y - TRUNK_Y) * 4.0,
            vel.x * 0.7,
            vel.z * 0.7 * mirror,
            hip_pitch,
            hip_roll * mirror * 2.0,
            knee,
            self.ankle_angle(me) * 2.0,
            other_hip,
            other_knee,
            if self.foot_down(me) { 1.0 } else { -1.0 },
            if self.foot_down(other) { 1.0 } else { -1.0 },
            1.0,
        ]
    }

    fn set_target(&mut self, joint: ImpulseJointHandle, axis: JointAxis, range: &JointRange, output: f32) {
        let target = range.center + range.span * output;
        if let Some(j) = self.world.impulse_joints.get_mut(joint, false) {
            let m = j.data.motor(axis).copied().unwrap_or_default();
            j.data.set_motor_position(axis, target, m.stiffness, m.damping);
        }
    }

    /// Runs the brain for both sides, sets the joint motor targets, and
    /// advances the physics by one step.
    pub fn step(&mut self, brain: &mut Brain) {
        if self.fallen {
            return;
        }
        // The left side runs last so the brain viewer shows its activations.
        for side in [1, 0] {
            let mirror = if side == 0 { 1.0 } else { -1.0 };
            let input = self.sense(side);
            let out = brain.think(&input);
            let s = &self.sides[side];
            let (hip, knee, ankle, shoulder) = (s.hip, s.knee, s.ankle, s.shoulder);
            self.set_target(hip, JointAxis::AngZ, &HIP_PITCH, out[0]);
            // Positive "hip out" moves the foot away from the body's midline.
            self.set_target(hip, JointAxis::AngX, &HIP_ROLL, -out[1] * mirror);
            self.set_target(knee, JointAxis::AngZ, &KNEE, out[2]);
            self.set_target(ankle, JointAxis::AngZ, &ANKLE, out[3]);
            self.set_target(shoulder, JointAxis::AngZ, &SHOULDER, out[4]);
        }

        for _ in 0..SUBSTEPS {
            self.world.step();
        }
        self.time += DT;

        let trunk = self.body(self.trunk);
        let pos = trunk.translation();
        let up = *trunk.rotation() * Vec3::Y;
        let head_y = trunk.position().transform_point(Vec3::new(0.0, HEAD_OFFSET, 0.0)).y;
        self.distance = pos.x;
        self.sideways_drift = pos.z;
        if pos.y < 0.75 || up.y < 0.45 || head_y < 1.0 || !pos.is_finite() {
            self.fallen = true;
        }
    }

    pub fn score(&self) -> Score {
        let mut fitness = self.distance + 0.1 * self.time - 0.3 * self.sideways_drift.abs();
        if !self.fallen {
            fitness += 1.0;
        }
        Score { fitness, distance: self.distance, time: self.time }
    }

    /// Whether the left and right feet are on the ground.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn feet_down(&self) -> (bool, bool) {
        (self.foot_down(&self.sides[0]), self.foot_down(&self.sides[1]))
    }

    pub fn trunk_position(&self) -> [f32; 3] {
        self.body(self.trunk).translation().to_array()
    }

    pub fn parts(&self) -> Vec<PartPose> {
        let mut parts = Vec::with_capacity(9);
        let mut push = |kind, h: RigidBodyHandle| {
            let b = &self.world.bodies[h];
            parts.push(PartPose { kind, pos: b.translation().to_array(), rot: b.rotation().to_array() });
        };
        push(PartKind::Trunk, self.trunk);
        for s in &self.sides {
            push(PartKind::Thigh, s.thigh);
            push(PartKind::Shin, s.shin);
            push(PartKind::Foot, s.foot);
            push(PartKind::Arm, s.arm);
        }
        parts
    }
}

/// Runs one full training attempt.
pub fn evaluate(brain: &Brain) -> Score {
    let mut brain = brain.clone();
    let mut walker = Walker::new();
    while !walker.fallen && walker.time < EPISODE_SECONDS {
        walker.step(&mut brain);
    }
    walker.score()
}

