//! 3D rendering of the pitch, the players and the ball. Macroquad has no
//! built-in lighting, so meshes are lit and fogged per vertex on the CPU and
//! submitted in large batches.

use macroquad::models::{Mesh, Vertex};
use macroquad::prelude::*;
use macroquad::rand::gen_range;

use crate::pose::{
    PartKind, PartPose, ARM_HALF_LEN, ARM_RADIUS, FOOT_HALF, HEAD_OFFSET, HEAD_RADIUS, SHIN_HALF_LEN, SHIN_RADIUS,
    THIGH_HALF_LEN, THIGH_RADIUS, TRUNK_HALF,
};
use crate::soccer::{BALL_RADIUS, GOAL_DEPTH, GOAL_HALF_WIDTH, GOAL_HEIGHT, HALF_LENGTH, HALF_WIDTH};

pub const SKY_TOP: Color = Color::new(0.33, 0.56, 0.86, 1.0);
pub const HORIZON: Color = Color::new(0.78, 0.86, 0.93, 1.0);
const BATCH_VERTS: usize = 30_000;
/// Must stay below the window's draw_call_index_capacity.
const BATCH_INDICES: usize = 150_000;

const SKIN: Color = Color::new(0.93, 0.74, 0.6, 1.0);
const BOOT: Color = Color::new(0.12, 0.12, 0.14, 1.0);
const EYE: Color = Color::new(0.05, 0.05, 0.07, 1.0);
const WHITE: Color = Color::new(0.95, 0.95, 0.92, 1.0);

/// Team colors of one player.
struct Kit {
    shirt: Color,
    shorts: Color,
    socks: Color,
    hair: Color,
}

const KITS: [Kit; 2] = [
    Kit {
        shirt: Color::new(0.2, 0.45, 0.9, 1.0),
        shorts: Color::new(0.95, 0.95, 0.95, 1.0),
        socks: Color::new(0.2, 0.45, 0.9, 1.0),
        hair: Color::new(0.27, 0.17, 0.1, 1.0),
    },
    Kit {
        shirt: Color::new(0.88, 0.2, 0.18, 1.0),
        shorts: Color::new(0.14, 0.14, 0.18, 1.0),
        socks: Color::new(0.95, 0.95, 0.95, 1.0),
        hair: Color::new(0.85, 0.68, 0.35, 1.0),
    },
];

/// Per-frame lighting context.
pub struct Shading {
    cam_pos: Vec3,
    light_dir: Vec3,
}

impl Shading {
    pub fn new(cam_pos: Vec3) -> Self {
        Self { cam_pos, light_dir: vec3(-0.4, 1.0, 0.55).normalize() }
    }

    fn shade(&self, pos: Vec3, normal: Vec3, base: Color) -> Color {
        let diffuse = normal.dot(self.light_dir).max(0.0);
        // Sky light from above plus a little bounce light from the ground.
        let ambient = 0.42 + 0.12 * normal.y;
        let light = ambient + 0.62 * diffuse;
        self.fog(pos, Color::new(base.r * light, base.g * light, base.b * light, base.a))
    }

    /// Distant things fade into the hazy horizon.
    fn fog(&self, pos: Vec3, c: Color) -> Color {
        let dist = pos.distance(self.cam_pos);
        let f = 1.0 - (-(dist - 45.0).max(0.0) * 0.011).exp();
        Color::new(
            c.r + (HORIZON.r - c.r) * f,
            c.g + (HORIZON.g - c.g) * f,
            c.b + (HORIZON.b - c.b) * f,
            c.a,
        )
    }
}

/// Accumulates triangles and flushes them as one `draw_mesh` call.
pub struct Batch {
    mesh: Mesh,
}

impl Batch {
    pub fn new() -> Self {
        Self { mesh: Mesh { vertices: Vec::with_capacity(BATCH_VERTS), indices: Vec::new(), texture: None } }
    }

    fn reserve(&mut self, verts: usize, indices: usize) -> u16 {
        if self.mesh.vertices.len() + verts > BATCH_VERTS || self.mesh.indices.len() + indices > BATCH_INDICES {
            self.flush();
        }
        self.mesh.vertices.len() as u16
    }

    fn vertex(&mut self, pos: Vec3, color: Color) {
        self.mesh.vertices.push(Vertex::new2(pos, Vec2::ZERO, color));
    }

    pub fn flush(&mut self) {
        if !self.mesh.vertices.is_empty() {
            draw_mesh(&self.mesh);
            self.mesh.vertices.clear();
            self.mesh.indices.clear();
        }
    }

    /// A flat quad, lit with the given normal.
    fn quad_lit(&mut self, sh: &Shading, p: [Vec3; 4], n: Vec3, color: Color) {
        let base = self.reserve(4, 6);
        for v in p {
            let c = sh.shade(v, n, color);
            self.vertex(v, c);
        }
        self.mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// A flat quad, lit with its own face normal (turned upward).
    fn quad(&mut self, sh: &Shading, p: [Vec3; 4], color: Color) {
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        self.quad_lit(sh, p, if n.y < 0.0 { -n } else { n }, color);
    }

    /// A flat strip on the ground from `a` to `b`, `width` wide.
    fn ground_line(&mut self, sh: &Shading, a: Vec2, b: Vec2, width: f32, y: f32, color: Color) {
        let side = (b - a).perp().normalize_or_zero() * (width * 0.5);
        let p = |v: Vec2| vec3(v.x, y, v.y);
        self.quad_lit(sh, [p(a - side), p(b - side), p(b + side), p(a + side)], Vec3::Y, color);
    }

    /// An axis-aligned box.
    fn cuboid(&mut self, sh: &Shading, center: Vec3, half: Vec3, color: Color) {
        for (n, u, v) in CUBE_FACES {
            let corner = |su: f32, sv: f32| center + (n + u * su + v * sv) * half;
            self.quad_lit(sh, [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)], n, color);
        }
    }

    /// A flat disc on the ground, for blob shadows. Not lit or fogged.
    fn disc(&mut self, center: Vec3, rx: f32, rz: f32, color: Color) {
        let n = 16;
        let base = self.reserve(n + 1, 3 * n);
        self.vertex(center, color);
        let rim = Color::new(color.r, color.g, color.b, 0.0);
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            self.vertex(center + vec3(a.cos() * rx, 0.0, a.sin() * rz), rim);
        }
        for i in 0..n as u16 {
            self.mesh.indices.extend_from_slice(&[base, base + 1 + i, base + 1 + (i + 1) % n as u16]);
        }
    }
}

/// Faces of a cube as (normal, u, v) axes.
const CUBE_FACES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y, Vec3::Z),
    (Vec3::NEG_X, Vec3::Y, Vec3::NEG_Z),
    (Vec3::Y, Vec3::Z, Vec3::X),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    (Vec3::Z, Vec3::X, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y, Vec3::X),
];

/// A mesh in local space with per-vertex colors and normals.
pub struct Model {
    verts: Vec<(Vec3, Vec3, Color)>,
    indices: Vec<u16>,
}

impl Model {
    fn new() -> Self {
        Self { verts: Vec::new(), indices: Vec::new() }
    }

    /// A capsule along the y axis. `color(y, normal)` picks each vertex's
    /// color, which is how clothes, skin and hair are painted on.
    fn capsule(&mut self, center: Vec3, half_len: f32, radius: f32, color: impl Fn(f32, Vec3) -> Color) {
        let (half_rings, slices) = (5, 14);
        let start = self.verts.len() as u16;
        let mut rows = 0;
        for (lo, hi, y_off) in [(-1.0f32, 0.0f32, -half_len), (0.0, 1.0, half_len)] {
            for i in 0..=half_rings {
                let theta = (lo + (hi - lo) * i as f32 / half_rings as f32) * std::f32::consts::FRAC_PI_2;
                let (st, ct) = theta.sin_cos();
                for j in 0..slices {
                    let phi = j as f32 / slices as f32 * std::f32::consts::TAU;
                    let n = vec3(ct * phi.cos(), st, ct * phi.sin());
                    let local = vec3(n.x * radius, y_off + st * radius, n.z * radius);
                    self.verts.push((center + local, n, color(local.y, n)));
                }
                rows += 1;
            }
        }
        for r in 0..rows - 1 {
            for j in 0..slices {
                let a = start + (r * slices + j) as u16;
                let b = start + (r * slices + (j + 1) % slices) as u16;
                let s = slices as u16;
                self.indices.extend_from_slice(&[a, a + s, b, b, a + s, b + s]);
            }
        }
    }

    fn sphere(&mut self, center: Vec3, radius: f32, color: impl Fn(f32, Vec3) -> Color) {
        self.capsule(center, 0.0, radius, color);
    }

    fn cuboid(&mut self, center: Vec3, half: Vec3, color: Color) {
        for (n, u, v) in CUBE_FACES {
            let start = self.verts.len() as u16;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = (n + u * su + v * sv) * half;
                self.verts.push((center + p, n, color));
            }
            self.indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }

    fn draw(&self, batch: &mut Batch, sh: &Shading, pos: Vec3, rot: Quat, scale: Vec3) {
        let base = batch.reserve(self.verts.len(), self.indices.len());
        for &(p, n, c) in &self.verts {
            let world = pos + rot * (p * scale);
            let color = sh.shade(world, (rot * (n / scale)).normalize(), c);
            batch.vertex(world, color);
        }
        batch.mesh.indices.extend(self.indices.iter().map(|&i| base + i));
    }
}

fn solid(c: Color) -> impl Fn(f32, Vec3) -> Color {
    move |_, _| c
}

/// The body part meshes of one player, painted in its team's kit.
pub struct PersonModels {
    trunk: Model,
    thigh: Model,
    shin: Model,
    foot: Model,
    arm: Model,
}

impl PersonModels {
    fn new(kit: &Kit) -> Self {
        let mut trunk = Model::new();
        let [hx, hy, hz] = TRUNK_HALF;
        let waist = -hy + 0.18;
        trunk.cuboid(vec3(0.0, (waist + hy) * 0.5, 0.0), vec3(hx, (hy - waist) * 0.5, hz), kit.shirt);
        trunk.cuboid(vec3(0.0, (waist - hy) * 0.5, 0.0), vec3(hx * 1.02, (waist + hy) * 0.5, hz * 1.02), kit.shorts);
        trunk.capsule(vec3(0.0, hy + 0.05, 0.0), 0.03, 0.05, solid(SKIN));
        let hair = kit.hair;
        trunk.sphere(vec3(0.0, HEAD_OFFSET, 0.0), HEAD_RADIUS, move |_, n| {
            if n.y > 0.3 || n.x < -0.25 { hair } else { SKIN }
        });
        for side in [-1.0, 1.0] {
            trunk.sphere(vec3(HEAD_RADIUS * 0.88, HEAD_OFFSET + 0.02, 0.04 * side), 0.018, solid(EYE));
        }

        let mut thigh = Model::new();
        let shorts = kit.shorts;
        thigh.capsule(Vec3::ZERO, THIGH_HALF_LEN, THIGH_RADIUS, move |y, _| if y > -0.02 { shorts } else { SKIN });

        let mut shin = Model::new();
        let socks = kit.socks;
        shin.capsule(Vec3::ZERO, SHIN_HALF_LEN, SHIN_RADIUS, move |y, _| if y < 0.1 { socks } else { SKIN });

        let mut foot = Model::new();
        let [fx, fy, fz] = FOOT_HALF;
        foot.cuboid(vec3(0.0, 0.006, 0.0), vec3(fx, fy - 0.006, fz), BOOT);
        foot.cuboid(vec3(0.0, -fy + 0.006, 0.0), vec3(fx * 1.02, 0.006, fz * 1.05), WHITE);

        let mut arm = Model::new();
        let shirt = kit.shirt;
        arm.capsule(Vec3::ZERO, ARM_HALF_LEN - 0.04, ARM_RADIUS, move |y, _| if y > 0.1 { shirt } else { SKIN });
        arm.sphere(vec3(0.0, -ARM_HALF_LEN + 0.01, 0.0), 0.05, solid(SKIN));

        Self { trunk, thigh, shin, foot, arm }
    }
}

pub struct Models {
    pub people: [PersonModels; 2],
    ball: Model,
    tree: Model,
    rock: Model,
}

impl Models {
    pub fn new() -> Self {
        // A classic ball: black patches around the 12 corners of an icosahedron.
        let phi = (1.0 + 5f32.sqrt()) * 0.5;
        let mut patches = Vec::new();
        for a in [-1.0, 1.0] {
            for b in [-phi, phi] {
                patches.extend([vec3(0.0, a, b), vec3(a, b, 0.0), vec3(b, 0.0, a)].map(|v| v.normalize()));
            }
        }
        let mut ball = Model::new();
        ball.sphere(Vec3::ZERO, BALL_RADIUS, move |_, n| {
            let n = n.normalize();
            if patches.iter().any(|p| p.dot(n) > 0.92) { Color::new(0.1, 0.1, 0.12, 1.0) } else { WHITE }
        });

        let mut tree = Model::new();
        tree.capsule(vec3(0.0, 1.5, 0.0), 1.5, 0.18, solid(Color::new(0.4, 0.28, 0.18, 1.0)));
        let leaves = Color::new(0.22, 0.5, 0.2, 1.0);
        tree.sphere(vec3(0.0, 3.8, 0.0), 1.6, solid(leaves));
        tree.sphere(vec3(0.7, 3.2, 0.5), 1.1, solid(Color::new(0.25, 0.55, 0.22, 1.0)));
        tree.sphere(vec3(-0.6, 3.4, -0.6), 1.2, solid(Color::new(0.2, 0.45, 0.18, 1.0)));

        let mut rock = Model::new();
        rock.sphere(Vec3::ZERO, 1.0, solid(Color::new(0.52, 0.5, 0.47, 1.0)));

        Self { people: [PersonModels::new(&KITS[0]), PersonModels::new(&KITS[1])], ball, tree, rock }
    }
}

pub fn draw_person(batch: &mut Batch, sh: &Shading, models: &PersonModels, parts: &[PartPose]) {
    for p in parts {
        let model = match p.kind {
            PartKind::Trunk => &models.trunk,
            PartKind::Thigh => &models.thigh,
            PartKind::Shin => &models.shin,
            PartKind::Foot => &models.foot,
            PartKind::Arm => &models.arm,
        };
        model.draw(batch, sh, p.pos, p.rot, Vec3::ONE);
    }
}

pub fn draw_ball(batch: &mut Batch, sh: &Shading, models: &Models, pos: Vec3, rot: Quat) {
    models.ball.draw(batch, sh, pos, rot, Vec3::ONE);
}

/// Soft dark blobs under each body part and the ball; they shrink and fade
/// with height.
pub fn draw_shadows(batch: &mut Batch, parts: &[PartPose], ball: Vec3) {
    let blobs = parts.iter().map(|p| {
        let size = match p.kind {
            PartKind::Trunk => 0.3,
            PartKind::Thigh => 0.12,
            PartKind::Shin => 0.09,
            PartKind::Foot => 0.15,
            PartKind::Arm => 0.08,
        };
        (p.pos, size)
    });
    for (pos, size) in blobs.chain(std::iter::once((ball, BALL_RADIUS * 1.1))) {
        let fade = (1.0 - pos.y / 2.5).clamp(0.0, 1.0);
        batch.disc(vec3(pos.x - pos.y * 0.25, 0.012, pos.z + pos.y * 0.2), size * 1.3, size, Color::new(0.0, 0.0, 0.0, 0.35 * fade));
    }
}

/// The pitch with its lines, walls and goals, and trees and rocks around it.
pub struct Scenery {
    trees: Vec<(Vec3, f32, f32)>,
    rocks: Vec<(Vec3, Vec3)>,
}

impl Scenery {
    pub fn new() -> Self {
        // Anywhere around the pitch, but not on it.
        let around = |margin: f32| loop {
            let pos = vec3(gen_range(-60.0, 60.0), 0.0, gen_range(-45.0, 45.0));
            if pos.x.abs() > HALF_LENGTH + GOAL_DEPTH + margin || pos.z.abs() > HALF_WIDTH + margin {
                return pos;
            }
        };
        let trees = (0..70).map(|_| (around(8.0), gen_range(0.8, 1.5), gen_range(0.0, std::f32::consts::TAU))).collect();
        let rocks = (0..40)
            .map(|_| {
                let r = gen_range(0.2, 0.8);
                (around(3.0), vec3(r * gen_range(1.0, 1.6), r * gen_range(0.4, 0.8), r))
            })
            .collect();
        Self { trees, rocks }
    }

    pub fn draw(&self, batch: &mut Batch, sh: &Shading, models: &Models) {
        // Grass in 2 m tiles, mowed in stripes across the pitch.
        let tile = 2.0;
        for ix in -36i32..36 {
            for iz in -28..28 {
                let (x, z) = (ix as f32 * tile, iz as f32 * tile);
                let hash = ((ix * 73 + iz * 151) & 7) as f32 / 7.0;
                let on_pitch = x >= -HALF_LENGTH - 2.0 && x < HALF_LENGTH + 2.0 && z >= -HALF_WIDTH - 2.0 && z < HALF_WIDTH + 2.0;
                let g = if on_pitch {
                    let stripe = if ix.rem_euclid(2) == 0 { 0.0 } else { 0.045 };
                    Color::new(0.27 + stripe, 0.56 + stripe, 0.22 + stripe * 0.5, 1.0)
                } else {
                    Color::new(0.35 + hash * 0.03, 0.55 + hash * 0.05, 0.25 + hash * 0.02, 1.0)
                };
                batch.quad(
                    sh,
                    [vec3(x, 0.0, z), vec3(x + tile, 0.0, z), vec3(x + tile, 0.0, z + tile), vec3(x, 0.0, z + tile)],
                    g,
                );
            }
        }

        // Pitch markings.
        let (l, w, y, lw) = (HALF_LENGTH - 0.1, HALF_WIDTH - 0.1, 0.006, 0.1);
        let corners = [vec2(-l, -w), vec2(l, -w), vec2(l, w), vec2(-l, w)];
        for i in 0..4 {
            batch.ground_line(sh, corners[i], corners[(i + 1) % 4], lw, y, WHITE);
        }
        batch.ground_line(sh, vec2(0.0, -w), vec2(0.0, w), lw, y, WHITE);
        let ring = |batch: &mut Batch, c: Vec2, r: f32| {
            let n = 48;
            for i in 0..n {
                let a = |k: i32| k as f32 / n as f32 * std::f32::consts::TAU;
                batch.ground_line(sh, c + Vec2::from_angle(a(i)) * r, c + Vec2::from_angle(a(i + 1)) * r, lw, y, WHITE);
            }
        };
        ring(batch, Vec2::ZERO, 3.0);
        ring(batch, Vec2::ZERO, 0.12);
        for s in [-1.0f32, 1.0] {
            // Goal area and penalty spot.
            let (front, half) = (s * (l - 2.5), GOAL_HALF_WIDTH + 1.5);
            batch.ground_line(sh, vec2(s * l, -half), vec2(front, -half), lw, y, WHITE);
            batch.ground_line(sh, vec2(front, -half), vec2(front, half), lw, y, WHITE);
            batch.ground_line(sh, vec2(front, half), vec2(s * l, half), lw, y, WHITE);
            ring(batch, vec2(s * (l - 5.0), 0.0), 0.12);
        }

        self.draw_walls(batch, sh);
        for s in [-1.0, 1.0] {
            draw_goal(batch, sh, s);
        }

        for &(pos, scale, spin) in &self.trees {
            models.tree.draw(batch, sh, pos, Quat::from_rotation_y(spin), Vec3::splat(scale));
        }
        for &(pos, size) in &self.rocks {
            models.rock.draw(batch, sh, pos, Quat::IDENTITY, size);
        }
    }

    /// Low boards all around the pitch, open at the goals. The ball bounces
    /// off them.
    fn draw_walls(&self, batch: &mut Batch, sh: &Shading) {
        let board = Color::new(0.16, 0.3, 0.55, 1.0);
        let (l, w, t, h) = (HALF_LENGTH, HALF_WIDTH, 0.06, 0.9);
        for s in [-1.0, 1.0] {
            // Long sides.
            batch.cuboid(sh, vec3(0.0, h * 0.5, s * (w + t)), vec3(l + 2.0 * t, h * 0.5, t), board);
            batch.cuboid(sh, vec3(0.0, h + 0.02, s * (w + t)), vec3(l + 2.0 * t, 0.02, t * 1.2), WHITE);
            // Ends, on either side of the goal.
            let len = (w - GOAL_HALF_WIDTH) * 0.5;
            for zs in [-1.0, 1.0] {
                let zc = zs * (GOAL_HALF_WIDTH + len);
                batch.cuboid(sh, vec3(s * (l + t), h * 0.5, zc), vec3(t, h * 0.5, len), board);
                batch.cuboid(sh, vec3(s * (l + t), h + 0.02, zc), vec3(t * 1.2, 0.02, len), WHITE);
            }
        }
    }
}

/// Goal frame and net at the end `s` (+1 or -1) of the pitch.
fn draw_goal(batch: &mut Batch, sh: &Shading, s: f32) {
    let (x, gw, gh, d) = (s * HALF_LENGTH, GOAL_HALF_WIDTH, GOAL_HEIGHT, GOAL_DEPTH);
    let r = 0.06;
    for z in [-gw - r, gw + r] {
        batch.cuboid(sh, vec3(x, gh * 0.5, z), vec3(r, gh * 0.5 + r, r), WHITE);
    }
    batch.cuboid(sh, vec3(x, gh + r, 0.0), vec3(r, r, gw + 2.0 * r), WHITE);

    // The net: a grid of thin strings on the back, sides and roof.
    let net = Color::new(0.85, 0.87, 0.9, 1.0);
    let back = x + s * d;
    let k = 0.012;
    let gap = 0.25;
    let mut z = -gw;
    while z <= gw + 0.01 {
        batch.cuboid(sh, vec3(back, gh * 0.5, z), vec3(k, gh * 0.5, k), net);
        batch.cuboid(sh, vec3(x + s * d * 0.5, gh, z), vec3(d * 0.5, k, k), net);
        z += gap;
    }
    let mut yy = 0.0;
    while yy <= gh + 0.01 {
        batch.cuboid(sh, vec3(back, yy, 0.0), vec3(k, k, gw), net);
        for zs in [-gw, gw] {
            batch.cuboid(sh, vec3(x + s * d * 0.5, yy, zs), vec3(d * 0.5, k, k), net);
        }
        yy += gap;
    }
    let mut dx = 0.0;
    while dx <= d + 0.01 {
        let xx = x + s * dx;
        batch.cuboid(sh, vec3(xx, gh, 0.0), vec3(k, k, gw), net);
        for zs in [-gw, gw] {
            batch.cuboid(sh, vec3(xx, gh * 0.5, zs), vec3(k, gh * 0.5, k), net);
        }
        dx += gap;
    }
}

/// Vertical sky gradient, drawn in screen space before the 3D scene.
pub fn draw_sky() {
    let (w, h) = (screen_width(), screen_height());
    let v = |x: f32, y: f32, c: Color| Vertex::new(x, y, 0.0, 0.0, 0.0, c);
    let mesh = Mesh {
        vertices: vec![v(0.0, 0.0, SKY_TOP), v(w, 0.0, SKY_TOP), v(w, h * 0.55, HORIZON), v(0.0, h * 0.55, HORIZON)],
        indices: vec![0, 1, 2, 0, 2, 3],
        texture: None,
    };
    draw_mesh(&mesh);
}
