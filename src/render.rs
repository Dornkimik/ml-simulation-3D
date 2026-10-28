//! 3D rendering of the field and the person. Macroquad has no built-in
//! lighting, so meshes are lit and fogged per vertex on the CPU and submitted
//! in large batches.

use macroquad::models::{Mesh, Vertex};
use macroquad::prelude::*;
use macroquad::rand::gen_range;

use crate::walker::{
    PartKind, PartPose, ARM_HALF_LEN, ARM_RADIUS, FOOT_HALF, HEAD_OFFSET, HEAD_RADIUS, SHIN_HALF_LEN, SHIN_RADIUS,
    THIGH_HALF_LEN, THIGH_RADIUS, TRUNK_HALF,
};

pub const SKY_TOP: Color = Color::new(0.33, 0.56, 0.86, 1.0);
pub const HORIZON: Color = Color::new(0.78, 0.86, 0.93, 1.0);
const BATCH_VERTS: usize = 30_000;

const SKIN: Color = Color::new(0.93, 0.74, 0.6, 1.0);
const SHIRT: Color = Color::new(0.2, 0.5, 0.85, 1.0);
const SHORTS: Color = Color::new(0.16, 0.18, 0.27, 1.0);
const SHOE: Color = Color::new(0.85, 0.27, 0.2, 1.0);
const SOCK: Color = Color::new(0.95, 0.95, 0.95, 1.0);
const HAIR: Color = Color::new(0.27, 0.17, 0.1, 1.0);
const EYE: Color = Color::new(0.05, 0.05, 0.07, 1.0);

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
        let f = 1.0 - (-(dist - 25.0).max(0.0) * 0.011).exp();
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

    fn reserve(&mut self, verts: usize) -> u16 {
        if self.mesh.vertices.len() + verts > BATCH_VERTS {
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

    /// A flat quad, lit with its own face normal.
    fn quad(&mut self, sh: &Shading, p: [Vec3; 4], color: Color) {
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let n = if n.y < 0.0 { -n } else { n };
        let base = self.reserve(4);
        for v in p {
            let c = sh.shade(v, n, color);
            self.vertex(v, c);
        }
        self.mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// A flat disc on the ground, for blob shadows. Not lit or fogged.
    fn disc(&mut self, center: Vec3, rx: f32, rz: f32, color: Color) {
        let n = 16;
        let base = self.reserve(n + 1);
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
        let faces = [
            (Vec3::X, Vec3::Y, Vec3::Z),
            (Vec3::NEG_X, Vec3::Y, Vec3::NEG_Z),
            (Vec3::Y, Vec3::Z, Vec3::X),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y, Vec3::X),
        ];
        for (n, u, v) in faces {
            let start = self.verts.len() as u16;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = (n + u * su + v * sv) * half;
                self.verts.push((center + p, n, color));
            }
            self.indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }

    fn draw(&self, batch: &mut Batch, sh: &Shading, pos: Vec3, rot: Quat, scale: Vec3) {
        let base = batch.reserve(self.verts.len());
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

pub struct Models {
    trunk: Model,
    thigh: Model,
    shin: Model,
    foot: Model,
    arm: Model,
    tree: Model,
    rock: Model,
}

impl Models {
    pub fn new() -> Self {
        let mut trunk = Model::new();
        let [hx, hy, hz] = TRUNK_HALF;
        let waist = -hy + 0.18;
        trunk.cuboid(vec3(0.0, (waist + hy) * 0.5, 0.0), vec3(hx, (hy - waist) * 0.5, hz), SHIRT);
        trunk.cuboid(vec3(0.0, (waist - hy) * 0.5, 0.0), vec3(hx * 1.02, (waist + hy) * 0.5, hz * 1.02), SHORTS);
        trunk.capsule(vec3(0.0, hy + 0.05, 0.0), 0.03, 0.05, solid(SKIN));
        trunk.sphere(vec3(0.0, HEAD_OFFSET, 0.0), HEAD_RADIUS, |_, n| {
            if n.y > 0.3 || n.x < -0.25 { HAIR } else { SKIN }
        });
        for side in [-1.0, 1.0] {
            trunk.sphere(vec3(HEAD_RADIUS * 0.88, HEAD_OFFSET + 0.02, 0.04 * side), 0.018, solid(EYE));
        }

        let mut thigh = Model::new();
        thigh.capsule(Vec3::ZERO, THIGH_HALF_LEN, THIGH_RADIUS, |y, _| if y > -0.02 { SHORTS } else { SKIN });

        let mut shin = Model::new();
        shin.capsule(Vec3::ZERO, SHIN_HALF_LEN, SHIN_RADIUS, |y, _| if y < -0.12 { SOCK } else { SKIN });

        let mut foot = Model::new();
        let [fx, fy, fz] = FOOT_HALF;
        foot.cuboid(vec3(0.0, 0.006, 0.0), vec3(fx, fy - 0.006, fz), SHOE);
        foot.cuboid(vec3(0.0, -fy + 0.006, 0.0), vec3(fx * 1.02, 0.006, fz * 1.05), SOCK);

        let mut arm = Model::new();
        arm.capsule(Vec3::ZERO, ARM_HALF_LEN - 0.04, ARM_RADIUS, |y, _| if y > 0.1 { SHIRT } else { SKIN });
        arm.sphere(vec3(0.0, -ARM_HALF_LEN + 0.01, 0.0), 0.05, solid(SKIN));

        let mut tree = Model::new();
        tree.capsule(vec3(0.0, 1.5, 0.0), 1.5, 0.18, solid(Color::new(0.4, 0.28, 0.18, 1.0)));
        let leaves = Color::new(0.22, 0.5, 0.2, 1.0);
        tree.sphere(vec3(0.0, 3.8, 0.0), 1.6, solid(leaves));
        tree.sphere(vec3(0.7, 3.2, 0.5), 1.1, solid(Color::new(0.25, 0.55, 0.22, 1.0)));
        tree.sphere(vec3(-0.6, 3.4, -0.6), 1.2, solid(Color::new(0.2, 0.45, 0.18, 1.0)));

        let mut rock = Model::new();
        rock.sphere(Vec3::ZERO, 1.0, solid(Color::new(0.52, 0.5, 0.47, 1.0)));

        Self { trunk, thigh, shin, foot, arm, tree, rock }
    }
}

fn quat(rot: [f32; 4]) -> Quat {
    Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3])
}

pub fn draw_person(batch: &mut Batch, sh: &Shading, models: &Models, parts: &[PartPose]) {
    for p in parts {
        let model = match p.kind {
            PartKind::Trunk => &models.trunk,
            PartKind::Thigh => &models.thigh,
            PartKind::Shin => &models.shin,
            PartKind::Foot => &models.foot,
            PartKind::Arm => &models.arm,
        };
        model.draw(batch, sh, Vec3::from(p.pos), quat(p.rot), Vec3::ONE);
    }
}

/// Soft dark blobs under each body part; they shrink and fade with height.
pub fn draw_shadows(batch: &mut Batch, parts: &[PartPose]) {
    for p in parts {
        let pos = Vec3::from(p.pos);
        let size = match p.kind {
            PartKind::Trunk => 0.3,
            PartKind::Thigh => 0.12,
            PartKind::Shin => 0.09,
            PartKind::Foot => 0.15,
            PartKind::Arm => 0.08,
        };
        let fade = (1.0 - pos.y / 2.5).clamp(0.0, 1.0);
        batch.disc(vec3(pos.x - pos.y * 0.25, 0.012, pos.z + pos.y * 0.2), size * 1.3, size, Color::new(0.0, 0.0, 0.0, 0.35 * fade));
    }
}

/// The field: grass, a running track with distance lines, trees and rocks.
pub struct Scenery {
    trees: Vec<(Vec3, f32, f32)>,
    rocks: Vec<(Vec3, Vec3)>,
}

impl Scenery {
    pub fn new() -> Self {
        let side = || if gen_range(0, 2) == 0 { -1.0 } else { 1.0 };
        let trees = (0..260)
            .map(|_| {
                let pos = vec3(gen_range(-80.0, 700.0), 0.0, side() * gen_range(6.0, 45.0));
                (pos, gen_range(0.8, 1.5), gen_range(0.0, std::f32::consts::TAU))
            })
            .collect();
        let rocks = (0..120)
            .map(|_| {
                let pos = vec3(gen_range(-80.0, 700.0), 0.0, side() * gen_range(2.5, 30.0));
                let r = gen_range(0.2, 0.8);
                (pos, vec3(r * gen_range(1.0, 1.6), r * gen_range(0.4, 0.8), r))
            })
            .collect();
        Self { trees, rocks }
    }

    /// Draws everything within view of the point `focus_x` along the track.
    /// `record` is the best distance reached so far, marked on the track.
    pub fn draw(&self, batch: &mut Batch, sh: &Shading, models: &Models, focus_x: f32, record: Option<f32>) {
        // Grass in 2 m tiles with a gentle checker so motion is easy to see.
        let tile = 2.0;
        let x0 = ((focus_x - 60.0) / tile).floor() as i32;
        let x1 = ((focus_x + 110.0) / tile).ceil() as i32;
        for ix in x0..x1 {
            for iz in -30..30 {
                let (x, z) = (ix as f32 * tile, iz as f32 * tile);
                let hash = ((ix * 73 + iz * 151) & 7) as f32 / 7.0;
                let checker = if (ix + iz) % 2 == 0 { 0.0 } else { 0.025 };
                let g = Color::new(0.33 + checker + hash * 0.03, 0.58 + checker + hash * 0.05, 0.25 + hash * 0.02, 1.0);
                batch.quad(
                    sh,
                    [vec3(x, 0.0, z), vec3(x + tile, 0.0, z), vec3(x + tile, 0.0, z + tile), vec3(x, 0.0, z + tile)],
                    g,
                );
            }
        }

        // Running track along +x.
        let (tx0, tx1) = (focus_x - 60.0, focus_x + 110.0);
        let lane = 1.1;
        let y = 0.004;
        let track = Color::new(0.72, 0.36, 0.26, 1.0);
        let white = Color::new(0.95, 0.95, 0.92, 1.0);
        let mut seg = tx0;
        while seg < tx1 {
            let e = (seg + 10.0).min(tx1);
            batch.quad(sh, [vec3(seg, y, -lane), vec3(e, y, -lane), vec3(e, y, lane), vec3(seg, y, lane)], track);
            for zl in [-lane, lane] {
                batch.quad(sh, [vec3(seg, y * 2.0, zl - 0.04), vec3(e, y * 2.0, zl - 0.04), vec3(e, y * 2.0, zl + 0.04), vec3(seg, y * 2.0, zl + 0.04)], white);
            }
            seg = e;
        }
        let first = tx0.ceil().max(0.0) as i32;
        for m in first..tx1 as i32 {
            let x = m as f32;
            let w = if m == 0 { 0.12 } else if m % 5 == 0 { 0.06 } else { 0.015 };
            let reach = if m % 5 == 0 { lane } else { 0.25 };
            batch.quad(sh, [vec3(x - w, y * 2.0, -reach), vec3(x + w, y * 2.0, -reach), vec3(x + w, y * 2.0, reach), vec3(x - w, y * 2.0, reach)], white);
            if m % 5 == 0 {
                // Distance post beside the track.
                let post = vec3(x, 0.0, -lane - 0.4);
                batch.quad(sh, [post + vec3(-0.04, 0.0, 0.0), post + vec3(0.04, 0.0, 0.0), post + vec3(0.04, 0.9, 0.0), post + vec3(-0.04, 0.9, 0.0)], white);
                batch.quad(sh, [post + vec3(-0.25, 0.9, 0.0), post + vec3(0.25, 0.9, 0.0), post + vec3(0.25, 1.2, 0.0), post + vec3(-0.25, 1.2, 0.0)], white);
            }
        }

        // The current record, as a gold line and flag.
        if let Some(r) = record {
            let gold = Color::new(1.0, 0.8, 0.15, 1.0);
            batch.quad(sh, [vec3(r - 0.05, y * 3.0, -lane), vec3(r + 0.05, y * 3.0, -lane), vec3(r + 0.05, y * 3.0, lane), vec3(r - 0.05, y * 3.0, lane)], gold);
            let pole = vec3(r, 0.0, lane + 0.3);
            batch.quad(sh, [pole + vec3(-0.025, 0.0, 0.0), pole + vec3(0.025, 0.0, 0.0), pole + vec3(0.025, 1.8, 0.0), pole + vec3(-0.025, 1.8, 0.0)], white);
            batch.quad(sh, [pole + vec3(0.0, 1.8, 0.0), pole + vec3(0.6, 1.65, 0.0), pole + vec3(0.6, 1.4, 0.0), pole + vec3(0.0, 1.35, 0.0)], gold);
        }

        for &(pos, scale, spin) in &self.trees {
            if pos.x > tx0 && pos.x < tx1 {
                let s = Vec3::splat(scale);
                models.tree.draw(batch, sh, pos, Quat::from_rotation_y(spin), s);
            }
        }
        for &(pos, size) in &self.rocks {
            if pos.x > tx0 && pos.x < tx1 {
                models.rock.draw(batch, sh, pos, Quat::IDENTITY, size);
            }
        }
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
