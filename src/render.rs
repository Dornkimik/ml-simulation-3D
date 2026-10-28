//! 3D rendering. Macroquad has no built-in lighting, so meshes are lit and
//! fogged per vertex on the CPU and submitted in large batches.

use macroquad::models::{Mesh, Vertex};
use macroquad::prelude::*;
use macroquad::rand::gen_range;

use crate::creature::{Creature, Species};
use crate::world::{World, TANK_MAX, TANK_MIN};

pub const WATER_COLOR: Color = Color::new(0.02, 0.16, 0.27, 1.0);
const BATCH_VERTS: usize = 30_000;

/// Per-frame lighting context.
pub struct Shading {
    cam_pos: Vec3,
    light_dir: Vec3,
}

impl Shading {
    pub fn new(cam_pos: Vec3) -> Self {
        Self { cam_pos, light_dir: vec3(0.35, 1.0, 0.25).normalize() }
    }

    fn shade(&self, pos: Vec3, normal: Vec3, base: Color, two_sided: bool) -> Color {
        let mut d = normal.dot(self.light_dir);
        if two_sided {
            d = d.abs();
        }
        let diffuse = d.max(0.0);
        // Water absorbs sunlight, so deeper things are darker.
        let depth = 0.5 + 0.5 * ((pos.y - TANK_MIN.y) / (TANK_MAX.y - TANK_MIN.y)).clamp(0.0, 1.0);
        let light = (0.32 + 0.68 * diffuse) * depth;
        let lit = Color::new(base.r * light, base.g * light, base.b * light, base.a);
        self.fog(pos, lit)
    }

    fn fog(&self, pos: Vec3, c: Color) -> Color {
        let dist = pos.distance(self.cam_pos);
        let f = 1.0 - (-(dist - 10.0).max(0.0) * 0.009).exp();
        Color::new(
            c.r + (WATER_COLOR.r - c.r) * f,
            c.g + (WATER_COLOR.g - c.g) * f,
            c.b + (WATER_COLOR.b - c.b) * f,
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

    fn quad(&mut self, sh: &Shading, p: [Vec3; 4], color: Color) {
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let base = self.reserve(4);
        for v in p {
            let c = sh.shade(v, n, color, true);
            self.vertex(v, c);
        }
        self.mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mat {
    Body,
    Belly,
    Fin,
    Eye,
}

struct ModelVert {
    pos: Vec3,
    normal: Vec3,
    mat: Mat,
}

/// A creature mesh in local space: +x forward, +y up, +z right.
pub struct Model {
    verts: Vec<ModelVert>,
    indices: Vec<u16>,
}

impl Model {
    fn new() -> Self {
        Self { verts: Vec::new(), indices: Vec::new() }
    }

    /// Adds a body of revolution along the x axis. `profile(t)` gives the
    /// radius (0..1) at t = +1 (nose) through t = -1 (tail).
    fn body(&mut self, half_len: f32, height: f32, width: f32, profile: impl Fn(f32) -> f32) {
        let (segs, ring) = (14, 10);
        let start = self.verts.len() as u16;
        for i in 0..=segs {
            let t = 1.0 - 2.0 * i as f32 / segs as f32;
            let r = profile(t);
            for j in 0..ring {
                let a = j as f32 / ring as f32 * std::f32::consts::TAU;
                let (s, c) = a.sin_cos();
                let pos = vec3(t * half_len, c * r * height, s * r * width);
                let normal = vec3(t * 0.4, c / height, s / width).normalize();
                let mat = if c < -0.35 { Mat::Belly } else { Mat::Body };
                self.verts.push(ModelVert { pos, normal, mat });
            }
        }
        for i in 0..segs {
            for j in 0..ring {
                let a = start + (i * ring + j) as u16;
                let b = start + (i * ring + (j + 1) % ring) as u16;
                let c = a + ring as u16;
                let d = b + ring as u16;
                self.indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    }

    /// Adds a dome (jellyfish bell) around the +y axis.
    fn dome(&mut self, radius: f32, height: f32) {
        let (rings, slices) = (8, 16);
        let start = self.verts.len() as u16;
        for i in 0..=rings {
            let theta = i as f32 / rings as f32 * 1.75;
            let (st, ct) = theta.sin_cos();
            for j in 0..slices {
                let a = j as f32 / slices as f32 * std::f32::consts::TAU;
                let (s, c) = a.sin_cos();
                let pos = vec3(c * st * radius, ct * height, s * st * radius);
                let normal = vec3(c * st / radius, ct / height, s * st / radius).normalize();
                let mat = if i == rings { Mat::Belly } else { Mat::Body };
                self.verts.push(ModelVert { pos, normal, mat });
            }
        }
        for i in 0..rings {
            for j in 0..slices {
                let a = start + (i * slices + j) as u16;
                let b = start + (i * slices + (j + 1) % slices) as u16;
                self.indices.extend_from_slice(&[a, a + slices as u16, b, b, a + slices as u16, b + slices as u16]);
            }
        }
    }

    fn fin(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        let n = (b - a).cross(c - a).normalize_or_zero();
        let start = self.verts.len() as u16;
        for pos in [a, b, c] {
            self.verts.push(ModelVert { pos, normal: n, mat: Mat::Fin });
        }
        self.indices.extend_from_slice(&[start, start + 1, start + 2]);
    }

    fn eye(&mut self, center: Vec3, r: f32) {
        let start = self.verts.len() as u16;
        for d in OCTAHEDRON {
            self.verts.push(ModelVert { pos: center + d * r, normal: d, mat: Mat::Eye });
        }
        self.indices.extend(OCTAHEDRON_TRIS.iter().map(|&i| start + i));
    }
}

const OCTAHEDRON: [Vec3; 6] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(-1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, -1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(0.0, 0.0, -1.0),
];
const OCTAHEDRON_TRIS: [u16; 24] = [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5];

pub struct Models {
    fish: Model,
    shark: Model,
    jelly: Model,
}

impl Models {
    pub fn new() -> Self {
        let mut fish = Model::new();
        fish.body(1.0, 0.42, 0.22, |t| {
            let s = (1.0 - t * t).max(0.0).powf(0.45);
            if t < 0.0 { s * (1.0 + t * 0.5) } else { s }
        });
        fish.fin(vec3(-0.8, 0.0, 0.0), vec3(-1.5, 0.48, 0.0), vec3(-1.28, 0.0, 0.0));
        fish.fin(vec3(-0.8, 0.0, 0.0), vec3(-1.28, 0.0, 0.0), vec3(-1.5, -0.48, 0.0));
        fish.fin(vec3(0.25, 0.36, 0.0), vec3(-0.35, 0.3, 0.0), vec3(-0.2, 0.62, 0.0));
        fish.fin(vec3(0.0, -0.3, 0.0), vec3(-0.3, -0.25, 0.0), vec3(-0.25, -0.45, 0.0));
        for side in [-1.0, 1.0] {
            fish.eye(vec3(0.62, 0.1, 0.13 * side), 0.07);
        }

        let mut shark = Model::new();
        shark.body(1.0, 0.3, 0.26, |t| {
            let s = (1.0 - t * t).max(0.0).powf(0.55);
            if t < 0.0 { s * (1.0 + t * 0.62) } else { s * (1.0 - t * t * 0.15) }
        });
        shark.fin(vec3(0.18, 0.27, 0.0), vec3(-0.3, 0.24, 0.0), vec3(-0.22, 0.78, 0.0));
        shark.fin(vec3(-0.8, 0.02, 0.0), vec3(-1.55, 0.72, 0.0), vec3(-1.15, 0.0, 0.0));
        shark.fin(vec3(-0.8, -0.02, 0.0), vec3(-1.15, 0.0, 0.0), vec3(-1.38, -0.42, 0.0));
        shark.fin(vec3(-0.45, 0.16, 0.0), vec3(-0.6, 0.12, 0.0), vec3(-0.62, 0.3, 0.0));
        for side in [-1.0, 1.0] {
            shark.fin(
                vec3(0.32, -0.12, 0.2 * side),
                vec3(0.02, -0.15, 0.2 * side),
                vec3(-0.15, -0.32, 0.8 * side),
            );
            shark.eye(vec3(0.72, 0.07, 0.15 * side), 0.045);
        }

        let mut jelly = Model::new();
        jelly.dome(1.0, 0.8);

        Self { fish, shark, jelly }
    }
}

/// Orthonormal frame (forward, up, right) facing `dir`.
fn basis(dir: Vec3) -> (Vec3, Vec3, Vec3) {
    let f = if dir.length_squared() > 1e-6 { dir.normalize() } else { Vec3::X };
    let mut r = f.cross(Vec3::Y);
    if r.length_squared() < 1e-4 {
        r = f.cross(Vec3::Z);
    }
    let r = r.normalize();
    let u = r.cross(f);
    (f, u, r)
}

fn draw_model(
    batch: &mut Batch,
    sh: &Shading,
    model: &Model,
    origin: Vec3,
    frame: (Vec3, Vec3, Vec3),
    scale: f32,
    color: Color,
    deform: impl Fn(Vec3) -> Vec3,
) {
    let (f, u, r) = frame;
    let base = batch.reserve(model.verts.len());
    let belly = Color::new(
        color.r + (1.0 - color.r) * 0.45,
        color.g + (1.0 - color.g) * 0.45,
        color.b + (1.0 - color.b) * 0.45,
        color.a,
    );
    let fin = Color::new(color.r * 0.8, color.g * 0.8, color.b * 0.85, color.a * 0.9);
    let eye = Color::new(0.03, 0.03, 0.05, 1.0);
    for v in &model.verts {
        let lp = deform(v.pos);
        let world = origin + (f * lp.x + u * lp.y + r * lp.z) * scale;
        let n = f * v.normal.x + u * v.normal.y + r * v.normal.z;
        let (base_color, two_sided) = match v.mat {
            Mat::Body => (color, false),
            Mat::Belly => (belly, false),
            Mat::Fin => (fin, true),
            Mat::Eye => (eye, false),
        };
        let c = sh.shade(world, n, base_color, two_sided);
        batch.vertex(world, c);
    }
    batch.mesh.indices.extend(model.indices.iter().map(|&i| base + i));
}

pub fn draw_creature(batch: &mut Batch, sh: &Shading, models: &Models, c: &Creature) {
    let p = c.species.params();
    let color = c.color();
    let phase = c.phase;
    match c.species {
        Species::Fish | Species::Shark => {
            let model = if c.species == Species::Fish { &models.fish } else { &models.shark };
            let amp = if c.species == Species::Fish { 0.22 } else { 0.14 };
            draw_model(batch, sh, model, c.pos, basis(c.heading), p.size, color, |v| {
                if v.x < 0.3 {
                    let k = 0.3 - v.x;
                    vec3(v.x, v.y, v.z + (phase - v.x * 2.5).sin() * amp * k)
                } else {
                    v
                }
            });
        }
        Species::Jellyfish => {}
    }
}

/// Jellyfish are translucent, so they are drawn in a later pass.
pub fn draw_jellyfish(batch: &mut Batch, sh: &Shading, models: &Models, c: &Creature, time: f32) {
    let p = c.species.params();
    let color = c.color();
    let pulse = c.phase.sin();
    // The bell leads, so its axis tilts toward the swimming direction.
    let axis = (Vec3::Y * 1.5 + c.heading).normalize();
    let (f, u, r) = basis(axis);
    // basis() returns forward = axis; remap so the model's +y follows the axis.
    let frame = (u, f, r);
    draw_model(batch, sh, &models.jelly, c.pos, frame, p.size, color, |v| {
        vec3(v.x * (1.0 + 0.13 * pulse), v.y * (1.0 - 0.15 * pulse), v.z * (1.0 + 0.13 * pulse))
    });

    // Tentacles trail behind the bell as wavy ribbons.
    let rim = p.size * 0.85;
    let tentacle_color = Color::new(color.r, color.g, color.b, 0.55);
    let down = -axis;
    for k in 0..8 {
        let a = k as f32 / 8.0 * std::f32::consts::TAU;
        let mut prev = c.pos + (u * a.cos() + r * a.sin()) * rim - axis * 0.1 * p.size;
        for s in 1..=7 {
            let sf = s as f32;
            let sway = (time * 1.7 + sf * 0.6 + k as f32).sin() * 0.12 * sf;
            let next = prev + down * 0.45 * p.size + (u * a.cos() + r * a.sin()) * sway * 0.15
                - c.vel * 0.02 * sf;
            let w = 0.06 * p.size * (1.0 - sf / 9.0);
            let side = r * w;
            batch.quad(sh, [prev - side, prev + side, next + side, next - side], tentacle_color);
            prev = next;
        }
    }
}

pub fn draw_plankton(batch: &mut Batch, sh: &Shading, world: &World, time: f32) {
    for (i, p) in world.plankton.iter().enumerate() {
        let twinkle = 0.75 + 0.25 * (time * 3.0 + i as f32).sin();
        let color = Color::new(0.35 * twinkle, 0.95 * twinkle, 0.45 * twinkle, 1.0);
        let base = batch.reserve(6);
        let r = 0.16;
        for d in OCTAHEDRON {
            let pos = p.pos + d * r;
            // Plankton glows faintly, so it is only fogged, not shaded.
            batch.vertex(pos, sh.fog(pos, color));
        }
        batch.mesh.indices.extend(OCTAHEDRON_TRIS.iter().map(|&i| base + i));
    }
}

/// Static and ambient decoration: sand, rocks, seaweed, bubbles, surface.
pub struct Scenery {
    ground: StaticMesh,
    seaweed: Vec<(Vec3, f32, f32, Color)>,
    bubbles: Vec<Vec3>,
    vents: Vec<Vec3>,
}

struct StaticMesh {
    verts: Vec<(Vec3, Vec3, Color)>,
    indices: Vec<u32>,
}

pub fn floor_height(x: f32, z: f32) -> f32 {
    -0.3 + 0.45 * (x * 0.21).sin() * (z * 0.17).cos() + 0.25 * (x * 0.07 + z * 0.13).sin()
}

impl Scenery {
    pub fn new() -> Self {
        let mut ground = StaticMesh { verts: Vec::new(), indices: Vec::new() };

        // Sandy floor.
        let (nx, nz) = (60, 40);
        for iz in 0..=nz {
            for ix in 0..=nx {
                let x = TANK_MIN.x + (TANK_MAX.x - TANK_MIN.x) * ix as f32 / nx as f32;
                let z = TANK_MIN.z + (TANK_MAX.z - TANK_MIN.z) * iz as f32 / nz as f32;
                let y = floor_height(x, z);
                let e = 0.05;
                let n = vec3(
                    floor_height(x - e, z) - floor_height(x + e, z),
                    2.0 * e,
                    floor_height(x, z - e) - floor_height(x, z + e),
                )
                .normalize();
                let speck = gen_range(-0.04, 0.04);
                ground.verts.push((vec3(x, y, z), n, Color::new(0.76 + speck, 0.68 + speck, 0.5 + speck, 1.0)));
            }
        }
        for iz in 0..nz {
            for ix in 0..nx {
                let a = (iz * (nx + 1) + ix) as u32;
                let b = a + 1;
                let c = a + nx as u32 + 1;
                let d = c + 1;
                ground.indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }

        // Rocks: lumpy squashed spheres.
        for _ in 0..9 {
            let center = vec3(
                gen_range(TANK_MIN.x + 3.0, TANK_MAX.x - 3.0),
                0.0,
                gen_range(TANK_MIN.z + 3.0, TANK_MAX.z - 3.0),
            );
            let center = vec3(center.x, floor_height(center.x, center.z), center.z);
            let radii = vec3(gen_range(1.2, 3.2), gen_range(0.8, 2.2), gen_range(1.2, 3.0));
            let g = gen_range(0.3, 0.45);
            let color = Color::new(g, g * 0.95, g * 0.9, 1.0);
            let seed = gen_range(0.0, 100.0);
            let start = ground.verts.len() as u32;
            let (rings, slices) = (8, 12);
            for i in 0..=rings {
                let theta = i as f32 / rings as f32 * std::f32::consts::PI;
                for j in 0..slices {
                    let a = j as f32 / slices as f32 * std::f32::consts::TAU;
                    let dir = vec3(theta.sin() * a.cos(), theta.cos(), theta.sin() * a.sin());
                    let lump = 1.0 + 0.18 * (dir.x * 3.1 + seed).sin() * (dir.z * 2.7 + seed).cos();
                    ground.verts.push((center + dir * radii * lump, dir, color));
                }
            }
            for i in 0..rings {
                for j in 0..slices {
                    let a = start + (i * slices + j) as u32;
                    let b = start + (i * slices + (j + 1) % slices) as u32;
                    ground.indices.extend_from_slice(&[a, b, a + slices as u32, b, b + slices as u32, a + slices as u32]);
                }
            }
        }

        let seaweed = (0..34)
            .map(|_| {
                let x = gen_range(TANK_MIN.x + 1.0, TANK_MAX.x - 1.0);
                let z = gen_range(TANK_MIN.z + 1.0, TANK_MAX.z - 1.0);
                let g = gen_range(0.45, 0.75);
                (
                    vec3(x, floor_height(x, z), z),
                    gen_range(4.0, 11.0),
                    gen_range(0.0, 10.0),
                    Color::new(0.15, g, 0.25, 1.0),
                )
            })
            .collect();

        let vents = (0..3)
            .map(|_| {
                let x = gen_range(TANK_MIN.x + 4.0, TANK_MAX.x - 4.0);
                let z = gen_range(TANK_MIN.z + 4.0, TANK_MAX.z - 4.0);
                vec3(x, floor_height(x, z), z)
            })
            .collect();

        Self { ground, seaweed, bubbles: Vec::new(), vents }
    }

    pub fn update(&mut self, dt: f32) {
        for v in &self.vents {
            if gen_range(0.0, 1.0) < dt * 6.0 {
                self.bubbles.push(*v + vec3(gen_range(-0.3, 0.3), 0.0, gen_range(-0.3, 0.3)));
            }
        }
        for b in &mut self.bubbles {
            b.y += dt * 3.0;
            b.x += (b.y * 1.3).sin() * dt * 0.4;
        }
        self.bubbles.retain(|b| b.y < TANK_MAX.y);
    }

    pub fn draw_opaque(&self, batch: &mut Batch, sh: &Shading, time: f32) {
        // Ground can exceed one batch, so submit it in chunks of whole triangles.
        let mut tri = 0;
        let idx = &self.ground.indices;
        while tri < idx.len() {
            let end = (tri + 6000).min(idx.len());
            let base = batch.reserve(end - tri);
            for &i in &idx[tri..end] {
                let (pos, n, color) = self.ground.verts[i as usize];
                batch.vertex(pos, sh.shade(pos, n, color, false));
            }
            batch.mesh.indices.extend((0..(end - tri) as u16).map(|i| base + i));
            tri = end;
        }

        // Swaying seaweed ribbons.
        for &(root, height, seed, color) in &self.seaweed {
            let segs = 10;
            let mut prev = root;
            for s in 1..=segs {
                let t = s as f32 / segs as f32;
                let sway = (time * 0.9 + seed + t * 2.5).sin() * 0.9 * t;
                let next = root + vec3(sway, height * t, (time * 0.7 + seed * 1.3 + t * 2.0).cos() * 0.5 * t);
                let w0 = 0.32 * (1.0 - (t - 1.0 / segs as f32) * 0.8);
                let w1 = 0.32 * (1.0 - t * 0.8);
                let side = vec3(0.4, 0.0, 1.0).normalize();
                batch.quad(sh, [prev - side * w0, prev + side * w0, next + side * w1, next - side * w1], color);
                prev = next;
            }
        }
    }

    pub fn draw_translucent(&self, batch: &mut Batch, sh: &Shading, time: f32) {
        let bubble = Color::new(0.8, 0.95, 1.0, 0.45);
        for b in &self.bubbles {
            let base = batch.reserve(6);
            for d in OCTAHEDRON {
                let pos = *b + d * 0.12;
                batch.vertex(pos, sh.fog(pos, bubble));
            }
            batch.mesh.indices.extend(OCTAHEDRON_TRIS.iter().map(|&i| base + i));
        }

        // Rippling water surface.
        let (nx, nz) = (30, 20);
        let surf = Color::new(0.35, 0.65, 0.8, 0.35);
        let h = |x: f32, z: f32| TANK_MAX.y + 0.2 * (x * 0.4 + time).sin() * (z * 0.35 - time * 0.8).cos();
        for iz in 0..nz {
            for ix in 0..nx {
                let x0 = TANK_MIN.x + (TANK_MAX.x - TANK_MIN.x) * ix as f32 / nx as f32;
                let x1 = TANK_MIN.x + (TANK_MAX.x - TANK_MIN.x) * (ix + 1) as f32 / nx as f32;
                let z0 = TANK_MIN.z + (TANK_MAX.z - TANK_MIN.z) * iz as f32 / nz as f32;
                let z1 = TANK_MIN.z + (TANK_MAX.z - TANK_MIN.z) * (iz + 1) as f32 / nz as f32;
                batch.quad(
                    sh,
                    [vec3(x0, h(x0, z0), z0), vec3(x1, h(x1, z0), z0), vec3(x1, h(x1, z1), z1), vec3(x0, h(x0, z1), z1)],
                    surf,
                );
            }
        }

        // Faint shafts of sunlight.
        for k in 0..7 {
            let kf = k as f32;
            let x = TANK_MIN.x + 6.0 + kf * 8.0 + (time * 0.1 + kf).sin() * 2.0;
            let z = -10.0 + (kf * 2.3).sin() * 8.0;
            let top = vec3(x, TANK_MAX.y, z);
            let bottom = top + vec3(-6.0, -TANK_MAX.y, 2.0);
            let alpha = 0.035 + 0.02 * (time * 0.5 + kf * 1.7).sin();
            let c = Color::new(0.7, 0.9, 1.0, alpha);
            let w = vec3(1.6, 0.0, 0.0);
            let base = batch.reserve(4);
            for v in [top - w, top + w, bottom + w * 1.8, bottom - w * 1.8] {
                batch.vertex(v, c);
            }
            batch.mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
}

pub fn draw_tank_frame() {
    let (a, b) = (TANK_MIN, TANK_MAX);
    let c = Color::new(0.6, 0.85, 1.0, 0.25);
    let corners = [
        vec3(a.x, a.y, a.z),
        vec3(b.x, a.y, a.z),
        vec3(b.x, a.y, b.z),
        vec3(a.x, a.y, b.z),
        vec3(a.x, b.y, a.z),
        vec3(b.x, b.y, a.z),
        vec3(b.x, b.y, b.z),
        vec3(a.x, b.y, b.z),
    ];
    for i in 0..4 {
        draw_line_3d(corners[i], corners[(i + 1) % 4], c);
        draw_line_3d(corners[i + 4], corners[(i + 1) % 4 + 4], c);
        draw_line_3d(corners[i], corners[i + 4], c);
    }
}
