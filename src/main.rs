//! Underwater neuroevolution: fish, jellyfish and sharks controlled by small
//! neural networks evolve in a 3D tank through natural selection.

mod brain;
mod creature;
mod hud;
mod render;
mod world;

use macroquad::prelude::*;

use creature::Species;
use hud::HudState;
use render::{Batch, Models, Scenery, Shading, WATER_COLOR};
use world::{World, DT, TANK_MAX, TANK_MIN};

const SPEEDS: [f32; 7] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
/// Cap on simulation steps per frame outside turbo mode.
const MAX_STEPS_PER_FRAME: u32 = 400;

fn window_conf() -> macroquad::conf::Conf {
    macroquad::conf::Conf {
        miniquad_conf: macroquad::window::Conf {
            window_title: "Underwater Neuroevolution".to_owned(),
            window_width: 1600,
            window_height: 900,
            sample_count: 4,
            window_resizable: true,
            ..Default::default()
        },
        draw_call_vertex_capacity: 64_000,
        draw_call_index_capacity: 160_000,
        ..Default::default()
    }
}

struct OrbitCam {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl OrbitCam {
    fn position(&self) -> Vec3 {
        self.target + vec3(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos()) * self.dist
    }

    fn camera(&self) -> Camera3D {
        Camera3D { position: self.position(), target: self.target, up: Vec3::Y, ..Default::default() }
    }
}

/// World-space ray under the mouse cursor.
fn mouse_ray(cam: &Camera3D) -> (Vec3, Vec3) {
    let (mx, my) = mouse_position();
    let ndc = vec2(mx / screen_width() * 2.0 - 1.0, 1.0 - my / screen_height() * 2.0);
    let inv = cam.matrix().inverse();
    let near = inv.project_point3(vec3(ndc.x, ndc.y, -1.0));
    let far = inv.project_point3(vec3(ndc.x, ndc.y, 1.0));
    (near, (far - near).normalize())
}

fn pick(world: &World, origin: Vec3, dir: Vec3) -> Option<u64> {
    let mut best: Option<(f32, u64)> = None;
    for c in &world.creatures {
        let radius = c.species.params().size * 1.4;
        let oc = c.pos - origin;
        let t = oc.dot(dir);
        if t < 0.0 {
            continue;
        }
        let miss2 = oc.length_squared() - t * t;
        if miss2 <= radius * radius && best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, c.id));
        }
    }
    best.map(|(_, id)| id)
}

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand((miniquad::date::now() * 1000.0) as u64);

    let mut world = World::new();
    let models = Models::new();
    let mut scenery = Scenery::new();
    let mut batch = Batch::new();
    let tank_center = (TANK_MIN + TANK_MAX) * 0.5;
    let mut cam = OrbitCam { target: tank_center, yaw: 0.55, pitch: 0.22, dist: 58.0 };

    let mut state = HudState {
        speed: 1.0,
        paused: false,
        turbo: false,
        steps_per_sec: 0.0,
        follow: false,
        show_help: true,
        show_graph: true,
    };
    let mut speed_idx = 2;
    let mut selected: Option<u64> = None;
    let mut step_acc = 0.0f32;
    let mut drag_start: Option<Vec2> = None;
    let mut dragged = false;
    let mut last_mouse = Vec2::from(mouse_position());
    let mut steps_counter = 0u32;
    let mut steps_timer = 0.0f32;

    loop {
        let frame_dt = get_frame_time().min(0.1);
        let mouse = Vec2::from(mouse_position());
        let mouse_delta = mouse - last_mouse;
        last_mouse = mouse;

        // --- Keyboard ---
        if is_key_pressed(KeyCode::Space) {
            state.paused = !state.paused;
        }
        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) || is_key_pressed(KeyCode::Up) {
            speed_idx = (speed_idx + 1).min(SPEEDS.len() - 1);
            state.turbo = false;
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) || is_key_pressed(KeyCode::Down) {
            speed_idx = speed_idx.saturating_sub(1);
            state.turbo = false;
        }
        state.speed = SPEEDS[speed_idx];
        if is_key_pressed(KeyCode::T) {
            state.turbo = !state.turbo;
        }
        if is_key_pressed(KeyCode::H) {
            state.show_help = !state.show_help;
        }
        if is_key_pressed(KeyCode::G) {
            state.show_graph = !state.show_graph;
        }
        if is_key_pressed(KeyCode::F) && selected.is_some() {
            state.follow = !state.follow;
        }
        if is_key_pressed(KeyCode::Escape) {
            selected = None;
            state.follow = false;
        }
        if is_key_pressed(KeyCode::LeftBracket) {
            world.plankton_rate = (world.plankton_rate - 4.0).max(0.0);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            world.plankton_rate = (world.plankton_rate + 4.0).min(120.0);
        }
        if is_key_pressed(KeyCode::R) {
            world = World::new();
            selected = None;
            state.follow = false;
        }
        if is_key_pressed(KeyCode::B) {
            selected = world
                .creatures
                .iter()
                .max_by(|a, b| a.fitness().total_cmp(&b.fitness()))
                .map(|c| c.id);
        }
        if is_key_pressed(KeyCode::N) && !world.creatures.is_empty() {
            let i = rand::gen_range(0, world.creatures.len());
            selected = Some(world.creatures[i].id);
        }

        // --- Camera ---
        let pan_speed = cam.dist * 0.6 * frame_dt;
        let forward = vec3(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
        let right = vec3(-forward.z, 0.0, forward.x);
        let mut pan = Vec3::ZERO;
        if is_key_down(KeyCode::W) {
            pan += forward;
        }
        if is_key_down(KeyCode::S) {
            pan -= forward;
        }
        if is_key_down(KeyCode::D) {
            pan += right;
        }
        if is_key_down(KeyCode::A) {
            pan -= right;
        }
        if is_key_down(KeyCode::E) {
            pan += Vec3::Y;
        }
        if is_key_down(KeyCode::Q) {
            pan -= Vec3::Y;
        }
        if pan != Vec3::ZERO {
            cam.target += pan.normalize() * pan_speed;
            state.follow = false;
        }
        let wheel = mouse_wheel().1;
        if wheel != 0.0 {
            cam.dist = (cam.dist * if wheel > 0.0 { 0.9 } else { 1.1 }).clamp(4.0, 160.0);
        }

        let over_panel = selected.is_some() && hud::brain_panel_rect().contains(mouse);
        if is_mouse_button_pressed(MouseButton::Left) && !over_panel {
            drag_start = Some(mouse);
            dragged = false;
        }
        if let Some(start) = drag_start {
            if is_mouse_button_down(MouseButton::Left) {
                if mouse.distance(start) > 4.0 {
                    dragged = true;
                }
                if dragged {
                    cam.yaw -= mouse_delta.x * 0.006;
                    cam.pitch = (cam.pitch + mouse_delta.y * 0.006).clamp(-1.4, 1.4);
                }
            }
            if is_mouse_button_released(MouseButton::Left) {
                if !dragged {
                    let (o, d) = mouse_ray(&cam.camera());
                    selected = pick(&world, o, d);
                    if selected.is_none() {
                        state.follow = false;
                    }
                }
                drag_start = None;
            }
        }

        // --- Simulation ---
        if !state.paused {
            if state.turbo {
                // Run as many steps as fit in ~28ms, keeping the window responsive.
                let start = get_time();
                while get_time() - start < 0.028 {
                    for _ in 0..10 {
                        world.step();
                    }
                    steps_counter += 10;
                }
            } else {
                step_acc += frame_dt * state.speed;
                let mut steps = 0;
                while step_acc >= DT && steps < MAX_STEPS_PER_FRAME {
                    world.step();
                    step_acc -= DT;
                    steps += 1;
                }
                if steps == MAX_STEPS_PER_FRAME {
                    step_acc = 0.0;
                }
                steps_counter += steps;
            }
        }
        steps_timer += frame_dt;
        if steps_timer >= 0.5 {
            state.steps_per_sec = steps_counter as f32 / steps_timer;
            steps_counter = 0;
            steps_timer = 0.0;
        }
        scenery.update(frame_dt);

        let selected_creature = selected.and_then(|id| world.find(id));
        if selected.is_some() && selected_creature.is_none() {
            // It died; keep following its lineage's region but drop the selection.
            selected = None;
            state.follow = false;
        }
        if state.follow {
            if let Some(c) = selected_creature {
                cam.target += (c.pos - cam.target) * (frame_dt * 4.0).min(1.0);
            }
        }

        // --- Render 3D ---
        clear_background(WATER_COLOR);
        let cam3d = cam.camera();
        set_camera(&cam3d);
        let sh = Shading::new(cam3d.position);
        let time = get_time() as f32;

        scenery.draw_opaque(&mut batch, &sh, time);
        render::draw_plankton(&mut batch, &sh, &world, time);
        for c in &world.creatures {
            render::draw_creature(&mut batch, &sh, &models, c);
        }
        batch.flush();
        render::draw_tank_frame();

        if let Some(c) = selected_creature {
            let p = c.species.params();
            draw_sphere_wires(c.pos, p.size * 2.0, None, Color::new(1.0, 1.0, 1.0, 0.35));
            if let Some(f) = c.seen_food {
                draw_line_3d(c.pos, f, Color::new(0.3, 1.0, 0.4, 0.9));
            }
            if let Some(t) = c.seen_threat {
                draw_line_3d(c.pos, t, Color::new(1.0, 0.3, 0.25, 0.9));
            }
        }

        for c in world.creatures.iter().filter(|c| c.species == Species::Jellyfish) {
            render::draw_jellyfish(&mut batch, &sh, &models, c, world.time);
        }
        scenery.draw_translucent(&mut batch, &sh, time);
        batch.flush();

        // --- HUD ---
        set_default_camera();
        hud::draw(&world, &state, selected_creature);

        next_frame().await;
    }
}
