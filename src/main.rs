//! A 3D person learns to walk. Evolution strategies train a neural-network
//! brain in the background; the window shows the best brain so far
//! attempting to walk down a track.

mod brain;
mod hud;
mod render;
mod trainer;
mod walker;

use std::sync::atomic::Ordering;

use macroquad::prelude::*;

use brain::Brain;
use hud::{HudState, Showcase};
use render::{Batch, Models, Scenery, Shading, HORIZON};
use trainer::Trainer;
use walker::{Walker, DT, EPISODE_SECONDS};

const SPEEDS: [f32; 6] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
/// How long to linger on the final pose before replaying.
const REPLAY_DELAY: f32 = 1.5;

fn window_conf() -> macroquad::conf::Conf {
    macroquad::conf::Conf {
        miniquad_conf: macroquad::window::Conf {
            window_title: "Learning to Walk".to_owned(),
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
    fn camera(&self) -> Camera3D {
        let offset = vec3(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos());
        Camera3D { position: self.target + offset * self.dist, target: self.target, up: Vec3::Y, ..Default::default() }
    }
}

/// The single person on screen, replaying the current champion brain.
struct Show {
    walker: Walker,
    brain: Brain,
    generation: Option<u32>,
    ended_for: f32,
}

impl Show {
    fn new(trainer: &Trainer) -> Self {
        let champion = trainer.progress.lock().unwrap().champion.clone();
        let (brain, generation) = match champion {
            Some(c) => (c.brain, Some(c.generation)),
            None => (Brain::random(), None),
        };
        Self { walker: Walker::new(), brain, generation, ended_for: 0.0 }
    }

    fn finished(&self) -> bool {
        !self.walker.fallen && self.walker.time >= EPISODE_SECONDS
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand((miniquad::date::now() * 1000.0) as u64);

    let mut trainer = Trainer::start();
    let models = Models::new();
    let scenery = Scenery::new();
    let mut batch = Batch::new();
    let mut show = Show::new(&trainer);
    let mut cam = OrbitCam { target: vec3(0.0, 0.95, 0.0), yaw: 0.55, pitch: 0.16, dist: 6.0 };

    let mut state = HudState {
        speed: 1.0,
        paused: false,
        training_paused: false,
        show_help: true,
        show_graph: true,
        show_brain: true,
    };
    let mut speed_idx = 2;
    let mut step_acc = 0.0f32;
    let mut last_mouse = Vec2::from(mouse_position());

    loop {
        let frame_dt = get_frame_time().min(0.1);
        let mouse = Vec2::from(mouse_position());
        let mouse_delta = mouse - last_mouse;
        last_mouse = mouse;

        // --- Input ---
        if is_key_pressed(KeyCode::Space) {
            state.paused = !state.paused;
        }
        // Read +/- as typed characters so they work on any keyboard layout
        // (on a German keyboard, for example, + is not where KeyCode::Equal is).
        let (mut faster, mut slower) = (is_key_pressed(KeyCode::Up), is_key_pressed(KeyCode::Down));
        while let Some(c) = get_char_pressed() {
            match c {
                '+' | '=' => faster = true,
                '-' | '_' => slower = true,
                _ => {}
            }
        }
        if faster {
            speed_idx = (speed_idx + 1).min(SPEEDS.len() - 1);
        }
        if slower {
            speed_idx = speed_idx.saturating_sub(1);
        }
        state.speed = SPEEDS[speed_idx];
        if is_key_pressed(KeyCode::H) {
            state.show_help = !state.show_help;
        }
        if is_key_pressed(KeyCode::G) {
            state.show_graph = !state.show_graph;
        }
        if is_key_pressed(KeyCode::B) {
            state.show_brain = !state.show_brain;
        }
        if is_key_pressed(KeyCode::P) {
            state.training_paused = !state.training_paused;
            trainer.paused.store(state.training_paused, Ordering::Relaxed);
        }
        if is_key_pressed(KeyCode::R) {
            trainer.restart_from_scratch();
            trainer.paused.store(state.training_paused, Ordering::Relaxed);
            show = Show::new(&trainer);
        }
        if is_key_pressed(KeyCode::N) {
            show = Show::new(&trainer);
        }
        if is_mouse_button_down(MouseButton::Left) {
            cam.yaw -= mouse_delta.x * 0.006;
            cam.pitch = (cam.pitch + mouse_delta.y * 0.006).clamp(0.02, 1.4);
        }
        let wheel = mouse_wheel().1;
        if wheel != 0.0 {
            cam.dist = (cam.dist * if wheel > 0.0 { 0.9 } else { 1.1 }).clamp(2.0, 40.0);
        }

        // --- Simulation of the on-screen person ---
        if !state.paused {
            if show.walker.fallen || show.finished() {
                show.ended_for += frame_dt;
                if show.ended_for > REPLAY_DELAY {
                    show = Show::new(&trainer);
                }
            } else {
                step_acc += frame_dt * state.speed;
                while step_acc >= DT {
                    show.walker.step(&mut show.brain);
                    step_acc -= DT;
                    if show.walker.fallen || show.finished() {
                        step_acc = 0.0;
                        break;
                    }
                }
            }
        }

        let [px, _, pz] = show.walker.trunk_position();
        let follow = vec3(px, 0.95, pz);
        cam.target += (follow - cam.target) * (frame_dt * 3.0).min(1.0);

        // --- Render ---
        clear_background(HORIZON);
        render::draw_sky();
        let cam3d = cam.camera();
        set_camera(&cam3d);
        let sh = Shading::new(cam3d.position);
        let record = trainer.progress.lock().unwrap().champion.as_ref().map(|c| c.score.distance);
        let parts = show.walker.parts();

        scenery.draw(&mut batch, &sh, &models, cam.target.x, record);
        render::draw_person(&mut batch, &sh, &models, &parts);
        batch.flush();
        render::draw_shadows(&mut batch, &parts);
        batch.flush();

        set_default_camera();
        let view_proj = cam3d.matrix();
        let first = ((cam.target.x - 15.0) / 5.0).ceil().max(0.0) as i32;
        for k in first..first + 8 {
            let m = k as f32 * 5.0;
            hud::label_3d(&view_proj, vec3(m, 1.35, -1.5), &format!("{m:.0} m"), Color::new(1.0, 1.0, 1.0, 0.9));
        }
        if let Some(r) = record {
            hud::label_3d(&view_proj, vec3(r, 2.05, 1.4), &format!("record {r:.1} m"), Color::new(1.0, 0.85, 0.2, 1.0));
        }

        let showcase = Showcase {
            generation: show.generation,
            time: show.walker.time,
            distance: show.walker.distance,
            fallen: show.walker.fallen,
            finished: show.finished(),
        };
        {
            let progress = trainer.progress.lock().unwrap();
            hud::draw(&progress, &showcase, &state, &show.brain);
        }

        next_frame().await;
    }
}
