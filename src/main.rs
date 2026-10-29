//! Two people learn to play soccer against each other. Each player has its
//! own neural-network brain, trained in the background with evolution
//! strategies by playing against the other; the window shows the latest
//! brains playing a match. Brains can be exported to files and imported back.

mod brain;
mod hud;
mod pose;
mod render;
mod soccer;
mod trainer;

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use macroquad::prelude::*;

use brain::Brain;
use hud::{Button, HudState, MatchView, TEAM_COLORS, TEAM_NAMES};
use pose::Anim;
use render::{Batch, Models, Scenery, Shading, HORIZON};
use soccer::{Match, BALL_RADIUS, DT};
use trainer::Trainer;

const SPEEDS: [f32; 6] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
const SHOW_MATCH_SECONDS: f32 = 45.0;
/// Pause after a goal, before the next kickoff.
const GOAL_CELEBRATION: f32 = 1.5;
/// How long to linger at full time before the next match.
const REPLAY_DELAY: f32 = 2.5;
const MESSAGE_SECONDS: f32 = 6.0;

fn window_conf() -> macroquad::conf::Conf {
    macroquad::conf::Conf {
        miniquad_conf: macroquad::window::Conf {
            window_title: "Learning to Play Soccer".to_owned(),
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

/// The match on screen, between the latest brains of both players.
struct Show {
    game: Match,
    brains: [Brain; 2],
    generations: [u32; 2],
    anims: [Anim; 2],
    ball_rot: Quat,
    number: u32,
    ended_for: f32,
    counted: bool,
}

impl Show {
    fn new(trainer: &Trainer, number: u32) -> Self {
        let (brains, generations) = {
            let p = trainer.progress.lock().unwrap();
            ([p.players[0].brain.clone(), p.players[1].brain.clone()], [p.players[0].generation, p.players[1].generation])
        };
        let game = Match::new(rand::gen_range(0, u32::MAX) as u64, SHOW_MATCH_SECONDS, GOAL_CELEBRATION);
        let anims = [Anim::new(&game.players[0]), Anim::new(&game.players[1])];
        Self { game, brains, generations, anims, ball_rot: Quat::IDENTITY, number, ended_for: 0.0, counted: false }
    }

    fn step(&mut self) {
        self.game.step(&mut self.brains);
        for (anim, p) in self.anims.iter_mut().zip(&self.game.players) {
            anim.update(p);
        }
        // Spin the ball as if it were rolling.
        let v = self.game.ball.vel;
        let spin = vec3(v.z, 0.0, -v.x) / BALL_RADIUS;
        if spin.length() > 1e-3 {
            self.ball_rot = (Quat::from_axis_angle(spin.normalize(), spin.length() * DT) * self.ball_rot).normalize();
        }
    }

    fn view(&self) -> MatchView {
        MatchView {
            number: self.number,
            score: self.game.score,
            time: self.game.time,
            duration: self.game.duration,
            generations: self.generations,
            celebrating: (self.game.celebrating > 0.0).then_some(self.game.last_scorer).flatten(),
            over: self.game.over(),
        }
    }
}

fn brain_file_dialog(title: &str) -> rfd::FileDialog {
    rfd::FileDialog::new()
        .set_title(title)
        .add_filter("Soccer brain", &["txt"])
        .set_directory(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Handles a click on (or the shortcut for) one of a player's buttons.
/// Returns a status message, and whether to restart the match on screen.
fn press(button: Button, player: usize, trainer: &Trainer) -> (String, bool) {
    let name = TEAM_NAMES[player];
    let (brain, generation, learning) = {
        let p = trainer.progress.lock().unwrap();
        let pp = &p.players[player];
        (pp.brain.clone(), pp.generation, pp.learning)
    };
    match button {
        Button::Export => {
            let file = format!("{}_brain_gen{generation}.txt", name.to_lowercase());
            let Some(path) = brain_file_dialog(&format!("Export the {name} brain")).set_file_name(file).save_file() else {
                return ("export cancelled".into(), false);
            };
            match brain.save(&path, generation) {
                Ok(()) => (format!("exported the {name} brain (generation {generation}) to {}", path.display()), false),
                Err(e) => (format!("could not export to {}: {e}", path.display()), false),
            }
        }
        Button::Import => {
            let Some(path) = brain_file_dialog(&format!("Import a brain for {name}")).pick_file() else {
                return ("import cancelled".into(), false);
            };
            match Brain::load(&path) {
                Ok((brain, generation)) => {
                    trainer.replace_brain(player, brain, generation);
                    (format!("{name} now plays with {} (generation {generation})", path.display()), true)
                }
                Err(e) => (format!("could not import {}: {e}", path.display()), false),
            }
        }
        Button::Learning => {
            trainer.set_learning(player, !learning);
            let msg = if learning { "is frozen: it keeps playing but stops learning" } else { "is learning again" };
            (format!("{name} {msg}"), false)
        }
        Button::Reset => {
            trainer.replace_brain(player, Brain::random(), 0);
            (format!("{name} starts over with a random brain"), true)
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand((miniquad::date::now() * 1000.0) as u64);

    let trainer = Trainer::start();
    let models = Models::new();
    let scenery = Scenery::new();
    let mut batch = Batch::new();
    let mut show = Show::new(&trainer, 1);
    let mut cam = OrbitCam { target: vec3(0.0, 0.5, 0.0), yaw: 0.0, pitch: 0.62, dist: 27.0 };

    let mut state = HudState {
        speed: 1.0,
        paused: false,
        training_paused: false,
        show_help: true,
        show_graph: true,
        show_brain: true,
        selected: 0,
        wins: [0, 0],
        draws: 0,
        message: None,
    };
    let mut speed_idx = 2;
    let mut step_acc = 0.0f32;
    let mut last_mouse = Vec2::from(mouse_position());
    // Whether the current mouse drag started on a panel (and so shouldn't
    // turn the camera).
    let mut dragging_ui = false;

    loop {
        let frame_dt = get_frame_time().min(0.1);
        let mouse = Vec2::from(mouse_position());
        let mouse_delta = mouse - last_mouse;
        last_mouse = mouse;

        // --- Input ---
        let mut pressed = None;
        if is_mouse_button_pressed(MouseButton::Left) {
            if let Some((player, button)) = hud::hit(mouse) {
                dragging_ui = true;
                state.selected = player;
                pressed = button.map(|b| (b, player));
            }
        }
        if !is_mouse_button_down(MouseButton::Left) {
            dragging_ui = false;
        }
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
        if is_key_pressed(KeyCode::Tab) {
            state.selected = 1 - state.selected;
        }
        if is_key_pressed(KeyCode::P) {
            state.training_paused = !state.training_paused;
            trainer.paused.store(state.training_paused, Ordering::Relaxed);
        }
        if is_key_pressed(KeyCode::N) {
            show = Show::new(&trainer, show.number + 1);
        }
        for (key, button) in [(KeyCode::I, Button::Import), (KeyCode::E, Button::Export), (KeyCode::F, Button::Learning), (KeyCode::R, Button::Reset)] {
            if is_key_pressed(key) {
                pressed = Some((button, state.selected));
            }
        }
        if let Some((button, player)) = pressed {
            let (msg, restart) = press(button, player, &trainer);
            state.message = Some((msg, MESSAGE_SECONDS));
            if restart {
                show = Show::new(&trainer, show.number + 1);
            }
            // A file dialog may have taken a while; don't count that time.
            step_acc = 0.0;
            last_mouse = Vec2::from(mouse_position());
            dragging_ui = false;
        }
        if let Some((_, left)) = &mut state.message {
            *left -= frame_dt;
            if *left <= 0.0 {
                state.message = None;
            }
        }
        if is_mouse_button_down(MouseButton::Left) && !dragging_ui {
            cam.yaw -= mouse_delta.x * 0.006;
            cam.pitch = (cam.pitch + mouse_delta.y * 0.006).clamp(0.05, 1.45);
        }
        let wheel = mouse_wheel().1;
        if wheel != 0.0 {
            cam.dist = (cam.dist * if wheel > 0.0 { 0.9 } else { 1.1 }).clamp(3.0, 70.0);
        }

        // --- Simulation of the match on screen ---
        if !state.paused {
            if show.game.over() {
                if !show.counted {
                    show.counted = true;
                    let [b, r] = show.game.score;
                    match b.cmp(&r) {
                        std::cmp::Ordering::Greater => state.wins[0] += 1,
                        std::cmp::Ordering::Less => state.wins[1] += 1,
                        std::cmp::Ordering::Equal => state.draws += 1,
                    }
                }
                show.ended_for += frame_dt;
                if show.ended_for > REPLAY_DELAY {
                    show = Show::new(&trainer, show.number + 1);
                }
            } else {
                step_acc += frame_dt * state.speed;
                while step_acc >= DT {
                    show.step();
                    step_acc -= DT;
                }
            }
        }

        // The camera drifts after the ball, but keeps most of the pitch in view.
        let ball = show.game.ball.pos;
        let follow = vec3(ball.x * 0.55, 0.5, ball.z * 0.35);
        cam.target += (follow - cam.target) * (frame_dt * 2.0).min(1.0);

        // --- Render ---
        clear_background(HORIZON);
        render::draw_sky();
        let cam3d = cam.camera();
        set_camera(&cam3d);
        let sh = Shading::new(cam3d.position);
        scenery.draw(&mut batch, &sh, &models);
        let parts: Vec<_> = (0..2).map(|i| show.anims[i].parts(&show.game.players[i])).collect();
        for (i, p) in parts.iter().enumerate() {
            render::draw_person(&mut batch, &sh, &models.people[i], p);
        }
        render::draw_ball(&mut batch, &sh, &models, ball, show.ball_rot);
        batch.flush();
        render::draw_shadows(&mut batch, &parts.concat(), ball);
        batch.flush();

        set_default_camera();
        let view_proj = cam3d.matrix();
        for (i, p) in show.game.players.iter().enumerate() {
            hud::label_3d(&view_proj, vec3(p.pos.x, 2.2, p.pos.y), TEAM_NAMES[i], TEAM_COLORS[i]);
        }
        {
            let progress = trainer.progress.lock().unwrap();
            hud::draw(&progress, &show.view(), &state, &show.brains[state.selected]);
        }

        next_frame().await;
    }
}
