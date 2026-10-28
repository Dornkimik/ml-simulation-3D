//! 2D overlay: training progress, learning curve, and the live brain.

use macroquad::prelude::*;

use crate::brain::{Brain, INPUT_LABELS, OUTPUT_LABELS};
use crate::trainer::{Progress, POPULATION};

const PANEL: Color = Color::new(0.02, 0.05, 0.1, 0.62);
const TEXT: Color = Color::new(0.93, 0.96, 1.0, 1.0);
const DIM: Color = Color::new(0.66, 0.76, 0.86, 1.0);
const ACCENT: Color = Color::new(1.0, 0.8, 0.2, 1.0);
const GOOD: Color = Color::new(0.45, 0.95, 0.5, 1.0);
const BAD: Color = Color::new(1.0, 0.45, 0.4, 1.0);

/// What the on-screen person is doing right now.
pub struct Showcase {
    pub generation: Option<u32>,
    pub time: f32,
    pub distance: f32,
    pub fallen: bool,
    pub finished: bool,
}

pub struct HudState {
    pub speed: f32,
    pub paused: bool,
    pub training_paused: bool,
    pub show_help: bool,
    pub show_graph: bool,
    pub show_brain: bool,
}

fn panel(x: f32, y: f32, w: f32, h: f32) {
    draw_rectangle(x, y, w, h, PANEL);
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(1.0, 1.0, 1.0, 0.18));
}

fn text(s: &str, x: f32, y: f32, size: f32, color: Color) {
    draw_text(s, x, y, size, color);
}

pub fn draw(progress: &Progress, show: &Showcase, state: &HudState, brain: &Brain) {
    draw_status(progress, show, state);
    if state.show_graph {
        draw_graph(progress);
    }
    if state.show_brain {
        draw_brain_panel(brain);
    }
    draw_help(state);
}

fn draw_status(progress: &Progress, show: &Showcase, state: &HudState) {
    let (x, y) = (12.0, 12.0);
    panel(x, y, 440.0, 192.0);
    text("LEARNING TO WALK", x + 12.0, y + 28.0, 28.0, TEXT);

    let training = if state.training_paused {
        "training paused".to_string()
    } else {
        format!("{:.0} attempts/s", progress.attempts_per_sec)
    };
    text(
        &format!("generation {}   {} attempts   {}", progress.generation, progress.generation as usize * POPULATION, training),
        x + 12.0,
        y + 52.0,
        17.0,
        DIM,
    );
    if progress.resumed_from_file {
        text("(resumed from walker_brain.txt)", x + 12.0, y + 70.0, 15.0, DIM);
    }

    match &progress.champion {
        Some(c) => text(
            &format!("record: {:.2} m in {:.0} s  (generation {})", c.score.distance, c.score.time, c.generation),
            x + 12.0,
            y + 96.0,
            21.0,
            ACCENT,
        ),
        None => text("record: waiting for the first generation...", x + 12.0, y + 96.0, 19.0, ACCENT),
    }

    let who = match show.generation {
        Some(g) => format!("showing the champion of generation {g}"),
        None => "showing an untrained random brain".to_string(),
    };
    text(&who, x + 12.0, y + 124.0, 17.0, TEXT);
    let (status, color) = if show.fallen {
        ("FELL", BAD)
    } else if show.finished {
        ("FINISHED", GOOD)
    } else {
        ("walking", TEXT)
    };
    text(&format!("attempt  {:>5.1} s   {:>6.2} m   ", show.time, show.distance), x + 12.0, y + 148.0, 20.0, TEXT);
    text(status, x + 300.0, y + 148.0, 20.0, color);

    let speed = if state.paused { "view paused".to_string() } else { format!("view speed x{}", state.speed) };
    text(&speed, x + 12.0, y + 176.0, 16.0, DIM);
}

fn draw_graph(progress: &Progress) {
    let (w, h) = (440.0, 170.0);
    let x = 12.0;
    let y = screen_height() - h - 12.0;
    panel(x, y, w, h);
    text("best distance per generation (m)", x + 10.0, y + 18.0, 16.0, ACCENT);
    let hist = &progress.history;
    if hist.len() < 2 {
        return;
    }
    let (gx, gy, gw, gh) = (x + 36.0, y + 30.0, w - 48.0, h - 50.0);
    let max = hist.iter().map(|s| s.best_distance).fold(1.0f32, f32::max).ceil();
    let n = hist.len();
    let px = |i: usize| gx + gw * i as f32 / (n - 1) as f32;
    let py = |v: f32| gy + gh * (1.0 - (v / max).clamp(0.0, 1.0));

    for k in 0..=4 {
        let v = max * k as f32 / 4.0;
        let ly = py(v);
        draw_line(gx, ly, gx + gw, ly, 1.0, Color::new(1.0, 1.0, 1.0, 0.08));
        text(&format!("{v:.0}"), x + 8.0, ly + 4.0, 14.0, DIM);
    }
    // Draw at most ~400 segments, however long training has run.
    let step = (n / 400).max(1);
    let mut prev = 0;
    let mut best_so_far = hist[0].best_distance;
    let mut prev_best = best_so_far;
    for i in (step..n).step_by(step).chain(std::iter::once(n - 1)) {
        for s in &hist[prev + 1..=i] {
            best_so_far = best_so_far.max(s.best_distance);
        }
        draw_line(px(prev), py(hist[prev].best_distance), px(i), py(hist[i].best_distance), 1.0, Color::new(0.5, 0.75, 1.0, 0.6));
        draw_line(px(prev), py(prev_best), px(i), py(best_so_far), 2.0, ACCENT);
        prev_best = best_so_far;
        prev = i;
    }
    text(&format!("gen {n}"), gx + gw - 50.0, y + h - 6.0, 14.0, DIM);
}

fn activation_color(a: f32) -> Color {
    let m = a.abs().min(1.0);
    if a >= 0.0 {
        Color::new(0.25 + 0.75 * m, 0.25 + 0.45 * m, 0.25, 1.0)
    } else {
        Color::new(0.25, 0.25 + 0.4 * m, 0.25 + 0.75 * m, 1.0)
    }
}

fn draw_brain_panel(brain: &Brain) {
    let w = 430.0;
    let r = Rect::new(screen_width() - w - 12.0, 12.0, w, 560.0);
    panel(r.x, r.y, r.w, r.h);
    text("BRAIN  (left leg, live)", r.x + 14.0, r.y + 26.0, 20.0, TEXT);
    text("the same network drives the right leg, mirrored", r.x + 14.0, r.y + 44.0, 14.0, DIM);
    let area = Rect::new(r.x + 90.0, r.y + 62.0, r.w - 170.0, r.h - 76.0);

    let acts = &brain.activations;
    let cols = acts.len();
    let node_pos = |layer: usize, i: usize| -> Vec2 {
        let count = acts[layer].len();
        let x = area.x + area.w * layer as f32 / (cols - 1) as f32;
        let spacing = area.h / count as f32;
        vec2(x, area.y + spacing * (i as f32 + 0.5))
    };
    for (li, layer) in brain.layers.iter().enumerate() {
        for o in 0..layer.outputs {
            let to = node_pos(li + 1, o);
            for i in 0..layer.inputs {
                let w = layer.weight(o, i);
                let signal = (w * acts[li][i]).abs();
                let alpha = (w.abs() * 0.1 + signal * 0.5).min(0.85);
                if alpha < 0.04 {
                    continue;
                }
                let from = node_pos(li, i);
                let color = if w > 0.0 { Color::new(1.0, 0.7, 0.3, alpha) } else { Color::new(0.35, 0.7, 1.0, alpha) };
                draw_line(from.x, from.y, to.x, to.y, 1.0 + signal.min(1.5), color);
            }
        }
    }
    for (li, layer_acts) in acts.iter().enumerate() {
        for (i, &a) in layer_acts.iter().enumerate() {
            let p = node_pos(li, i);
            draw_circle(p.x, p.y, 6.0, activation_color(a));
            draw_circle_lines(p.x, p.y, 6.0, 1.0, Color::new(1.0, 1.0, 1.0, 0.35));
            if li == 0 {
                let dims = measure_text(INPUT_LABELS[i], None, 14, 1.0);
                text(INPUT_LABELS[i], p.x - 12.0 - dims.width, p.y + 4.0, 14.0, DIM);
            } else if li == cols - 1 {
                text(OUTPUT_LABELS[i], p.x + 12.0, p.y + 4.0, 14.0, DIM);
            }
        }
    }
}

fn draw_help(state: &HudState) {
    let lines: &[&str] = if state.show_help {
        &[
            "drag: orbit camera   wheel: zoom",
            "N: replay with the latest champion",
            "Space: pause view   +/-: view speed",
            "P: pause training   R: forget & restart",
            "B: brain   G: graph   H: hide help",
        ]
    } else {
        &["H: help"]
    };
    let w = 330.0;
    let h = lines.len() as f32 * 20.0 + 14.0;
    let x = screen_width() - w - 12.0;
    let y = screen_height() - h - 12.0;
    panel(x, y, w, h);
    for (i, l) in lines.iter().enumerate() {
        text(l, x + 12.0, y + 24.0 + i as f32 * 20.0, 17.0, TEXT);
    }
}

/// Draws a text label at a 3D world position (e.g. distance markers).
pub fn label_3d(view_proj: &Mat4, pos: Vec3, s: &str, color: Color) {
    let clip = *view_proj * pos.extend(1.0);
    if clip.w <= 0.1 {
        return;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.x.abs() > 1.1 || ndc.y.abs() > 1.1 {
        return;
    }
    let sx = (ndc.x + 1.0) * 0.5 * screen_width();
    let sy = (1.0 - ndc.y) * 0.5 * screen_height();
    let size = (900.0 / clip.w).clamp(12.0, 26.0);
    let dims = measure_text(s, None, size as u16, 1.0);
    draw_text(s, sx - dims.width * 0.5, sy, size, color);
}
