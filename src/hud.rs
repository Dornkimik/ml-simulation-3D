//! 2D overlay: statistics, population graph, and the selected creature's brain.

use macroquad::prelude::*;

use crate::brain::{INPUT_LABELS, OUTPUT_LABELS};
use crate::creature::{Creature, Species};
use crate::world::World;

const PANEL: Color = Color::new(0.0, 0.04, 0.08, 0.62);
const TEXT: Color = Color::new(0.88, 0.95, 1.0, 1.0);
const DIM: Color = Color::new(0.6, 0.72, 0.8, 1.0);
const PLANKTON: Color = Color::new(0.35, 0.95, 0.45, 1.0);

pub struct HudState {
    pub speed: f32,
    pub paused: bool,
    pub turbo: bool,
    pub steps_per_sec: f32,
    pub follow: bool,
    pub show_help: bool,
    pub show_graph: bool,
}

fn species_color(s: Species) -> Color {
    let c = s.params().color;
    Color::new(c.r, c.g, c.b, 1.0)
}

fn panel(x: f32, y: f32, w: f32, h: f32) {
    draw_rectangle(x, y, w, h, PANEL);
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(0.5, 0.8, 1.0, 0.25));
}

fn text(s: &str, x: f32, y: f32, size: f32, color: Color) {
    draw_text(s, x, y, size, color);
}

/// Screen rectangle of the selected-creature panel, so clicks there don't
/// deselect it.
pub fn brain_panel_rect() -> Rect {
    let w = 430.0;
    Rect::new(screen_width() - w - 12.0, 12.0, w, 640.0)
}

pub fn draw(world: &World, state: &HudState, selected: Option<&Creature>) {
    draw_stats(world, state);
    if state.show_graph {
        draw_graph(world);
    }
    if let Some(c) = selected {
        draw_selected(c, state);
    }
    draw_help(state);
}

fn draw_stats(world: &World, state: &HudState) {
    let (x, y) = (12.0, 12.0);
    panel(x, y, 430.0, 196.0);
    text("UNDERWATER NEUROEVOLUTION", x + 12.0, y + 26.0, 24.0, TEXT);
    let mode = if state.paused {
        "PAUSED".to_string()
    } else if state.turbo {
        format!("TURBO  {:.0} steps/s", state.steps_per_sec)
    } else {
        format!("speed x{}", state.speed)
    };
    text(&format!("time {:>7.1}s   {}", world.time, mode), x + 12.0, y + 48.0, 18.0, DIM);

    let mut ly = y + 76.0;
    text("species     pop   gen  births  best", x + 34.0, ly, 16.0, DIM);
    for s in Species::ALL {
        ly += 22.0;
        let st = &world.stats[s.index()];
        draw_rectangle(x + 14.0, ly - 11.0, 12.0, 12.0, species_color(s));
        text(
            &format!("{:<10} {:>4} {:>5} {:>7} {:>5.1}", s.name(), st.count, st.max_generation, st.births, st.best_fitness),
            x + 34.0,
            ly,
            18.0,
            TEXT,
        );
    }
    ly += 26.0;
    draw_rectangle(x + 14.0, ly - 11.0, 12.0, 12.0, PLANKTON);
    text(
        &format!("Plankton   {:>4}   spawn {:.0}/s", world.plankton.len(), world.plankton_rate),
        x + 34.0,
        ly,
        18.0,
        TEXT,
    );
    ly += 22.0;
    text(&format!("shark kills {}", world.kills), x + 34.0, ly, 18.0, DIM);
}

fn draw_graph(world: &World) {
    let (w, h) = (420.0, 160.0);
    let x = 12.0;
    let y = screen_height() - h - 12.0;
    panel(x, y, w, h);
    text("population (each scaled to its cap)", x + 10.0, y + 18.0, 16.0, DIM);
    let gx = x + 10.0;
    let gy = y + 28.0;
    let gw = w - 20.0;
    let gh = h - 38.0;
    let hist = &world.history;
    if hist.len() < 2 {
        return;
    }
    let n = hist.len();
    let px = |i: usize| gx + gw * i as f32 / (n - 1) as f32;
    let py = |v: f32| gy + gh * (1.0 - v.clamp(0.0, 1.0));
    for i in 1..n {
        let (a, b) = (&hist[i - 1], &hist[i]);
        draw_line(px(i - 1), py(a.plankton as f32 / 900.0), px(i), py(b.plankton as f32 / 900.0), 1.0, Color::new(0.35, 0.95, 0.45, 0.5));
        for s in Species::ALL {
            let cap = s.params().max_pop as f32;
            let si = s.index();
            draw_line(px(i - 1), py(a.counts[si] as f32 / cap), px(i), py(b.counts[si] as f32 / cap), 2.0, species_color(s));
        }
    }
}

fn draw_selected(c: &Creature, state: &HudState) {
    let r = brain_panel_rect();
    panel(r.x, r.y, r.w, r.h);
    let p = c.species.params();
    let (x, mut y) = (r.x + 14.0, r.y + 28.0);
    text(&format!("{} #{}", c.species.name(), c.id), x, y, 26.0, species_color(c.species));
    text(&format!("generation {}", c.generation), x + 250.0, y, 18.0, DIM);
    y += 22.0;

    // Energy bar.
    let frac = (c.energy / p.max_energy).clamp(0.0, 1.0);
    draw_rectangle(x, y, r.w - 28.0, 10.0, Color::new(1.0, 1.0, 1.0, 0.1));
    let bar = Color::new(1.0 - frac, 0.3 + 0.6 * frac, 0.3, 1.0);
    draw_rectangle(x, y, (r.w - 28.0) * frac, 10.0, bar);
    y += 28.0;
    text(
        &format!("energy {:>5.1}/{:.0}   age {:>5.1}s   speed {:>4.1}", c.energy, p.max_energy, c.age, c.vel.length()),
        x,
        y,
        17.0,
        TEXT,
    );
    y += 20.0;
    text(
        &format!("meals {}   children {}   fitness {:.1}   {}", c.meals, c.children, c.fitness(), if state.follow { "[following]" } else { "" }),
        x,
        y,
        17.0,
        TEXT,
    );
    y += 22.0;
    text("neural network (live activations)", x, y, 16.0, DIM);
    draw_brain(c, Rect::new(r.x + 74.0, y + 12.0, r.w - 160.0, r.y + r.h - y - 24.0));
}

fn activation_color(a: f32) -> Color {
    let m = a.abs().min(1.0);
    if a >= 0.0 {
        Color::new(0.25 + 0.75 * m, 0.25 + 0.45 * m, 0.25, 1.0)
    } else {
        Color::new(0.25, 0.25 + 0.4 * m, 0.25 + 0.75 * m, 1.0)
    }
}

fn draw_brain(c: &Creature, area: Rect) {
    let acts = &c.brain.activations;
    let cols = acts.len();
    let node_pos = |layer: usize, i: usize| -> Vec2 {
        let count = acts[layer].len();
        let x = area.x + area.w * layer as f32 / (cols - 1) as f32;
        let spacing = area.h / count as f32;
        vec2(x, area.y + spacing * (i as f32 + 0.5))
    };

    for (li, layer) in c.brain.layers.iter().enumerate() {
        for o in 0..layer.outputs {
            let to = node_pos(li + 1, o);
            for i in 0..layer.inputs {
                let w = layer.weight(o, i);
                let signal = (w * acts[li][i]).abs();
                let alpha = (w.abs() * 0.12 + signal * 0.5).min(0.85);
                if alpha < 0.04 {
                    continue;
                }
                let from = node_pos(li, i);
                let color = if w > 0.0 {
                    Color::new(1.0, 0.7, 0.3, alpha)
                } else {
                    Color::new(0.35, 0.7, 1.0, alpha)
                };
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
                let label = INPUT_LABELS[i];
                let dims = measure_text(label, None, 14, 1.0);
                text(label, p.x - 12.0 - dims.width, p.y + 4.0, 14.0, DIM);
            } else if li == cols - 1 {
                text(OUTPUT_LABELS[i], p.x + 12.0, p.y + 4.0, 14.0, DIM);
            }
        }
    }
}

fn draw_help(state: &HudState) {
    let lines: &[&str] = if state.show_help {
        &[
            "drag: orbit   wheel: zoom   WASD/QE: pan",
            "click: select creature   F: follow",
            "B: select fittest   N: random   Esc: deselect",
            "Space: pause   +/-: speed   T: turbo",
            "[ ]: plankton rate   G: graph   R: reset",
            "H: hide help",
        ]
    } else {
        &["H: help"]
    };
    let w = 340.0;
    let h = lines.len() as f32 * 20.0 + 14.0;
    let x = screen_width() - w - 12.0;
    let y = screen_height() - h - 12.0;
    panel(x, y, w, h);
    for (i, l) in lines.iter().enumerate() {
        text(l, x + 12.0, y + 24.0 + i as f32 * 20.0, 17.0, TEXT);
    }
}
