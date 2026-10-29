//! 2D overlay: the scoreboard, a panel per player with its training status
//! and import/export buttons, the learning curve, and the live brain.

use macroquad::prelude::*;

use crate::brain::{Brain, INPUT_LABELS, OUTPUT_LABELS};
use crate::trainer::Progress;

pub const TEAM_NAMES: [&str; 2] = ["BLUE", "RED"];
pub const TEAM_COLORS: [Color; 2] = [Color::new(0.4, 0.65, 1.0, 1.0), Color::new(1.0, 0.45, 0.4, 1.0)];

const PANEL: Color = Color::new(0.02, 0.05, 0.1, 0.62);
const TEXT: Color = Color::new(0.93, 0.96, 1.0, 1.0);
const DIM: Color = Color::new(0.66, 0.76, 0.86, 1.0);
const ACCENT: Color = Color::new(1.0, 0.8, 0.2, 1.0);

const PLAYER_PANEL: Vec2 = vec2(390.0, 158.0);

/// The buttons on each player's panel.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Button {
    Import,
    Export,
    Learning,
    Reset,
}

const BUTTONS: [Button; 4] = [Button::Import, Button::Export, Button::Learning, Button::Reset];

/// The match on screen.
pub struct MatchView {
    pub number: u32,
    pub score: [u32; 2],
    pub time: f32,
    pub duration: f32,
    /// Brain generations playing this match.
    pub generations: [u32; 2],
    /// Who just scored, during the pause after a goal.
    pub celebrating: Option<usize>,
    pub over: bool,
}

pub struct HudState {
    pub speed: f32,
    pub paused: bool,
    pub training_paused: bool,
    pub show_help: bool,
    pub show_graph: bool,
    pub show_brain: bool,
    /// The player whose brain is shown and whom the keyboard shortcuts act on.
    pub selected: usize,
    /// Matches on screen won by blue and by red, and drawn.
    pub wins: [u32; 2],
    pub draws: u32,
    /// A status line (e.g. from an export) and how many seconds it stays.
    pub message: Option<(String, f32)>,
}

fn panel(r: Rect) {
    draw_rectangle(r.x, r.y, r.w, r.h, PANEL);
    draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.0, Color::new(1.0, 1.0, 1.0, 0.18));
}

fn text(s: &str, x: f32, y: f32, size: f32, color: Color) {
    draw_text(s, x, y, size, color);
}

fn centered(s: &str, cx: f32, y: f32, size: f32, color: Color) {
    let dims = measure_text(s, None, size as u16, 1.0);
    draw_text(s, cx - dims.width * 0.5, y, size, color);
}

pub fn player_panel(player: usize) -> Rect {
    let x = if player == 0 { 12.0 } else { screen_width() - PLAYER_PANEL.x - 12.0 };
    Rect::new(x, 12.0, PLAYER_PANEL.x, PLAYER_PANEL.y)
}

fn button_rect(player: usize, index: usize) -> Rect {
    let r = player_panel(player);
    let gap = 8.0;
    let w = (r.w - 24.0 - gap * 3.0) / 4.0;
    Rect::new(r.x + 12.0 + index as f32 * (w + gap), r.y + r.h - 40.0, w, 28.0)
}

/// What is under the mouse: a player's panel, and maybe one of its buttons.
pub fn hit(mouse: Vec2) -> Option<(usize, Option<Button>)> {
    for player in 0..2 {
        if player_panel(player).contains(mouse) {
            let button = (0..BUTTONS.len()).find(|&i| button_rect(player, i).contains(mouse)).map(|i| BUTTONS[i]);
            return Some((player, button));
        }
    }
    None
}

pub fn draw(progress: &Progress, view: &MatchView, state: &HudState, brain: &Brain) {
    draw_scoreboard(progress, view, state);
    for player in 0..2 {
        draw_player_panel(progress, state, player);
    }
    if state.show_graph {
        draw_graph(progress);
    }
    if state.show_brain {
        draw_brain_panel(brain, state.selected);
    }
    draw_help(state);
    if let Some((msg, _)) = &state.message {
        let dims = measure_text(msg, None, 18, 1.0);
        let w = dims.width + 28.0;
        let r = Rect::new((screen_width() - w) * 0.5, screen_height() - 50.0, w, 34.0);
        panel(r);
        text(msg, r.x + 14.0, r.y + 23.0, 18.0, TEXT);
    }
}

fn clock(seconds: f32) -> String {
    let s = seconds.max(0.0) as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

fn draw_scoreboard(progress: &Progress, view: &MatchView, state: &HudState) {
    let w = 420.0;
    let r = Rect::new((screen_width() - w) * 0.5, 12.0, w, 108.0);
    panel(r);
    let cx = r.x + r.w * 0.5;
    let size = 44.0;
    let score = format!("{}  :  {}", view.score[0], view.score[1]);
    centered(&score, cx, r.y + 46.0, size, TEXT);
    let half = measure_text(&score, None, size as u16, 1.0).width * 0.5;
    let blue = measure_text(TEAM_NAMES[0], None, 28, 1.0).width;
    text(TEAM_NAMES[0], cx - half - 22.0 - blue, r.y + 42.0, 28.0, TEAM_COLORS[0]);
    text(TEAM_NAMES[1], cx + half + 22.0, r.y + 42.0, 28.0, TEAM_COLORS[1]);

    let status = if view.over { "full time".to_string() } else { format!("{} / {}", clock(view.time), clock(view.duration)) };
    centered(
        &format!("match {}   {}   (gen {} vs gen {})", view.number, status, view.generations[0], view.generations[1]),
        cx,
        r.y + 72.0,
        17.0,
        DIM,
    );
    let training = if state.training_paused {
        "training paused".to_string()
    } else {
        format!("training in the background: {:.0} matches/s", progress.matches_per_sec)
    };
    centered(&training, cx, r.y + 94.0, 16.0, DIM);

    if let Some(team) = view.celebrating {
        let msg = format!("GOAL {}!", TEAM_NAMES[team]);
        let y = screen_height() * 0.3;
        centered(&msg, screen_width() * 0.5 + 3.0, y + 3.0, 90.0, Color::new(0.0, 0.0, 0.0, 0.5));
        centered(&msg, screen_width() * 0.5, y, 90.0, TEAM_COLORS[team]);
    }
}

fn draw_player_panel(progress: &Progress, state: &HudState, player: usize) {
    let r = player_panel(player);
    let p = &progress.players[player];
    let color = TEAM_COLORS[player];
    panel(r);
    if state.selected == player {
        draw_rectangle_lines(r.x, r.y, r.w, r.h, 2.0, color);
    }
    text(TEAM_NAMES[player], r.x + 12.0, r.y + 28.0, 28.0, color);
    let status = if !p.learning {
        "frozen"
    } else if state.training_paused {
        "training paused"
    } else {
        "learning"
    };
    text(&format!("brain generation {}   {status}", p.generation), r.x + 90.0, r.y + 26.0, 18.0, TEXT);
    if p.resumed_from_file {
        text(&format!("(resumed from {})", crate::trainer::SAVE_FILES[player]), r.x + 90.0, r.y + 44.0, 14.0, DIM);
    }
    if let Some(stats) = progress.history.last() {
        let s = stats[player];
        text(
            &format!("training matches: {:.2} goals for, {:.2} against", s.goals_for, s.goals_against),
            r.x + 12.0,
            r.y + 70.0,
            17.0,
            DIM,
        );
    }
    text(
        &format!("matches on screen: {} won, {} drawn, {} lost", state.wins[player], state.draws, state.wins[1 - player]),
        r.x + 12.0,
        r.y + 92.0,
        17.0,
        DIM,
    );

    let mouse = Vec2::from(mouse_position());
    for (i, &b) in BUTTONS.iter().enumerate() {
        let br = button_rect(player, i);
        let hover = br.contains(mouse);
        let fill = if hover { Color::new(color.r, color.g, color.b, 0.45) } else { Color::new(1.0, 1.0, 1.0, 0.1) };
        draw_rectangle(br.x, br.y, br.w, br.h, fill);
        draw_rectangle_lines(br.x, br.y, br.w, br.h, 1.0, Color::new(color.r, color.g, color.b, 0.8));
        let label = match b {
            Button::Import => "Import",
            Button::Export => "Export",
            Button::Learning if p.learning => "Freeze",
            Button::Learning => "Train",
            Button::Reset => "Reset",
        };
        centered(label, br.x + br.w * 0.5, br.y + 19.0, 18.0, TEXT);
    }
}

fn draw_graph(progress: &Progress) {
    let (w, h) = (440.0, 170.0);
    let x = 12.0;
    let y = screen_height() - h - 12.0;
    panel(Rect::new(x, y, w, h));
    text("goals per training match (vs the other's latest brain)", x + 10.0, y + 18.0, 16.0, ACCENT);
    let hist = &progress.history;
    if hist.len() < 2 {
        return;
    }
    // Smoothed, since single generations are noisy.
    let mut smooth = [0.0f32; 2];
    let lines: Vec<[f32; 2]> = hist
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let a = if i == 0 { 1.0 } else { 0.08 };
            for p in 0..2 {
                smooth[p] += (s[p].goals_for - smooth[p]) * a;
            }
            smooth
        })
        .collect();

    let (gx, gy, gw, gh) = (x + 36.0, y + 30.0, w - 48.0, h - 50.0);
    let max = lines.iter().flatten().fold(1.0f32, |m, &v| m.max(v)).ceil();
    let n = lines.len();
    let px = |i: usize| gx + gw * i as f32 / (n - 1) as f32;
    let py = |v: f32| gy + gh * (1.0 - (v / max).clamp(0.0, 1.0));
    for k in 0..=4 {
        let v = max * k as f32 / 4.0;
        let ly = py(v);
        draw_line(gx, ly, gx + gw, ly, 1.0, Color::new(1.0, 1.0, 1.0, 0.08));
        text(&format!("{v:.1}"), x + 6.0, ly + 4.0, 14.0, DIM);
    }
    // Draw at most ~400 segments, however long training has run.
    let step = (n / 400).max(1);
    let mut prev = 0;
    for i in (step..n).step_by(step).chain(std::iter::once(n - 1)) {
        for p in 0..2 {
            draw_line(px(prev), py(lines[prev][p]), px(i), py(lines[i][p]), 2.0, TEAM_COLORS[p]);
        }
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

fn draw_brain_panel(brain: &Brain, player: usize) {
    let w = 430.0;
    let top = player_panel(player).bottom() + 10.0;
    let x = if player == 0 { 12.0 } else { screen_width() - w - 12.0 };
    let h = (screen_height() - top - 210.0).clamp(300.0, 540.0);
    let r = Rect::new(x, top, w, h);
    panel(r);
    text(&format!("{} BRAIN  (live)", TEAM_NAMES[player]), r.x + 14.0, r.y + 26.0, 20.0, TEAM_COLORS[player]);
    text("everything it senses is relative to itself", r.x + 14.0, r.y + 44.0, 14.0, DIM);
    let area = Rect::new(r.x + 100.0, r.y + 62.0, r.w - 170.0, r.h - 76.0);

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
            "Space: pause view   +/- or Up/Down: speed",
            "N: new match with the latest brains",
            "Tab or click a panel: select player",
            "I / E: import / export selected brain",
            "F: freeze/train selected   R: reset it",
            "P: pause all training",
            "B: brain   G: graph   H: hide help",
        ]
    } else {
        &["H: help"]
    };
    let w = 340.0;
    let h = lines.len() as f32 * 20.0 + 14.0;
    let x = screen_width() - w - 12.0;
    let y = screen_height() - h - 12.0;
    panel(Rect::new(x, y, w, h));
    for (i, l) in lines.iter().enumerate() {
        text(l, x + 12.0, y + 24.0 + i as f32 * 20.0, 17.0, TEXT);
    }
}

/// Draws a text label at a 3D world position (e.g. player names).
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
    let size = (900.0 / clip.w).clamp(14.0, 26.0);
    let dims = measure_text(s, None, size as u16, 1.0);
    draw_text(s, sx - dims.width * 0.5, sy, size, color);
}
