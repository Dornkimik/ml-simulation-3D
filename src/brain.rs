//! A small feed-forward neural network that controls one soccer player.
//! Everything it senses is measured relative to the player itself (ahead of
//! it, to its left), so the same brain can play for either team.
//!
//! The weights are found by evolution strategies (see trainer.rs), not by
//! backpropagation. Brains can be saved to and loaded from text files.

use std::path::Path;

use macroquad::rand::gen_range;

/// Names of the sensor inputs, shown in the brain viewer.
pub const INPUT_LABELS: [&str; 18] = [
    "ball ahead", "ball left", "ball dist", "ball v ahead", "ball v left", "ball height", "opp ahead",
    "opp left", "goal ahead", "goal left", "own goal ahd", "own goal left", "speed ahead", "speed left",
    "field x", "field z", "ball close", "bias",
];
pub const OUTPUT_LABELS: [&str; 5] = ["run", "strafe", "turn", "kick", "lift"];

pub const INPUTS: usize = INPUT_LABELS.len();
pub const HIDDEN: usize = 24;
pub const OUTPUTS: usize = OUTPUT_LABELS.len();

const FILE_HEADER: &str = "soccer-brain v1";

/// A fully connected layer. Weights are stored row-major, one row per
/// output neuron, with the bias as the last element of each row.
#[derive(Clone)]
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Vec<f32>,
}

impl Layer {
    fn random(inputs: usize, outputs: usize, gain: f32) -> Self {
        // Scaled init keeps tanh units out of saturation at the start.
        let scale = gain / (inputs as f32).sqrt();
        let weights = (0..outputs * (inputs + 1))
            .map(|_| gen_range(-1.0, 1.0) * scale)
            .collect();
        Self { inputs, outputs, weights }
    }

    pub fn weight(&self, out: usize, inp: usize) -> f32 {
        self.weights[out * (self.inputs + 1) + inp]
    }

    fn forward(&self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        let stride = self.inputs + 1;
        for o in 0..self.outputs {
            let row = &self.weights[o * stride..(o + 1) * stride];
            let mut sum = row[self.inputs];
            for (w, x) in row.iter().zip(input) {
                sum += w * x;
            }
            output.push(sum.tanh());
        }
    }
}

#[derive(Clone)]
pub struct Brain {
    pub layers: Vec<Layer>,
    /// Activations of every layer from the last `think` call, input layer
    /// first. Kept so the HUD can visualize what the player is "thinking".
    pub activations: Vec<Vec<f32>>,
}

impl Brain {
    pub fn random() -> Self {
        let layers = vec![Layer::random(INPUTS, HIDDEN, 1.5), Layer::random(HIDDEN, OUTPUTS, 1.0)];
        Self::from_layers(layers)
    }

    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.layers.iter().flat_map(|l| l.weights.iter().copied())
    }

    /// Rebuilds a brain from `weights()` output, or None if the size doesn't
    /// match the current network shape or a weight isn't a finite number.
    pub fn from_weights(weights: &[f32]) -> Option<Self> {
        let mut layers = vec![Layer::random(INPUTS, HIDDEN, 1.0), Layer::random(HIDDEN, OUTPUTS, 1.0)];
        let total: usize = layers.iter().map(|l| l.weights.len()).sum();
        if weights.len() != total || !weights.iter().all(|w| w.is_finite()) {
            return None;
        }
        let mut rest = weights;
        for l in &mut layers {
            let (head, tail) = rest.split_at(l.weights.len());
            l.weights.copy_from_slice(head);
            rest = tail;
        }
        Some(Self::from_layers(layers))
    }

    fn from_layers(layers: Vec<Layer>) -> Self {
        let mut activations = vec![vec![0.0; layers[0].inputs]];
        activations.extend(layers.iter().map(|l| vec![0.0; l.outputs]));
        Self { layers, activations }
    }

    pub fn think(&mut self, input: &[f32; INPUTS]) -> [f32; OUTPUTS] {
        self.activations[0].clear();
        self.activations[0].extend_from_slice(input);
        for i in 0..self.layers.len() {
            let (done, rest) = self.activations.split_at_mut(i + 1);
            self.layers[i].forward(&done[i], &mut rest[0]);
        }
        let mut out = [0.0; OUTPUTS];
        out.copy_from_slice(self.activations.last().unwrap());
        out
    }

    /// The brain as a text file: a header, the number of generations it has
    /// trained for, the network shape, and then all weights.
    pub fn to_text(&self, generation: u32) -> String {
        let weights: Vec<String> = self.weights().map(|w| w.to_string()).collect();
        format!("{FILE_HEADER}\n{generation}\n{INPUTS} {HIDDEN} {OUTPUTS}\n{}\n", weights.join(" "))
    }

    /// Parses `to_text` output into a brain and its generation.
    pub fn from_text(text: &str) -> Result<(Self, u32), String> {
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some(FILE_HEADER) {
            return Err("not a soccer brain file".into());
        }
        let generation = lines
            .next()
            .and_then(|l| l.trim().parse().ok())
            .ok_or("the generation number is missing")?;
        let shape = lines.next().unwrap_or("").trim().to_string();
        let expected = format!("{INPUTS} {HIDDEN} {OUTPUTS}");
        if shape != expected {
            return Err(format!("the network shape is {shape}, but this version needs {expected}"));
        }
        let weights: Vec<f32> = lines
            .next()
            .unwrap_or("")
            .split_whitespace()
            .map(|w| w.parse())
            .collect::<Result<_, _>>()
            .map_err(|_| "a weight is not a number")?;
        let brain = Self::from_weights(&weights).ok_or("the weights are damaged or incomplete")?;
        Ok((brain, generation))
    }

    pub fn save(&self, path: &Path, generation: u32) -> std::io::Result<()> {
        std::fs::write(path, self.to_text(generation))
    }

    pub fn load(path: &Path) -> Result<(Self, u32), String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_text(&text)
    }
}

/// Standard normal sample via Box-Muller.
pub fn gaussian() -> f32 {
    let u1: f32 = gen_range(1e-6, 1.0);
    let u2: f32 = gen_range(0.0, 1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_round_trip() {
        let brain = Brain::random();
        let (loaded, generation) = Brain::from_text(&brain.to_text(42)).unwrap();
        assert_eq!(generation, 42);
        assert!(brain.weights().eq(loaded.weights()));
        assert!(Brain::from_text("walker-brain v1\n3\n1 2 3\n").is_err());
    }
}
