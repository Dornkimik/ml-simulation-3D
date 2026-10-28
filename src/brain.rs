//! A small feed-forward neural network that drives one side of the walker's
//! body. The same network runs for the left and the right side (with mirrored
//! inputs and half a gait cycle apart), so gaits come out symmetric.
//!
//! The weights are found by evolution strategies (see trainer.rs), not by
//! backpropagation.

use macroquad::rand::gen_range;

/// Names of the sensor inputs, shown in the brain viewer.
pub const INPUT_LABELS: [&str; 18] = [
    "phase sin", "phase cos", "lean fwd", "lean side", "lean fwd v", "lean side v", "height",
    "speed fwd", "speed side", "hip", "hip out", "knee", "ankle", "other hip", "other knee",
    "foot down", "other foot", "bias",
];
pub const OUTPUT_LABELS: [&str; 5] = ["hip", "hip out", "knee", "ankle", "shoulder"];

pub const INPUTS: usize = INPUT_LABELS.len();
pub const HIDDEN: usize = 16;
pub const OUTPUTS: usize = OUTPUT_LABELS.len();

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
    /// first. Kept so the HUD can visualize what the creature is "thinking".
    pub activations: Vec<Vec<f32>>,
}

impl Brain {
    /// A random brain with small output weights, so it starts out holding
    /// roughly the default standing pose instead of flailing.
    pub fn random() -> Self {
        let layers = vec![Layer::random(INPUTS, HIDDEN, 1.5), Layer::random(HIDDEN, OUTPUTS, 0.4)];
        Self::from_layers(layers)
    }

    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.layers.iter().flat_map(|l| l.weights.iter().copied())
    }

    /// Rebuilds a brain from `weights()` output, or None if the size doesn't
    /// match the current network shape.
    pub fn from_weights(weights: &[f32]) -> Option<Self> {
        let mut layers = vec![Layer::random(INPUTS, HIDDEN, 1.0), Layer::random(HIDDEN, OUTPUTS, 1.0)];
        let total: usize = layers.iter().map(|l| l.weights.len()).sum();
        if weights.len() != total {
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
}

/// Standard normal sample via Box-Muller.
pub fn gaussian() -> f32 {
    let u1: f32 = gen_range(1e-6, 1.0);
    let u2: f32 = gen_range(0.0, 1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
