//! A small feed-forward neural network used as each creature's brain.
//!
//! Brains are never trained with gradients. They improve through
//! neuroevolution: creatures that eat and survive long enough reproduce,
//! passing on a mutated copy of their weights.

use macroquad::rand::gen_range;

/// Names of the sensor inputs, shown in the brain viewer.
pub const INPUT_LABELS: [&str; 19] = [
    "food x", "food y", "food z", "threat x", "threat y", "threat z", "kin x", "kin y", "kin z",
    "vel x", "vel y", "vel z", "pos x", "pos y", "pos z", "energy", "clock", "memory", "bias",
];
pub const OUTPUT_LABELS: [&str; 5] = ["thrust x", "thrust y", "thrust z", "throttle", "memory"];

pub const INPUTS: usize = INPUT_LABELS.len();
pub const HIDDEN: usize = 14;
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
    fn random(inputs: usize, outputs: usize) -> Self {
        // Scaled init keeps tanh units out of saturation at the start.
        let scale = 1.0 / (inputs as f32).sqrt() * 1.5;
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
    pub fn random() -> Self {
        let layers = vec![Layer::random(INPUTS, HIDDEN), Layer::random(HIDDEN, OUTPUTS)];
        Self::from_layers(layers)
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

    /// Returns a copy with gaussian noise added to a fraction of the weights,
    /// plus the occasional weight replaced outright.
    pub fn mutated(&self, rate: f32, strength: f32) -> Self {
        let mut child = self.clone();
        for layer in &mut child.layers {
            for w in &mut layer.weights {
                let roll = gen_range(0.0, 1.0);
                if roll < rate * 0.1 {
                    *w = gen_range(-1.5, 1.5);
                } else if roll < rate {
                    *w = (*w + gaussian() * strength).clamp(-4.0, 4.0);
                }
            }
        }
        child
    }

    /// Uniform crossover per neuron, so each neuron's incoming weights stay
    /// together as a functional unit.
    pub fn crossover(a: &Brain, b: &Brain) -> Self {
        let layers = a
            .layers
            .iter()
            .zip(&b.layers)
            .map(|(la, lb)| {
                let stride = la.inputs + 1;
                let mut weights = la.weights.clone();
                for o in 0..la.outputs {
                    if gen_range(0, 2) == 1 {
                        let r = o * stride..(o + 1) * stride;
                        weights[r.clone()].copy_from_slice(&lb.weights[r]);
                    }
                }
                Layer { inputs: la.inputs, outputs: la.outputs, weights }
            })
            .collect();
        Self::from_layers(layers)
    }
}

/// Standard normal sample via Box-Muller.
pub fn gaussian() -> f32 {
    let u1: f32 = gen_range(1e-6, 1.0);
    let u2: f32 = gen_range(0.0, 1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
