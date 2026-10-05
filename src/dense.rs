//! Reference (weights-based) blocks that Evelyn converts. These define the
//! ground truth every spiking conversion is verified against.

use crate::Rng;

/// Fully connected layer `y = W x + b`, `W` row-major `[out][in]`.
#[derive(Clone, Debug)]
pub struct Dense {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Dense {
    pub fn new(inputs: usize, outputs: usize, weights: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            inputs * outputs,
            "weights must be outputs x inputs"
        );
        assert_eq!(bias.len(), outputs);
        Self {
            inputs,
            outputs,
            weights,
            bias,
        }
    }

    /// He-style random init, for tests and synthetic verification.
    pub fn random(inputs: usize, outputs: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / inputs as f32).sqrt();
        let weights = (0..inputs * outputs)
            .map(|_| rng.normal() * scale)
            .collect();
        let bias = (0..outputs).map(|_| rng.normal() * 0.05).collect();
        Self::new(inputs, outputs, weights, bias)
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.inputs);
        (0..self.outputs)
            .map(|o| {
                let row = &self.weights[o * self.inputs..(o + 1) * self.inputs];
                row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>() + self.bias[o]
            })
            .collect()
    }
}

pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

/// `Dense -> ReLU -> ... -> Dense` (no activation after the last layer).
#[derive(Clone, Debug)]
pub struct ReluMlp {
    pub layers: Vec<Dense>,
}

impl ReluMlp {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i + 1 < self.layers.len() {
                h = relu(&h);
            }
        }
        h
    }

    /// Activations after each hidden ReLU (used for threshold balancing).
    pub fn hidden_activations(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let mut h = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i + 1 < self.layers.len() {
                h = relu(&h);
                out.push(h.clone());
            }
        }
        out
    }
}

/// The feed-forward block of modern open-weights LLMs (Llama/Qwen/Gemma
/// style): `down( act(gate x) * (up x) )`.
#[derive(Clone, Debug)]
pub struct SwiGluMlp {
    pub gate: Dense,
    pub up: Dense,
    pub down: Dense,
}

impl SwiGluMlp {
    pub fn random(model: usize, hidden: usize, rng: &mut Rng) -> Self {
        Self {
            gate: Dense::random(model, hidden, rng),
            up: Dense::random(model, hidden, rng),
            down: Dense::random(hidden, model, rng),
        }
    }

    /// Exact SwiGLU (SiLU gate), the model's true behaviour.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let g = silu(&self.gate.forward(x));
        let u = self.up.forward(x);
        let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a * b).collect();
        self.down.forward(&h)
    }

    /// ReLU-gated variant (`ReGLU`): what a rate-coded spiking gate computes
    /// natively. The gap to [`Self::forward`] is the "ReLU-fication" error that
    /// stage 1 must close (fine-tuning or a SiLU-shaped neuron response).
    pub fn forward_relu_gate(&self, x: &[f32]) -> Vec<f32> {
        let g = relu(&self.gate.forward(x));
        let u = self.up.forward(x);
        let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a * b).collect();
        self.down.forward(&h)
    }
}
