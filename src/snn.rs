//! Rate-coded conversion of feed-forward blocks into spiking populations.
//!
//! * Hidden ReLU units become integrate-and-fire (IF) neurons with
//!   reset-by-subtraction. Thresholds are set by data-based normalisation: the
//!   99.9th-percentile activation on a calibration set (Diehl 2015,
//!   Rueckauer 2017). A neuron's firing rate times its threshold converges to
//!   its ReLU activation, clipped at the threshold.
//! * The output layer is a non-spiking integrator (membrane readout), so
//!   signed outputs need no sign coding.
//! * Gated blocks (ReGLU/SwiGLU) use coincidence detection. Stochastic gate
//!   and up neurons fire independently, so a coincidence unit's rate is the
//!   product of their rates. Signed up-projections are split into
//!   excitatory/inhibitory (positive/negative) channels.
//!
//! Every conversion is verified against [`crate::dense`] by the tests.

use crate::Rng;
use crate::dense::{Dense, ReluMlp, SwiGluMlp};

/// Percentile of the absolute values of a batch of activation vectors.
fn percentile_abs(samples: &[Vec<f32>], index: usize, p: f32) -> f32 {
    let mut v: Vec<f32> = samples.iter().map(|s| s[index].abs()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((v.len() as f32 - 1.0) * p).round() as usize;
    v[k].max(1e-6)
}

/// Per-neuron thresholds from calibration activations.
fn thresholds(samples: &[Vec<f32>], p: f32) -> Vec<f32> {
    (0..samples[0].len())
        .map(|i| percentile_abs(samples, i, p))
        .collect()
}

/// Spiking equivalent of a [`ReluMlp`].
#[derive(Clone, Debug)]
pub struct SpikingReluMlp {
    layers: Vec<Dense>,
    /// `theta[l][i]` is the threshold of hidden neuron `i` in hidden layer `l`.
    theta: Vec<Vec<f32>>,
}

impl SpikingReluMlp {
    /// Convert, balancing thresholds on `calibration` inputs.
    pub fn convert(mlp: &ReluMlp, calibration: &[Vec<f32>], percentile: f32) -> Self {
        assert!(mlp.layers.len() >= 2, "need at least one hidden layer");
        let acts: Vec<Vec<Vec<f32>>> = calibration
            .iter()
            .map(|x| mlp.hidden_activations(x))
            .collect();
        let theta = (0..mlp.layers.len() - 1)
            .map(|l| {
                let per_sample: Vec<Vec<f32>> = acts.iter().map(|a| a[l].clone()).collect();
                thresholds(&per_sample, percentile)
            })
            .collect();
        Self {
            layers: mlp.layers.clone(),
            theta,
        }
    }

    /// Simulate `steps` timesteps with the input as a constant analog current.
    /// Returns the readout and the total hidden spike count (an energy proxy).
    pub fn run(&self, x: &[f32], steps: usize) -> (Vec<f32>, u64) {
        let hidden = self.layers.len() - 1;
        let mut v: Vec<Vec<f32>> = self
            .theta
            .iter()
            .map(|t| t.iter().map(|th| th * 0.5).collect())
            .collect();
        let last = &self.layers[hidden];
        let mut acc = vec![0.0f32; last.outputs];
        let mut spikes_total = 0u64;
        for _ in 0..steps {
            // `drive` is the per-step input to the next layer: analog x for the
            // first layer, then spike events weighted by their threshold.
            let mut drive = x.to_vec();
            for l in 0..hidden {
                let current = self.layers[l].forward(&drive);
                let mut out = vec![0.0f32; current.len()];
                for i in 0..current.len() {
                    v[l][i] += current[i];
                    if v[l][i] >= self.theta[l][i] {
                        v[l][i] -= self.theta[l][i];
                        out[i] = self.theta[l][i];
                        spikes_total += 1;
                    }
                }
                drive = out;
            }
            for (a, c) in acc.iter_mut().zip(last.forward(&drive)) {
                *a += c;
            }
        }
        (
            acc.into_iter().map(|a| a / steps as f32).collect(),
            spikes_total,
        )
    }
}

/// Spiking equivalent of a gated feed-forward block via coincidence detection.
#[derive(Clone, Debug)]
pub struct SpikingGatedMlp {
    gate: Dense,
    up: Dense,
    down: Dense,
    theta_gate: Vec<f32>,
    theta_up: Vec<f32>,
}

impl SpikingGatedMlp {
    pub fn convert(mlp: &SwiGluMlp, calibration: &[Vec<f32>], percentile: f32) -> Self {
        let g: Vec<Vec<f32>> = calibration
            .iter()
            .map(|x| crate::dense::relu(&mlp.gate.forward(x)))
            .collect();
        let u: Vec<Vec<f32>> = calibration.iter().map(|x| mlp.up.forward(x)).collect();
        Self {
            gate: mlp.gate.clone(),
            up: mlp.up.clone(),
            down: mlp.down.clone(),
            theta_gate: thresholds(&g, percentile),
            theta_up: thresholds(&u, percentile),
        }
    }

    /// Simulate `steps` timesteps. The gate and up neurons fire stochastically
    /// at rate `clip(a / theta, 0, 1)`; each coincidence of gate and up spikes
    /// injects `+/- theta_gate * theta_up` into the down-projection integrator.
    pub fn run(&self, x: &[f32], steps: usize, rng: &mut Rng) -> (Vec<f32>, u64) {
        let pg: Vec<f32> = self
            .gate
            .forward(x)
            .iter()
            .zip(&self.theta_gate)
            .map(|(a, t)| (a / t).clamp(0.0, 1.0))
            .collect();
        let pu: Vec<f32> = self
            .up
            .forward(x)
            .iter()
            .zip(&self.theta_up)
            .map(|(a, t)| (a / t).clamp(-1.0, 1.0))
            .collect();
        let hidden = pg.len();
        let mut counts = vec![0.0f32; hidden];
        let mut spikes = 0u64;
        for _ in 0..steps {
            for i in 0..hidden {
                let g = rng.uniform() < pg[i];
                let u = rng.uniform() < pu[i].abs();
                spikes += g as u64 + u as u64;
                if g && u {
                    counts[i] += pu[i].signum();
                }
            }
        }
        let h: Vec<f32> = (0..hidden)
            .map(|i| counts[i] / steps as f32 * self.theta_gate[i] * self.theta_up[i])
            .collect();
        (self.down.forward(&h), spikes)
    }
}
