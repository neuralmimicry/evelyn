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

/// Gated block whose gate uses any activation (stage 1), encoded as a fitted
/// heterogeneous-threshold population per hidden unit. Since
/// `act(z) * u = tonic * u + sum_k w_k * unit_k(z) * u`, each population unit
/// gets its own coincidence detector with the up neuron. The model's true
/// SiLU/GELU gate is reproduced without ReLU-fication or fine-tuning.
#[derive(Clone, Debug)]
pub struct SpikingPopulationGatedMlp {
    gate: Dense,
    up: Dense,
    down: Dense,
    code: crate::activation::PopulationCode,
    /// Per-unit saturation threshold (max drive over the calibrated range).
    theta_unit: Vec<f32>,
    theta_up: Vec<f32>,
}

impl SpikingPopulationGatedMlp {
    pub fn convert(
        mlp: &SwiGluMlp,
        act: crate::activation::Activation,
        calibration: &[Vec<f32>],
        percentile: f32,
        knots: usize,
    ) -> Self {
        Self::convert_with_headroom(mlp, act, calibration, percentile, knots, 1.0)
    }

    /// As [`Self::convert`], with every dynamic range (gate input range, up
    /// thresholds) widened by `headroom` (>= 1). Real LLM activations are
    /// heavy-tailed, so ranges calibrated on a finite sample clip outliers.
    /// Headroom trades that saturation error for lower firing rates, meaning
    /// more spiking sampling noise at a given simulation length.
    pub fn convert_with_headroom(
        mlp: &SwiGluMlp,
        act: crate::activation::Activation,
        calibration: &[Vec<f32>],
        percentile: f32,
        knots: usize,
        headroom: f32,
    ) -> Self {
        assert!(headroom >= 1.0, "headroom must be >= 1");
        // Calibration forwards are independent, so they run in parallel across
        // all available cores.
        let (gate_z, up_u) = parallel_forwards(mlp, calibration);
        let z: Vec<f32> = gate_z.into_iter().flatten().collect();
        let mut sorted = z.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f32| sorted[((sorted.len() as f32 - 1.0) * p).round() as usize];
        let (lo, hi) = (q(1.0 - percentile) * headroom, q(percentile) * headroom);
        let code = crate::activation::PopulationCode::fit(act, lo, hi, knots);
        let theta_unit = code
            .units
            .iter()
            .map(|u| u.drive(lo).max(u.drive(hi)).max(1e-6))
            .collect();
        let u = up_u;
        Self {
            gate: mlp.gate.clone(),
            up: mlp.up.clone(),
            down: mlp.down.clone(),
            code,
            theta_unit,
            theta_up: thresholds(&u, percentile)
                .into_iter()
                .map(|t| t * headroom)
                .collect(),
        }
    }

    /// Deterministic, dithered spike timing. Each stream fires on a
    /// low-discrepancy (Weyl) phase sequence instead of independent random
    /// draws. Up streams advance by the golden-ratio conjugate and population
    /// units by sqrt(2) - 1, which are rationally independent, so each
    /// (up, unit) coincidence pair is equidistributed. The product-rate error
    /// therefore decays roughly as 1/T rather than 1/sqrt(T): the same
    /// accuracy in far fewer timesteps, which is what keeps latency low. This
    /// models regular-spiking neurons; per-stream phase offsets keep different
    /// neurons desynchronised.
    pub fn run_dithered(&self, x: &[f32], steps: usize) -> (Vec<f32>, u64) {
        const BETA_UP: f64 = 0.618_033_988_749_894_9; // golden-ratio conjugate
        const BETA_UNIT: f64 = 0.414_213_562_373_095_1; // sqrt(2) - 1
        let z = self.gate.forward(x);
        let pu: Vec<f64> = self
            .up
            .forward(x)
            .iter()
            .zip(&self.theta_up)
            .map(|(a, t)| (a / t).clamp(-1.0, 1.0) as f64)
            .collect();
        let units = &self.code.units;
        let hidden = z.len();
        let mut spikes = 0u64;
        let h: Vec<f32> = (0..hidden)
            .map(|i| {
                let pk: Vec<f64> = units
                    .iter()
                    .zip(&self.theta_unit)
                    .map(|(u, t)| (u.drive(z[i]) / t).clamp(0.0, 1.0) as f64)
                    .collect();
                // Per-stream phase offsets (themselves a Weyl sequence over
                // neuron and unit indices) desynchronise different neurons.
                let off_up = (i as f64 * 0.754_877_666_246_692_7).fract();
                let mut acc = 0.0f64;
                let sign = pu[i].signum();
                for t in 0..steps {
                    let tf = t as f64;
                    if (off_up + tf * BETA_UP).fract() >= pu[i].abs() {
                        continue;
                    }
                    spikes += 1;
                    acc += sign * self.code.tonic as f64;
                    for (k, u) in units.iter().enumerate() {
                        let off = ((i * units.len() + k) as f64 * 0.569_840_290_998_053_3).fract();
                        if (off + tf * BETA_UNIT).fract() < pk[k] {
                            spikes += 1;
                            acc += sign * (u.weight * self.theta_unit[k]) as f64;
                        }
                    }
                }
                (acc / steps as f64) as f32 * self.theta_up[i]
            })
            .collect();
        (self.down.forward(&h), spikes)
    }

    /// Infinite-time (rate-limit) output: isolates population-code fitting and
    /// saturation error from spiking sampling noise.
    pub fn run_analog(&self, x: &[f32]) -> Vec<f32> {
        let z = self.gate.forward(x);
        let u = self.up.forward(x);
        let h: Vec<f32> = z
            .iter()
            .zip(u.iter().zip(&self.theta_up))
            .map(|(zi, (ui, tu))| {
                let g: f32 = self.code.tonic
                    + self
                        .code
                        .units
                        .iter()
                        .zip(&self.theta_unit)
                        .map(|(k, t)| k.weight * k.drive(*zi).min(*t))
                        .sum::<f32>();
                // The up neuron saturates at its threshold, exactly as when spiking.
                g * ui.clamp(-tu, *tu)
            })
            .collect();
        self.down.forward(&h)
    }

    /// Error decomposition for diagnosis: outputs with (a) the population fit
    /// alone (no saturation), (b) plus gate-unit saturation, (c) plus
    /// up-neuron saturation. (c) equals [`Self::run_analog`].
    pub fn run_analog_decomposed(&self, x: &[f32]) -> [Vec<f32>; 3] {
        let z = self.gate.forward(x);
        let u = self.up.forward(x);
        let fit = |zi: f32, clip: bool| -> f32 {
            self.code.tonic
                + self
                    .code
                    .units
                    .iter()
                    .zip(&self.theta_unit)
                    .map(|(k, t)| {
                        k.weight
                            * if clip {
                                k.drive(zi).min(*t)
                            } else {
                                k.drive(zi)
                            }
                    })
                    .sum::<f32>()
        };
        let mk = |clip_gate: bool, clip_up: bool| -> Vec<f32> {
            let h: Vec<f32> = (0..z.len())
                .map(|i| {
                    let up = if clip_up {
                        u[i].clamp(-self.theta_up[i], self.theta_up[i])
                    } else {
                        u[i]
                    };
                    fit(z[i], clip_gate) * up
                })
                .collect();
            self.down.forward(&h)
        };
        [mk(false, false), mk(true, false), mk(true, true)]
    }

    pub fn population_size(&self) -> usize {
        self.code.units.len() + 1
    }

    pub fn run(&self, x: &[f32], steps: usize, rng: &mut Rng) -> (Vec<f32>, u64) {
        let z = self.gate.forward(x);
        let pu: Vec<f32> = self
            .up
            .forward(x)
            .iter()
            .zip(&self.theta_up)
            .map(|(a, t)| (a / t).clamp(-1.0, 1.0))
            .collect();
        let hidden = z.len();
        let units = &self.code.units;
        // Firing probability of every population unit for every hidden neuron.
        let pk: Vec<Vec<f32>> = z
            .iter()
            .map(|zi| {
                units
                    .iter()
                    .zip(&self.theta_unit)
                    .map(|(u, t)| (u.drive(*zi) / t).clamp(0.0, 1.0))
                    .collect()
            })
            .collect();
        let mut acc = vec![0.0f32; hidden];
        let mut spikes = 0u64;
        for _ in 0..steps {
            for i in 0..hidden {
                if rng.uniform() >= pu[i].abs() {
                    continue; // no up spike: no coincidence possible this step
                }
                spikes += 1;
                let sign = pu[i].signum();
                acc[i] += sign * self.code.tonic; // tonic unit always coincides
                for (k, u) in units.iter().enumerate() {
                    if rng.uniform() < pk[i][k] {
                        spikes += 1;
                        acc[i] += sign * u.weight * self.theta_unit[k];
                    }
                }
            }
        }
        let h: Vec<f32> = (0..hidden)
            .map(|i| acc[i] / steps as f32 * self.theta_up[i])
            .collect();
        (self.down.forward(&h), spikes)
    }
}

/// Gate and up projections of every calibration sample, computed in parallel
/// with scoped threads (one chunk per available core).
fn parallel_forwards(mlp: &SwiGluMlp, calibration: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = calibration.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = calibration
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|x| (mlp.gate.forward(x), mlp.up.forward(x)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("calibration worker panicked"))
            .unzip()
    })
}
