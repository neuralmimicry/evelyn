//! Evelyn: a transformer language system whose feed-forward ("perceptron")
//! knowledge layers run as spiking neuron/synapse populations in AARNN.
//!
//! Stage 0 (this crate today) is the verified conversion core, pure Rust with
//! no dependencies:
//! * [`dense`] - reference ANN blocks (dense layers, ReLU MLPs, SwiGLU MLPs)
//! * [`snn`] - integrate-and-fire populations, data-based threshold balancing,
//!   rate-coded ANN->SNN conversion and coincidence-detection gating
//!
//! * [`activation`] - model-agnostic activation mapping: any activation is
//!   fitted as a heterogeneous-threshold neuron population (stage 1)
//!
//! See `docs/ARCHITECTURE.md` for the staged plan and verification gates.

pub mod activation;
pub mod curve;
pub mod dense;
pub mod gguf;
pub mod import;
pub mod mesh;
pub mod snn;

/// Root-mean-square error between two equal-length vectors.
pub fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let s: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (s / a.len() as f32).sqrt()
}

/// RMSE normalised by the reference vector's RMS (scale-free error).
pub fn relative_error(reference: &[f32], approx: &[f32]) -> f32 {
    let rms = (reference.iter().map(|x| x * x).sum::<f32>() / reference.len() as f32).sqrt();
    if rms == 0.0 {
        rmse(reference, approx)
    } else {
        rmse(reference, approx) / rms
    }
}

/// Small deterministic PRNG (xorshift64*), so tests and conversions are
/// reproducible without external crates.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Approximately standard normal (sum of 4 uniforms, variance-corrected).
    pub fn normal(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.uniform()).sum();
        (s - 2.0) * (3.0f32).sqrt()
    }
}
