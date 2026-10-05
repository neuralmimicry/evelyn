//! Model-agnostic activation mapping (stage 1).
//!
//! Any scalar activation `f` is encoded as a heterogeneous-threshold neuron
//! population, with no per-activation code:
//!
//! ```text
//! f(z) ~= c + sum_k  a_k * relu(z - b_k)   (rising units, threshold b_k)
//!           + sum_k  d_k * relu(b_k - z)   (falling units, mirrored)
//! ```
//!
//! Each unit is a rate-coded neuron whose rate tracks its rectified input. The
//! coefficients are signed synaptic weights: excitatory if positive,
//! inhibitory if negative. `c` is a tonically active unit. The coefficients
//! come from a ridge-regularised least-squares fit to `f` sampled over the
//! calibrated input range, so ReLU, GELU, SiLU, tanh, sigmoid or any custom
//! function convert the same way. The basis is linear beyond the end knots, so
//! extrapolation stays bounded and monotone.

/// Activation functions found in open-weights models, plus arbitrary ones.
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Silu,
    /// tanh-approximation GELU, as used by GPT-2/Gemma "gelu_pytorch_tanh".
    GeluTanh,
    /// exact-erf GELU (approximated to 1e-7 here), as used by BERT/"gelu".
    Gelu,
    Tanh,
    Sigmoid,
    Custom(fn(f32) -> f32),
}

impl Activation {
    /// Parse a Hugging Face `hidden_act` / GGUF activation name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "relu" => Self::Relu,
            "silu" | "swish" | "swiglu" => Self::Silu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" | "geglu" => Self::GeluTanh,
            "gelu" | "gelu_exact" => Self::Gelu,
            "tanh" => Self::Tanh,
            "sigmoid" => Self::Sigmoid,
            _ => return None,
        })
    }

    pub fn apply(&self, z: f32) -> f32 {
        match self {
            Self::Relu => z.max(0.0),
            Self::Silu => z / (1.0 + (-z).exp()),
            Self::GeluTanh => 0.5 * z * (1.0 + (0.797_884_6 * (z + 0.044_715 * z * z * z)).tanh()),
            Self::Gelu => 0.5 * z * (1.0 + erf(z / std::f32::consts::SQRT_2)),
            Self::Tanh => z.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            Self::Custom(f) => f(z),
        }
    }
}

/// Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    y.copysign(x)
}

/// One population unit: `relu(direction * (z - knot))`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit {
    pub knot: f32,
    /// +1 for rising units, -1 for falling (mirrored) units.
    pub direction: f32,
    /// Synaptic weight onto the population output (sign = E/I).
    pub weight: f32,
}

impl Unit {
    pub fn drive(&self, z: f32) -> f32 {
        (self.direction * (z - self.knot)).max(0.0)
    }
}

/// A fitted population code for one activation.
#[derive(Clone, Debug)]
pub struct PopulationCode {
    pub tonic: f32,
    pub units: Vec<Unit>,
    pub range: (f32, f32),
}

impl PopulationCode {
    /// Fit `act` over `[lo, hi]` with `knots` thresholds per direction
    /// (`2 * knots` units in total). Units whose weight is negligible are
    /// pruned, keeping the mesh sparse.
    pub fn fit(act: Activation, lo: f32, hi: f32, knots: usize) -> Self {
        assert!(hi > lo && knots >= 2);
        let mut ks = curvature_knots(act, lo, hi, knots);
        // Model activations bend most at the origin (ReLU's kink, the SiLU/GELU
        // knee), so a threshold always sits exactly there when it is in range.
        if lo < 0.0 && hi > 0.0 && !ks.iter().any(|k| k.abs() < 1e-6) {
            ks.push(0.0);
            ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let mut basis: Vec<Unit> = Vec::with_capacity(2 * knots);
        for &k in &ks {
            basis.push(Unit {
                knot: k,
                direction: 1.0,
                weight: 0.0,
            });
            basis.push(Unit {
                knot: k,
                direction: -1.0,
                weight: 0.0,
            });
        }
        let samples = 16 * knots + 1;
        let zs: Vec<f32> = (0..samples)
            .map(|i| lo + (hi - lo) * i as f32 / (samples - 1) as f32)
            .collect();
        // Design matrix columns: tonic (1) + basis units.
        let n = basis.len() + 1;
        let row = |z: f32| -> Vec<f64> {
            std::iter::once(1.0)
                .chain(basis.iter().map(|u| u.drive(z) as f64))
                .collect()
        };
        let mut ata = vec![0.0f64; n * n];
        let mut aty = vec![0.0f64; n];
        for &z in &zs {
            let r = row(z);
            let y = act.apply(z) as f64;
            for i in 0..n {
                aty[i] += r[i] * y;
                for j in 0..n {
                    ata[i * n + j] += r[i] * r[j];
                }
            }
        }
        let ridge = 1e-6 * zs.len() as f64;
        for i in 1..n {
            ata[i * n + i] += ridge;
        }
        let coef = solve(&mut ata, &mut aty, n);
        let scale = coef[1..].iter().fold(0.0f64, |m, c| m.max(c.abs()));
        let units = basis
            .iter()
            .zip(&coef[1..])
            .filter(|(_, c)| c.abs() > 1e-6 * scale.max(1e-12))
            .map(|(u, c)| Unit {
                weight: *c as f32,
                ..*u
            })
            .collect();
        Self {
            tonic: coef[0] as f32,
            units,
            range: (lo, hi),
        }
    }

    /// Analog (infinite-time) output of the population.
    pub fn eval(&self, z: f32) -> f32 {
        self.tonic
            + self
                .units
                .iter()
                .map(|u| u.weight * u.drive(z))
                .sum::<f32>()
    }

    /// Max absolute error against `act` on `points` evenly spaced in range.
    pub fn max_error(&self, act: Activation, points: usize) -> f32 {
        let (lo, hi) = self.range;
        (0..points)
            .map(|i| lo + (hi - lo) * i as f32 / (points - 1) as f32)
            .map(|z| (self.eval(z) - act.apply(z)).abs())
            .fold(0.0, f32::max)
    }
}

/// Thresholds equidistributed in `sqrt(|f''|)` (optimal for piecewise-linear
/// approximation), mixed with a uniform floor so straight regions still get
/// units. `f''` is estimated numerically, so this works for any activation.
fn curvature_knots(act: Activation, lo: f32, hi: f32, knots: usize) -> Vec<f32> {
    let n = 4096;
    let h = (hi - lo) / n as f32;
    let eps = (h * 4.0).max(1e-3);
    let density: Vec<f64> = (0..=n)
        .map(|i| {
            let z = lo + h * i as f32;
            let f2 = (act.apply(z + eps) - 2.0 * act.apply(z) + act.apply(z - eps)) / (eps * eps);
            f2.abs().sqrt() as f64
        })
        .collect();
    let mean = density.iter().sum::<f64>() / density.len() as f64;
    let floor = 0.25 * mean.max(1e-9);
    let cum: Vec<f64> = density
        .iter()
        .scan(0.0, |acc, d| {
            *acc += d + floor;
            Some(*acc)
        })
        .collect();
    let total = *cum.last().unwrap();
    let mut ks = Vec::with_capacity(knots);
    let mut idx = 0;
    for k in 0..knots {
        let target = total * k as f64 / (knots - 1) as f64;
        while idx < n && cum[idx] < target {
            idx += 1;
        }
        ks.push(lo + h * idx as f32);
    }
    ks[0] = lo;
    ks[knots - 1] = hi;
    ks.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    ks
}

/// Gaussian elimination with partial pivoting (small dense systems).
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Vec<f64> {
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| {
                a[i * n + col]
                    .abs()
                    .partial_cmp(&a[j * n + col].abs())
                    .unwrap()
            })
            .unwrap();
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        assert!(d.abs() > 1e-300, "singular system");
        for r in col + 1..n {
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for k in col..n {
                    a[r * n + k] -= f * a[col * n + k];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r * n + k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r * n + r];
    }
    x
}
