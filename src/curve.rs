//! Stage 3a: population codes fitted against AARNN's *measured* neuron
//! transfer curves.
//!
//! Stages 1 and 2 used an idealised rectified-linear unit response. AARNN's
//! real neurons (LIF, Izhikevich/AARNN at each computational detail depth)
//! respond differently. They have a rheobase below which they are silent, a
//! curved rise, refractory saturation and discrete-time ripple. aarnn_rust
//! measures these curves with its own transition kernels
//! (`aarnn-knowledge-curves`) and exports them as data. Here each population
//! unit's basis is that measured curve, shifted and scaled into its own
//! operating range, and the unit weights are refitted by least squares.
//!
//! The approximation therefore targets the dynamics AARNN actually runs
//! (design principle 2), for any neuron model and any detail depth, with no
//! model-specific code.

use crate::activation::Activation;
use serde::Deserialize;
use std::path::Path;

/// A transfer curve as exported by `aarnn-knowledge-curves`.
#[derive(Clone, Debug, Deserialize)]
pub struct MeasuredCurve {
    pub neuron: String,
    pub currents: Vec<f64>,
    pub rates: Vec<f64>,
}

impl MeasuredCurve {
    /// Load every curve from an exported JSON file.
    pub fn load_all(path: impl AsRef<Path>) -> Result<Vec<Self>, String> {
        let text =
            std::fs::read_to_string(path.as_ref()).map_err(|e| format!("read curves: {e}"))?;
        let curves: Vec<Self> =
            serde_json::from_str(&text).map_err(|e| format!("parse curves: {e}"))?;
        for c in &curves {
            if c.currents.len() != c.rates.len() || c.currents.len() < 8 {
                return Err(format!("curve {}: malformed", c.neuron));
            }
        }
        Ok(curves.into_iter().map(Self::smoothed).collect())
    }

    /// Remove finite-sample measurement jitter, keeping the curve's shape:
    /// a centred moving average, then monotone (running-maximum) enforcement.
    /// The jitter comes from counting a finite number of spikes, not from the
    /// neuron's dynamics. A population of such neurons averages it away, so
    /// smoothing represents the population response more faithfully than the
    /// raw single-neuron count does.
    pub fn smoothed(mut self) -> Self {
        let n = self.rates.len();
        let w = (n / 100).max(2);
        let raw = self.rates.clone();
        for i in 0..n {
            let (a, b) = (i.saturating_sub(w), (i + w + 1).min(n));
            self.rates[i] = raw[a..b].iter().sum::<f64>() / (b - a) as f64;
        }
        let mut peak = 0.0f64;
        for r in &mut self.rates {
            peak = peak.max(*r);
            *r = peak;
        }
        self
    }

    /// Lowest current with non-zero firing (the rheobase).
    pub fn rheobase(&self) -> f64 {
        self.currents
            .iter()
            .zip(&self.rates)
            .find(|(_, r)| **r > 0.0)
            .map_or(self.currents[0], |(c, _)| *c)
    }

    pub fn max_rate(&self) -> f64 {
        self.rates.iter().cloned().fold(0.0, f64::max)
    }

    /// Lowest current reaching 98 % of the maximum rate (saturation onset).
    pub fn saturation(&self) -> f64 {
        let target = 0.98 * self.max_rate();
        self.currents
            .iter()
            .zip(&self.rates)
            .find(|(_, r)| **r >= target)
            .map_or(*self.currents.last().unwrap(), |(c, _)| *c)
    }

    /// Normalised rate (0..1) at `current`, by linear interpolation; clamped
    /// at the measured ends.
    pub fn response(&self, current: f64) -> f64 {
        let (c, r) = (&self.currents, &self.rates);
        let max = self.max_rate().max(1e-12);
        if current <= c[0] {
            return r[0] / max;
        }
        if current >= c[c.len() - 1] {
            return r[r.len() - 1] / max;
        }
        let i = c
            .partition_point(|x| *x <= current)
            .saturating_sub(1)
            .min(c.len() - 2);
        let t = (current - c[i]) / (c[i + 1] - c[i]);
        (r[i] + t * (r[i + 1] - r[i])) / max
    }
}

/// A population unit driven through a measured AARNN curve.
#[derive(Clone, Copy, Debug)]
pub struct CurveUnit {
    pub knot: f32,
    pub direction: f32,
    /// Input gain (current per unit of pre-activation), mapping the unit's
    /// operating range onto the neuron's rheobase-to-saturation span.
    pub gain: f64,
    pub weight: f32,
}

/// A population code whose units respond as real AARNN neurons.
#[derive(Clone, Debug)]
pub struct CurvePopulationCode {
    pub tonic: f32,
    pub units: Vec<CurveUnit>,
    pub range: (f32, f32),
    curve: MeasuredCurve,
    rheobase: f64,
}

impl CurvePopulationCode {
    /// Normalised response of `unit` to pre-activation `z`.
    pub fn unit_response(&self, unit: &CurveUnit, z: f32) -> f64 {
        let drive = (unit.direction * (z - unit.knot)) as f64;
        if drive <= 0.0 {
            return 0.0;
        }
        self.curve.response(self.rheobase + unit.gain * drive)
    }

    /// Fit `act` on `[lo, hi]` using `knots` thresholds per direction. Each
    /// unit spans `span_knots` knot spacings between rheobase and saturation,
    /// so neighbouring units overlap smoothly.
    pub fn fit(
        act: Activation,
        lo: f32,
        hi: f32,
        knots: usize,
        span_knots: f32,
        curve: &MeasuredCurve,
    ) -> Self {
        // Fit over a domain 15 % wider on each side, so the requested range's
        // edges are interior. Units near a hard boundary would otherwise not
        // have finished ramping, which produces edge error.
        let (req_lo, req_hi) = (lo, hi);
        let margin = 0.15 * (hi - lo);
        let (lo, hi) = (lo - margin, hi + margin);
        // Thresholds are evenly spaced. Shifted copies of one saturating
        // response sum to a near-exact line, which is what an activation's
        // linear regions need. Curvature-adaptive spacing suits ideal
        // rectifiers, but leaves saturating neurons rippling in sparse tails.
        let ks: Vec<f32> = (0..knots)
            .map(|i| lo + (hi - lo) * i as f32 / (knots - 1) as f32)
            .collect();
        let rheobase = curve.rheobase();
        let dynamic_span = curve.saturation() - rheobase;
        // Gain follows the *local* knot gap in each unit's direction, because
        // curvature-adaptive thresholds are dense near the knee and sparse in
        // the tails. A unit's response then spans `span_knots` of its own
        // neighbourhood, leaving no uncovered gaps.
        let fallback = (hi - lo) / (knots.max(2) - 1) as f32;
        // Floor the gap so no unit ramps faster than the fitting grid can
        // resolve; otherwise the fit is blind between samples and overshoots.
        let min_gap = (hi - lo) / (2 * knots.max(2)) as f32;
        let mut basis = Vec::with_capacity(ks.len() * 2);
        for (i, k) in ks.iter().enumerate() {
            let up_gap = ks.get(i + 1).map_or(fallback, |n| n - k).max(min_gap);
            let down_gap = if i > 0 {
                (k - ks[i - 1]).max(min_gap)
            } else {
                fallback
            };
            for (direction, gap) in [(1.0f32, up_gap), (-1.0f32, down_gap)] {
                let gain = dynamic_span / (span_knots * gap) as f64;
                basis.push(CurveUnit {
                    knot: *k,
                    direction,
                    gain,
                    weight: 0.0,
                });
            }
        }
        let mut code = Self {
            tonic: 0.0,
            units: basis,
            range: (req_lo, req_hi),
            curve: curve.clone(),
            rheobase,
        };
        let samples = 128 * knots + 1;
        let zs: Vec<f32> = (0..samples)
            .map(|i| lo + (hi - lo) * i as f32 / (samples - 1) as f32)
            .collect();
        let n = code.units.len() + 1;
        let mut ata = vec![0.0f64; n * n];
        let mut aty = vec![0.0f64; n];
        for &z in &zs {
            let row: Vec<f64> = std::iter::once(1.0)
                .chain(code.units.iter().map(|u| code.unit_response(u, z)))
                .collect();
            let y = act.apply(z) as f64;
            for i in 0..n {
                aty[i] += row[i] * y;
                for j in 0..n {
                    ata[i * n + j] += row[i] * row[j];
                }
            }
        }
        let ridge = 1e-6 * zs.len() as f64;
        for i in 1..n {
            ata[i * n + i] += ridge;
        }
        let coef = crate::activation::solve_dense(&mut ata, &mut aty, n);
        code.tonic = coef[0] as f32;
        for (u, c) in code.units.iter_mut().zip(&coef[1..]) {
            u.weight = *c as f32;
        }
        code
    }

    /// Smallest population (from an increasing schedule of threshold counts
    /// and response widths) that meets `tolerance` maximum absolute error on
    /// `[lo, hi]`. This is how the conversion adapts automatically to the
    /// AARNN neuron model and detail depth in use. Returns the best fit found
    /// and whether it met the tolerance.
    pub fn fit_to_tolerance(
        act: Activation,
        lo: f32,
        hi: f32,
        curve: &MeasuredCurve,
        tolerance: f32,
        max_knots: usize,
    ) -> (Self, bool) {
        let mut best: Option<(Self, f32)> = None;
        for knots in [16usize, 24, 32, 48, 64, 96, 128, 192, 256]
            .into_iter()
            .filter(|k| *k <= max_knots)
        {
            for span in [2.0f32, 4.0] {
                let code = Self::fit(act, lo, hi, knots, span, curve);
                let err = code.max_error(act, 4001);
                if err < tolerance {
                    return (code, true);
                }
                if best.as_ref().is_none_or(|(_, e)| err < *e) {
                    best = Some((code, err));
                }
            }
        }
        (best.expect("schedule is non-empty").0, false)
    }

    pub fn eval(&self, z: f32) -> f32 {
        self.tonic
            + self
                .units
                .iter()
                .map(|u| u.weight * self.unit_response(u, z) as f32)
                .sum::<f32>()
    }

    pub fn max_error(&self, act: Activation, points: usize) -> f32 {
        let (lo, hi) = self.range;
        (0..points)
            .map(|i| lo + (hi - lo) * i as f32 / (points - 1) as f32)
            .map(|z| (self.eval(z) - act.apply(z)).abs())
            .fold(0.0, f32::max)
    }
}
