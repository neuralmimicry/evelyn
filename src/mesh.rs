//! Stage 3b: knowledge-region mesh descriptions for AARNN.
//!
//! Evelyn converts one feed-forward layer into an `FfnMesh` (the format of
//! `aarnn_rust::knowledge_region`). It contains the input and readout
//! synapse matrices, the fitted gate and up population codes, and the exact
//! AARNN neuron configuration and membrane noise the codes were fitted to.
//! AARNN instantiates and runs the mesh. Evelyn holds no AARNN code, so the
//! two stay decoupled and each remains the single owner of its part.
//!
//! The gate population encodes the layer's activation on the calibrated
//! gate-current range. The up population encodes the identity on the
//! up-current range. Both share one code across all hidden channels; only
//! the synapses differ between channels.

use crate::activation::Activation;
use crate::curve::{CurvePopulationCode, MeasuredCurve};
use crate::dense::{Dense, SwiGluMlp};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Write values as little-endian f32.
pub fn write_f32(path: &Path, values: &[f32]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()
}

fn identity(z: f32) -> f32 {
    z
}

/// Fitted codes plus the ranges they cover, before export.
pub struct MeshPlan {
    pub gate_code: CurvePopulationCode,
    pub up_code: CurvePopulationCode,
    pub gate_range: (f32, f32),
    /// Normalised up range, always `[-1, 1]` (per-channel synaptic scaling).
    pub up_range: (f32, f32),
    /// Per-channel up synaptic scale: calibrated `max |u_i|` times headroom.
    pub up_scale: Vec<f32>,
    pub gate_met: bool,
    pub up_met: bool,
}

/// Fit the gate and up population codes for `mlp` on AARNN neuron `curve`,
/// using calibration activations. The gate range is the calibrated extreme
/// widened by `headroom` (gain control against heavy-tailed activations).
pub fn plan(
    mlp: &SwiGluMlp,
    act: Activation,
    calibration: &[Vec<f32>],
    curve: &MeasuredCurve,
    headroom: f32,
) -> MeshPlan {
    let hidden = mlp.gate.outputs;
    let (mut zlo, mut zhi) = (0.0f32, 0.0f32);
    let mut umax = vec![0.0f32; hidden];
    for x in calibration {
        for z in mlp.gate.forward(x) {
            zlo = zlo.min(z);
            zhi = zhi.max(z);
        }
        for (m, u) in umax.iter_mut().zip(mlp.up.forward(x)) {
            *m = m.max(u.abs());
        }
    }
    let gate_range = (zlo * headroom, zhi * headroom);
    // Synaptic scaling: each channel's up current is normalised by its own
    // calibrated range, so one identity code on [-1, 1] serves every channel
    // at the same relative precision.
    let up_scale: Vec<f32> = umax.iter().map(|m| (m * headroom).max(1e-6)).collect();
    let up_range = (-1.0f32, 1.0f32);
    // The gate tolerance is absolute (activation units). Most gate currents
    // lie near zero, where the activation is small, so a span-relative
    // tolerance would swamp them.
    let (gate_code, gate_met) =
        CurvePopulationCode::fit_to_tolerance(act, gate_range.0, gate_range.1, curve, 0.01, 256);
    let (up_code, up_met) = CurvePopulationCode::fit_to_tolerance(
        Activation::Custom(identity),
        up_range.0,
        up_range.1,
        curve,
        0.005,
        256,
    );
    MeshPlan {
        gate_code,
        up_code,
        gate_range,
        up_range,
        up_scale,
        gate_met,
        up_met,
    }
}

fn code_json(code: &CurvePopulationCode) -> Value {
    json!({
        "tonic": code.tonic,
        "units": code.units.iter().map(|u| json!({
            "knot": u.knot, "direction": u.direction, "gain": u.gain, "weight": u.weight
        })).collect::<Vec<_>>(),
        "rheobase": code.rheobase(),
        "max_rate": code.max_rate(),
    })
}

fn dense_json(d: &Dense, file: &str) -> Value {
    json!({ "inputs": d.inputs, "outputs": d.outputs, "weights_file": file })
}

/// Write `mesh.json` and its weight files into `dir`.
pub fn export(
    dir: &Path,
    name: &str,
    mlp: &SwiGluMlp,
    plan: &MeshPlan,
    curve: &MeasuredCurve,
    steps: usize,
    warmup: usize,
) -> io::Result<()> {
    let spec = curve.spec.clone().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "curve has no AARNN neuron spec; re-export curves",
        )
    })?;
    std::fs::create_dir_all(dir)?;
    for (d, f) in [
        (&mlp.gate, "gate.f32"),
        (&mlp.up, "up.f32"),
        (&mlp.down, "down.f32"),
    ] {
        write_f32(&dir.join(f), &d.weights)?;
    }
    write_f32(&dir.join("up_scale.f32"), &plan.up_scale)?;
    let mesh = json!({
        "name": name,
        "neuron": spec,
        "noise_std": curve.noise_std,
        "steps": steps,
        "warmup": warmup,
        "gate": dense_json(&mlp.gate, "gate.f32"),
        "up": dense_json(&mlp.up, "up.f32"),
        "down": dense_json(&mlp.down, "down.f32"),
        "gate_code": code_json(&plan.gate_code),
        "up_code": code_json(&plan.up_code),
        "up_scale_file": "up_scale.f32",
    });
    std::fs::write(dir.join("mesh.json"), serde_json::to_vec_pretty(&mesh)?)?;
    Ok(())
}

/// The mesh's infinite-time (analog) output: what AARNN converges to as
/// `steps` grows. It separates fitting error from spiking sampling noise.
pub fn analog_output(mlp: &SwiGluMlp, plan: &MeshPlan, x: &[f32]) -> Vec<f32> {
    let z = mlp.gate.forward(x);
    let u = mlp.up.forward(x);
    let h: Vec<f32> = (0..z.len())
        .map(|i| {
            let g = plan
                .gate_code
                .eval(z[i].clamp(plan.gate_range.0, plan.gate_range.1));
            let s = plan.up_scale[i];
            let v = plan
                .up_code
                .eval((u[i] / s).clamp(plan.up_range.0, plan.up_range.1))
                * s;
            g * v
        })
        .collect();
    mlp.down.forward(&h)
}
