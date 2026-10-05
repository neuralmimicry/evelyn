//! `evelyn-export-mesh <model.gguf> <layer> <curves.json> <neuron-label> <outdir> [samples] [steps] [calibration] [headroom]`
//!
//! Stage 3b, Evelyn side. The tool:
//! 1. imports one real feed-forward layer;
//! 2. fits gate and up population codes on the chosen AARNN neuron's
//!    measured transfer curve;
//! 3. writes the knowledge-region mesh (`mesh.json` plus weight files);
//! 4. writes test inputs, the exact layer outputs (`reference.f32`) and the
//!    mesh's analog prediction (`analog.f32`).
//!
//! AARNN then runs the mesh with `aarnn-knowledge-run`, and `evelyn-compare`
//! scores its output.

use evelyn::curve::MeasuredCurve;
use evelyn::gguf::Gguf;
use evelyn::import::{import_ffn, rms_norm};
use evelyn::mesh::{analog_output, export, plan, write_f32};
use evelyn::{Rng, relative_error};
use std::path::Path;
use std::process::ExitCode;

fn arg<T: std::str::FromStr>(a: &[String], i: usize, d: T, n: &str) -> Result<T, String> {
    a.get(i).map_or(Ok(d), |s| {
        s.parse().map_err(|_| format!("invalid {n}: {s:?}"))
    })
}

fn run(a: &[String]) -> Result<(), String> {
    if a.len() < 6 {
        return Err("usage: evelyn-export-mesh <model.gguf> <layer> <curves.json> <neuron-label> <outdir> [samples] [steps] [calibration] [headroom]".into());
    }
    let layer: usize = arg(a, 2, 0, "layer")?;
    let samples: usize = arg(a, 6, 4, "samples")?;
    let steps: usize = arg(a, 7, 4000, "steps")?;
    let calibration: usize = arg(a, 8, 512, "calibration")?;
    let headroom: f32 = arg(a, 9, 2.0, "headroom")?;
    let curves = MeasuredCurve::load_all(&a[3])?;
    let curve = curves
        .iter()
        .find(|c| c.neuron == a[4])
        .ok_or_else(|| format!("no curve labelled {}", a[4]))?;
    let mut g = Gguf::open(&a[1]).map_err(|e| format!("open: {e}"))?;
    let ffn = import_ffn(&mut g, layer).map_err(|e| format!("import: {e}"))?;
    let vocab = g
        .tensors
        .get("token_embd.weight")
        .ok_or("no token_embd.weight")?
        .dims[1];
    let mut rng = Rng::new(2026);
    let ids: Vec<u64> = (0..calibration + samples)
        .map(|_| rng.next_u64() % vocab)
        .collect();
    let emb = g
        .tensor_rows_f32("token_embd.weight", &ids)
        .map_err(|e| format!("embeddings: {e}"))?;
    let xs: Vec<Vec<f32>> = emb
        .iter()
        .map(|e| rms_norm(e, ffn.norm.as_deref(), ffn.norm_eps))
        .collect();
    let (calib, test) = xs.split_at(calibration);
    let p = plan(&ffn.mlp, ffn.activation, calib, curve, headroom);
    println!(
        "plan on {}: gate {} units (met {}), up {} units (met {}), gate range {:?}, up range {:?}",
        curve.neuron,
        p.gate_code.units.len() + 1,
        p.gate_met,
        p.up_code.units.len() + 1,
        p.up_met,
        p.gate_range,
        p.up_range
    );
    let dir = Path::new(&a[5]);
    export(
        dir,
        &format!("{}-layer{layer}-{}", ffn.arch, curve.neuron),
        &ffn.mlp,
        &p,
        curve,
        steps,
        steps / 10,
    )
    .map_err(|e| format!("export: {e}"))?;
    let reference: Vec<f32> = test
        .iter()
        .flat_map(|x| ffn.mlp.forward_act(x, ffn.activation))
        .collect();
    let analog: Vec<f32> = test
        .iter()
        .flat_map(|x| analog_output(&ffn.mlp, &p, x))
        .collect();
    write_f32(&dir.join("inputs.f32"), &test.concat()).map_err(|e| e.to_string())?;
    write_f32(&dir.join("reference.f32"), &reference).map_err(|e| e.to_string())?;
    write_f32(&dir.join("analog.f32"), &analog).map_err(|e| e.to_string())?;
    let width = ffn.mlp.down.outputs;
    let err: f32 = reference
        .chunks(width)
        .zip(analog.chunks(width))
        .map(|(r, a)| relative_error(r, a))
        .sum::<f32>()
        / samples as f32;
    println!("analog (infinite-time) mesh relative error vs exact layer: {err:.4}");
    println!("exported {} samples to {}", samples, dir.display());
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    match run(&a) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("evelyn-export-mesh: {e}");
            ExitCode::from(2)
        }
    }
}
