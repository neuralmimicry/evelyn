//! `evelyn-verify-layer <model.gguf> [layer] [steps] [samples] [percentile] [knots] [calibration] [headroom]`
//!
//! Stage 2 verification gate. The tool:
//! 1. imports one real feed-forward layer from an open-weights checkpoint;
//! 2. converts it into the spiking population mesh;
//! 3. measures conversion error against the original layer.
//!
//! Inputs are realistic proxies: real token embeddings RMS-normalised with
//! the layer's own learned norm weights. Genuine mid-network activations
//! become available with the stage 4 runtime.
//!
//! It reports the analog population error (fit and saturation) and the
//! dithered-spiking error. Samples run in parallel across all cores, and
//! the process exits non-zero if the gate fails.
//!
//! Recommended settings (stage 2, Qwen 3.5 9B): percentile 1.0, 24 knots,
//! 512 calibration samples, headroom 2.0, 4096 steps.

use evelyn::gguf::Gguf;
use evelyn::import::{import_ffn, rms_norm};
use evelyn::snn::SpikingPopulationGatedMlp;
use evelyn::{Rng, relative_error};
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Instant;

const GATE: f32 = 0.05;

/// Parse an optional positional argument, with a clear error on bad input.
fn arg<T: FromStr>(args: &[String], i: usize, default: T, name: &str) -> Result<T, String> {
    match args.get(i) {
        None => Ok(default),
        Some(s) => s.parse().map_err(|_| format!("invalid {name}: {s:?}")),
    }
}

fn run(args: &[String]) -> Result<bool, String> {
    let path = args.get(1).ok_or("usage: evelyn-verify-layer <model.gguf> [layer] [steps] [samples] [percentile] [knots] [calibration] [headroom]")?;
    let layer: usize = arg(args, 2, 0, "layer")?;
    let steps: usize = arg(args, 3, 4096, "steps")?;
    let samples: usize = arg(args, 4, 8, "samples")?;
    let percentile: f32 = arg(args, 5, 1.0, "percentile")?;
    let knots: usize = arg(args, 6, 24, "knots")?;
    let calibration: usize = arg(args, 7, 512, "calibration")?;
    let headroom: f32 = arg(args, 8, 2.0, "headroom")?;
    if !(0.5..=1.0).contains(&percentile) || knots < 2 || headroom < 1.0 || samples == 0 {
        return Err(
            "out-of-range argument (percentile 0.5-1, knots >= 2, headroom >= 1, samples >= 1)"
                .into(),
        );
    }

    let t0 = Instant::now();
    let mut g = Gguf::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let ffn = import_ffn(&mut g, layer).map_err(|e| format!("import layer {layer}: {e}"))?;
    println!(
        "imported {} layer {layer}: gate {}x{}, activation {} ({:.1}s)",
        ffn.arch,
        ffn.mlp.gate.outputs,
        ffn.mlp.gate.inputs,
        ffn.activation_name,
        t0.elapsed().as_secs_f32()
    );
    let vocab = g
        .tensors
        .get("token_embd.weight")
        .ok_or("no token_embd.weight")?
        .dims[1];
    let mut rng = Rng::new(2026);
    let ids: Vec<u64> = (0..(calibration + samples))
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

    let t1 = Instant::now();
    let snn = SpikingPopulationGatedMlp::convert_with_headroom(
        &ffn.mlp,
        ffn.activation,
        calib,
        percentile,
        knots,
        headroom,
    );
    println!(
        "converted: {} population units per hidden neuron ({:.1}s)",
        snn.population_size(),
        t1.elapsed().as_secs_f32()
    );

    let t2 = Instant::now();
    let results: Vec<(f32, f32, u64)> = std::thread::scope(|s| {
        let handles: Vec<_> = test
            .iter()
            .map(|x| {
                let (snn, mlp, act) = (&snn, &ffn.mlp, ffn.activation);
                s.spawn(move || {
                    let reference = mlp.forward_act(x, act);
                    let analog = snn.run_analog(x);
                    let (spiking, spikes) = snn.run_dithered(x, steps);
                    (
                        relative_error(&reference, &analog),
                        relative_error(&reference, &spiking),
                        spikes,
                    )
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    if results.len() != test.len() {
        return Err(format!(
            "{} of {} sample workers failed",
            test.len() - results.len(),
            test.len()
        ));
    }
    let n = results.len() as f32;
    let analog = results.iter().map(|r| r.0).sum::<f32>() / n;
    let spiking = results.iter().map(|r| r.1).sum::<f32>() / n;
    let spikes = results.iter().map(|r| r.2).sum::<u64>() / results.len() as u64;
    println!(
        "layer {layer}: population (analog) relative error {analog:.4}; dithered spiking T={steps} relative error {spiking:.4}; {spikes} spikes/token; {:.1}s for {} samples",
        t2.elapsed().as_secs_f32(),
        results.len()
    );
    let pass = spiking < GATE;
    println!("GATE {}", if pass { "PASS (< 5%)" } else { "FAIL (>= 5%)" });
    Ok(pass)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("evelyn-verify-layer: {e}");
            ExitCode::from(2)
        }
    }
}
