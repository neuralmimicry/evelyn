//! `evelyn-layer-mesh <model.gguf> <layer> <curves.json> <neuron> <outdir> <llama-server-url> <text-file> [tokens] [headroom]`
//!
//! Stage 4b. It exports a knowledge-region mesh for one real layer,
//! calibrated on **real mid-network activations**: the text is run through
//! Evelyn's runtime and the layer's normalised FFN inputs are captured. This
//! replaces stage 2's proxy inputs.

use evelyn::curve::MeasuredCurve;
use evelyn::gguf::Gguf;
use evelyn::import::import_ffn;
use evelyn::mesh::{analog_output, export, plan};
use evelyn::relative_error;
use evelyn::remote::CaptureFfn;
use evelyn::runtime::{Model, Session};
use serde_json::json;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Mutex;

fn run(a: &[String]) -> Result<(), String> {
    if a.len() < 8 {
        return Err("usage: evelyn-layer-mesh <model.gguf> <layer> <curves.json> <neuron> <outdir> <llama-server-url> <text-file> [tokens] [headroom]".into());
    }
    let layer: usize = a[2].parse().map_err(|_| "invalid layer")?;
    let max_tokens: usize = a
        .get(8)
        .map_or(Ok(256), |s| s.parse())
        .map_err(|_| "invalid tokens")?;
    let headroom: f32 = a
        .get(9)
        .map_or(Ok(2.0), |s| s.parse())
        .map_err(|_| "invalid headroom")?;
    let curves = MeasuredCurve::load_all(&a[3])?;
    let curve = curves
        .iter()
        .find(|c| c.neuron == a[4])
        .ok_or_else(|| format!("no curve {}", a[4]))?;
    let text = std::fs::read_to_string(&a[7]).map_err(|e| format!("text: {e}"))?;
    let toks: Vec<u32> = ureq::post(&format!("{}/tokenize", a[6].trim_end_matches('/')))
        .send_json(json!({"content": text, "add_special": false}))
        .map_err(|e| format!("tokenize: {e}"))?
        .into_json::<serde_json::Value>()
        .map_err(|e| e.to_string())?["tokens"]
        .as_array()
        .ok_or("tokenize: no tokens")?
        .iter()
        .filter_map(|t| t.as_u64().map(|v| v as u32))
        .take(max_tokens)
        .collect();
    let model = Model::load(&a[1]).map_err(|e| format!("load: {e}"))?;
    let cap = CaptureFfn {
        model: &model,
        layer,
        captured: Mutex::new(Vec::new()),
    };
    let mut s = Session::new(&model);
    for t in &toks {
        s.step(*t, &cap).map_err(|e| e.to_string())?;
    }
    let xs = cap
        .captured
        .into_inner()
        .map_err(|_| "capture lock poisoned")?;
    // Hold back every eighth vector for evaluation; calibrate on the rest.
    let (mut calib, mut test) = (Vec::new(), Vec::new());
    for (i, x) in xs.into_iter().enumerate() {
        if i % 8 == 7 {
            test.push(x)
        } else {
            calib.push(x)
        }
    }
    let mut g = Gguf::open(&a[1]).map_err(|e| format!("open: {e}"))?;
    let ffn = import_ffn(&mut g, layer).map_err(|e| format!("import: {e}"))?;
    let p = plan(&ffn.mlp, ffn.activation, &calib, curve, headroom);
    let err: f32 = test
        .iter()
        .map(|x| {
            relative_error(
                &ffn.mlp.forward_act(x, ffn.activation),
                &analog_output(&ffn.mlp, &p, x),
            )
        })
        .sum::<f32>()
        / test.len().max(1) as f32;
    println!(
        "layer {layer}: {} real activations ({} calibration, {} held out); gate {} units, up {} units; analog mesh error on held-out real activations {err:.4}",
        calib.len() + test.len(),
        calib.len(),
        test.len(),
        p.gate_code.units.len() + 1,
        p.up_code.units.len() + 1
    );
    export(
        Path::new(&a[5]),
        &format!("{}-layer{layer}-{}", ffn.arch, curve.neuron),
        &ffn.mlp,
        &p,
        curve,
        4000,
        400,
    )
    .map_err(|e| format!("export: {e}"))?;
    println!("exported mesh to {}", a[5]);
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    match run(&a) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("evelyn-layer-mesh: {e}");
            ExitCode::from(2)
        }
    }
}
