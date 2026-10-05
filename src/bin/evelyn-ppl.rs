//! `evelyn-ppl <model.gguf> <llama-server-url> <text-file> <tokens> [layer=host:port ...]`
//!
//! Stage 4b gate. It computes teacher-forced perplexity on the text twice:
//! once fully dense, and once with the listed layers served by AARNN
//! knowledge regions (`aarnn-knowledge-serve`). It reports the relative
//! perplexity change, top-1 next-token agreement, latency and dense
//! fallbacks, and exits non-zero if perplexity rises by 5 % or more.

use evelyn::remote::RemoteAarnnFfn;
use evelyn::runtime::{DenseFfn, FfnBackend, Model, Session, argmax, log_prob};
use serde_json::json;
use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::{Duration, Instant};

fn score(
    model: &Model,
    toks: &[u32],
    ffn: &dyn FfnBackend,
) -> Result<(f64, Vec<u32>, f64), String> {
    let mut s = Session::new(model);
    let (mut nll, mut top1) = (0.0f64, Vec::with_capacity(toks.len()));
    let t = Instant::now();
    for (i, tok) in toks.iter().enumerate() {
        let logits = s.step(*tok, ffn).map_err(|e| e.to_string())?;
        top1.push(argmax(&logits));
        if let Some(next) = toks.get(i + 1) {
            nll -= log_prob(&logits, *next) as f64;
        }
    }
    Ok((
        (nll / (toks.len() - 1).max(1) as f64).exp(),
        top1,
        t.elapsed().as_secs_f64(),
    ))
}

fn run(a: &[String]) -> Result<bool, String> {
    if a.len() < 5 {
        return Err("usage: evelyn-ppl <model.gguf> <llama-server-url> <text-file> <tokens> [layer=host:port ...]".into());
    }
    let n: usize = a[4].parse().map_err(|_| "invalid tokens")?;
    let mut layers = BTreeMap::new();
    for spec in &a[5..] {
        let (l, addr) = spec
            .split_once('=')
            .ok_or_else(|| format!("bad layer spec {spec}"))?;
        layers.insert(
            l.parse::<usize>().map_err(|_| format!("bad layer {l}"))?,
            addr.to_string(),
        );
    }
    let text = std::fs::read_to_string(&a[3]).map_err(|e| format!("text: {e}"))?;
    let toks: Vec<u32> = ureq::post(&format!("{}/tokenize", a[2].trim_end_matches('/')))
        .send_json(json!({"content": text, "add_special": false}))
        .map_err(|e| format!("tokenize: {e}"))?
        .into_json::<serde_json::Value>()
        .map_err(|e| e.to_string())?["tokens"]
        .as_array()
        .ok_or("tokenize: no tokens")?
        .iter()
        .filter_map(|t| t.as_u64().map(|v| v as u32))
        .take(n)
        .collect();
    let model = Model::load(&a[1]).map_err(|e| format!("load: {e}"))?;
    let (ppl_dense, top_dense, t_dense) = score(&model, &toks, &DenseFfn(&model))?;
    println!(
        "dense:  perplexity {ppl_dense:.4} over {} tokens ({:.2}s/token)",
        toks.len(),
        t_dense / toks.len() as f64
    );
    if layers.is_empty() {
        return Ok(true);
    }
    let remote = RemoteAarnnFfn::new(&model, layers.clone(), Duration::from_secs(900));
    let (ppl_aarnn, top_aarnn, t_aarnn) = score(&model, &toks, &remote)?;
    let agree = top_dense
        .iter()
        .zip(&top_aarnn)
        .filter(|(a, b)| a == b)
        .count() as f64
        / toks.len() as f64;
    let delta = ppl_aarnn / ppl_dense - 1.0;
    let fallbacks = *remote.fallbacks.lock().map_err(|_| "lock")?;
    println!(
        "aarnn:  perplexity {ppl_aarnn:.4} with layers {:?} on AARNN ({:.2}s/token); change {:+.2}%; top-1 agreement {:.1}%; dense fallbacks {fallbacks}",
        layers.keys().collect::<Vec<_>>(),
        t_aarnn / toks.len() as f64,
        delta * 100.0,
        agree * 100.0
    );
    let pass = delta < 0.05 && fallbacks == 0;
    println!(
        "GATE {}",
        if pass {
            "PASS (perplexity change < 5%, no fallbacks)"
        } else {
            "FAIL"
        }
    );
    Ok(pass)
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    match run(&a) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("evelyn-ppl: {e}");
            ExitCode::from(2)
        }
    }
}
