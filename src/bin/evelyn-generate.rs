//! `evelyn-generate <model.gguf> <llama-server-url> <tokens> <prompt>`
//!
//! Stage 4a verification gate. It:
//! 1. tokenises `prompt` with the reference llama.cpp server (reusing its
//!    tokeniser rather than duplicating one);
//! 2. computes Evelyn's teacher-forced perplexity on the prompt;
//! 3. greedily generates `tokens` tokens with Evelyn's pure-Rust runtime;
//! 4. compares the result with llama.cpp's own greedy (temperature 0)
//!    continuation of the same prompt.
//!
//! It prints the agreement, the perplexity and the speed, and exits non-zero
//! below 90 % agreement.

use evelyn::runtime::{DenseFfn, Model, Session, argmax, log_prob};
use serde_json::{Value, json};
use std::process::ExitCode;
use std::time::Instant;

fn post(url: &str, body: Value) -> Result<Value, String> {
    ureq::post(url)
        .timeout(std::time::Duration::from_secs(600))
        .send_json(body)
        .map_err(|e| format!("{url}: {e}"))?
        .into_json()
        .map_err(|e| format!("{url}: bad JSON: {e}"))
}

fn run(a: &[String]) -> Result<bool, String> {
    if a.len() < 5 {
        return Err(
            "usage: evelyn-generate <model.gguf> <llama-server-url> <tokens> <prompt>".into(),
        );
    }
    let (path, server) = (&a[1], a[2].trim_end_matches('/'));
    let n: usize = a[3].parse().map_err(|_| "invalid token count")?;
    let prompt = a[4..].join(" ");
    let toks: Vec<u32> = post(
        &format!("{server}/tokenize"),
        json!({"content": prompt, "add_special": false}),
    )?["tokens"]
        .as_array()
        .ok_or("tokenize: no tokens")?
        .iter()
        .filter_map(|t| t.as_u64().map(|v| v as u32))
        .collect();
    let reference = post(
        &format!("{server}/completion"),
        json!({"prompt": toks, "n_predict": n, "temperature": 0.0, "top_k": 1, "return_tokens": true, "cache_prompt": false}),
    )?;
    let ref_tokens: Vec<u32> = reference["tokens"]
        .as_array()
        .ok_or("completion: no tokens (needs return_tokens)")?
        .iter()
        .filter_map(|t| t.as_u64().map(|v| v as u32))
        .collect();

    let t0 = Instant::now();
    let model = Model::load(path).map_err(|e| format!("load: {e}"))?;
    println!(
        "loaded {} ({} layers, dim {}, ffn {}, heads {}/{}) in {:.1}s",
        model.cfg.arch,
        model.cfg.layers,
        model.cfg.dim,
        model.cfg.ffn,
        model.cfg.heads,
        model.cfg.kv_heads,
        t0.elapsed().as_secs_f32()
    );
    let ffn = DenseFfn(&model);
    let mut s = Session::new(&model);
    let t1 = Instant::now();
    let mut nll = 0.0f64;
    let mut logits = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        logits = s.step(*t, &ffn).map_err(|e| e.to_string())?;
        if let Some(next) = toks.get(i + 1) {
            nll -= log_prob(&logits, *next) as f64;
        }
    }
    let ppl = (nll / (toks.len().saturating_sub(1).max(1)) as f64).exp();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let next = argmax(&logits);
        out.push(next);
        logits = s.step(next, &ffn).map_err(|e| e.to_string())?;
    }
    let secs = t1.elapsed().as_secs_f64();
    let compared = out.len().min(ref_tokens.len());
    // Agreement up to the first divergence (greedy paths diverge permanently).
    let prefix = out
        .iter()
        .zip(&ref_tokens)
        .take_while(|(a, b)| a == b)
        .count();
    let agree = if compared == 0 {
        0.0
    } else {
        prefix as f64 / compared as f64
    };
    println!("prompt tokens {}, prompt perplexity {ppl:.3}", toks.len());
    println!(
        "greedy agreement with llama.cpp: {prefix}/{compared} tokens before first divergence ({:.0}%)",
        agree * 100.0
    );
    println!(
        "speed: {:.2} tokens/s ({:.1}s for {} steps)",
        (toks.len() + n) as f64 / secs,
        secs,
        toks.len() + n
    );
    println!("evelyn   : {out:?}\nllama.cpp: {ref_tokens:?}");
    let pass = agree >= 0.9;
    println!(
        "GATE {}",
        if pass {
            "PASS (>= 90% greedy agreement)"
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
            eprintln!("evelyn-generate: {e}");
            ExitCode::from(2)
        }
    }
}
