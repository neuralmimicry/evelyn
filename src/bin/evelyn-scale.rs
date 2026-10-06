//! `evelyn-scale <model.gguf> <llama-server-url> <text-file> <tokens> <routes.json> <layer-counts>`
//!
//! Stage 4c sweep. It scores the same teacher-forced text with progressively
//! larger sets of AARNN-served layers, then reports perplexity and the latency
//! of every remote layer. The largest sweep must span at least two hosts.

use evelyn::remote::RemoteAarnnFfn;
use evelyn::runtime::{DenseFfn, FfnBackend, Model, Session, argmax, log_prob, validate_token_ids};
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;
use std::time::{Duration, Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteManifest {
    schema_version: u32,
    routes: BTreeMap<String, String>,
}

struct Score {
    perplexity: f64,
    top1: Vec<u32>,
    elapsed: Duration,
}

fn score(model: &Model, tokens: &[u32], ffn: &dyn FfnBackend) -> Result<Score, String> {
    let mut session = Session::new(model);
    let mut nll = 0.0f64;
    let mut top1 = Vec::with_capacity(tokens.len());
    let started = Instant::now();
    for (i, token) in tokens.iter().enumerate() {
        let logits = session.step(*token, ffn).map_err(|e| e.to_string())?;
        top1.push(argmax(&logits));
        if let Some(next) = tokens.get(i + 1) {
            nll -= log_prob(&logits, *next) as f64;
        }
    }
    Ok(Score {
        perplexity: (nll / (tokens.len() - 1) as f64).exp(),
        top1,
        elapsed: started.elapsed(),
    })
}

fn endpoint_host(endpoint: &str) -> Result<&str, String> {
    if endpoint.trim() != endpoint || endpoint.contains(char::is_whitespace) {
        return Err(format!("endpoint contains whitespace: {endpoint:?}"));
    }
    let (host, port) = if let Some(rest) = endpoint.strip_prefix('[') {
        let end = rest
            .find(']')
            .ok_or_else(|| format!("invalid bracketed endpoint: {endpoint:?}"))?;
        let host = &rest[..end];
        let suffix = &rest[end + 1..];
        let port = suffix
            .strip_prefix(':')
            .ok_or_else(|| format!("endpoint has no port: {endpoint:?}"))?;
        (host, port)
    } else {
        endpoint
            .rsplit_once(':')
            .ok_or_else(|| format!("endpoint must be host:port: {endpoint:?}"))?
    };
    let port: u16 = port
        .parse()
        .map_err(|_| format!("invalid endpoint port in {endpoint:?}"))?;
    if host.is_empty() || port == 0 {
        return Err(format!("invalid endpoint: {endpoint:?}"));
    }
    Ok(host)
}

fn parse_routes(path: &str) -> Result<BTreeMap<usize, String>, String> {
    let source = std::fs::read_to_string(path).map_err(|e| format!("routes: {e}"))?;
    let manifest: RouteManifest =
        serde_json::from_str(&source).map_err(|e| format!("routes: {e}"))?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported route schema {}; expected 1",
            manifest.schema_version
        ));
    }
    if manifest.routes.is_empty() {
        return Err("route manifest has no layers".into());
    }
    let mut routes = BTreeMap::new();
    let mut endpoints = BTreeSet::new();
    for (layer, endpoint) in manifest.routes {
        let layer = layer
            .parse::<usize>()
            .map_err(|_| format!("invalid layer index {layer:?}"))?;
        endpoint_host(&endpoint)?;
        if routes.contains_key(&layer) {
            return Err(format!("duplicate route for layer {layer}"));
        }
        if !endpoints.insert(endpoint.clone()) {
            return Err(format!(
                "endpoint {endpoint:?} is assigned to multiple layers"
            ));
        }
        routes.insert(layer, endpoint);
    }
    Ok(routes)
}

fn parse_counts(source: &str, available: usize) -> Result<Vec<usize>, String> {
    let mut counts = source
        .split(',')
        .map(|part| {
            let count = part
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid layer count {part:?}"))?;
            if count == 0 || count > available {
                return Err(format!("layer count {count} is outside 1..={available}"));
            }
            Ok(count)
        })
        .collect::<Result<Vec<_>, String>>()?;
    counts.sort_unstable();
    counts.dedup();
    if counts.is_empty() {
        return Err("specify at least one layer count".into());
    }
    Ok(counts)
}

fn run(args: &[String]) -> Result<bool, String> {
    if args.len() != 7 {
        return Err("usage: evelyn-scale <model.gguf> <llama-server-url> <text-file> <tokens> <routes.json> <layer-counts>".into());
    }
    let token_limit: usize = args[4].parse().map_err(|_| "invalid token count")?;
    let routes = parse_routes(&args[5])?;
    let counts = parse_counts(&args[6], routes.len())?;
    let max_count = *counts.last().expect("non-empty counts");
    let hosts: BTreeSet<&str> = routes
        .values()
        .take(max_count)
        .map(|endpoint| endpoint_host(endpoint))
        .collect::<Result<_, _>>()?;
    if hosts.len() < 2 {
        return Err(format!(
            "largest sweep uses {} host; stage 4c requires routes across at least two hosts",
            hosts.len()
        ));
    }

    let text = std::fs::read_to_string(&args[3]).map_err(|e| format!("text: {e}"))?;
    let http = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build();
    let tokens: Vec<u32> = http
        .post(&format!("{}/tokenize", args[2].trim_end_matches('/')))
        .send_json(json!({ "content": text, "add_special": false }))
        .map_err(|e| format!("tokenize: {e}"))?
        .into_json::<serde_json::Value>()
        .map_err(|e| e.to_string())?["tokens"]
        .as_array()
        .ok_or("tokenize: no tokens")?
        .iter()
        .filter_map(|token| token.as_u64().map(|v| v as u32))
        .take(token_limit)
        .collect();
    if tokens.len() < 2 {
        return Err(format!(
            "need at least two tokens; tokenizer returned {}",
            tokens.len()
        ));
    }

    let model = Model::load(&args[1]).map_err(|e| format!("load: {e}"))?;
    validate_token_ids(&tokens, model.cfg.vocab).map_err(|e| e.to_string())?;
    for layer in routes.keys() {
        if *layer >= model.cfg.layers {
            return Err(format!(
                "route references layer {layer}, but model has {} layers",
                model.cfg.layers
            ));
        }
    }

    let dense = score(&model, &tokens, &DenseFfn(&model))?;
    println!(
        "dense baseline: ppl={:.5}, {:.3}s/token over {} tokens",
        dense.perplexity,
        dense.elapsed.as_secs_f64() / tokens.len() as f64,
        tokens.len()
    );
    println!(
        "N,remote_layers,hosts,ppl,delta_ppl_pct,top1_agreement_pct,seconds_per_token,retries,fallbacks,gate"
    );

    let mut all_pass = true;
    let mut fastest_multihost: Option<(f64, usize)> = None;
    for count in counts {
        let selected: BTreeMap<usize, String> = routes
            .iter()
            .take(count)
            .map(|(layer, endpoint)| (*layer, endpoint.clone()))
            .collect();
        let selected_hosts: BTreeSet<String> = selected
            .values()
            .map(|endpoint| endpoint_host(endpoint).map(str::to_owned))
            .collect::<Result<_, _>>()?;
        let remote = RemoteAarnnFfn::new(&model, selected, Duration::from_secs(900));
        let candidate = score(&model, &tokens, &remote)?;
        let delta = candidate.perplexity / dense.perplexity - 1.0;
        let agreement = dense
            .top1
            .iter()
            .zip(&candidate.top1)
            .filter(|(a, b)| a == b)
            .count() as f64
            / tokens.len() as f64;
        let fallbacks = *remote
            .fallbacks
            .lock()
            .map_err(|_| "fallback lock poisoned")?;
        let stats = remote.endpoint_stats();
        let retries: u64 = stats.values().map(|(_, s)| s.retries).sum();
        let seconds_per_token = candidate.elapsed.as_secs_f64() / tokens.len() as f64;
        let pass = delta.abs() < 0.05 && fallbacks == 0;
        all_pass &= pass;
        if pass && count >= 2 && selected_hosts.len() >= 2 {
            fastest_multihost = Some(match fastest_multihost {
                Some((best_time, best_count)) if best_time <= seconds_per_token => {
                    (best_time, best_count)
                }
                _ => (seconds_per_token, count),
            });
        }
        println!(
            "{count},{count},{},{:.5},{:+.3},{:.2},{seconds_per_token:.3},{retries},{fallbacks},{}",
            selected_hosts.len(),
            candidate.perplexity,
            delta * 100.0,
            agreement * 100.0,
            if pass { "PASS" } else { "FAIL" }
        );
        for (layer, (endpoint, stat)) in stats {
            let host = endpoint_host(&endpoint)?;
            let mean_ms = if stat.calls == 0 {
                0.0
            } else {
                stat.total_ns as f64 / stat.calls as f64 / 1_000_000.0
            };
            println!(
                "  layer={layer} host={host} calls={} mean={mean_ms:.2}ms max={:.2}ms retries={} failures={}",
                stat.calls,
                stat.max_ns as f64 / 1_000_000.0,
                stat.retries,
                stat.failures
            );
        }
    }
    println!(
        "4c GATE {} (absolute ΔPPL < 5%, zero dense fallbacks at every N; largest sweep spans {} hosts)",
        if all_pass { "PASS" } else { "FAIL" },
        hosts.len()
    );
    match fastest_multihost {
        Some((seconds, layers)) => {
            println!("FASTEST passing multi-host point: N={layers}, {seconds:.3}s/token")
        }
        None => println!("FASTEST passing multi-host point: none"),
    }
    Ok(all_pass)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("evelyn-scale: {error}");
            ExitCode::from(2)
        }
    }
}
