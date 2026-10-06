//! `evelyn-serve <model.gguf> <tokenizer-url> <routes.json> [bind]`
//! OpenAI-compatible, authenticated shadow endpoint consumed by Gail.

use evelyn::runtime::Model;
use evelyn::server::{ServerConfig, read_routes, serve};
use std::process::ExitCode;

fn run(args: &[String]) -> Result<(), String> {
    if args.len() < 4 || args.len() > 5 {
        return Err("usage: evelyn-serve <model.gguf> <tokenizer-url> <routes.json> [bind]".into());
    }
    let routes = read_routes(&args[3]).map_err(|error| format!("route manifest: {error}"))?;
    let api_key =
        std::env::var("EVELYN_API_KEY").map_err(|_| "EVELYN_API_KEY is required".to_string())?;
    let max_tokens = std::env::var("EVELYN_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let model = Model::load(&args[1]).map_err(|error| format!("model load: {error}"))?;
    let model_id =
        std::env::var("EVELYN_MODEL_ID").unwrap_or_else(|_| "evelyn/qwen3.5-9b-aarnn".to_string());
    let config = ServerConfig {
        bind: args
            .get(4)
            .cloned()
            .unwrap_or_else(|| "0.0.0.0:8080".to_string()),
        tokenizer_url: args[2].clone(),
        model_id,
        api_key,
        max_tokens,
        routes,
    };
    serve(model, config).map_err(|error| format!("server: {error}"))
}

fn main() -> ExitCode {
    match run(&std::env::args().collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("evelyn-serve: {error}");
            ExitCode::FAILURE
        }
    }
}
