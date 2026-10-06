//! `evelyn-serve <model.gguf> <tokenizer-url> <routes.json> [bind]`
//! OpenAI-compatible, authenticated shadow endpoint consumed by Gail.

use evelyn::runtime::Model;
use evelyn::server::{ServerConfig, read_routes, serve};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::ExitCode;
use std::time::Duration;

fn healthcheck() -> Result<(), String> {
    healthcheck_at(SocketAddr::from(([127, 0, 0, 1], 8080)))
}

fn healthcheck_at(address: SocketAddr) -> Result<(), String> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))
        .map_err(|error| format!("health check connection: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("health check timeout: {error}"))?;
    stream
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .map_err(|error| format!("health check request: {error}"))?;
    let mut status = String::new();
    BufReader::new(stream)
        .read_line(&mut status)
        .map_err(|error| format!("health check response: {error}"))?;
    if status.starts_with("HTTP/1.1 200 ") || status.starts_with("HTTP/1.0 200 ") {
        Ok(())
    } else {
        Err(format!("health check returned {status:?}"))
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.get(1).is_some_and(|arg| arg == "--healthcheck") && args.len() == 2 {
        return healthcheck();
    }
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

#[cfg(test)]
mod tests {
    use super::healthcheck_at;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn healthcheck_with_response(status: &'static str) -> Result<(), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 128];
            let _ = stream.read(&mut request).unwrap();
            write!(stream, "{status}\r\nContent-Length: 0\r\n\r\n").unwrap();
        });

        let result = healthcheck_at(address);
        server.join().unwrap();
        result
    }

    #[test]
    fn healthcheck_accepts_http_200() {
        assert!(healthcheck_with_response("HTTP/1.1 200 OK").is_ok());
    }

    #[test]
    fn healthcheck_rejects_http_503() {
        assert!(healthcheck_with_response("HTTP/1.1 503 Unavailable").is_err());
    }
}
