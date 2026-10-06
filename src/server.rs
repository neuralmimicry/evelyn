//! OpenAI-compatible shadow endpoint for Gail (Evelyn stage 5).
//!
//! The endpoint uses the Rust transformer runtime and the configured AARNN
//! route manifest. It is deliberately an explicit provider alias in Gail so
//! shadow traffic cannot replace existing routes. Dense FFN is retained as a
//! diagnostic fallback inside the runtime, but a request that needed any such
//! fallback is rejected and is never returned to Gail.

use crate::remote::RemoteAarnnFfn;
use crate::runtime::{Model, Session, argmax};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_HTTP_BODY: usize = 2 * 1024 * 1024;
const DEFAULT_MAX_TOKENS: usize = 512;
const HARD_MAX_TOKENS: usize = 2048;
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct ServerConfig {
    pub bind: String,
    pub tokenizer_url: String,
    pub model_id: String,
    pub api_key: String,
    pub max_tokens: usize,
    pub routes: BTreeMap<usize, String>,
}

struct AppState {
    model: Arc<Model>,
    backend: RemoteAarnnFfn<'static>,
    tokenizer_url: String,
    model_id: String,
    api_key: String,
    max_tokens: usize,
    inference_lock: Mutex<()>,
}

/// Read and validate Evelyn's versioned AARNN route manifest.
pub fn read_routes(path: &str) -> io::Result<BTreeMap<usize, String>> {
    let manifest: Value = serde_json::from_slice(&std::fs::read(path)?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if manifest["schema_version"].as_u64() != Some(1) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "route manifest schema_version must be 1",
        ));
    }
    let route_values = manifest["routes"].as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "route manifest needs a routes object",
        )
    })?;
    let mut routes = BTreeMap::new();
    for (layer, endpoint) in route_values {
        let layer = layer.parse::<usize>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid layer index: {error}"),
            )
        })?;
        let endpoint = endpoint
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("layer {layer} has no endpoint"),
                )
            })?;
        if !endpoint
            .rsplit_once(':')
            .is_some_and(|(host, port)| !host.is_empty() && port.parse::<u16>().is_ok())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("layer {layer} endpoint must be host:port"),
            ));
        }
        routes.insert(layer, endpoint.to_string());
    }
    if routes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "route manifest is empty",
        ));
    }
    Ok(routes)
}

/// The stage-5 route must retain the multi-host property verified in stage 4c.
pub fn validate_multihost(routes: &BTreeMap<usize, String>) -> io::Result<()> {
    let hosts: BTreeSet<&str> = routes
        .values()
        .filter_map(|endpoint| endpoint.rsplit_once(':').map(|(host, _)| host))
        .collect();
    if hosts.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Evelyn shadow serving requires AARNN routes on at least two hosts",
        ));
    }
    Ok(())
}

/// Start the blocking HTTP service. Requests are serialized because the first
/// stage-5 target is a CPU-bound 9B runtime with one Gail admission slot.
pub fn serve(model: Model, config: ServerConfig) -> io::Result<()> {
    validate_multihost(&config.routes)?;
    if config.api_key.trim().len() < 32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "EVELYN_API_KEY must contain at least 32 bytes",
        ));
    }
    if config.routes.keys().any(|layer| *layer >= model.cfg.layers) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "route manifest contains a layer outside the loaded model",
        ));
    }
    let listener = TcpListener::bind(&config.bind)?;
    eprintln!(
        "Evelyn shadow provider listening on {} for {} ({} routed layers, {} hosts)",
        config.bind,
        config.model_id,
        config.routes.len(),
        config
            .routes
            .values()
            .filter_map(|route| route.rsplit_once(':').map(|(host, _)| host))
            .collect::<BTreeSet<_>>()
            .len()
    );
    let model = Arc::new(model);
    let backend = RemoteAarnnFfn::new_shared(
        Arc::clone(&model),
        config.routes.clone(),
        Duration::from_secs(120),
    );
    let state = Arc::new(AppState {
        model,
        backend,
        tokenizer_url: config.tokenizer_url.trim_end_matches('/').to_string(),
        model_id: config.model_id,
        api_key: config.api_key,
        max_tokens: config.max_tokens.clamp(1, HARD_MAX_TOKENS),
        inference_lock: Mutex::new(()),
    });
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, &state) {
                        eprintln!("Evelyn HTTP request failed: {error}");
                    }
                });
            }
            Err(error) => eprintln!("Evelyn listener accept failed: {error}"),
        }
    }
    Ok(())
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> io::Result<HttpRequest> {
    let mut reader = BufReader::new(stream.try_clone()?);
    reader
        .get_ref()
        .set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    if method.is_empty() || path.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "malformed request line",
        ));
    }
    let mut headers = BTreeMap::new();
    let mut header_bytes = line.len();
    loop {
        line.clear();
        let count = reader.read_line(&mut line)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete HTTP headers",
            ));
        }
        header_bytes += count;
        if header_bytes > 32 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP headers too large",
            ));
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    if headers.contains_key("transfer-encoding") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "chunked requests are unsupported",
        ));
    }
    let body_len = headers
        .get("content-length")
        .map_or(Ok(0), |value| value.parse::<usize>())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if body_len > MAX_HTTP_BODY {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request body too large",
        ));
    }
    let mut body = vec![0; body_len];
    reader.read_exact(&mut body)?;
    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn handle_connection(mut stream: TcpStream, state: &AppState) -> io::Result<()> {
    let response = match read_request(&stream) {
        Ok(request) => route_request(request, state),
        Err(error) => HttpResponse::error(400, error.to_string()),
    };
    let body = serde_json::to_vec(&response.body).map_err(io::Error::other)?;
    let reason = match response.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        reason,
        body.len()
    )?;
    stream.write_all(&body)
}

struct HttpResponse {
    status: u16,
    body: Value,
}

impl HttpResponse {
    fn error(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            body: json!({"error": {"message": message.into(), "type": "evelyn_error"}}),
        }
    }
}

fn route_request(request: HttpRequest, state: &AppState) -> HttpResponse {
    if request.method == "GET" && request.path == "/healthz" {
        return HttpResponse {
            status: 200,
            body: json!({"status": "ok", "model": state.model_id}),
        };
    }
    if !authorized(request.headers.get("authorization"), &state.api_key) {
        return HttpResponse::error(401, "invalid Evelyn API key");
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models") => HttpResponse {
            status: 200,
            body: json!({"object":"list","data":[{"id":state.model_id,"object":"model","owned_by":"neuralmimicry"}]}),
        },
        ("POST", "/v1/chat/completions") => match complete(&request.body, state) {
            Ok(body) => HttpResponse { status: 200, body },
            Err((status, message)) => HttpResponse::error(status, message),
        },
        _ => HttpResponse::error(404, "unknown Evelyn endpoint"),
    }
}

fn authorized(header: Option<&String>, expected: &str) -> bool {
    let Some(supplied) = header.and_then(|value| value.strip_prefix("Bearer ")) else {
        return false;
    };
    constant_time_eq(supplied.as_bytes(), expected.as_bytes())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for i in 0..left.len().max(right.len()) {
        difference |=
            (left.get(i).copied().unwrap_or(0) ^ right.get(i).copied().unwrap_or(0)) as usize;
    }
    difference == 0
}

fn sample_token(logits: &[f32], temperature: f32, state: &mut u64) -> u32 {
    if temperature <= f32::EPSILON {
        return argmax(logits);
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|value| ((*value - maximum) / temperature).exp() as f64)
        .collect();
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return argmax(logits);
    }
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let unit = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64;
    let threshold = unit * total;
    let mut cumulative = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative >= threshold {
            return index as u32;
        }
    }
    argmax(logits)
}

/// Render the text-only Qwen chat template. Image/audio/tool messages are
/// rejected instead of being silently degraded to text.
pub fn render_chat_prompt(messages: &[Value]) -> Result<String, String> {
    if messages.is_empty() {
        return Err("messages must be a non-empty array".into());
    }
    let mut prompt = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or("message role is required")?;
        if !matches!(role, "system" | "user" | "assistant") {
            return Err(format!("unsupported chat role `{role}`"));
        }
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or("message content must be plain text")?;
        if content.len() > 256 * 1024 {
            return Err("message content is too large".into());
        }
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

fn add_system_instruction(messages: &mut Vec<Value>, instruction: &str) -> Result<(), String> {
    if let Some(system) = messages
        .iter_mut()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("system"))
    {
        let content = system
            .get("content")
            .and_then(Value::as_str)
            .ok_or("system message content must be plain text")?;
        system["content"] = json!(format!("{content}\n{instruction}"));
    } else {
        messages.insert(0, json!({"role":"system","content":instruction}));
    }
    Ok(())
}

fn complete(body: &[u8], state: &AppState) -> Result<Value, (u16, String)> {
    let request: Value = serde_json::from_slice(body).map_err(|error| (400, error.to_string()))?;
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_id);
    if model != state.model_id
        && model
            != state
                .model_id
                .strip_prefix("evelyn/")
                .unwrap_or(&state.model_id)
    {
        return Err((400, format!("unsupported model `{model}`")));
    }
    if request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err((
            400,
            "streaming responses are not supported by Evelyn shadow serving".into(),
        ));
    }
    let mut messages = request
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .ok_or((400, "messages must be an array".into()))?;
    let response_format = request
        .get("response_format")
        .filter(|value| !value.is_null());
    let json_object = match response_format
        .and_then(|format| format.get("type"))
        .and_then(Value::as_str)
    {
        None => false,
        Some("json_object") => true,
        Some(value) => return Err((400, format!("unsupported response format `{value}`"))),
    };
    if request["chat_template_kwargs"]["enable_thinking"] == false {
        add_system_instruction(&mut messages, "/no_think").map_err(|message| (400, message))?;
    }
    if json_object {
        add_system_instruction(
            &mut messages,
            "Return only one valid JSON object without markdown or commentary.",
        )
        .map_err(|message| (400, message))?;
    }
    let prompt = render_chat_prompt(&messages).map_err(|message| (400, message))?;
    let max_tokens = request
        .get("max_tokens")
        .or_else(|| request.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map_or(DEFAULT_MAX_TOKENS, |value| value as usize)
        .clamp(1, state.max_tokens);
    let temperature = request
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.2) as f32;
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err((400, "temperature must be between 0 and 2".into()));
    }

    let _guard = state
        .inference_lock
        .lock()
        .map_err(|_| (503, "Evelyn inference lock is unavailable".into()))?;
    let started = Instant::now();
    let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let mut random_state = (id
        ^ std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64)
        .max(1);
    let tokenized = ureq::post(&format!("{}/tokenize", state.tokenizer_url))
        .timeout(Duration::from_secs(120))
        .send_json(json!({"content":prompt,"add_special":false,"parse_special":true}))
        .map_err(|error| (503, format!("tokenizer unavailable: {error}")))?
        .into_json::<Value>()
        .map_err(|error| (503, format!("tokenizer returned invalid JSON: {error}")))?;
    let tokens = tokenized["tokens"]
        .as_array()
        .ok_or((503, "tokenizer response has no token array".into()))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or((503, "tokenizer returned an invalid token ID".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err((400, "prompt tokenized to an empty sequence".into()));
    }

    let initial_fallbacks = *state
        .backend
        .fallbacks
        .lock()
        .map_err(|_| (503, "AARNN fallback counter unavailable".into()))?;
    let mut session = Session::new(&state.model);
    let mut logits = Vec::new();
    for token in &tokens {
        logits = session
            .step(*token, &state.backend)
            .map_err(|error| (503, format!("Evelyn prompt inference failed: {error}")))?;
    }
    let mut generated = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let next = sample_token(&logits, temperature, &mut random_state);
        if state.model.cfg.eos_token_id == Some(next) {
            break;
        }
        generated.push(next);
        logits = session
            .step(next, &state.backend)
            .map_err(|error| (503, format!("Evelyn generation failed: {error}")))?;
    }
    let fallbacks = *state
        .backend
        .fallbacks
        .lock()
        .map_err(|_| (503, "AARNN fallback counter unavailable".into()))?
        - initial_fallbacks;
    if fallbacks != 0 {
        return Err((
            503,
            format!("AARNN unavailable; rejected output after {fallbacks} dense fallback(s)"),
        ));
    }
    let detokenized = ureq::post(&format!("{}/detokenize", state.tokenizer_url))
        .timeout(Duration::from_secs(120))
        .send_json(json!({"tokens":generated}))
        .map_err(|error| (503, format!("tokenizer detokenize failed: {error}")))?
        .into_json::<Value>()
        .map_err(|error| {
            (
                503,
                format!("tokenizer returned invalid detokenize JSON: {error}"),
            )
        })?;
    let content = detokenized["content"]
        .as_str()
        .ok_or((503, "detokenize response has no content".into()))?;
    if json_object {
        let parsed: Value = serde_json::from_str(content.trim())
            .map_err(|_| (503, "model did not satisfy JSON object output mode".into()))?;
        if !parsed.is_object() {
            return Err((503, "model JSON output is not an object".into()));
        }
    }
    eprintln!(
        "evelyn shadow completion id={id} prompt_tokens={} output_tokens={} elapsed_ms={} aarnn_fallbacks=0",
        tokens.len(),
        generated.len(),
        started.elapsed().as_millis()
    );
    Ok(json!({
        "id": format!("chatcmpl-evelyn-{id}"),
        "object": "chat.completion",
        "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model": state.model_id,
        "choices": [{"index":0,"message":{"role":"assistant","content":content},"finish_reason":if generated.len() < max_tokens {"stop"} else {"length"}}],
        "usage": {"prompt_tokens":tokens.len(),"completion_tokens":generated.len(),"total_tokens":tokens.len()+generated.len()}
    }))
}

#[cfg(test)]
mod tests {
    use super::{authorized, render_chat_prompt, sample_token, validate_multihost};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::io;

    #[test]
    fn qwen_template_keeps_roles_and_opens_assistant_turn() {
        let prompt = render_chat_prompt(&[
            json!({"role":"system","content":"Be concise."}),
            json!({"role":"user","content":"Hi"}),
        ])
        .unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn system_instruction_is_added_without_dropping_existing_policy() {
        let mut messages = vec![
            json!({"role":"system","content":"Follow policy."}),
            json!({"role":"user","content":"Hi"}),
        ];
        super::add_system_instruction(&mut messages, "/no_think").unwrap();
        assert_eq!(messages[0]["content"], "Follow policy.\n/no_think");
    }

    #[test]
    fn multimodal_and_tool_messages_are_rejected_explicitly() {
        assert!(
            render_chat_prompt(&[json!({"role":"user","content":[{"type":"text","text":"Hi"}]})])
                .is_err()
        );
        assert!(render_chat_prompt(&[json!({"role":"tool","content":"done"})]).is_err());
    }

    #[test]
    fn stage_five_requires_routes_on_two_hosts() {
        let one = BTreeMap::from([
            (18, "qc04:38118".to_string()),
            (20, "qc04:38120".to_string()),
        ]);
        assert_eq!(
            validate_multihost(&one).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let two = BTreeMap::from([
            (18, "qc04:38118".to_string()),
            (19, "qc05:38119".to_string()),
        ]);
        assert!(validate_multihost(&two).is_ok());
    }

    #[test]
    fn only_the_configured_bearer_key_is_accepted() {
        let good = "Bearer stage-five-test-key".to_string();
        let wrong = "Bearer another-key".to_string();
        assert!(authorized(Some(&good), "stage-five-test-key"));
        assert!(!authorized(Some(&wrong), "stage-five-test-key"));
        assert!(!authorized(None, "stage-five-test-key"));
    }

    #[test]
    fn zero_temperature_uses_greedy_decoding() {
        let mut seed = 1;
        assert_eq!(sample_token(&[1.0, 3.0, 2.0], 0.0, &mut seed), 1);
    }
}
