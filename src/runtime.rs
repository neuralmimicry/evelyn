//! Stage 4: Evelyn's transformer runtime (pure Rust, from GGUF).
//!
//! It runs the standard dense decoder (Llama/Qwen3 family): token
//! embeddings, then per layer RMSNorm, Q/K/V projections, optional per-head
//! Q/K RMSNorm, rotary position embedding, grouped-query causal attention
//! with a KV cache, an output projection, and a feed-forward block. A final
//! norm and the language-model head follow.
//!
//! Weights stay in their GGUF quantised form in memory and are dequantised
//! row by row inside a parallel matrix-vector product, so an 8 B model needs
//! about its file size in RAM rather than 32 GB of f32.
//!
//! The feed-forward block is pluggable ([`FfnBackend`]). Any layer can be
//! served by AARNN instead of the dense weights. That substitution is what
//! Evelyn is for, and stage 4b measures its effect on perplexity.

use crate::activation::Activation;
use crate::gguf::{Gguf, TensorInfo, Value, dequantize, type_bytes};
use rayon::prelude::*;
use std::io;

/// A quantised weight matrix `[rows = outputs][cols = inputs]`.
pub struct QMatrix {
    pub rows: usize,
    pub cols: usize,
    ggml_type: u32,
    row_bytes: usize,
    data: Vec<u8>,
}

impl QMatrix {
    fn from_raw(t: &TensorInfo, data: Vec<u8>) -> io::Result<Self> {
        let (cols, rows) = match t.dims.as_slice() {
            [c, r] => (*c as usize, *r as usize),
            [c] => (*c as usize, 1),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}: unsupported rank", t.name),
                ));
            }
        };
        let row_bytes = type_bytes(t.ggml_type, cols)?;
        Ok(Self {
            rows,
            cols,
            ggml_type: t.ggml_type,
            row_bytes,
            data,
        })
    }

    /// One dequantised row (used for embedding lookups).
    pub fn row(&self, r: usize) -> io::Result<Vec<f32>> {
        if r >= self.rows {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "embedding row {r} is outside matrix vocabulary {}",
                    self.rows
                ),
            ));
        }
        dequantize(
            self.ggml_type,
            &self.data[r * self.row_bytes..(r + 1) * self.row_bytes],
            self.cols,
        )
    }

    /// `y = W x`, parallel over output rows.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0f32; self.rows];
        y.par_chunks_mut(64).enumerate().for_each(|(chunk, out)| {
            for (j, o) in out.iter_mut().enumerate() {
                let r = chunk * 64 + j;
                let row = dequantize(
                    self.ggml_type,
                    &self.data[r * self.row_bytes..(r + 1) * self.row_bytes],
                    self.cols,
                )
                .expect("row size validated at load");
                *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        });
        y
    }
}

/// Model hyper-parameters read from GGUF metadata.
#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub layers: usize,
    pub dim: usize,
    pub ffn: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub rope_dim: usize,
    pub rope_interleaved: bool,
    pub eps: f32,
    pub vocab: usize,
    pub full_attention_interval: usize,
    pub ssm: Option<SsmConfig>,
    /// End-of-sequence token from GGUF tokenizer metadata, when present.
    pub eos_token_id: Option<u32>,
}

/// Hyper-parameters for the Qwen3.5 gated-delta (linear-attention) blocks.
#[derive(Clone, Copy, Debug)]
pub struct SsmConfig {
    pub conv_kernel: usize,
    pub inner: usize,
    pub state: usize,
    pub groups: usize,
    pub value_heads: usize,
}

/// Reject token IDs from a tokenizer whose vocabulary does not match the
/// loaded GGUF. This keeps a model/tokenizer mismatch from becoming an index
/// panic during embedding lookup.
pub fn validate_token_ids(tokens: &[u32], vocab: usize) -> io::Result<()> {
    if let Some(id) = tokens.iter().find(|&&id| id as usize >= vocab) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "token ID {id} is outside model vocabulary ({vocab}); check that tokenizer and GGUF match"
            ),
        ));
    }
    Ok(())
}

/// Feed-forward provider for one layer: dense weights or an AARNN region.
pub trait FfnBackend: Send + Sync {
    fn ffn(&self, layer: usize, x: &[f32]) -> Vec<f32>;
}

enum Attention {
    Full {
        q: QMatrix,
        k: QMatrix,
        v: QMatrix,
        o: QMatrix,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        gated: bool,
    },
    GatedDelta(GatedDelta),
}

struct GatedDelta {
    qkv: QMatrix,
    z: QMatrix,
    conv: Vec<f32>,
    alpha: QMatrix,
    beta: QMatrix,
    a: Vec<f32>,
    dt: Vec<f32>,
    norm: Vec<f32>,
    out: QMatrix,
    conv_kernel: usize,
    key_dim: usize,
    key_heads: usize,
    value_dim: usize,
    value_heads: usize,
}

struct Layer {
    attn_norm: Vec<f32>,
    attention: Attention,
    /// Qwen3.5 calls this `post_attention_norm`; other supported models use
    /// `ffn_norm`. Both are applied to the residual before the FFN.
    ffn_norm: Vec<f32>,
    gate: QMatrix,
    up: QMatrix,
    down: QMatrix,
}

/// The loaded model (weights shared, cache per [`Session`]).
pub struct Model {
    pub cfg: Config,
    pub activation: Activation,
    embed: QMatrix,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    lm_head: QMatrix,
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

impl Model {
    pub fn load(path: &str) -> io::Result<Self> {
        let mut g = Gguf::open(path)?;
        let arch = g.architecture().unwrap_or("llama").to_string();
        let need = |g: &Gguf, k: &str| {
            g.arch_u64(k).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, format!("missing {arch}.{k}"))
            })
        };
        let dim = need(&g, "embedding_length")? as usize;
        let heads = need(&g, "attention.head_count")? as usize;
        let kv_heads = g
            .arch_u64("attention.head_count_kv")
            .or_else(|| {
                g.meta(&format!("{arch}.attention.head_count_kv"))
                    .and_then(|value| match value {
                        Value::Array(values) => values
                            .iter()
                            .filter_map(Value::as_u64)
                            .find(|&value| value > 0),
                        _ => None,
                    })
            })
            .unwrap_or(heads as u64) as usize;
        let head_dim = g
            .arch_u64("attention.key_length")
            .unwrap_or((dim / heads) as u64) as usize;
        let ssm = if arch == "qwen35" {
            Some(SsmConfig {
                conv_kernel: need(&g, "ssm.conv_kernel")? as usize,
                inner: need(&g, "ssm.inner_size")? as usize,
                state: need(&g, "ssm.state_size")? as usize,
                groups: need(&g, "ssm.group_count")? as usize,
                value_heads: need(&g, "ssm.time_step_rank")? as usize,
            })
        } else {
            None
        };
        let cfg = Config {
            layers: need(&g, "block_count")? as usize,
            dim,
            ffn: need(&g, "feed_forward_length")? as usize,
            heads,
            kv_heads,
            head_dim,
            rope_base: g
                .meta(&format!("{arch}.rope.freq_base"))
                .and_then(Value::as_f64)
                .unwrap_or(10000.0) as f32,
            rope_dim: g
                .arch_u64("rope.dimension_count")
                .unwrap_or(head_dim as u64) as usize,
            rope_interleaved: matches!(
                g.meta(&format!("{arch}.rope.mrope_interleaved")),
                Some(Value::Bool(true))
            ),
            eps: g
                .meta(&format!("{arch}.attention.layer_norm_rms_epsilon"))
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            full_attention_interval: g.arch_u64("full_attention_interval").unwrap_or(4) as usize,
            ssm,
            vocab: g
                .tensors
                .get("token_embd.weight")
                .map_or(0, |t| t.dims[1] as usize),
            eos_token_id: g
                .meta("tokenizer.ggml.eos_token_id")
                .and_then(Value::as_u64)
                .and_then(|id| u32::try_from(id).ok()),
            arch: arch.clone(),
        };
        let mat = |g: &mut Gguf, n: &str| -> io::Result<QMatrix> {
            let (t, raw) = g.tensor_raw(n)?;
            QMatrix::from_raw(&t, raw)
        };
        let embed = mat(&mut g, "token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("blk.{i}.");
            let has_gated_delta = g.tensors.contains_key(&format!("{p}attn_qkv.weight"));
            if arch == "qwen35" && cfg.full_attention_interval > 0 {
                let expected_gated_delta = (i + 1) % cfg.full_attention_interval != 0;
                if has_gated_delta != expected_gated_delta {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{p}attention type does not match qwen35.full_attention_interval={}",
                            cfg.full_attention_interval
                        ),
                    ));
                }
            }
            let opt = |g: &mut Gguf, n: &str| {
                if g.tensors.contains_key(n) {
                    g.tensor_f32(n).map(Some)
                } else {
                    Ok(None)
                }
            };
            let attention = if has_gated_delta {
                let s = cfg.ssm.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "gated-delta tensors found without SSM architecture metadata",
                    )
                })?;
                let key_dim = s.state;
                let key_heads = s.groups;
                let value_heads = s.value_heads;
                if value_heads == 0 || s.inner % value_heads != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Qwen3.5 SSM inner size must divide evenly across value heads",
                    ));
                }
                let value_dim = s.inner / value_heads;
                let qkv = mat(&mut g, &format!("{p}attn_qkv.weight"))?;
                let z = mat(&mut g, &format!("{p}attn_gate.weight"))?;
                let conv = g.tensor_f32(&format!("{p}ssm_conv1d.weight"))?;
                let alpha = mat(&mut g, &format!("{p}ssm_alpha.weight"))?;
                let beta = mat(&mut g, &format!("{p}ssm_beta.weight"))?;
                let a = g.tensor_f32(&format!("{p}ssm_a"))?;
                let dt = g.tensor_f32(&format!("{p}ssm_dt"))?;
                let norm = g.tensor_f32(&format!("{p}ssm_norm.weight"))?;
                let out = mat(&mut g, &format!("{p}ssm_out.weight"))?;
                let conv_channels = 2 * key_dim * key_heads + value_dim * value_heads;
                if qkv.rows != conv_channels
                    || z.rows != value_dim * value_heads
                    || conv.len() != s.conv_kernel * conv_channels
                    || alpha.rows != value_heads
                    || beta.rows != value_heads
                    || a.len() != value_heads
                    || dt.len() != value_heads
                    || norm.len() != value_dim
                    || out.cols != value_dim * value_heads
                    || out.rows != dim
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{p}gated-delta tensor shapes do not match Qwen3.5 metadata"),
                    ));
                }
                Attention::GatedDelta(GatedDelta {
                    qkv,
                    z,
                    conv,
                    alpha,
                    beta,
                    a,
                    dt,
                    norm,
                    out,
                    conv_kernel: s.conv_kernel,
                    key_dim,
                    key_heads,
                    value_dim,
                    value_heads,
                })
            } else {
                let q = mat(&mut g, &format!("{p}attn_q.weight"))?;
                let k = mat(&mut g, &format!("{p}attn_k.weight"))?;
                let v = mat(&mut g, &format!("{p}attn_v.weight"))?;
                let o = mat(&mut g, &format!("{p}attn_output.weight"))?;
                let q_norm = opt(&mut g, &format!("{p}attn_q_norm.weight"))?;
                let k_norm = opt(&mut g, &format!("{p}attn_k_norm.weight"))?;
                Attention::Full {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    gated: arch == "qwen35",
                }
            };
            let ffn_norm_name = if g
                .tensors
                .contains_key(&format!("{p}post_attention_norm.weight"))
            {
                format!("{p}post_attention_norm.weight")
            } else {
                format!("{p}ffn_norm.weight")
            };
            layers.push(Layer {
                attn_norm: g.tensor_f32(&format!("{p}attn_norm.weight"))?,
                attention,
                ffn_norm: g.tensor_f32(&ffn_norm_name)?,
                gate: mat(&mut g, &format!("{p}ffn_gate.weight"))?,
                up: mat(&mut g, &format!("{p}ffn_up.weight"))?,
                down: mat(&mut g, &format!("{p}ffn_down.weight"))?,
            });
        }
        let out_norm = g.tensor_f32("output_norm.weight")?;
        let lm_head = if g.tensors.contains_key("output.weight") {
            mat(&mut g, "output.weight")?
        } else {
            mat(&mut g, "token_embd.weight")?
        };
        let activation = crate::import::default_activation(&arch).1;
        Ok(Self {
            cfg,
            activation,
            embed,
            layers,
            out_norm,
            lm_head,
        })
    }

    /// Dense (weights-based) FFN of layer `i`, the reference path.
    pub fn dense_ffn(&self, i: usize, x: &[f32]) -> Vec<f32> {
        let l = &self.layers[i];
        let (g, u) = rayon::join(|| l.gate.matvec(x), || l.up.matvec(x));
        let h: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(a, b)| self.activation.apply(*a) * b)
            .collect();
        l.down.matvec(&h)
    }

    /// Normalised FFN input of layer `i` for residual `x` (what an AARNN
    /// region receives).
    pub fn ffn_input(&self, i: usize, x: &[f32]) -> Vec<f32> {
        rms_norm(x, &self.layers[i].ffn_norm, self.cfg.eps)
    }
}

/// Uses the model's own dense FFN weights for every layer.
pub struct DenseFfn<'a>(pub &'a Model);

impl FfnBackend for DenseFfn<'_> {
    fn ffn(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        self.0.dense_ffn(layer, x)
    }
}

/// Apply the split-half (NeoX) rotary layout used by Qwen3 and Llama.
fn rope_neox(v: &mut [f32], heads: usize, hd: usize, rotary_dim: usize, pos: usize, base: f32) {
    let rotary_dim = rotary_dim.min(hd) & !1;
    let half = rotary_dim / 2;
    for h in 0..heads {
        let s = &mut v[h * hd..(h + 1) * hd];
        for i in 0..half {
            let theta = pos as f32 * base.powf(-2.0 * i as f32 / rotary_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (s[i], s[i + half]);
            s[i] = a * cos - b * sin;
            s[i + half] = a * sin + b * cos;
        }
    }
}

/// Apply adjacent-pair rotary positions used by Qwen3.5's interleaved MRoPE.
/// Text-only requests have equal temporal/height/width positions, so the
/// three MRoPE sections reduce to one rotary position over `rotary_dim`.
fn rope_interleaved(
    v: &mut [f32],
    heads: usize,
    hd: usize,
    rotary_dim: usize,
    pos: usize,
    base: f32,
) {
    let rotary_dim = rotary_dim.min(hd) & !1;
    for h in 0..heads {
        let s = &mut v[h * hd..(h + 1) * hd];
        for i in (0..rotary_dim).step_by(2) {
            let theta = pos as f32 * base.powf(-(i as f32) / rotary_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (s[i], s[i + 1]);
            s[i] = a * cos - b * sin;
            s[i + 1] = a * sin + b * cos;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// One autoregressive gated-delta update. State is row-major [key, value].
fn gated_delta_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
) -> Vec<f32> {
    let width = q.len();
    debug_assert_eq!(k.len(), width);
    debug_assert_eq!(state.len(), width * v.len());
    for value in 0..v.len() {
        for key in 0..width {
            state[key * v.len() + value] *= decay;
        }
    }
    let mut delta = vec![0.0; v.len()];
    for value in 0..v.len() {
        let predicted = (0..width)
            .map(|key| state[key * v.len() + value] * k[key])
            .sum::<f32>();
        delta[value] = (v[value] - predicted) * beta;
    }
    for key in 0..width {
        for value in 0..v.len() {
            state[key * v.len() + value] += k[key] * delta[value];
        }
    }
    (0..v.len())
        .map(|value| {
            (0..width)
                .map(|key| state[key * v.len() + value] * q[key])
                .sum::<f32>()
                / (width as f32).sqrt()
        })
        .collect()
}

struct GatedDeltaState {
    /// The last `kernel - 1` un-convolved Q/K/V projection rows.
    conv_history: Vec<f32>,
    /// One [key_dim, value_dim] recurrent matrix per value head.
    recurrent: Vec<f32>,
}

impl GatedDeltaState {
    fn new(attention: &GatedDelta) -> Self {
        let channels = 2 * attention.key_dim * attention.key_heads
            + attention.value_dim * attention.value_heads;
        Self {
            conv_history: vec![0.0; (attention.conv_kernel - 1) * channels],
            recurrent: vec![0.0; attention.value_heads * attention.key_dim * attention.value_dim],
        }
    }
}

fn gated_delta_attention(
    attention: &GatedDelta,
    state: &mut GatedDeltaState,
    x: &[f32],
    eps: f32,
) -> Vec<f32> {
    let key_width = attention.key_dim * attention.key_heads;
    let value_width = attention.value_dim * attention.value_heads;
    let channels = 2 * key_width + value_width;
    let projection = attention.qkv.matvec(x);
    let z = attention.z.matvec(x);

    // ggml_ssm_conv consumes the previous kernel-1 projections followed by
    // this token, then applies one depthwise causal convolution per channel.
    let history_len = state.conv_history.len();
    let mut conv_input = Vec::with_capacity(history_len + channels);
    conv_input.extend_from_slice(&state.conv_history);
    conv_input.extend_from_slice(&projection);
    let mut conv_output = vec![0.0; channels];
    for channel in 0..channels {
        let mut sum = 0.0;
        for tap in 0..attention.conv_kernel {
            sum += conv_input[tap * channels + channel]
                * attention.conv[channel * attention.conv_kernel + tap];
        }
        conv_output[channel] = silu(sum);
    }
    if history_len > 0 {
        state
            .conv_history
            .copy_from_slice(&conv_input[conv_input.len() - history_len..]);
    }

    let alpha = attention.alpha.matvec(x);
    let beta = attention.beta.matvec(x);
    let mut output = vec![0.0; value_width];
    for head in 0..attention.value_heads {
        let key_head = head % attention.key_heads;
        let mut q =
            conv_output[key_head * attention.key_dim..(key_head + 1) * attention.key_dim].to_vec();
        let mut k = conv_output[key_width + key_head * attention.key_dim
            ..key_width + (key_head + 1) * attention.key_dim]
            .to_vec();
        let v = &conv_output[2 * key_width + head * attention.value_dim
            ..2 * key_width + (head + 1) * attention.value_dim];
        let q_norm = (q.iter().map(|value| value * value).sum::<f32>() + eps).sqrt();
        let k_norm = (k.iter().map(|value| value * value).sum::<f32>() + eps).sqrt();
        q.iter_mut().for_each(|value| *value /= q_norm);
        k.iter_mut().for_each(|value| *value /= k_norm);
        let decay = (softplus(alpha[head] + attention.dt[head]) * attention.a[head]).exp();
        let beta = sigmoid(beta[head]);
        let state_start = head * attention.key_dim * attention.value_dim;
        let state_end = state_start + attention.key_dim * attention.value_dim;
        let head_output = gated_delta_step(
            &mut state.recurrent[state_start..state_end],
            &q,
            &k,
            v,
            decay,
            beta,
        );
        output[head * attention.value_dim..(head + 1) * attention.value_dim]
            .copy_from_slice(&head_output);
    }

    // Qwen3.5 applies per-head RMSNorm and the projected SiLU gate after the
    // recurrent update, then projects the concatenated value heads to hidden.
    for head in 0..attention.value_heads {
        let start = head * attention.value_dim;
        let end = start + attention.value_dim;
        let normalized = rms_norm(&output[start..end], &attention.norm, eps);
        for ((dst, value), gate) in output[start..end]
            .iter_mut()
            .zip(normalized)
            .zip(&z[start..end])
        {
            *dst = value * silu(*gate);
        }
    }
    attention.out.matvec(&output)
}

/// One decoding session over a shared [`Model`], with attention KV and
/// Qwen3.5 convolution/recurrent state kept per request.
pub struct Session<'m> {
    model: &'m Model,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    gated_delta_state: Vec<Option<GatedDeltaState>>,
    pub pos: usize,
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model) -> Self {
        let n = model.cfg.layers;
        let gated_delta_state = model
            .layers
            .iter()
            .map(|layer| match &layer.attention {
                Attention::GatedDelta(attention) => Some(GatedDeltaState::new(attention)),
                Attention::Full { .. } => None,
            })
            .collect();
        Self {
            model,
            k_cache: vec![Vec::new(); n],
            v_cache: vec![Vec::new(); n],
            gated_delta_state,
            pos: 0,
        }
    }
}

fn full_attention(
    layer: &Layer,
    xn: &[f32],
    c: &Config,
    pos: usize,
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
) -> Vec<f32> {
    let Attention::Full {
        q,
        k,
        v,
        o,
        q_norm,
        k_norm,
        gated,
    } = &layer.attention
    else {
        unreachable!("full_attention called for a gated-delta layer")
    };
    let projected_q = q.matvec(xn);
    let query_width = c.heads * c.head_dim;
    let mut queries = Vec::with_capacity(query_width);
    let mut query_gate = Vec::with_capacity(if *gated { query_width } else { 0 });
    if *gated {
        for head in 0..c.heads {
            let start = head * c.head_dim * 2;
            queries.extend_from_slice(&projected_q[start..start + c.head_dim]);
            query_gate.extend_from_slice(&projected_q[start + c.head_dim..start + c.head_dim * 2]);
        }
    } else {
        queries = projected_q;
    }
    let (mut qv, mut kv) = rayon::join(|| queries, || k.matvec(xn));
    let vv = v.matvec(xn);
    let kv_heads = k.rows / c.head_dim;
    let group = c.heads / kv_heads;
    if let Some(weights) = q_norm {
        for head in 0..c.heads {
            let start = head * c.head_dim;
            let normalized = rms_norm(&qv[start..start + c.head_dim], weights, c.eps);
            qv[start..start + c.head_dim].copy_from_slice(&normalized);
        }
    }
    if let Some(weights) = k_norm {
        for head in 0..kv_heads {
            let start = head * c.head_dim;
            let normalized = rms_norm(&kv[start..start + c.head_dim], weights, c.eps);
            kv[start..start + c.head_dim].copy_from_slice(&normalized);
        }
    }
    if c.rope_interleaved {
        rope_interleaved(&mut qv, c.heads, c.head_dim, c.rope_dim, pos, c.rope_base);
        rope_interleaved(&mut kv, kv_heads, c.head_dim, c.rope_dim, pos, c.rope_base);
    } else {
        rope_neox(&mut qv, c.heads, c.head_dim, c.rope_dim, pos, c.rope_base);
        rope_neox(&mut kv, kv_heads, c.head_dim, c.rope_dim, pos, c.rope_base);
    }
    k_cache.extend_from_slice(&kv);
    v_cache.extend_from_slice(&vv);
    let t = pos + 1;
    let kv_width = kv_heads * c.head_dim;
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    let mut att: Vec<f32> = (0..c.heads)
        .into_par_iter()
        .flat_map_iter(|head| {
            let kv_head = head / group;
            let qh = &qv[head * c.head_dim..(head + 1) * c.head_dim];
            let mut scores: Vec<f32> = (0..t)
                .map(|position| {
                    let kp = &k_cache[position * kv_width + kv_head * c.head_dim
                        ..position * kv_width + (kv_head + 1) * c.head_dim];
                    qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let maximum = scores.iter().copied().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                sum += *score;
            }
            let mut out = vec![0.0f32; c.head_dim];
            for (position, weight) in scores.iter().enumerate() {
                let vp = &v_cache[position * kv_width + kv_head * c.head_dim
                    ..position * kv_width + (kv_head + 1) * c.head_dim];
                for (value, vv) in out.iter_mut().zip(vp) {
                    *value += weight / sum * vv;
                }
            }
            if *gated {
                let gate_start = head * c.head_dim;
                for (value, gate) in out
                    .iter_mut()
                    .zip(&query_gate[gate_start..gate_start + c.head_dim])
                {
                    *value *= sigmoid(*gate);
                }
            }
            out
        })
        .collect();
    if att.len() != query_width {
        att.resize(query_width, 0.0);
    }
    o.matvec(&att)
}

impl<'m> Session<'m> {
    /// Feed one token; returns next-token logits. `ffn` serves every
    /// feed-forward block (dense, AARNN, or a mix).
    pub fn step(&mut self, token: u32, ffn: &dyn FfnBackend) -> io::Result<Vec<f32>> {
        let model = self.model;
        let c = &model.cfg;
        validate_token_ids(&[token], c.vocab)?;
        let mut x = model.embed.row(token as usize)?;
        for (li, layer) in model.layers.iter().enumerate() {
            let normalized = rms_norm(&x, &layer.attn_norm, c.eps);
            let attention_out = match &layer.attention {
                Attention::Full { .. } => full_attention(
                    layer,
                    &normalized,
                    c,
                    self.pos,
                    &mut self.k_cache[li],
                    &mut self.v_cache[li],
                ),
                Attention::GatedDelta(attention) => gated_delta_attention(
                    attention,
                    self.gated_delta_state[li]
                        .as_mut()
                        .expect("gated-delta session state missing"),
                    &normalized,
                    c.eps,
                ),
            };
            for (residual, output) in x.iter_mut().zip(attention_out) {
                *residual += output;
            }
            let ffn_input = rms_norm(&x, &layer.ffn_norm, c.eps);
            for (residual, output) in x.iter_mut().zip(ffn.ffn(li, &ffn_input)) {
                *residual += output;
            }
        }
        self.pos += 1;
        Ok(model.lm_head.matvec(&rms_norm(&x, &model.out_norm, c.eps)))
    }
}

#[cfg(test)]
mod token_validation_tests {
    use super::{QMatrix, gated_delta_step, rope_interleaved, validate_token_ids};
    use std::io;

    #[test]
    fn accepts_ids_inside_the_vocabulary() {
        assert!(validate_token_ids(&[0, 17, 99], 100).is_ok());
    }

    #[test]
    fn rejects_a_token_from_a_different_vocabulary() {
        let error = validate_token_ids(&[17, 100], 100).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("tokenizer and GGUF match"));
    }

    #[test]
    fn embedding_row_bounds_are_reported_as_an_error() {
        let matrix = QMatrix {
            rows: 1,
            cols: 1,
            ggml_type: 0,
            row_bytes: 4,
            data: 1.0f32.to_le_bytes().to_vec(),
        };
        assert_eq!(
            matrix.row(1).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn gated_delta_updates_state_before_query_readout() {
        let mut state = vec![0.0; 4];
        let out = gated_delta_step(&mut state, &[1.0, 0.0], &[1.0, 0.0], &[2.0, -4.0], 1.0, 1.0);
        let scale = 1.0 / 2.0f32.sqrt();
        assert!((out[0] - 2.0 * scale).abs() < 1e-6);
        assert!((out[1] + 4.0 * scale).abs() < 1e-6);
        assert_eq!(state, vec![2.0, -4.0, 0.0, 0.0]);
    }

    #[test]
    fn interleaved_rope_leaves_non_rotary_dimensions_untouched() {
        let mut values = vec![1.0, 0.0, 3.0, 4.0];
        rope_interleaved(&mut values, 1, 4, 2, 1, 10000.0);
        assert!((values[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((values[1] - 1.0f32.sin()).abs() < 1e-6);
        assert_eq!(&values[2..], &[3.0, 4.0]);
    }
}

/// Index of the largest logit.
pub fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |b, (i, x)| if *x > b.1 { (i, *x) } else { b },
        )
        .0 as u32
}

/// Log-probability of `target` under `logits` (numerically stable).
pub fn log_prob(logits: &[f32], target: u32) -> f32 {
    let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
    let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
    logits[target as usize] - lse
}
