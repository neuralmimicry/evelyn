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
    pub eps: f32,
    pub vocab: usize,
    /// End-of-sequence token from GGUF tokenizer metadata, when present.
    pub eos_token_id: Option<u32>,
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

struct Layer {
    attn_norm: Vec<f32>,
    q: QMatrix,
    k: QMatrix,
    v: QMatrix,
    o: QMatrix,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
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
            .unwrap_or(heads as u64) as usize;
        let head_dim = g
            .arch_u64("attention.key_length")
            .unwrap_or((dim / heads) as u64) as usize;
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
            eps: g
                .meta(&format!("{arch}.attention.layer_norm_rms_epsilon"))
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
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
            let opt = |g: &mut Gguf, n: &str| {
                if g.tensors.contains_key(n) {
                    g.tensor_f32(n).map(Some)
                } else {
                    Ok(None)
                }
            };
            layers.push(Layer {
                attn_norm: g.tensor_f32(&format!("{p}attn_norm.weight"))?,
                q: mat(&mut g, &format!("{p}attn_q.weight"))?,
                k: mat(&mut g, &format!("{p}attn_k.weight"))?,
                v: mat(&mut g, &format!("{p}attn_v.weight"))?,
                o: mat(&mut g, &format!("{p}attn_output.weight"))?,
                q_norm: opt(&mut g, &format!("{p}attn_q_norm.weight"))?,
                k_norm: opt(&mut g, &format!("{p}attn_k_norm.weight"))?,
                ffn_norm: g.tensor_f32(&format!("{p}ffn_norm.weight"))?,
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

/// One decoding session (KV cache) over a shared [`Model`].
pub struct Session<'m> {
    model: &'m Model,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pub pos: usize,
}

fn rope_neox(v: &mut [f32], heads: usize, hd: usize, pos: usize, base: f32) {
    let half = hd / 2;
    for h in 0..heads {
        let s = &mut v[h * hd..(h + 1) * hd];
        for i in 0..half {
            let theta = pos as f32 * base.powf(-2.0 * i as f32 / hd as f32);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (s[i], s[i + half]);
            s[i] = a * cos - b * sin;
            s[i + half] = a * sin + b * cos;
        }
    }
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model) -> Self {
        let n = model.cfg.layers;
        Self {
            model,
            k_cache: vec![Vec::new(); n],
            v_cache: vec![Vec::new(); n],
            pos: 0,
        }
    }

    /// Feed one token; returns next-token logits. `ffn` serves every
    /// feed-forward block (dense, AARNN, or a mix).
    pub fn step(&mut self, token: u32, ffn: &dyn FfnBackend) -> io::Result<Vec<f32>> {
        let m = self.model;
        let c = &m.cfg;
        validate_token_ids(&[token], c.vocab)?;
        let mut x = m.embed.row(token as usize)?;
        let group = c.heads / c.kv_heads;
        for (li, l) in m.layers.iter().enumerate() {
            let xn = rms_norm(&x, &l.attn_norm, c.eps);
            let (q, (k, v)) = rayon::join(
                || l.q.matvec(&xn),
                || rayon::join(|| l.k.matvec(&xn), || l.v.matvec(&xn)),
            );
            let (mut q, mut k) = (q, k);
            if let Some(w) = &l.q_norm {
                for h in 0..c.heads {
                    let s = rms_norm(&q[h * c.head_dim..(h + 1) * c.head_dim], w, c.eps);
                    q[h * c.head_dim..(h + 1) * c.head_dim].copy_from_slice(&s);
                }
            }
            if let Some(w) = &l.k_norm {
                for h in 0..c.kv_heads {
                    let s = rms_norm(&k[h * c.head_dim..(h + 1) * c.head_dim], w, c.eps);
                    k[h * c.head_dim..(h + 1) * c.head_dim].copy_from_slice(&s);
                }
            }
            rope_neox(&mut q, c.heads, c.head_dim, self.pos, c.rope_base);
            rope_neox(&mut k, c.kv_heads, c.head_dim, self.pos, c.rope_base);
            self.k_cache[li].extend_from_slice(&k);
            self.v_cache[li].extend_from_slice(&v);
            let t = self.pos + 1;
            let kvw = c.kv_heads * c.head_dim;
            let scale = 1.0 / (c.head_dim as f32).sqrt();
            let (kc, vc) = (&self.k_cache[li], &self.v_cache[li]);
            let att: Vec<f32> = (0..c.heads)
                .into_par_iter()
                .flat_map_iter(|h| {
                    let kvh = h / group;
                    let qh = &q[h * c.head_dim..(h + 1) * c.head_dim];
                    let mut scores: Vec<f32> = (0..t)
                        .map(|p| {
                            let kp =
                                &kc[p * kvw + kvh * c.head_dim..p * kvw + (kvh + 1) * c.head_dim];
                            qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale
                        })
                        .collect();
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mut sum = 0.0;
                    for s in &mut scores {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let mut out = vec![0.0f32; c.head_dim];
                    for (p, w) in scores.iter().enumerate() {
                        let vp = &vc[p * kvw + kvh * c.head_dim..p * kvw + (kvh + 1) * c.head_dim];
                        for (o, vv) in out.iter_mut().zip(vp) {
                            *o += w / sum * vv;
                        }
                    }
                    out
                })
                .collect();
            for (xi, oi) in x.iter_mut().zip(l.o.matvec(&att)) {
                *xi += oi;
            }
            let fin = rms_norm(&x, &l.ffn_norm, c.eps);
            for (xi, fi) in x.iter_mut().zip(ffn.ffn(li, &fin)) {
                *xi += fi;
            }
        }
        self.pos += 1;
        Ok(m.lm_head.matvec(&rms_norm(&x, &m.out_norm, c.eps)))
    }
}

#[cfg(test)]
mod token_validation_tests {
    use super::{QMatrix, validate_token_ids};
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
