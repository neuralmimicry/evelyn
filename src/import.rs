//! Stage 2 importer: extract one transformer layer's feed-forward
//! ("perceptron") block from an open-weights checkpoint, architecture-agnostic.
//!
//! The FFN layout is discovered from tensor names, not from per-model code:
//! * gated FFN: `blk.N.ffn_gate` + `blk.N.ffn_up` + `blk.N.ffn_down` (Llama,
//!   Qwen, Mistral, Gemma, ...)
//! * plain FFN: `blk.N.ffn_up` + `blk.N.ffn_down` (GPT-2/Falcon style)
//!
//! The activation comes from metadata when the checkpoint declares it, otherwise
//! from the architecture family's documented default.

use crate::activation::Activation;
use crate::dense::{Dense, SwiGluMlp};
use crate::gguf::{Gguf, Value};
use std::io;

pub struct ImportedFfn {
    pub arch: String,
    pub layer: usize,
    pub mlp: SwiGluMlp,
    pub activation: Activation,
    pub activation_name: String,
    /// The pre-FFN RMSNorm weight (used to build realistic layer inputs).
    pub norm: Option<Vec<f32>>,
    pub norm_eps: f32,
}

/// Documented FFN activation per architecture family (used only when the
/// checkpoint does not say). Gemma uses tanh-GELU; the Llama/Qwen/Mistral
/// families use SiLU (SwiGLU).
pub fn default_activation(arch: &str) -> (&'static str, Activation) {
    let a = arch.to_ascii_lowercase();
    if a.starts_with("gemma") {
        ("gelu_pytorch_tanh", Activation::GeluTanh)
    } else if a.starts_with("gpt2") || a.starts_with("falcon") || a.starts_with("bloom") {
        ("gelu", Activation::Gelu)
    } else {
        ("silu", Activation::Silu)
    }
}

fn dense(g: &mut Gguf, name: &str) -> io::Result<Dense> {
    let t = g
        .tensors
        .get(name)
        .cloned()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?;
    if t.dims.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name}: expected 2-D"),
        ));
    }
    let (inputs, outputs) = (t.dims[0] as usize, t.dims[1] as usize);
    let w = g.tensor_f32(name)?;
    let bias_name = name.replace(".weight", ".bias");
    let bias = if g.tensors.contains_key(&bias_name) {
        g.tensor_f32(&bias_name)?
    } else {
        vec![0.0; outputs]
    };
    Ok(Dense::new(inputs, outputs, w, bias))
}

pub fn import_ffn(g: &mut Gguf, layer: usize) -> io::Result<ImportedFfn> {
    let arch = g.architecture().unwrap_or("unknown").to_string();
    let p = format!("blk.{layer}.");
    let declared = g
        .meta(&format!("{arch}.feed_forward_activation"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let (name, activation) = match declared
        .as_deref()
        .and_then(|n| Activation::from_name(n).map(|a| (n.to_string(), a)))
    {
        Some(x) => x,
        None => {
            let (n, a) = default_activation(&arch);
            (n.to_string(), a)
        }
    };
    let gate_name = format!("{p}ffn_gate.weight");
    if !g.tensors.contains_key(&gate_name) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "layer {layer}: no {gate_name}; plain or MoE FFNs are converted in a later stage"
            ),
        ));
    }
    let gate = dense(g, &gate_name)?;
    let up = dense(g, &format!("{p}ffn_up.weight"))?;
    let down = dense(g, &format!("{p}ffn_down.weight"))?;
    // Pre-FFN norm: name varies by family.
    let norm_name = ["post_attention_norm.weight", "ffn_norm.weight"]
        .iter()
        .map(|n| format!("{p}{n}"))
        .find(|n| g.tensors.contains_key(n));
    let norm = match norm_name {
        Some(n) => Some(g.tensor_f32(&n)?),
        None => None,
    };
    let norm_eps = g
        .meta(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(Value::as_f64)
        .unwrap_or(1e-6) as f32;
    Ok(ImportedFfn {
        arch,
        layer,
        mlp: SwiGluMlp { gate, up, down },
        activation,
        activation_name: name,
        norm,
        norm_eps,
    })
}

/// RMSNorm with a learned per-channel weight.
pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    match weight {
        Some(w) => x.iter().zip(w).map(|(v, g)| v * inv * g).collect(),
        None => x.iter().map(|v| v * inv).collect(),
    }
}
