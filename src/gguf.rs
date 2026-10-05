//! Minimal GGUF (v2/v3) reader for the stage-2 importer, pure Rust.
//!
//! Reads the header, metadata and tensor table, then only the bytes of the
//! tensors that are asked for, so a 9 B checkpoint never has to fit in memory.
//! Dequantises F32, F16, BF16, Q8_0, Q4_K, Q5_K and Q6_K (the types in
//! llama.cpp Q4_K_M/Q5_K_M/Q6_K/Q8_0 exports) to f32.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            Self::U64(v) => Some(*v as f64),
            Self::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: `dims[0]` is the innermost (contiguous) dimension.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    file: BufReader<File>,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    data_start: u64,
}

fn rd<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn rd_u32(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(rd(r)?))
}
fn rd_u64(r: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(rd(r)?))
}
fn rd_str(r: &mut impl Read) -> io::Result<String> {
    let n = rd_u64(r)? as usize;
    if n > 1 << 24 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "string too long",
        ));
    }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn rd_value(r: &mut impl Read, ty: u32) -> io::Result<Value> {
    Ok(match ty {
        0 => Value::U64(rd::<1>(r)?[0] as u64),
        1 => Value::I64(rd::<1>(r)?[0] as i8 as i64),
        2 => Value::U64(u16::from_le_bytes(rd(r)?) as u64),
        3 => Value::I64(i16::from_le_bytes(rd(r)?) as i64),
        4 => Value::U64(rd_u32(r)? as u64),
        5 => Value::I64(i32::from_le_bytes(rd(r)?) as i64),
        6 => Value::F64(f32::from_le_bytes(rd(r)?) as f64),
        7 => Value::Bool(rd::<1>(r)?[0] != 0),
        8 => Value::Str(rd_str(r)?),
        9 => {
            let et = rd_u32(r)?;
            let n = rd_u64(r)?;
            let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
            for _ in 0..n {
                v.push(rd_value(r, et)?);
            }
            Value::Array(v)
        }
        10 => Value::U64(rd_u64(r)?),
        11 => Value::I64(i64::from_le_bytes(rd(r)?)),
        12 => Value::F64(f64::from_le_bytes(rd(r)?)),
        t => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown GGUF value type {t}"),
            ));
        }
    })
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        if &rd::<4>(&mut r)? != b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a GGUF file",
            ));
        }
        let version = rd_u32(&mut r)?;
        if version < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF v1 unsupported",
            ));
        }
        let n_tensors = rd_u64(&mut r)?;
        let n_kv = rd_u64(&mut r)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let k = rd_str(&mut r)?;
            let ty = rd_u32(&mut r)?;
            metadata.insert(k, rd_value(&mut r, ty)?);
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = rd_str(&mut r)?;
            let nd = rd_u32(&mut r)? as usize;
            let dims = (0..nd)
                .map(|_| rd_u64(&mut r))
                .collect::<io::Result<Vec<_>>>()?;
            let ggml_type = rd_u32(&mut r)?;
            let offset = rd_u64(&mut r)?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ggml_type,
                    offset,
                },
            );
        }
        let align = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32);
        let pos = r.stream_position()?;
        let data_start = pos.div_ceil(align) * align;
        Ok(Self {
            file: r,
            version,
            metadata,
            tensors,
            data_start,
        })
    }

    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.meta("general.architecture").and_then(Value::as_str)
    }

    /// `<arch>.<key>` metadata, e.g. `arch_u64("feed_forward_length")`.
    pub fn arch_u64(&self, key: &str) -> Option<u64> {
        let a = self.architecture()?;
        self.meta(&format!("{a}.{key}")).and_then(Value::as_u64)
    }

    /// Read and dequantise one tensor to f32 (GGUF element order).
    pub fn tensor_f32(&mut self, name: &str) -> io::Result<Vec<f32>> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} not found"))
            })?
            .clone();
        let n = t.elements() as usize;
        let bytes = type_bytes(t.ggml_type, n)?;
        self.file
            .seek(SeekFrom::Start(self.data_start + t.offset))?;
        let mut raw = vec![0u8; bytes];
        self.file.read_exact(&mut raw)?;
        dequantize(t.ggml_type, &raw, n)
    }
}

impl Gguf {
    /// Read selected rows of a 2-D tensor (row = `dims[0]` contiguous values),
    /// e.g. a few token embeddings without loading the whole table.
    pub fn tensor_rows_f32(&mut self, name: &str, rows: &[u64]) -> io::Result<Vec<Vec<f32>>> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} not found"))
            })?
            .clone();
        let width = t.dims[0] as usize;
        let row_bytes = type_bytes(t.ggml_type, width)?;
        let mut out = Vec::with_capacity(rows.len());
        for &r in rows {
            self.file.seek(SeekFrom::Start(
                self.data_start + t.offset + r * row_bytes as u64,
            ))?;
            let mut raw = vec![0u8; row_bytes];
            self.file.read_exact(&mut raw)?;
            out.push(dequantize(t.ggml_type, &raw, width)?);
        }
        Ok(out)
    }

    /// Byte offset just past a tensor's data (to check a partial file).
    pub fn tensor_end(&self, name: &str) -> Option<u64> {
        let t = self.tensors.get(name)?;
        Some(
            self.data_start
                + t.offset
                + type_bytes(t.ggml_type, t.elements() as usize).ok()? as u64,
        )
    }
}

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q5_K: u32 = 13;
pub const GGML_Q6_K: u32 = 14;
pub const GGML_BF16: u32 = 30;

pub fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        30 => "BF16",
        _ => "other",
    }
}

fn unsupported(t: u32) -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!("ggml type {t} ({}) not supported yet", type_name(t)),
    )
}

fn type_bytes(t: u32, n: usize) -> io::Result<usize> {
    Ok(match t {
        GGML_F32 => n * 4,
        GGML_F16 | GGML_BF16 => n * 2,
        GGML_Q8_0 => n / 32 * 34,
        GGML_Q4_K => n / 256 * 144,
        GGML_Q5_K => n / 256 * 176,
        GGML_Q6_K => n / 256 * 210,
        _ => return Err(unsupported(t)),
    })
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut f = f;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

fn h(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// K-quant 6-bit scale/min unpacking (`get_scale_min_k4` in ggml).
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (
            ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4)) as f32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32,
        )
    }
}

pub fn dequantize(t: u32, raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    let mut y = Vec::with_capacity(n);
    match t {
        GGML_F32 => y.extend(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        ),
        GGML_F16 => y.extend(
            raw.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        ),
        GGML_BF16 => y.extend(
            raw.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)),
        ),
        GGML_Q8_0 => {
            for b in raw.chunks_exact(34) {
                let d = h(b, 0);
                y.extend(b[2..34].iter().map(|q| d * (*q as i8) as f32));
            }
        }
        GGML_Q4_K => {
            for b in raw.chunks_exact(144) {
                let (d, dmin) = (h(b, 0), h(b, 2));
                let sc = &b[4..16];
                let qs = &b[16..144];
                for (blk, is) in (0..4).zip((0..8).step_by(2)) {
                    let q = &qs[blk * 32..blk * 32 + 32];
                    let (s1, m1) = scale_min_k4(is, sc);
                    let (s2, m2) = scale_min_k4(is + 1, sc);
                    y.extend(q.iter().map(|v| d * s1 * (v & 0xf) as f32 - dmin * m1));
                    y.extend(q.iter().map(|v| d * s2 * (v >> 4) as f32 - dmin * m2));
                }
            }
        }
        GGML_Q5_K => {
            for b in raw.chunks_exact(176) {
                let (d, dmin) = (h(b, 0), h(b, 2));
                let sc = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                let (mut u1, mut u2) = (1u8, 2u8);
                for (blk, is) in (0..4).zip((0..8).step_by(2)) {
                    let q = &qs[blk * 32..blk * 32 + 32];
                    let (s1, m1) = scale_min_k4(is, sc);
                    let (s2, m2) = scale_min_k4(is + 1, sc);
                    y.extend((0..32).map(|l| {
                        d * s1 * ((q[l] & 0xf) + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32
                            - dmin * m1
                    }));
                    y.extend((0..32).map(|l| {
                        d * s2 * ((q[l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32
                            - dmin * m2
                    }));
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GGML_Q6_K => {
            for b in raw.chunks_exact(210) {
                let (ql_all, qh_all, sc_all) = (&b[0..128], &b[128..192], &b[192..208]);
                let d = h(b, 208);
                for half in 0..2 {
                    let ql = &ql_all[half * 64..];
                    let qh = &qh_all[half * 32..];
                    let sc = &sc_all[half * 8..];
                    let mut out = [0f32; 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        out[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        out[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        out[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        out[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                    y.extend_from_slice(&out);
                }
            }
        }
        _ => return Err(unsupported(t)),
    }
    if y.len() != n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("dequantised {} of {n} values", y.len()),
        ));
    }
    Ok(y)
}
