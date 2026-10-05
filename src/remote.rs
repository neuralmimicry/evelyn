//! Stage 4b: feed-forward backends that capture real activations or delegate
//! layers to AARNN knowledge regions hosted anywhere on the estate.

use crate::runtime::{FfnBackend, Model};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::Duration;

/// Dense FFN that also records the normalised FFN input of one layer. The
/// recorded vectors are real mid-network activations, used to calibrate a
/// knowledge region.
pub struct CaptureFfn<'a> {
    pub model: &'a Model,
    pub layer: usize,
    pub captured: Mutex<Vec<Vec<f32>>>,
}

impl FfnBackend for CaptureFfn<'_> {
    fn ffn(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        if layer == self.layer {
            self.captured
                .lock()
                .expect("capture lock poisoned")
                .push(x.to_vec());
        }
        self.model.dense_ffn(layer, x)
    }
}

/// One persistent connection to an `aarnn-knowledge-serve` endpoint.
struct Endpoint {
    addr: String,
    stream: Mutex<Option<TcpStream>>,
    timeout: Duration,
}

impl Endpoint {
    fn connect(&self) -> std::io::Result<TcpStream> {
        let s = TcpStream::connect(&self.addr)?;
        s.set_nodelay(true)?;
        s.set_read_timeout(Some(self.timeout))?;
        s.set_write_timeout(Some(Duration::from_secs(30)))?;
        Ok(s)
    }

    fn exchange(s: &mut TcpStream, x: &[f32]) -> std::io::Result<Vec<f32>> {
        let mut buf = Vec::with_capacity(4 + x.len() * 4);
        buf.extend_from_slice(&(x.len() as u32).to_le_bytes());
        for v in x {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        s.write_all(&buf)?;
        let mut h = [0u8; 8];
        s.read_exact(&mut h)?;
        let (status, m) = (
            u32::from_le_bytes([h[0], h[1], h[2], h[3]]),
            u32::from_le_bytes([h[4], h[5], h[6], h[7]]),
        );
        if status != 0 {
            return Err(std::io::Error::other(format!(
                "region returned status {status}"
            )));
        }
        let mut out = vec![0u8; m as usize * 4];
        s.read_exact(&mut out)?;
        Ok(out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Run one request, reconnecting once if the connection has dropped.
    fn call(&self, x: &[f32]) -> std::io::Result<Vec<f32>> {
        let mut guard = self.stream.lock().expect("endpoint lock poisoned");
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(self.connect()?);
            }
            match Self::exchange(guard.as_mut().expect("connected"), x) {
                Ok(y) => return Ok(y),
                Err(e) if attempt == 0 => {
                    eprintln!("aarnn region {}: {e}; reconnecting", self.addr);
                    *guard = None;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("loop returns on the second attempt")
    }
}

/// Serves selected layers from AARNN knowledge regions and the rest densely.
/// If a region fails after its retry, the layer falls back to the dense
/// weights for that token and the failure is counted, so generation degrades
/// rather than aborting.
pub struct RemoteAarnnFfn<'a> {
    pub model: &'a Model,
    endpoints: BTreeMap<usize, Endpoint>,
    pub fallbacks: Mutex<u64>,
}

impl<'a> RemoteAarnnFfn<'a> {
    /// `layers` maps a layer index to a `host:port` region address.
    pub fn new(model: &'a Model, layers: BTreeMap<usize, String>, timeout: Duration) -> Self {
        let endpoints = layers
            .into_iter()
            .map(|(l, addr)| {
                (
                    l,
                    Endpoint {
                        addr,
                        stream: Mutex::new(None),
                        timeout,
                    },
                )
            })
            .collect();
        Self {
            model,
            endpoints,
            fallbacks: Mutex::new(0),
        }
    }
}

impl FfnBackend for RemoteAarnnFfn<'_> {
    fn ffn(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        match self.endpoints.get(&layer) {
            None => self.model.dense_ffn(layer, x),
            Some(ep) => match ep.call(x) {
                Ok(y) if y.len() == x.len() => y,
                Ok(y) => {
                    eprintln!(
                        "aarnn region {}: width {} != {}; dense fallback",
                        ep.addr,
                        y.len(),
                        x.len()
                    );
                    *self.fallbacks.lock().expect("lock") += 1;
                    self.model.dense_ffn(layer, x)
                }
                Err(e) => {
                    eprintln!("aarnn region {}: {e}; dense fallback", ep.addr);
                    *self.fallbacks.lock().expect("lock") += 1;
                    self.model.dense_ffn(layer, x)
                }
            },
        }
    }
}
