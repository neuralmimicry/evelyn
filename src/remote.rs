//! Stage 4b: feed-forward backends that capture real activations or delegate
//! layers to AARNN knowledge regions hosted anywhere on the estate.

use crate::runtime::{FfnBackend, Model};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Network work performed for one configured AARNN layer.
#[derive(Clone, Debug, Default)]
pub struct EndpointStats {
    /// FFN requests made by the runtime.
    pub calls: u64,
    /// Calls that failed after the single reconnect attempt.
    pub failures: u64,
    /// Additional connection/exchange attempts made after the first try.
    pub retries: u64,
    /// Wall time spent in all calls, including retries.
    pub total_ns: u128,
    /// Longest single FFN call, including retries.
    pub max_ns: u128,
}

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
    stats: Mutex<EndpointStats>,
}

impl Endpoint {
    fn connect(&self) -> std::io::Result<TcpStream> {
        let addr = self
            .addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| std::io::Error::other("endpoint resolved to no addresses"))?;
        let s = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
        s.set_nodelay(true)?;
        s.set_read_timeout(Some(self.timeout))?;
        s.set_write_timeout(Some(Duration::from_secs(30)))?;
        Ok(s)
    }

    fn exchange(s: &mut TcpStream, x: &[f32]) -> std::io::Result<Vec<f32>> {
        let width = u32::try_from(x.len())
            .map_err(|_| std::io::Error::other("input width exceeds protocol limit"))?;
        let mut buf = Vec::with_capacity(4 + x.len() * 4);
        buf.extend_from_slice(&width.to_le_bytes());
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
        if m != width {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("region returned width {m}; expected {width}"),
            ));
        }
        let byte_len = (m as usize)
            .checked_mul(4)
            .ok_or_else(|| std::io::Error::other("response width overflow"))?;
        let mut out = vec![0u8; byte_len];
        s.read_exact(&mut out)?;
        let values: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "region returned a non-finite value",
            ));
        }
        Ok(values)
    }

    /// Run one request, reconnecting once if the connection has dropped.
    fn call(&self, x: &[f32]) -> std::io::Result<Vec<f32>> {
        let started = Instant::now();
        let mut attempts = 0u64;
        let result = (|| {
            let mut guard = self
                .stream
                .lock()
                .map_err(|_| std::io::Error::other("endpoint stream lock poisoned"))?;
            let mut last_error = None;
            for _ in 0..2 {
                attempts += 1;
                if guard.is_none() {
                    match self.connect() {
                        Ok(s) => *guard = Some(s),
                        Err(e) => {
                            last_error = Some(e);
                            continue;
                        }
                    }
                }
                match Self::exchange(guard.as_mut().expect("connected"), x) {
                    Ok(y) => return Ok(y),
                    Err(e) => {
                        *guard = None;
                        last_error = Some(e);
                    }
                }
            }
            Err(last_error.unwrap_or_else(|| std::io::Error::other("request failed")))
        })();

        let elapsed_ns = started.elapsed().as_nanos();
        if let Ok(mut stats) = self.stats.lock() {
            stats.calls += 1;
            stats.retries += attempts.saturating_sub(1);
            stats.total_ns += elapsed_ns;
            stats.max_ns = stats.max_ns.max(elapsed_ns);
            if result.is_err() {
                stats.failures += 1;
            }
        }
        result
    }
}

/// Serves selected layers from AARNN knowledge regions and the rest densely.
/// If a region fails after its retry, the layer falls back to the dense
/// weights for that token and the failure is counted, so generation degrades
/// rather than aborting.
pub struct RemoteAarnnFfn<'a> {
    model: ModelHandle<'a>,
    endpoints: BTreeMap<usize, Endpoint>,
    pub fallbacks: Mutex<u64>,
}

enum ModelHandle<'a> {
    Borrowed(&'a Model),
    Shared(Arc<Model>),
}

impl ModelHandle<'_> {
    fn get(&self) -> &Model {
        match self {
            Self::Borrowed(model) => model,
            Self::Shared(model) => model,
        }
    }
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
                        stats: Mutex::new(EndpointStats::default()),
                    },
                )
            })
            .collect();
        Self {
            model: ModelHandle::Borrowed(model),
            endpoints,
            fallbacks: Mutex::new(0),
        }
    }

    /// Own a shared model reference so long-lived services can also keep the
    /// endpoint TCP connections alive between completions.
    pub fn new_shared(
        model: Arc<Model>,
        layers: BTreeMap<usize, String>,
        timeout: Duration,
    ) -> RemoteAarnnFfn<'static> {
        let endpoints = layers
            .into_iter()
            .map(|(layer, addr)| {
                (
                    layer,
                    Endpoint {
                        addr,
                        stream: Mutex::new(None),
                        timeout,
                        stats: Mutex::new(EndpointStats::default()),
                    },
                )
            })
            .collect();
        RemoteAarnnFfn {
            model: ModelHandle::Shared(model),
            endpoints,
            fallbacks: Mutex::new(0),
        }
    }

    /// Snapshot request latency and retry counts, keyed by model layer.
    pub fn endpoint_stats(&self) -> BTreeMap<usize, (String, EndpointStats)> {
        self.endpoints
            .iter()
            .map(|(layer, endpoint)| {
                let stats = endpoint.stats.lock().map(|s| s.clone()).unwrap_or_default();
                (*layer, (endpoint.addr.clone(), stats))
            })
            .collect()
    }
}

impl FfnBackend for RemoteAarnnFfn<'_> {
    fn ffn(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        let model = self.model.get();
        match self.endpoints.get(&layer) {
            None => model.dense_ffn(layer, x),
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
                    model.dense_ffn(layer, x)
                }
                Err(e) => {
                    eprintln!("aarnn region {}: {e}; dense fallback", ep.addr);
                    *self.fallbacks.lock().expect("lock") += 1;
                    model.dense_ffn(layer, x)
                }
            },
        }
    }
}
