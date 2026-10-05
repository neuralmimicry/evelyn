//! `evelyn-compare <reference.f32> <candidate.f32> <width> [gate]`
//!
//! Mean relative error between two row-major f32 files of `width`-wide
//! vectors. Exits non-zero if the error is at or above `gate` (default 0.05).

use evelyn::relative_error;
use std::process::ExitCode;

fn read(path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    let res = (|| -> Result<bool, String> {
        if a.len() < 4 {
            return Err(
                "usage: evelyn-compare <reference.f32> <candidate.f32> <width> [gate]".into(),
            );
        }
        let (r, c) = (read(&a[1])?, read(&a[2])?);
        let width: usize = a[3].parse().map_err(|_| "invalid width")?;
        let gate: f32 = a
            .get(4)
            .map_or(Ok(0.05), |s| s.parse())
            .map_err(|_| "invalid gate")?;
        if r.len() != c.len() || width == 0 || r.len() % width != 0 {
            return Err(format!(
                "length mismatch: {} vs {} (width {width})",
                r.len(),
                c.len()
            ));
        }
        let n = r.len() / width;
        let err: f32 = r
            .chunks(width)
            .zip(c.chunks(width))
            .map(|(x, y)| relative_error(x, y))
            .sum::<f32>()
            / n as f32;
        println!(
            "mean relative error over {n} vectors: {err:.4}  GATE {}",
            if err < gate { "PASS" } else { "FAIL" }
        );
        Ok(err < gate)
    })();
    match res {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("evelyn-compare: {e}");
            ExitCode::from(2)
        }
    }
}
