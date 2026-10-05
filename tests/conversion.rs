//! Stage 0 verification gates: spiking conversions must reproduce their
//! weights-based reference within stated error bounds, and the error must fall
//! as simulation time grows.

use evelyn::dense::{Dense, ReluMlp, SwiGluMlp};
use evelyn::snn::{SpikingGatedMlp, SpikingReluMlp};
use evelyn::{Rng, relative_error};

fn inputs(n: usize, dim: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.normal()).collect())
        .collect()
}

fn mean_rel_error(mut f: impl FnMut(&[f32]) -> (Vec<f32>, Vec<f32>), xs: &[Vec<f32>]) -> f32 {
    xs.iter()
        .map(|x| {
            let (r, a) = f(x);
            relative_error(&r, &a)
        })
        .sum::<f32>()
        / xs.len() as f32
}

#[test]
fn relu_mlp_converges_to_reference() {
    let mut rng = Rng::new(7);
    let mlp = ReluMlp {
        layers: vec![
            Dense::random(32, 64, &mut rng),
            Dense::random(64, 64, &mut rng),
            Dense::random(64, 16, &mut rng),
        ],
    };
    let calib = inputs(200, 32, &mut rng);
    let test = inputs(20, 32, &mut rng);
    let snn = SpikingReluMlp::convert(&mlp, &calib, 0.999);
    let err = |t| mean_rel_error(|x| (mlp.forward(x), snn.run(x, t).0), &test);
    let (e64, e512, e2048) = (err(64), err(512), err(2048));
    println!("relu mlp relative error: T=64 {e64:.4}  T=512 {e512:.4}  T=2048 {e2048:.4}");
    assert!(e512 < e64, "error must fall with time");
    assert!(e2048 < 0.05, "T=2048 error {e2048} above 5%");
}

#[test]
fn gated_block_matches_relu_gated_reference() {
    let mut rng = Rng::new(11);
    let mlp = SwiGluMlp::random(32, 96, &mut rng);
    let calib = inputs(200, 32, &mut rng);
    let test = inputs(10, 32, &mut rng);
    let snn = SpikingGatedMlp::convert(&mlp, &calib, 0.999);
    let mut srng = Rng::new(99);
    let mut err = |t| {
        mean_rel_error(
            |x| (mlp.forward_relu_gate(x), snn.run(x, t, &mut srng).0),
            &test,
        )
    };
    let (e256, e4096) = (err(256), err(4096));
    println!("gated (coincidence) relative error vs ReGLU: T=256 {e256:.4}  T=4096 {e4096:.4}");
    assert!(e4096 < e256, "error must fall with time");
    assert!(e4096 < 0.10, "T=4096 error {e4096} above 10%");
}

#[test]
fn relufication_gap_is_measured() {
    // Not a pass/fail gate: records how far a ReLU gate is from the model's
    // true SiLU gate, the gap stage 1 has to close.
    let mut rng = Rng::new(5);
    let mlp = SwiGluMlp::random(32, 96, &mut rng);
    let test = inputs(50, 32, &mut rng);
    let gap = mean_rel_error(|x| (mlp.forward(x), mlp.forward_relu_gate(x)), &test);
    println!("ReLU-fication gap (SwiGLU vs ReGLU, untuned random weights): {gap:.4}");
    assert!(gap.is_finite());
}

#[test]
fn conversion_is_deterministic() {
    let mut rng = Rng::new(3);
    let mlp = ReluMlp {
        layers: vec![
            Dense::random(8, 16, &mut rng),
            Dense::random(16, 4, &mut rng),
        ],
    };
    let calib = inputs(50, 8, &mut rng);
    let x = inputs(1, 8, &mut rng).remove(0);
    let a = SpikingReluMlp::convert(&mlp, &calib, 0.999).run(&x, 256);
    let b = SpikingReluMlp::convert(&mlp, &calib, 0.999).run(&x, 256);
    assert_eq!(a, b);
}
