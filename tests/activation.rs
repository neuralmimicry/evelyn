//! Stage 1 verification gates: any activation maps onto a neuron population,
//! and the model's true gate (SiLU, GELU) converts without ReLU-fication.

use evelyn::activation::{Activation, PopulationCode};
use evelyn::dense::SwiGluMlp;
use evelyn::snn::SpikingPopulationGatedMlp;
use evelyn::{Rng, relative_error};

#[test]
fn population_code_fits_every_model_activation() {
    for (name, act) in [
        ("relu", Activation::Relu),
        ("silu", Activation::Silu),
        ("gelu", Activation::Gelu),
        ("gelu_pytorch_tanh", Activation::GeluTanh),
        ("tanh", Activation::Tanh),
        ("sigmoid", Activation::Sigmoid),
    ] {
        let code = PopulationCode::fit(act, -8.0, 8.0, 24);
        let err = code.max_error(act, 2001);
        println!(
            "{name:>18}: {} units, max |error| on [-8,8] = {err:.5}",
            code.units.len() + 1
        );
        assert!(err < 0.01, "{name}: max error {err}");
        assert_eq!(Activation::from_name(name).is_some(), true);
    }
}

#[test]
fn population_code_handles_custom_activations() {
    fn softplus(z: f32) -> f32 {
        (1.0 + z.exp()).ln()
    }
    let act = Activation::Custom(softplus);
    let err = PopulationCode::fit(act, -6.0, 6.0, 24).max_error(act, 1001);
    println!("softplus (custom): max |error| = {err:.5}");
    assert!(err < 0.01);
}

fn inputs(n: usize, dim: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.normal()).collect())
        .collect()
}

#[test]
fn true_swiglu_and_geglu_convert_without_relufication() {
    let mut rng = Rng::new(11);
    let mlp = SwiGluMlp::random(32, 96, &mut rng);
    let calib = inputs(200, 32, &mut rng);
    let test = inputs(6, 32, &mut rng);
    for (name, act) in [
        ("SwiGLU", Activation::Silu),
        ("GeGLU", Activation::GeluTanh),
    ] {
        let snn = SpikingPopulationGatedMlp::convert(&mlp, act, &calib, 0.999, 16);
        let mut srng = Rng::new(42);
        let mut err = |t| {
            test.iter()
                .map(|x| relative_error(&mlp.forward_act(x, act), &snn.run(x, t, &mut srng).0))
                .sum::<f32>()
                / test.len() as f32
        };
        let (e512, e8192) = (err(512), err(8192));
        let gap = test
            .iter()
            .map(|x| relative_error(&mlp.forward_act(x, act), &mlp.forward_relu_gate(x)))
            .sum::<f32>()
            / test.len() as f32;
        println!(
            "{name}: population {} units/neuron; rel. error T=512 {e512:.4}, T=8192 {e8192:.4} (ReLU-fication gap was {gap:.4})",
            snn.population_size()
        );
        assert!(e8192 < e512, "{name}: error must fall with time");
        assert!(e8192 < 0.05, "{name}: T=8192 error {e8192} above 5%");
        assert!(e8192 < gap, "{name}: must beat ReLU-fication");
    }
}
