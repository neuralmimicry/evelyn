//! Stage 3a gate: activations are fitted as populations of real AARNN neurons,
//! using transfer curves that aarnn_rust measured with its own kernels and
//! membrane noise (`aarnn-knowledge-curves`). The population size is chosen
//! automatically for each neuron model and detail depth.

use evelyn::activation::Activation;
use evelyn::curve::{CurvePopulationCode, MeasuredCurve};

#[test]
fn activations_fit_on_measured_aarnn_neurons() {
    let curves = MeasuredCurve::load_all(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/aarnn-transfer-curves.json"
    ))
    .expect("curves exported by aarnn-knowledge-curves");
    assert!(
        curves.len() >= 3,
        "expected LIF and Izhikevich (depth 0 and 2) curves"
    );
    for curve in &curves {
        for (name, act) in [
            ("silu", Activation::Silu),
            ("gelu", Activation::Gelu),
            ("tanh", Activation::Tanh),
        ] {
            let (code, met) =
                CurvePopulationCode::fit_to_tolerance(act, -8.0, 8.0, curve, 0.01, 256);
            let err = code.max_error(act, 4001);
            println!(
                "{:>14} {name:>5}: {:>3} units, max |error| = {err:.5}",
                curve.neuron,
                code.units.len() + 1
            );
            assert!(
                met,
                "{} {name}: best max error {err} did not meet 0.01",
                curve.neuron
            );
        }
    }
}
