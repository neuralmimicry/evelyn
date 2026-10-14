//! Verify the attachment contract without changing converted weights or codes.
use evelyn::Rng;
use evelyn::activation::Activation;
use evelyn::curve::{CurvePopulationCode, MeasuredCurve};
use evelyn::dense::SwiGluMlp;
use evelyn::mesh::{MeshPlan, export};

#[test]
fn exported_mesh_declares_adjacency_and_preserves_conversion_data() {
    let curve = MeasuredCurve::load_all(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/aarnn-transfer-curves.json"
    ))
    .unwrap()
    .remove(0);
    let code = CurvePopulationCode::fit(Activation::Relu, -1.0, 1.0, 4, 2.0, &curve);
    let plan = MeshPlan {
        gate_code: code.clone(),
        up_code: code,
        gate_range: (-1.0, 1.0),
        up_range: (-1.0, 1.0),
        up_scale: vec![1.0],
        gate_met: false,
        up_met: false,
    };
    let mlp = SwiGluMlp::random(2, 1, &mut Rng::new(20261013));
    let dir = std::env::temp_dir().join(format!("evelyn-export-adjacency-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
    export(&dir, "fixture", &mlp, &plan, &curve, 64, 8).unwrap();
    let mesh: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("mesh.json")).unwrap()).unwrap();
    assert_eq!(
        mesh["input_placement_policy"],
        "sensory_near_existing_outputs"
    );
    assert_eq!(mesh["neuron"], curve.spec.unwrap());
    assert_eq!(mesh["noise_std"].as_f64().unwrap(), curve.noise_std);
    assert_eq!(mesh["gate"]["inputs"], 2);
    assert_eq!(mesh["down"]["outputs"], 2);
    for (dense, file) in [
        (&mlp.gate, "gate.f32"),
        (&mlp.up, "up.f32"),
        (&mlp.down, "down.f32"),
    ] {
        let expected: Vec<u8> = dense
            .weights
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(std::fs::read(dir.join(file)).unwrap(), expected);
    }
    assert_eq!(
        std::fs::read(dir.join("up_scale.f32")).unwrap(),
        1.0f32.to_le_bytes()
    );
}
