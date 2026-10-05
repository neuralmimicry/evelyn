//! Known-answer tests for the GGUF dequantisers, using blocks constructed by
//! hand so the expected values follow directly from the ggml formats.

use evelyn::gguf::{GGML_BF16, GGML_F16, GGML_Q4_K, GGML_Q6_K, GGML_Q8_0, dequantize, f16_to_f32};

fn f16(v: f32) -> [u8; 2] {
    // Exact for the small powers of two and halves used below.
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let frac = ((bits >> 13) & 0x3ff) as u16;
    let h = if v == 0.0 {
        sign
    } else {
        sign | ((exp as u16) << 10) | frac
    };
    h.to_le_bytes()
}

#[test]
fn f16_and_bf16_round_trip() {
    for v in [0.0f32, 1.0, -2.5, 0.5, 65504.0] {
        assert_eq!(f16_to_f32(u16::from_le_bytes(f16(v))), v);
    }
    assert!(f16_to_f32(0x0001) > 0.0 && f16_to_f32(0x0001) < 1e-7); // subnormal
    let raw = f16(1.5);
    assert_eq!(dequantize(GGML_F16, &raw, 1).unwrap(), vec![1.5]);
    let bf = ((-3.0f32).to_bits() >> 16) as u16;
    assert_eq!(
        dequantize(GGML_BF16, &bf.to_le_bytes(), 1).unwrap(),
        vec![-3.0]
    );
}

#[test]
fn q8_0_block() {
    let mut b = Vec::new();
    b.extend_from_slice(&f16(0.5));
    b.extend((0..32).map(|i| (i as i8 - 16) as u8));
    let y = dequantize(GGML_Q8_0, &b, 32).unwrap();
    assert_eq!(y[0], -8.0);
    assert_eq!(y[31], 7.5);
}

#[test]
fn q4_k_block() {
    // d = 1, dmin = 0.5, every sub-block scale 2 and min 1, every quant 3 / 5:
    // value = d * 2 * q - dmin * 1.
    let mut b = Vec::new();
    b.extend_from_slice(&f16(1.0));
    b.extend_from_slice(&f16(0.5));
    let mut sc = [0u8; 12];
    for j in 0..4 {
        sc[j] = 2; // scales 0-3
        sc[j + 4] = 1; // mins 0-3
    }
    for j in 8..12 {
        sc[j] = 2 | (1 << 4); // scales/mins 4-7 (low nibbles), high bits zero
    }
    b.extend_from_slice(&sc);
    b.extend(std::iter::repeat_n(0x53u8, 128)); // low nibble 3, high nibble 5
    let y = dequantize(GGML_Q4_K, &b, 256).unwrap();
    assert_eq!(y[0], 2.0 * 3.0 - 0.5); // first 32 use the low nibble
    assert_eq!(y[32], 2.0 * 5.0 - 0.5); // next 32 use the high nibble
    assert_eq!(y[255], 2.0 * 5.0 - 0.5);
}

#[test]
fn q6_k_block() {
    // All quant bits zero gives q = -32; scales 1, d = 0.25: value = -8.
    let mut b = vec![0u8; 128 + 64];
    b.extend(std::iter::repeat_n(1u8, 16));
    b.extend_from_slice(&f16(0.25));
    let y = dequantize(GGML_Q6_K, &b, 256).unwrap();
    assert!(y.iter().all(|v| *v == -8.0));
}

#[test]
fn length_mismatch_is_an_error() {
    assert!(dequantize(GGML_Q8_0, &[0u8; 34], 64).is_err());
}
