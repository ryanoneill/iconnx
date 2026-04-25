//! WS-4 M4.3 contract test: htod/dtoh round-trip for INT8 / UINT8
//! GpuTensor must be bit-identical to the source.

#![cfg(feature = "cuda")]

use iconnx::cuda::{IconnxCudaContext, GpuTensor};

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_int8_roundtrip_bit_identical() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let host: Vec<i8> = (-128_i8..=127).step_by(7).collect();
    let g = GpuTensor::from_host_i8(&ctx, &host, vec![host.len()]).expect("from_host_i8");
    assert_eq!(g.shape(), &[host.len()]);
    let back = g.to_host_i8(&ctx).expect("to_host_i8");
    assert_eq!(back, host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_uint8_roundtrip_bit_identical() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let host: Vec<u8> = (0_u8..=255).step_by(11).collect();
    let g = GpuTensor::from_host_u8(&ctx, &host, vec![host.len()]).expect("from_host_u8");
    let back = g.to_host_u8(&ctx).expect("to_host_u8");
    assert_eq!(back, host);
}

/// Wrong-variant accessor must error rather than silently misroute. This
/// freezes the contract that DistilBERT-INT8's quantize/dequantize boundary
/// will rely on at op-execution time (M4.4–M4.6).
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_int8_uint8_wrong_variant_accessors_error() {
    let ctx = IconnxCudaContext::new().expect("ctx");

    let i8_tensor = GpuTensor::from_host_i8(&ctx, &[-1_i8, 0, 1], vec![3]).expect("from_host_i8");
    assert!(i8_tensor.data_u8().is_err(), "i8 tensor must reject data_u8");
    assert!(i8_tensor.to_host_u8(&ctx).is_err(), "i8 tensor must reject to_host_u8");
    assert!(i8_tensor.data_f32().is_err(), "i8 tensor must reject data_f32");

    let u8_tensor = GpuTensor::from_host_u8(&ctx, &[0_u8, 128, 255], vec![3]).expect("from_host_u8");
    assert!(u8_tensor.data_i8().is_err(), "u8 tensor must reject data_i8");
    assert!(u8_tensor.to_host_i8(&ctx).is_err(), "u8 tensor must reject to_host_i8");
    assert!(u8_tensor.data_i32().is_err(), "u8 tensor must reject data_i32");
}
