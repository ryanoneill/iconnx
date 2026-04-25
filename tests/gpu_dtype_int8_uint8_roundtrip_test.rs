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
