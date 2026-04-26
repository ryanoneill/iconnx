//! WS-3 M3.6 — Conv FP16-throughout regression tests.
//!
//! Locks in the no-silent-coercion contract at the
//! `gpu_conv2d → gpu_im2col → gpu_gemm` seam. The hand-computed test
//! is the load-bearing one: auditable math + dtype contract assertion
//! at a scale where the expected output is computable by hand. The
//! Whisper-encoder-shape cross-check sits next to it for numerical
//! realism at a real model's first conv.

#![cfg(feature = "cuda")]

use half::f16;
use iconnx::cuda::{
    gpu_conv1d, gpu_conv2d, Conv1dParams, Conv2dParams, ConvKernelCache, DType, GpuMemoryPool,
    GpuTensor, IconnxCudaContext,
};

fn setup() -> (IconnxCudaContext, ConvKernelCache, GpuMemoryPool) {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let cache = ConvKernelCache::new(&ctx).expect("conv kernel cache");
    let pool = GpuMemoryPool::new();
    (ctx, cache, pool)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn conv_float16_stays_float16_through_im2col_gemm() {
    // 1×1×4×4 input × 2×1×3×3 kernel, stride 1 / pad 0 → 1×2×2×2 output.
    // All values in f16 representable range; expected hand-computed.
    let (ctx, cache, mut pool) = setup();

    let input_f16: Vec<f16> = (1..=16).map(|x| f16::from_f32(x as f32)).collect();
    // Two output channels: weight[0] = 0.1..0.9, weight[1] = 1.0..1.8.
    // Note: weight[1] starts at 1.0, not 1.9 — chosen so the hand-computed
    // values stay representable in f16 (stays well below 65504).
    let weight_f16: Vec<f16> = (0..18)
        .map(|i| {
            // First 9 elements: 0.1, 0.2, ..., 0.9
            // Next 9 elements: 1.0, 1.1, ..., 1.8
            let v = if i < 9 {
                (i + 1) as f32 * 0.1
            } else {
                1.0 + (i - 9) as f32 * 0.1
            };
            f16::from_f32(v)
        })
        .collect();

    let input = GpuTensor::from_host_f16(&ctx, &input_f16, vec![1, 1, 4, 4]).unwrap();
    let weight = GpuTensor::from_host_f16(&ctx, &weight_f16, vec![2, 1, 3, 3]).unwrap();

    let params = Conv2dParams {
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };

    let out = gpu_conv2d(&ctx, &cache, &mut pool, &input, &weight, None, &params).unwrap();

    // PRIMARY CONTRACT: FP16 input + FP16 weight → FP16 output. Locks in
    // the no-silent-coercion guarantee at the im2col → GEMM seam.
    assert_eq!(
        out.dtype(),
        DType::Float16,
        "Conv must stay FP16 end-to-end; got {} — silent f32 coercion regression",
        out.dtype().name()
    );
    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    // Sanity-check: out[0, 0, 0, 0] should be the dot product of the
    // top-left 3×3 patch of the input with weight[0]:
    //   patch = [1,2,3, 5,6,7, 9,10,11]
    //   weight[0] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    //   dot = 1*0.1 + 2*0.2 + 3*0.3 + 5*0.4 + 6*0.5 + 7*0.6 + 9*0.7 + 10*0.8 + 11*0.9
    //       = 0.1 + 0.4 + 0.9 + 2.0 + 3.0 + 4.2 + 6.3 + 8.0 + 9.9
    //       = 34.8
    let host = out.to_host_f16(&ctx).unwrap();
    let got_0 = host[0].to_f32();
    assert!(
        (got_0 - 34.8).abs() < 0.5,
        "out[0, 0, 0, 0] = {} (expected ~34.8); precision regression",
        got_0
    );

    // Spot-check out[0, 1, 0, 0] = dot of same patch with weight[1]:
    //   weight[1] = [1.0, 1.1, ..., 1.8]
    //   = 1*1.0 + 2*1.1 + 3*1.2 + 5*1.3 + 6*1.4 + 7*1.5 + 9*1.6 + 10*1.7 + 11*1.8
    //   = 1.0 + 2.2 + 3.6 + 6.5 + 8.4 + 10.5 + 14.4 + 17.0 + 19.8
    //   = 83.4
    // Output index for [0, 1, 0, 0] in a [1, 2, 2, 2] tensor: 1 * 4 + 0 = 4.
    let got_4 = host[4].to_f32();
    assert!(
        (got_4 - 83.4).abs() < 1.0,
        "out[0, 1, 0, 0] = {} (expected ~83.4)",
        got_4
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn conv1d_float16_stays_float16_through_im2col_gemm() {
    // Whisper-Tiny encoder's first Conv is 1-D: [1, 80, 3000] × [384, 80, 3].
    // This test is a small-scale 1-D analog: [1, 2, 6] × [3, 2, 2]. Pure
    // dtype-contract + correctness signal; no padding/stride.
    let (ctx, cache, mut pool) = setup();

    // Input [1, 2, 6]: channel 0 = [1..6], channel 1 = [10, 20, ..., 60].
    let input_f16: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
        f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0),
        f16::from_f32(0.5), f16::from_f32(1.0), f16::from_f32(1.5),
        f16::from_f32(2.0), f16::from_f32(2.5), f16::from_f32(3.0),
    ];
    // Weight [3, 2, 2]: 3 output channels, 2 input channels, kernel size 2.
    // All weights = 0.5 to keep the math auditable.
    let weight_f16: Vec<f16> = vec![f16::from_f32(0.5); 3 * 2 * 2];

    let input = GpuTensor::from_host_f16(&ctx, &input_f16, vec![1, 2, 6]).unwrap();
    let weight = GpuTensor::from_host_f16(&ctx, &weight_f16, vec![3, 2, 2]).unwrap();

    let params = Conv1dParams {
        kernel_size: 2,
        stride: 1,
        padding: 0,
        dilation: 1,
    };

    let out = gpu_conv1d(&ctx, &cache, &mut pool, &input, &weight, None, &params).unwrap();

    // PRIMARY CONTRACT.
    assert_eq!(
        out.dtype(),
        DType::Float16,
        "Conv1D must stay FP16 end-to-end; got {}",
        out.dtype().name()
    );
    // Output shape: [1, 3, 5] (out_length = (6 - 2)/1 + 1 = 5)
    assert_eq!(out.shape(), &[1, 3, 5]);

    // Each output element = 0.5 * (sum of 4 input elements: 2 from each channel).
    // out[0, 0, 0] = 0.5 * (input[0] + input[1] + input[6] + input[7])
    //             = 0.5 * (1 + 2 + 0.5 + 1) = 0.5 * 4.5 = 2.25
    let host = out.to_host_f16(&ctx).unwrap();
    assert!(
        (host[0].to_f32() - 2.25).abs() < 0.05,
        "out[0, 0, 0] = {} (expected 2.25)",
        host[0].to_f32()
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn conv2d_float16_with_bias_stays_float16() {
    // Confirm bias add stays in FP16 too — gpu_conv2d → add_bias_f16_kernel.
    let (ctx, cache, mut pool) = setup();

    // [1, 1, 3, 3] input, [1, 1, 2, 2] kernel (identity-ish), [1] bias.
    let input_f16: Vec<f16> = (1..=9).map(|x| f16::from_f32(x as f32)).collect();
    // weight = [[1, 0], [0, 1]] (2x2 diagonal)
    let weight_f16: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(0.0),
        f16::from_f32(0.0), f16::from_f32(1.0),
    ];
    let bias_f16: Vec<f16> = vec![f16::from_f32(10.0)];

    let input = GpuTensor::from_host_f16(&ctx, &input_f16, vec![1, 1, 3, 3]).unwrap();
    let weight = GpuTensor::from_host_f16(&ctx, &weight_f16, vec![1, 1, 2, 2]).unwrap();
    let bias = GpuTensor::from_host_f16(&ctx, &bias_f16, vec![1]).unwrap();

    let params = Conv2dParams {
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };

    let out = gpu_conv2d(&ctx, &cache, &mut pool, &input, &weight, Some(&bias), &params).unwrap();
    assert_eq!(out.dtype(), DType::Float16);
    assert_eq!(out.shape(), &[1, 1, 2, 2]);

    // out[0, 0, 0, 0] = 1 + 5 + 10 (bias) = 16
    // out[0, 0, 0, 1] = 2 + 6 + 10 = 18
    // out[0, 0, 1, 0] = 4 + 8 + 10 = 22
    // out[0, 0, 1, 1] = 5 + 9 + 10 = 24
    let host = out.to_host_f16(&ctx).unwrap();
    assert!((host[0].to_f32() - 16.0).abs() < 0.1);
    assert!((host[1].to_f32() - 18.0).abs() < 0.1);
    assert!((host[2].to_f32() - 22.0).abs() < 0.1);
    assert!((host[3].to_f32() - 24.0).abs() < 0.1);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn conv2d_float16_matches_float32_within_tolerance() {
    // Cross-check: same conv computed in f32 and f16, results must agree
    // within FP16 ULP-shaped tolerance. Confirms the FP16 path's accumulator
    // strategy (gemm_fp16_acc32 + memcpy-class im2col) doesn't drift from
    // the f32 reference.
    let (ctx, cache, mut pool) = setup();

    // Small fixture — values chosen for benign f16 round-trip.
    let n = 1usize;
    let c_in = 4usize;
    let h = 8usize;
    let w = 8usize;
    let c_out = 8usize;
    let kh = 3usize;
    let kw = 3usize;

    let input_f32: Vec<f32> = (0..n * c_in * h * w)
        .map(|i| ((i % 11) as f32 + 1.0) * 0.1)
        .collect();
    let weight_f32: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    let input_f16: Vec<f16> = input_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let weight_f16: Vec<f16> = weight_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let params = Conv2dParams {
        kernel_h: kh,
        kernel_w: kw,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
    };

    let input_32 = GpuTensor::from_host_f32(&ctx, &input_f32, vec![n, c_in, h, w]).unwrap();
    let weight_32 =
        GpuTensor::from_host_f32(&ctx, &weight_f32, vec![c_out, c_in, kh, kw]).unwrap();

    let input_16 = GpuTensor::from_host_f16(&ctx, &input_f16, vec![n, c_in, h, w]).unwrap();
    let weight_16 =
        GpuTensor::from_host_f16(&ctx, &weight_f16, vec![c_out, c_in, kh, kw]).unwrap();

    let out_32 =
        gpu_conv2d(&ctx, &cache, &mut pool, &input_32, &weight_32, None, &params).unwrap();
    let out_16 =
        gpu_conv2d(&ctx, &cache, &mut pool, &input_16, &weight_16, None, &params).unwrap();

    assert_eq!(out_16.dtype(), DType::Float16);
    assert_eq!(out_32.dtype(), DType::Float32);
    assert_eq!(out_16.shape(), out_32.shape());

    let h32 = out_32.to_host_f32(&ctx).unwrap();
    let h16: Vec<f32> = out_16.to_host_f16(&ctx).unwrap().iter().map(|h| h.to_f32()).collect();

    // Tolerance shaped to f16 precision: relative error ≤ 1e-2 OR
    // absolute error ≤ 5e-3 (whichever is looser at each magnitude).
    for (i, (&f32_v, &f16_v)) in h32.iter().zip(h16.iter()).enumerate() {
        let abs_err = (f32_v - f16_v).abs();
        let rel_err = abs_err / f32_v.abs().max(1e-6);
        assert!(
            rel_err <= 1e-2 || abs_err <= 5e-3,
            "idx {}: f32={} f16={} (rel_err={} abs_err={})",
            i,
            f32_v,
            f16_v,
            rel_err,
            abs_err
        );
    }
}
