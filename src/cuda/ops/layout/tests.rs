use super::*;
use crate::cuda::memory_pool::GpuMemoryPool;

fn setup() -> (IconnxCudaContext, OpsKernelCache, GpuMemoryPool) {
    let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
    let cache = OpsKernelCache::new(&ctx).expect("Failed to compile kernels");
    let pool = GpuMemoryPool::new();
    (ctx, cache, pool)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_shape() {
    let (ctx, _cache, _pool) = setup();

    // Test shape of 1D tensor
    let input_1d = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0], vec![3]).unwrap();
    let shape_1d = gpu_shape(&ctx, &input_1d).unwrap();
    let result_1d = shape_1d.to_host_i64(&ctx).unwrap();
    assert_eq!(result_1d, vec![3i64]);
    assert_eq!(shape_1d.shape(), &[1]);

    // Test shape of 2D tensor
    let input_2d = GpuTensor::from_host_f32(&ctx, &[1.0; 6], vec![2, 3]).unwrap();
    let shape_2d = gpu_shape(&ctx, &input_2d).unwrap();
    let result_2d = shape_2d.to_host_i64(&ctx).unwrap();
    assert_eq!(result_2d, vec![2i64, 3i64]);
    assert_eq!(shape_2d.shape(), &[2]);

    // Test shape of 3D tensor
    let input_3d = GpuTensor::from_host_f32(&ctx, &[1.0; 24], vec![2, 3, 4]).unwrap();
    let shape_3d = gpu_shape(&ctx, &input_3d).unwrap();
    let result_3d = shape_3d.to_host_i64(&ctx).unwrap();
    assert_eq!(result_3d, vec![2i64, 3i64, 4i64]);
    assert_eq!(shape_3d.shape(), &[3]);

    // Test shape of Int64 tensor (Shape should work on any dtype)
    let input_i64 = GpuTensor::from_host_i64(&ctx, &[1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let shape_i64 = gpu_shape(&ctx, &input_i64).unwrap();
    let result_i64 = shape_i64.to_host_i64(&ctx).unwrap();
    assert_eq!(result_i64, vec![2i64, 3i64]);

    println!("Shape test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_constant_of_shape() {
    let (ctx, _cache, _pool) = setup();

    // Create shape tensor: [2, 3]
    let shape_tensor = GpuTensor::from_host_i64(&ctx, &[2, 3], vec![2]).unwrap();

    // Create tensor filled with 5.0
    let output = gpu_constant_of_shape(&ctx, &shape_tensor, 5.0).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(result.len(), 6);
    assert!(result.iter().all(|&v| (v - 5.0).abs() < 1e-6));

    // Test with default value 0.0
    let shape_tensor_3d = GpuTensor::from_host_i64(&ctx, &[2, 2, 2], vec![3]).unwrap();
    let output_zeros = gpu_constant_of_shape(&ctx, &shape_tensor_3d, 0.0).unwrap();
    let result_zeros = output_zeros.to_host_f32(&ctx).unwrap();

    assert_eq!(output_zeros.shape(), &[2, 2, 2]);
    assert_eq!(result_zeros.len(), 8);
    assert!(result_zeros.iter().all(|&v| v.abs() < 1e-6));

    // Test with negative value
    let shape_tensor_1d = GpuTensor::from_host_i64(&ctx, &[4], vec![1]).unwrap();
    let output_neg = gpu_constant_of_shape(&ctx, &shape_tensor_1d, -3.5).unwrap();
    let result_neg = output_neg.to_host_f32(&ctx).unwrap();

    assert_eq!(output_neg.shape(), &[4]);
    assert!(result_neg.iter().all(|&v| (v - (-3.5)).abs() < 1e-6));

    println!("ConstantOfShape test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_split_axis0_unequal_f32() {
    // Split a 6x2 Float32 input along axis 0 into [1, 3, 2] rows.
    let (ctx, cache, mut pool) = setup();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![6, 2]).expect("input");

    let outs = gpu_split(&ctx, &cache, &mut pool, &input, 0, &[1, 3, 2])
        .expect("gpu_split");
    assert_eq!(outs.len(), 3);
    assert_eq!(outs[0].shape(), &[1, 2]);
    assert_eq!(outs[1].shape(), &[3, 2]);
    assert_eq!(outs[2].shape(), &[2, 2]);

    let r0 = outs[0].to_host_f32(&ctx).unwrap();
    let r1 = outs[1].to_host_f32(&ctx).unwrap();
    let r2 = outs[2].to_host_f32(&ctx).unwrap();
    assert_eq!(r0, vec![0.0, 1.0]);
    assert_eq!(r1, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(r2, vec![8.0, 9.0, 10.0, 11.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_split_negative_axis_int64() {
    // Split a 2x6 Int64 input along axis -1 into [2, 2, 2] cols.
    let (ctx, cache, mut pool) = setup();
    let data: Vec<i64> = (0..12).collect();
    let input = GpuTensor::from_host_i64(&ctx, &data, vec![2, 6]).expect("input");

    let outs = gpu_split(&ctx, &cache, &mut pool, &input, -1, &[2, 2, 2])
        .expect("gpu_split");
    assert_eq!(outs.len(), 3);
    for out in &outs {
        assert_eq!(out.shape(), &[2, 2]);
    }
    assert_eq!(outs[0].to_host_i64(&ctx).unwrap(), vec![0, 1, 6, 7]);
    assert_eq!(outs[1].to_host_i64(&ctx).unwrap(), vec![2, 3, 8, 9]);
    assert_eq!(outs[2].to_host_i64(&ctx).unwrap(), vec![4, 5, 10, 11]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_split_size_sum_mismatch_errors() {
    let (ctx, cache, mut pool) = setup();
    let input = GpuTensor::from_host_f32(&ctx, &[0.0_f32; 6], vec![6]).expect("input");

    // Sizes sum to 5, not 6 — should error.
    match gpu_split(&ctx, &cache, &mut pool, &input, 0, &[2, 3]) {
        Err(CudaError::Kernel(msg)) => {
            assert!(
                msg.contains("Split") && msg.contains("sizes"),
                "expected Split sizes-sum error, got: {}",
                msg
            );
        }
        Err(other) => panic!("expected Kernel error, got: {}", other),
        Ok(_) => panic!("expected error for sizes-sum mismatch, got Ok"),
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_constant_of_shape_direct_i64_fills_with_int64_scalar() {
    let (ctx, cache, mut pool) = setup();

    // Target shape [3, 4] filled with i64 value 42.
    let output = gpu_constant_of_shape_direct_i64(&ctx, &cache, &mut pool, vec![3, 4], 42)
        .expect("gpu_constant_of_shape_direct_i64");
    assert_eq!(output.shape(), &[3, 4]);
    let result = output.to_host_i64(&ctx).expect("to_host_i64");
    assert_eq!(result.len(), 12);
    assert!(
        result.iter().all(|&v| v == 42),
        "expected all 42s, got {:?}",
        result
    );

    // Negative value + 1-D shape.
    let neg = gpu_constant_of_shape_direct_i64(&ctx, &cache, &mut pool, vec![5], -7)
        .expect("gpu_constant_of_shape_direct_i64 negative");
    assert_eq!(neg.shape(), &[5]);
    let neg_result = neg.to_host_i64(&ctx).expect("to_host_i64");
    assert!(neg_result.iter().all(|&v| v == -7));

    // Zero-element target shape — takes the `total_elements == 0`
    // early path. Must succeed without launching the kernel.
    let empty = gpu_constant_of_shape_direct_i64(&ctx, &cache, &mut pool, vec![0], 1)
        .expect("zero-element target shape");
    assert_eq!(empty.shape(), &[0]);
    assert_eq!(empty.to_host_i64(&ctx).unwrap().len(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_transpose_2d() {
    let (ctx, cache, mut pool) = setup();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![2, 3]).unwrap();

    let output = gpu_transpose_2d(&ctx, &cache, &mut pool, &input).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
    }

    println!("Transpose 2D test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_where() {
    let (ctx, cache, mut pool) = setup();

    // WS-3 M3.5 sub-B: condition is now native Bool.
    let condition = GpuTensor::from_host_bool(&ctx, &[true, false, true, false, true], vec![5]).unwrap();

    let x_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
    let x = GpuTensor::from_host_f32(&ctx, &x_data, vec![5]).unwrap();

    let y_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let y = GpuTensor::from_host_f32(&ctx, &y_data, vec![5]).unwrap();

    let output = gpu_where(&ctx, &cache, &mut pool, &condition, &x, &y).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    let expected = [10.0, 2.0, 30.0, 4.0, 50.0];

    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
    }

    println!("Where test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather() {
    let (ctx, cache, mut pool) = setup();

    let input_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![4, 3]).unwrap();

    let indices = vec![0i64, 2, 1];
    let indices_shape = vec![3]; // 1D shape
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &indices_shape).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[3, 3]);
    let expected = vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0];

    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
    }

    println!("Gather test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_contiguous() {
    let (ctx, cache, mut pool) = setup();

    let input_data: Vec<f32> = (0..10).map(|x| x as f32).collect();
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![10]).unwrap();

    let output = gpu_slice_contiguous(&ctx, &cache, &mut pool, &input, 3, 4, vec![4]).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[4]);
    let expected = [3.0, 4.0, 5.0, 6.0];

    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
    }

    println!("Slice contiguous test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_copy() {
    let (ctx, cache, mut pool) = setup();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    let output = gpu_copy(&ctx, &cache, &mut pool, &input).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(result, input_data);
    assert_eq!(output.shape(), input.shape());

    println!("Copy test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_copy_i64_polymorphic() {
    // gpu_copy must handle Int64 — shape-arithmetic graphs route Int64
    // tensors through Identity nodes, which used to panic with
    // `data_f32 called on int64 tensor`.
    let (ctx, cache, mut pool) = setup();
    let input_data: Vec<i64> = vec![10, -20, 30, -40, 50];
    let input = GpuTensor::from_host_i64(&ctx, &input_data, vec![5]).unwrap();

    let output = gpu_copy(&ctx, &cache, &mut pool, &input).unwrap();
    let result = output.to_host_i64(&ctx).unwrap();

    assert_eq!(result, input_data);
    assert_eq!(output.shape(), input.shape());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_copy_i32_polymorphic() {
    let (ctx, cache, mut pool) = setup();
    let input_data: Vec<i32> = vec![1, -2, 3, -4, 5];
    let input = GpuTensor::from_host_i32(&ctx, &input_data, vec![5]).unwrap();

    let output = gpu_copy(&ctx, &cache, &mut pool, &input).unwrap();
    let result = output.to_host_i32(&ctx).unwrap();

    assert_eq!(result, input_data);
    assert_eq!(output.shape(), input.shape());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_concat() {
    let (ctx, cache, mut pool) = setup();

    // Concat two tensors along axis 0
    let a_data = vec![1.0f32, 2.0, 3.0];
    let b_data = vec![4.0f32, 5.0, 6.0];
    let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![1, 3]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![1, 3]).unwrap();

    let output = gpu_concat(&ctx, &cache, &mut pool, &[&a, &b], 0).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(result, expected);

    // Concat along axis 1
    let c = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0], vec![2, 1]).unwrap();
    let d = GpuTensor::from_host_f32(&ctx, &[3.0f32, 4.0], vec![2, 1]).unwrap();

    let output2 = gpu_concat(&ctx, &cache, &mut pool, &[&c, &d], 1).unwrap();
    let result2 = output2.to_host_f32(&ctx).unwrap();

    assert_eq!(output2.shape(), &[2, 2]);
    assert_eq!(result2, vec![1.0, 3.0, 2.0, 4.0]);

    println!("Concat test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_expand() {
    let (ctx, cache, mut pool) = setup();

    // Expand [1, 3] to [2, 3]
    let input_data = vec![1.0f32, 2.0, 3.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 3]).unwrap();

    let output = gpu_expand(&ctx, &cache, &mut pool, &input, vec![2, 3]).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    let expected = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    assert_eq!(result, expected);

    // Expand [3] to [2, 3]
    let input2 = GpuTensor::from_host_f32(&ctx, &input_data, vec![3]).unwrap();
    let output2 = gpu_expand(&ctx, &cache, &mut pool, &input2, vec![2, 3]).unwrap();
    let result2 = output2.to_host_f32(&ctx).unwrap();

    assert_eq!(output2.shape(), &[2, 3]);
    assert_eq!(result2, expected);

    println!("Expand test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_pad() {
    let (ctx, cache, mut pool) = setup();

    // Pad [2, 2] with 1 on each side -> [4, 4]
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![2, 2]).unwrap();

    // pads: [begin_dim0, begin_dim1, end_dim0, end_dim1]
    let pads = vec![1i64, 1, 1, 1];
    let output = gpu_pad(&ctx, &cache, &mut pool, &input, &pads, 0.0, "constant").unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[4, 4]);
    // Expected layout:
    // 0 0 0 0
    // 0 1 2 0
    // 0 3 4 0
    // 0 0 0 0
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(result, expected);

    println!("Pad test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_resize() {
    let (ctx, cache, mut pool) = setup();

    // Test upscaling: [2, 2] -> [4, 4] (scale 2x) with asymmetric + floor
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![2, 2]).unwrap();

    let scales = vec![2.0f32, 2.0];
    let output = gpu_resize(
        &ctx,
        &cache,
        &mut pool,
        &input,
        &scales,
        ResizeCoordMode::Asymmetric,
        ResizeNearestMode::Floor,
        ResizeMode::Nearest,
    )
    .unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[4, 4]);
    // With asymmetric + floor, each input pixel maps to a 2x2 block
    // Expected: [[1,1,2,2], [1,1,2,2], [3,3,4,4], [3,3,4,4]]
    let expected = vec![
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(result, expected);

    // Test downscaling: [4, 4] -> [2, 2] (scale 0.5x)
    let input2_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let input2 = GpuTensor::from_host_f32(&ctx, &input2_data, vec![4, 4]).unwrap();

    let scales2 = vec![0.5f32, 0.5];
    let output2 = gpu_resize(
        &ctx,
        &cache,
        &mut pool,
        &input2,
        &scales2,
        ResizeCoordMode::Asymmetric,
        ResizeNearestMode::Floor,
        ResizeMode::Nearest,
    )
    .unwrap();
    let result2 = output2.to_host_f32(&ctx).unwrap();

    assert_eq!(output2.shape(), &[2, 2]);
    // Nearest neighbor samples at coordinates mapped back
    // Output[0,0] <- Input[0,0], Output[0,1] <- Input[0,2]
    // Output[1,0] <- Input[2,0], Output[1,1] <- Input[2,2]
    let expected2 = vec![0.0, 2.0, 8.0, 10.0];
    assert_eq!(result2, expected2);

    // Test 1D resize: [4] -> [8] (scale 2x)
    let input3_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input3 = GpuTensor::from_host_f32(&ctx, &input3_data, vec![4]).unwrap();

    let scales3 = vec![2.0f32];
    let output3 = gpu_resize(
        &ctx,
        &cache,
        &mut pool,
        &input3,
        &scales3,
        ResizeCoordMode::Asymmetric,
        ResizeNearestMode::Floor,
        ResizeMode::Nearest,
    )
    .unwrap();
    let result3 = output3.to_host_f32(&ctx).unwrap();

    assert_eq!(output3.shape(), &[8]);
    // [1,1,2,2,3,3,4,4]
    let expected3 = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    assert_eq!(result3, expected3);

    println!("Resize test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_resize_half_pixel() {
    let (ctx, cache, mut pool) = setup();

    // Test half_pixel + floor (common ONNX configuration)
    // For 2x upsampling: half_pixel maps (out + 0.5) / scale - 0.5
    // Input [0,1,2,3] with scale=2:
    //   out=0: (0.5)/2 - 0.5 = -0.25, floor -> 0, clamp -> 0
    //   out=1: (1.5)/2 - 0.5 = 0.25, floor -> 0
    //   out=2: (2.5)/2 - 0.5 = 0.75, floor -> 0
    //   out=3: (3.5)/2 - 0.5 = 1.25, floor -> 1
    //   out=4: (4.5)/2 - 0.5 = 1.75, floor -> 1
    //   out=5: (5.5)/2 - 0.5 = 2.25, floor -> 2
    //   out=6: (6.5)/2 - 0.5 = 2.75, floor -> 2
    //   out=7: (7.5)/2 - 0.5 = 3.25, floor -> 3
    // ORT output: [0, 0, 0, 1, 1, 2, 2, 3]

    let input_data = vec![0.0f32, 1.0, 2.0, 3.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 4]).unwrap();

    let scales = vec![1.0f32, 1.0, 2.0];
    let output = gpu_resize(
        &ctx,
        &cache,
        &mut pool,
        &input,
        &scales,
        ResizeCoordMode::HalfPixel,
        ResizeNearestMode::Floor,
        ResizeMode::Nearest,
    )
    .unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[1, 1, 8]);
    // ORT reference: [0, 0, 0, 1, 1, 2, 2, 3]
    let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
    assert_eq!(result, expected, "GPU half_pixel+floor should match ORT output");

    println!("Resize half_pixel test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_resize_half_pixel_300x() {
    let (ctx, cache, mut pool) = setup();

    // Test 300x upsampling like the Kokoro model uses for l_sin_gen path
    // Input: [1, 9, 70] with cumulative phase values
    // Output: [1, 9, 21000]
    // With half_pixel + floor

    // Create input with cumulative values (like phase)
    // Values from 0 to ~100 (70 values per channel)
    let mut input_data = Vec::with_capacity(9 * 70);
    for ch in 0..9 {
        for i in 0..70 {
            input_data.push((ch as f32 * 100.0) + (i as f32 * 1.5)); // 0, 1.5, 3, 4.5, ... up to ~105
        }
    }

    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 9, 70]).unwrap();

    let scales = vec![1.0f32, 1.0, 300.0];
    let output = gpu_resize(
        &ctx,
        &cache,
        &mut pool,
        &input,
        &scales,
        ResizeCoordMode::HalfPixel,
        ResizeNearestMode::Floor,
        ResizeMode::Nearest,
    )
    .unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[1, 9, 21000]);

    // Verify some key positions using half_pixel formula:
    // For scale=300, half_pixel: input_idx = floor((out_idx + 0.5) / 300 - 0.5)
    // out=0: (0.5)/300 - 0.5 = -0.4983, floor=-1, clamp=0
    // out=149: (149.5)/300 - 0.5 = -0.0017, floor=-1, clamp=0
    // out=150: (150.5)/300 - 0.5 = 0.00167, floor=0
    // out=449: (449.5)/300 - 0.5 = 0.998, floor=0
    // out=450: (450.5)/300 - 0.5 = 1.00167, floor=1

    // Check channel 0: input[0..70] maps to phases 0.0, 1.5, 3.0, ...
    // out[0..149] should all come from input[0] = 0.0
    for (i, &val) in result.iter().enumerate().take(150) {
        assert!(
            (val - 0.0).abs() < 0.001,
            "Position {} should be 0.0, got {}",
            i,
            val
        );
    }
    // out[150..449] should all come from input[0] = 0.0
    // Actually let me recalculate...
    // out=150: floor((150.5)/300 - 0.5) = floor(0.00167) = 0 -> input[0]
    // out=449: floor((449.5)/300 - 0.5) = floor(0.998) = 0 -> input[0]
    // out=450: floor((450.5)/300 - 0.5) = floor(1.00167) = 1 -> input[1] = 1.5
    for (i, &val) in result.iter().enumerate().take(450).skip(150) {
        assert!(
            (val - 0.0).abs() < 0.001,
            "Position {} should be 0.0 (from input[0]), got {}",
            i,
            val
        );
    }
    assert!(
        (result[450] - 1.5).abs() < 0.001,
        "Position 450 should be 1.5 (from input[1]), got {}",
        result[450]
    );

    println!("Resize half_pixel 300x test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_resize_gpu_vs_cpu() {
    use crate::attributes::NodeAttributes;
    use crate::operators::resize::Resize as CpuResize;
    use crate::tensor::Tensor;

    let (ctx, cache, mut pool) = setup();

    // Test half_pixel + floor (what Kokoro model uses for l_sin_gen path)
    let input_data: Vec<f32> = (0..70).map(|i| i as f32 * 1.5).collect();

    // GPU
    let gpu_input = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 70]).unwrap();
    let gpu_output = gpu_resize(
        &ctx,
        &cache,
        &mut pool,
        &gpu_input,
        &[1.0, 1.0, 300.0],
        ResizeCoordMode::HalfPixel,
        ResizeNearestMode::Floor,
        ResizeMode::Nearest,
    )
    .unwrap();
    let gpu_result = gpu_output.to_host_f32(&ctx).unwrap();

    // CPU
    let cpu_input = Tensor::from_vec_f32(input_data.clone(), vec![1, 1, 70]);
    let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 300.0], vec![3]);
    let roi = Tensor::from_vec_f32(vec![], vec![0]);
    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "nearest".to_string());
    attrs.add_string(
        "coordinate_transformation_mode".to_string(),
        "half_pixel".to_string(),
    );
    attrs.add_string("nearest_mode".to_string(), "floor".to_string());
    let cpu_output = CpuResize::forward(&[cpu_input, roi, scales], &attrs);
    let cpu_result = cpu_output.as_slice();

    // Compare
    assert_eq!(gpu_result.len(), cpu_result.len(), "Output lengths must match");

    let mut max_diff = 0.0f32;
    for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        let diff = (g - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-5 {
            println!(
                "Mismatch at position {}: GPU={}, CPU={}, diff={}",
                i, g, c, diff
            );
        }
    }

    println!("GPU vs CPU Resize max_diff: {}", max_diff);
    assert!(
        max_diff < 1e-5,
        "GPU and CPU Resize outputs should match, max_diff={}",
        max_diff
    );

    println!("GPU vs CPU Resize test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_nonzero() {
    let (ctx, _cache, _pool) = setup();

    // Test 1D input
    let input_data = vec![0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![6]).unwrap();

    let output = gpu_nonzero(&ctx, &input).unwrap();
    let result = output.to_host_i64(&ctx).unwrap();

    // Non-zero at indices 1, 3, 5
    assert_eq!(output.shape(), &[1, 3]);
    assert_eq!(result, vec![1i64, 3, 5]);

    // Test 2D input
    // [[0, 1, 2],
    //  [3, 0, 0]]
    let input_2d_data = vec![0.0f32, 1.0, 2.0, 3.0, 0.0, 0.0];
    let input_2d = GpuTensor::from_host_f32(&ctx, &input_2d_data, vec![2, 3]).unwrap();

    let output_2d = gpu_nonzero(&ctx, &input_2d).unwrap();
    let result_2d = output_2d.to_host_i64(&ctx).unwrap();

    // Non-zero at: (0,1), (0,2), (1,0)
    // Output shape: (2, 3) - 2 dims, 3 non-zeros
    assert_eq!(output_2d.shape(), &[2, 3]);
    // Row indices: [0, 0, 1], Col indices: [1, 2, 0]
    assert_eq!(result_2d, vec![0i64, 0, 1, 1, 2, 0]);

    // Test all zeros
    let zeros = GpuTensor::from_host_f32(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();
    let output_zeros = gpu_nonzero(&ctx, &zeros).unwrap();
    assert_eq!(output_zeros.shape(), &[1, 0]);

    // Test all non-zeros
    let all_nonzero = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0], vec![3]).unwrap();
    let output_all = gpu_nonzero(&ctx, &all_nonzero).unwrap();
    let result_all = output_all.to_host_i64(&ctx).unwrap();
    assert_eq!(output_all.shape(), &[1, 3]);
    assert_eq!(result_all, vec![0i64, 1, 2]);

    println!("NonZero test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_nonzero_bool() {
    // M3.7c: Kokoro's istft chain feeds a Bool comparison output into
    // NonZero. Pre-fix this site unconditionally read host bytes via
    // `to_host_f32` and panicked. Bool input must be accepted.
    let (ctx, _cache, _pool) = setup();

    let input = GpuTensor::from_host_bool(
        &ctx,
        &[false, true, false, true, true, false],
        vec![6],
    )
    .unwrap();
    let output = gpu_nonzero(&ctx, &input).unwrap();
    let result = output.to_host_i64(&ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_eq!(result, vec![1i64, 3, 4]);

    // 2D Bool: [[true, false], [false, true]] → indices (0,0) and (1,1).
    let input_2d =
        GpuTensor::from_host_bool(&ctx, &[true, false, false, true], vec![2, 2]).unwrap();
    let output_2d = gpu_nonzero(&ctx, &input_2d).unwrap();
    let result_2d = output_2d.to_host_i64(&ctx).unwrap();
    assert_eq!(output_2d.shape(), &[2, 2]);
    // Row indices [0, 1], Col indices [0, 1].
    assert_eq!(result_2d, vec![0i64, 1, 0, 1]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_where_i32() {
    let (ctx, cache, mut pool) = setup();

    // WS-3 M3.5 sub-B: condition is native Bool.
    let condition = GpuTensor::from_host_bool(&ctx, &[true, false, true, false, true], vec![5]).unwrap();

    let x_data = vec![10i32, 20, 30, 40, 50];
    let x = GpuTensor::from_host_i32(&ctx, &x_data, vec![5]).unwrap();

    let y_data = vec![1i32, 2, 3, 4, 5];
    let y = GpuTensor::from_host_i32(&ctx, &y_data, vec![5]).unwrap();

    let output = gpu_where(&ctx, &cache, &mut pool, &condition, &x, &y).unwrap();
    let result = output.to_host_i32(&ctx).unwrap();

    let expected = vec![10, 2, 30, 4, 50];
    assert_eq!(result, expected);

    println!("Where Int32 test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_concat_i32() {
    let (ctx, cache, mut pool) = setup();

    // Concat two Int32 tensors along axis 0
    let a_data = vec![1i32, 2, 3];
    let b_data = vec![4i32, 5, 6];

    let a = GpuTensor::from_host_i32(&ctx, &a_data, vec![3]).unwrap();
    let b = GpuTensor::from_host_i32(&ctx, &b_data, vec![3]).unwrap();

    let output = gpu_concat(&ctx, &cache, &mut pool, &[&a, &b], 0).unwrap();
    let result = output.to_host_i32(&ctx).unwrap();

    assert_eq!(output.shape(), &[6]);
    assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);

    println!("Concat Int32 test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_expand_i32() {
    let (ctx, cache, mut pool) = setup();

    // Expand [1, 3] to [2, 3]
    let input_data = vec![1i32, 2, 3];
    let input = GpuTensor::from_host_i32(&ctx, &input_data, vec![1, 3]).unwrap();

    let output = gpu_expand(&ctx, &cache, &mut pool, &input, vec![2, 3]).unwrap();
    let result = output.to_host_i32(&ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    let expected = vec![1, 2, 3, 1, 2, 3];
    assert_eq!(result, expected);

    println!("Expand Int32 test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_expand_bool() {
    // M3.7b: GPT-2 / DistilBERT-INT8 attention path broadcasts a Bool
    // mask via Expand. Bool is u8 underneath, byte-mirror kernel
    // produces identical broadcast.
    let (ctx, cache, mut pool) = setup();

    // Expand [1, 3] -> [2, 3]: each row repeats the input.
    let input_data = vec![true, false, true];
    let input = GpuTensor::from_host_bool(&ctx, &input_data, vec![1, 3]).unwrap();

    let output = gpu_expand(&ctx, &cache, &mut pool, &input, vec![2, 3]).unwrap();
    let result = output.to_host_bool(&ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(
        result,
        vec![true, false, true, true, false, true],
        "Expand Bool [1,3]→[2,3] should repeat the row"
    );

    // Trailing-1 broadcast: [3, 1] -> [3, 4].
    let trailing_data = vec![true, false, true];
    let trailing = GpuTensor::from_host_bool(&ctx, &trailing_data, vec![3, 1]).unwrap();
    let trailing_out = gpu_expand(&ctx, &cache, &mut pool, &trailing, vec![3, 4]).unwrap();
    let trailing_result = trailing_out.to_host_bool(&ctx).unwrap();
    assert_eq!(trailing_out.shape(), &[3, 4]);
    assert_eq!(
        trailing_result,
        vec![
            true, true, true, true,
            false, false, false, false,
            true, true, true, true,
        ],
        "Expand Bool [3,1]→[3,4] should broadcast each scalar across the new axis"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_nd_i32() {
    let (ctx, cache, mut pool) = setup();

    // Create a 2D Int32 tensor [4, 3]
    let input_data: Vec<i32> = (0..12).collect();
    let input = GpuTensor::from_host_i32(&ctx, &input_data, vec![4, 3]).unwrap();

    // Slice rows 1-3 (exclusive), all columns: out = input[1:3, :]
    let output = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[1], &[3], Some(&[0]), None).unwrap();
    let result = output.to_host_i32(&ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    // Expected: rows 1 and 2 => [3, 4, 5, 6, 7, 8]
    assert_eq!(result, vec![3, 4, 5, 6, 7, 8]);

    println!("Slice ND Int32 test passed!");
}

// =============================================================================
// Gather Edge Case Tests
// =============================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_negative_indices() {
    let (ctx, cache, mut pool) = setup();

    // Create a 2D tensor [4, 3]: 4 rows, 3 columns
    let input_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![4, 3]).unwrap();

    // Gather with negative indices: -1 = last row, -2 = second-to-last
    let indices = vec![-1i64, -2];
    let indices_shape = vec![2];
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &indices_shape).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    // -1 => row 3 (9, 10, 11), -2 => row 2 (6, 7, 8)
    let expected = vec![9.0, 10.0, 11.0, 6.0, 7.0, 8.0];
    assert_eq!(result, expected, "Negative indices should work");

    println!("Gather negative indices test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_boundary_indices() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [5]
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    // Gather first and last elements
    let indices = vec![0i64, 4, 0, 4];
    let indices_shape = vec![4];
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &indices_shape).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[4]);
    let expected = vec![10.0, 50.0, 10.0, 50.0];
    assert_eq!(result, expected, "Boundary indices should work");

    println!("Gather boundary indices test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_out_of_bounds_clamped() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [5]
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    // Out-of-bounds index should be clamped to last valid index
    let indices = vec![10i64]; // Way out of bounds
    let indices_shape = vec![1];
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &indices_shape).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[1]);
    // Index 10 should clamp to index 4 (last element = 50.0)
    assert_eq!(result, vec![50.0], "Out-of-bounds should clamp to last element");

    println!("Gather out-of-bounds clamped test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_very_negative_index_clamped() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [5]
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    // Very negative index: -100 + 5 = -95, should clamp to 0
    let indices = vec![-100i64];
    let indices_shape = vec![1];
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &indices_shape).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[1]);
    // Index -100 should clamp to index 0 (first element = 10.0)
    assert_eq!(result, vec![10.0], "Very negative index should clamp to first element");

    println!("Gather very negative index clamped test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_i64_negative_indices() {
    let (ctx, cache, mut pool) = setup();

    // Create a 2D Int64 tensor [3, 2]
    let input_data: Vec<i64> = vec![100, 101, 200, 201, 300, 301];
    let input = GpuTensor::from_host_i64(&ctx, &input_data, vec![3, 2]).unwrap();

    // Gather with negative index: -1 = last row
    let indices = vec![-1i64];
    let indices_shape = vec![1];
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &indices_shape).unwrap();
    let result = output.to_host_i64(&ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(result, vec![300, 301], "Int64 negative index should work");

    println!("Gather Int64 negative indices test passed!");
}

// =============================================================================
// Slice Edge Case Tests
// =============================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_nd_negative_indices() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [10]
    let input_data: Vec<f32> = (0..10).map(|x| x as f32).collect();
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![10]).unwrap();

    // Slice with negative end: [2:-2] = [2, 3, 4, 5, 6, 7]
    let output = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[2], &[-2], None, None).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[6]);
    let expected: Vec<f32> = (2..8).map(|x| x as f32).collect();
    assert_eq!(result, expected, "Negative end index should work");

    println!("Slice ND negative indices test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_nd_negative_start() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [5]
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    // Slice with negative start: [-3:] = [30, 40, 50]
    let output = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[-3], &[i64::MAX], None, None).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[3]);
    assert_eq!(result, vec![30.0, 40.0, 50.0], "Negative start should work");

    println!("Slice ND negative start test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_nd_negative_step() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [5]
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    // Reverse slice: [4:0:-1] = [50, 40, 30, 20]
    let output = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[4], &[0], None, Some(&[-1])).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[4]);
    assert_eq!(result, vec![50.0, 40.0, 30.0, 20.0], "Negative step should reverse");

    println!("Slice ND negative step test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_nd_empty_output() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [5]
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    // Empty slice: start >= end with positive step
    let output = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[3], &[2], None, None).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[0]);
    assert_eq!(result.len(), 0, "Empty slice should have no elements");

    println!("Slice ND empty output test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_nd_negative_axis() {
    let (ctx, cache, mut pool) = setup();

    // Create a 2D tensor [3, 4]
    let input_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![3, 4]).unwrap();

    // Slice with negative axis: axis=-1 (last axis = axis 1)
    // Slice columns 1:3
    let output = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[1], &[3], Some(&[-1]), None).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    // Each row gets columns 1 and 2
    let expected = vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0];
    assert_eq!(result, expected, "Negative axis should work");

    println!("Slice ND negative axis test passed!");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_slice_nd_boundary_slice() {
    let (ctx, cache, mut pool) = setup();

    // Create a 1D tensor [5]
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

    // Slice entire tensor: [0:5]
    let output = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[0], &[5], None, None).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(output.shape(), &[5]);
    assert_eq!(result, input_data, "Full slice should equal input");

    // Slice with out-of-bounds end (should clamp)
    let output2 = gpu_slice_nd(&ctx, &cache, &mut pool, &input, &[0], &[100], None, None).unwrap();
    let result2 = output2.to_host_f32(&ctx).unwrap();

    assert_eq!(output2.shape(), &[5]);
    assert_eq!(result2, input_data, "Out-of-bounds end should clamp");

    println!("Slice ND boundary test passed!");
}
