#![allow(clippy::too_many_arguments)] // cuBLAS GEMM requires many parameters
#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for matrix operations

use super::context::{CudaError, IconnxCudaContext};
use super::memory_pool::GpuMemoryPool;
use super::tensor::GpuTensor;
use core::ffi::c_longlong;
/// GPU Matrix Multiplication using cuBLAS
///
/// Provides MatMul and Gemm operations accelerated by cuBLAS.
use cudarc::cublas::{sys::cublasOperation_t, Gemm, GemmConfig, StridedBatchedConfig};

/// GPU MatMul: C = A @ B
///
/// Performs matrix multiplication of two 2D tensors.
/// For A[M, K] @ B[K, N] = C[M, N]
pub fn gpu_matmul(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    // Validate shapes
    assert_eq!(a.ndim(), 2, "MatMul: A must be 2D, got {}D", a.ndim());
    assert_eq!(b.ndim(), 2, "MatMul: B must be 2D, got {}D", b.ndim());

    let m = a.shape()[0]; // rows of A
    let k = a.shape()[1]; // cols of A = rows of B
    let n = b.shape()[1]; // cols of B

    assert_eq!(
        a.shape()[1],
        b.shape()[0],
        "MatMul: A columns ({}) must match B rows ({})",
        a.shape()[1],
        b.shape()[0]
    );

    // Allocate output: C[M, N]
    let mut c = pool.get_tensor_f32(ctx, vec![m, n])?;

    // cuBLAS uses column-major order, but our tensors are row-major.
    // For row-major C = A @ B, we compute column-major C^T = B^T @ A^T
    // which gives us the same result in row-major layout.
    //
    // So we call gemm with:
    // - transa = N (no transpose for B in column-major = B^T in row-major)
    // - transb = N (no transpose for A in column-major = A^T in row-major)
    // - m = n (cols of C = cols of B)
    // - n = m (rows of C = rows of A)
    // - k = k (shared dim)
    // - A_ptr = b_ptr (what cuBLAS sees as A is our B)
    // - B_ptr = a_ptr (what cuBLAS sees as B is our A)
    // - lda = n (leading dim of B in column-major)
    // - ldb = k (leading dim of A in column-major)
    // - ldc = n (leading dim of C in column-major)

    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32, // cols of result (in col-major view)
        n: m as i32, // rows of result (in col-major view)
        k: k as i32, // shared dimension
        alpha: 1.0f32,
        lda: n as i32, // leading dim of first operand (B in our case)
        ldb: k as i32, // leading dim of second operand (A in our case)
        beta: 0.0f32,
        ldc: n as i32, // leading dim of output
    };

    // Safety: gemm is unsafe because it operates on raw device pointers
    // We ensure the shapes and leading dimensions are correct above
    unsafe {
        ctx.cublas()
            .gemm(config, b.data_f32()?, a.data_f32()?, c.data_f32_mut()?)
            .map_err(|e| CudaError::Cublas(e.to_string()))?;
    }

    Ok(c)
}

/// GPU Gemm: C = alpha * A' @ B' + beta * C
///
/// Generalized matrix multiply with:
/// - alpha/beta scaling factors
/// - optional transposes on A and B
///
/// This matches the ONNX Gemm operator semantics.
pub fn gpu_gemm(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: Option<&GpuTensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<GpuTensor, CudaError> {
    // Validate shapes
    assert_eq!(a.ndim(), 2, "Gemm: A must be 2D");
    assert_eq!(b.ndim(), 2, "Gemm: B must be 2D");

    // Effective dimensions after transpose
    let (m, k_a) = if trans_a {
        (a.shape()[1], a.shape()[0])
    } else {
        (a.shape()[0], a.shape()[1])
    };

    let (k_b, n) = if trans_b {
        (b.shape()[1], b.shape()[0])
    } else {
        (b.shape()[0], b.shape()[1])
    };

    assert_eq!(
        k_a, k_b,
        "Gemm: A cols ({}) must match B rows ({}) after transpose",
        k_a, k_b
    );
    let k = k_a;

    // Allocate/prepare output
    // ONNX Gemm allows C to have various broadcastable shapes:
    // [N], [1, N], [M, 1], [M, N]
    let mut output = if let Some(c_tensor) = c {
        let c_shape = c_tensor.shape();
        let c_host = c_tensor.to_host_f32(ctx)?;

        if beta != 0.0 {
            // Broadcast C to [M, N]
            let mut expanded = vec![0.0f32; m * n];

            if c_shape == [m, n] {
                // [M, N] - direct copy
                expanded.copy_from_slice(&c_host);
            } else if c_shape == [n] || c_shape == [1, n] {
                // [N] or [1, N] - broadcast row to all rows
                for row in 0..m {
                    expanded[row * n..(row + 1) * n].copy_from_slice(&c_host);
                }
            } else if c_shape == [m, 1] {
                // [M, 1] - broadcast column to all columns
                for row in 0..m {
                    for col in 0..n {
                        expanded[row * n + col] = c_host[row];
                    }
                }
            } else if c_shape == [1] {
                // Scalar broadcast
                let val = c_host[0];
                for i in 0..(m * n) {
                    expanded[i] = val;
                }
            } else {
                return Err(CudaError::Kernel(format!(
                    "Gemm: C shape {:?} cannot broadcast to [{}, {}]",
                    c_shape, m, n
                )));
            }

            GpuTensor::from_host_f32(ctx, &expanded, vec![m, n])?
        } else {
            pool.get_tensor_f32(ctx, vec![m, n])?
        }
    } else {
        pool.get_tensor_f32(ctx, vec![m, n])?
    };

    // cuBLAS transpose flags
    // Note: cuBLAS uses column-major, so we swap the meaning
    let transa = if trans_b {
        cublasOperation_t::CUBLAS_OP_T
    } else {
        cublasOperation_t::CUBLAS_OP_N
    };
    let transb = if trans_a {
        cublasOperation_t::CUBLAS_OP_T
    } else {
        cublasOperation_t::CUBLAS_OP_N
    };

    // Leading dimensions (in row-major memory layout)
    let lda = b.shape()[1] as i32; // actual cols of B
    let ldb = a.shape()[1] as i32; // actual cols of A
    let ldc = n as i32;

    let config = GemmConfig {
        transa,
        transb,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha,
        lda,
        ldb,
        beta,
        ldc,
    };

    unsafe {
        ctx.cublas()
            .gemm(config, b.data_f32()?, a.data_f32()?, output.data_f32_mut()?)
            .map_err(|e| CudaError::Cublas(e.to_string()))?;
    }

    Ok(output)
}

/// GPU MatMul 2D x 3D: A[M, K] @ B[batch, K, N] = C[batch, M, N]
///
/// Broadcasts the 2D matrix A across all batches of B.
/// Uses strided batched GEMM with stride_a=0 for efficient broadcast.
pub fn gpu_matmul_2d_3d(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    // Validate shapes
    assert_eq!(a.ndim(), 2, "MatMul2D3D: A must be 2D, got {}D", a.ndim());
    assert_eq!(b.ndim(), 3, "MatMul2D3D: B must be 3D, got {}D", b.ndim());

    let m = a.shape()[0]; // rows of A
    let k = a.shape()[1]; // cols of A = rows of each B matrix
    let batch = b.shape()[0];
    let k_b = b.shape()[1]; // rows of each B matrix
    let n = b.shape()[2]; // cols of each B matrix

    assert_eq!(
        k, k_b,
        "MatMul2D3D: A cols ({}) must match B rows ({})",
        k, k_b
    );

    // Allocate output: C[batch, M, N]
    let mut c = pool.get_tensor_f32(ctx, vec![batch, m, n])?;

    // Strides for strided batched GEMM
    // stride_a = 0 broadcasts A across all batches
    // stride_b = K*N advances through B batches
    // stride_c = M*N advances through C batches
    let stride_a: c_longlong = 0; // Broadcast A
    let stride_b = (k * n) as c_longlong;
    let stride_c = (m * n) as c_longlong;

    // Same row-major to column-major conversion as gpu_matmul
    // For row-major C = A @ B, we compute column-major C^T = B^T @ A^T
    let gemm_config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };

    let config = StridedBatchedConfig {
        gemm: gemm_config,
        batch_size: batch as i32,
        stride_a: stride_b, // Note: swapped because we swap A and B for row-major
        stride_b: stride_a, // stride_a=0 broadcasts the 2D matrix
        stride_c,
    };

    // Safety: gemm_strided_batched is unsafe because it operates on raw device pointers
    // We ensure shapes, strides, and leading dimensions are correct above
    unsafe {
        ctx.cublas()
            .gemm_strided_batched(config, b.data_f32()?, a.data_f32()?, c.data_f32_mut()?)
            .map_err(|e| CudaError::Cublas(e.to_string()))?;
    }

    Ok(c)
}

/// GPU Batched MatMul: C[i] = A[i] @ B[i] for i in 0..batch_size
///
/// Performs batched matrix multiplication of 3D tensors.
/// For A[batch, M, K] @ B[batch, K, N] = C[batch, M, N]
///
/// This is more efficient than calling gpu_matmul in a loop.
pub fn gpu_batched_matmul(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    // Validate shapes - must be 3D for batched operation
    assert_eq!(
        a.ndim(),
        3,
        "BatchedMatMul: A must be 3D, got {}D",
        a.ndim()
    );
    assert_eq!(
        b.ndim(),
        3,
        "BatchedMatMul: B must be 3D, got {}D",
        b.ndim()
    );

    let batch = a.shape()[0];
    let m = a.shape()[1]; // rows of each A matrix
    let k = a.shape()[2]; // cols of A = rows of B
    let n = b.shape()[2]; // cols of each B matrix

    assert_eq!(
        a.shape()[0],
        b.shape()[0],
        "BatchedMatMul: batch sizes must match (A: {}, B: {})",
        a.shape()[0],
        b.shape()[0]
    );
    assert_eq!(
        a.shape()[2],
        b.shape()[1],
        "BatchedMatMul: A cols ({}) must match B rows ({})",
        a.shape()[2],
        b.shape()[1]
    );

    // Allocate output: C[batch, M, N]
    let mut c = pool.get_tensor_f32(ctx, vec![batch, m, n])?;

    // Strides for accessing consecutive batch matrices
    // Each matrix in A is M*K elements apart
    // Each matrix in B is K*N elements apart
    // Each matrix in C is M*N elements apart
    let stride_a = (m * k) as c_longlong;
    let stride_b = (k * n) as c_longlong;
    let stride_c = (m * n) as c_longlong;

    // Same row-major to column-major conversion as gpu_matmul
    let gemm_config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };

    let config = StridedBatchedConfig {
        gemm: gemm_config,
        batch_size: batch as i32,
        stride_a: stride_b, // Note: swapped because we swap A and B for row-major
        stride_b: stride_a,
        stride_c,
    };

    // Safety: gemm_strided_batched is unsafe because it operates on raw device pointers
    // We ensure shapes, strides, and leading dimensions are correct above
    unsafe {
        ctx.cublas()
            .gemm_strided_batched(config, b.data_f32()?, a.data_f32()?, c.data_f32_mut()?)
            .map_err(|e| CudaError::Cublas(e.to_string()))?;
    }

    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (IconnxCudaContext, GpuMemoryPool) {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let pool = GpuMemoryPool::new();
        (ctx, pool)
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_matmul_simple() {
        let (ctx, mut pool) = setup();

        // A = [[1, 2],
        //      [3, 4]]  (2x2)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![2, 2]).unwrap();

        // B = [[5, 6],
        //      [7, 8]]  (2x2)
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![2, 2]).unwrap();

        // C = A @ B = [[19, 22],
        //              [43, 50]]
        let c = gpu_matmul(&ctx, &mut pool, &a, &b).expect("MatMul failed");
        let result = c.to_host_f32(&ctx).expect("Failed to copy result");

        assert_eq!(c.shape(), &[2, 2]);
        let expected = [19.0, 22.0, 43.0, 50.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_matmul_rectangular() {
        let (ctx, mut pool) = setup();

        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![2, 3]).unwrap();

        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]  (3x2)
        let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![3, 2]).unwrap();

        // C = A @ B = [[22, 28],
        //              [49, 64]]  (2x2)
        let c = gpu_matmul(&ctx, &mut pool, &a, &b).expect("MatMul failed");
        let result = c.to_host_f32(&ctx).expect("Failed to copy result");

        assert_eq!(c.shape(), &[2, 2]);
        let expected = [22.0, 28.0, 49.0, 64.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_gemm_with_alpha() {
        let (ctx, mut pool) = setup();

        // A = [[1, 2],
        //      [3, 4]]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![2, 2]).unwrap();

        // B = [[1, 0],
        //      [0, 1]]  (identity)
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![2, 2]).unwrap();

        // C = 2.0 * A @ B = [[2, 4], [6, 8]]
        let c = gpu_gemm(&ctx, &mut pool, &a, &b, None, 2.0, 0.0, false, false).expect("Gemm failed");
        let result = c.to_host_f32(&ctx).expect("Failed to copy result");

        let expected = [2.0, 4.0, 6.0, 8.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_matmul_larger() {
        let (ctx, mut pool) = setup();

        // Test with 64x64 matrices
        let size = 64;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 3) % 10) as f32).collect();

        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![size, size]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![size, size]).unwrap();

        let c = gpu_matmul(&ctx, &mut pool, &a, &b).expect("MatMul failed");
        let result = c.to_host_f32(&ctx).expect("Failed to copy result");

        assert_eq!(result.len(), size * size);

        // Verify first element manually: sum of row 0 of A * col 0 of B
        let mut expected_00 = 0.0f32;
        for k in 0..size {
            expected_00 += a_data[k] * b_data[k * size];
        }
        assert!(
            (result[0] - expected_00).abs() < 1e-3,
            "First element mismatch: got {}, expected {}",
            result[0],
            expected_00
        );
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_batched_matmul() {
        let (ctx, mut pool) = setup();

        // Batch of 3 matrix multiplications
        // A[3, 2, 3] @ B[3, 3, 2] = C[3, 2, 2]
        //
        // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        // Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
        // Batch 2: all ones @ all ones = [[3,3],[3,3]]

        let a_data: Vec<f32> = vec![
            // Batch 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 2
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![3, 2, 3]).unwrap();

        let b_data: Vec<f32> = vec![
            // Batch 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 2
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![3, 3, 2]).unwrap();

        let c = gpu_batched_matmul(&ctx, &mut pool, &a, &b).expect("BatchedMatMul failed");
        let result = c.to_host_f32(&ctx).expect("Failed to copy result");

        assert_eq!(c.shape(), &[3, 2, 2]);

        // Expected results for all batches
        let expected: Vec<f32> = vec![
            // Batch 0: [[22,28],[49,64]]
            22.0, 28.0, 49.0, 64.0, // Batch 1: [[220,244],[301,334]]
            220.0, 244.0, 301.0, 334.0, // Batch 2: [[3,3],[3,3]]
            3.0, 3.0, 3.0, 3.0,
        ];

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "Mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_matmul_2d_x_3d() {
        // Test 2D x 3D: A[M, K] @ B[batch, K, N] = C[batch, M, N]
        // This broadcasts A across the batch dimension
        let (ctx, mut pool) = setup();

        // A = [[1, 2, 3],
        //      [4, 5, 6]]  shape: [2, 3] (M=2, K=3)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![2, 3]).unwrap();

        // B has 2 batches, each [3, 2] (K=3, N=2)
        // Batch 0: [[1, 2], [3, 4], [5, 6]]
        // Batch 1: [[7, 8], [9, 10], [11, 12]]
        let b_data: Vec<f32> = vec![
            // Batch 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![2, 3, 2]).unwrap();

        // Expected: C[batch, M, N] = [2, 2, 2]
        // Batch 0: A @ B[0] = [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
        //                   = [[22, 28], [49, 64]]
        // Batch 1: A @ B[1] = [[1,2,3], [4,5,6]] @ [[7,8], [9,10], [11,12]]
        //                   = [[58, 64], [139, 154]]
        let c = gpu_matmul_2d_3d(&ctx, &mut pool, &a, &b).expect("MatMul 2Dx3D failed");
        let result = c.to_host_f32(&ctx).expect("Failed to copy result");

        assert_eq!(c.shape(), &[2, 2, 2]);

        let expected: Vec<f32> = vec![
            // Batch 0
            22.0, 28.0, 49.0, 64.0, // Batch 1
            58.0, 64.0, 139.0, 154.0,
        ];

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "Mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA GPU - run with --nocapture to see benchmark output
    fn benchmark_gpu_vs_cpu_matmul() {
        use std::time::Instant;

        let (ctx, mut pool) = setup();

        // Test at different matrix sizes
        let sizes = [64, 128, 256, 512, 1024];

        println!("\n=== GPU vs CPU MatMul Benchmark ===\n");
        println!(
            "{:>6} | {:>12} | {:>12} | {:>10}",
            "Size", "GPU (ms)", "CPU (ms)", "Speedup"
        );
        println!("{}", "-".repeat(50));

        for &size in &sizes {
            // Create test data
            let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32 / 100.0).collect();
            let b_data: Vec<f32> = (0..size * size)
                .map(|i| ((i + 7) % 100) as f32 / 100.0)
                .collect();

            // GPU benchmark
            let a_gpu = GpuTensor::from_host_f32(&ctx, &a_data, vec![size, size]).unwrap();
            let b_gpu = GpuTensor::from_host_f32(&ctx, &b_data, vec![size, size]).unwrap();

            // Warmup
            let _ = gpu_matmul(&ctx, &mut pool, &a_gpu, &b_gpu).unwrap();
            ctx.sync().unwrap();

            // Timed GPU runs
            let iterations = 10;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = gpu_matmul(&ctx, &mut pool, &a_gpu, &b_gpu).unwrap();
            }
            ctx.sync().unwrap();
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

            // CPU benchmark (using ndarray for comparison)
            let a_cpu =
                ndarray::Array2::<f32>::from_shape_vec((size, size), a_data.clone()).unwrap();
            let b_cpu =
                ndarray::Array2::<f32>::from_shape_vec((size, size), b_data.clone()).unwrap();

            // Warmup
            let _ = a_cpu.dot(&b_cpu);

            // Timed CPU runs
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = a_cpu.dot(&b_cpu);
            }
            let cpu_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

            let speedup = cpu_time / gpu_time;

            println!(
                "{:>6} | {:>12.3} | {:>12.3} | {:>10.2}x",
                size, gpu_time, cpu_time, speedup
            );
        }

        println!("\nNote: GPU times include kernel launch overhead but not data transfer.");
        println!("For production use, keep data on GPU to avoid transfer costs.\n");
    }
}
