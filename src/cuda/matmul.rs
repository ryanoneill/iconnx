#![allow(clippy::too_many_arguments)] // cuBLAS GEMM requires many parameters
#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for matrix operations

//! GPU Matrix Multiplication via garboard's `BlasContext` (cuBLAS under the hood).
//!
//! MatMul / Gemm / BatchedMatMul call into `BlasContext::gemm` and
//! `BlasContext::gemm_strided_batched`, passing garboard `DeviceSlice`
//! references directly.

use core::ffi::c_longlong;

use garboard::{GemmBf16Config, GemmConfig, GemmFp16Acc32Config, GemmStridedBatchedConfig, Transpose};

use super::context::{CudaError, IconnxCudaContext};
use super::memory_pool::GpuMemoryPool;
use super::tensor::{DType, GpuTensor};

/// GPU MatMul: C = A @ B.
///
/// Performs matrix multiplication. Supports the 2D × 2D base case and
/// the 3D × 2D batched broadcast pattern Whisper attention uses
/// (`[B, M, K] × [K, N] → [B, M, N]`). The 3D × 2D path reshapes A to
/// `[B*M, K]`, runs a 2-D GEMM, then reshapes the result back — WS-4
/// M4.7 retrospective: same wrapper-level pattern that closed the
/// MatMulInteger 3-D × 2-D gap.
///
/// Float32 routes through cuBLAS `gemm`; Float16 routes through
/// `gemm_fp16_acc32` (FP16 IO, FP32 accumulator) — see WS-3 M3.6
/// branch decision.
pub fn gpu_matmul(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    // 3D × 2D: wrapper-level reshape to 2D, dispatch, reshape back.
    if a.ndim() == 3 && b.ndim() == 2 {
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[1];
        assert_eq!(
            k,
            b.shape()[0],
            "MatMul3D2D: A inner ({}) must match B rows ({})",
            k,
            b.shape()[0]
        );
        let a_2d = a
            .clone()
            .reshape(vec![batch * m, k])
            .ok_or_else(|| CudaError::Kernel("MatMul3D2D: failed to reshape A".to_string()))?;
        let c_2d = gpu_matmul_2d(ctx, pool, &a_2d, b)?;
        let c_3d = c_2d
            .reshape(vec![batch, m, n])
            .ok_or_else(|| CudaError::Kernel("MatMul3D2D: failed to reshape result".to_string()))?;
        return Ok(c_3d);
    }

    assert_eq!(a.ndim(), 2, "MatMul: A must be 2D, got {}D", a.ndim());
    assert_eq!(b.ndim(), 2, "MatMul: B must be 2D, got {}D", b.ndim());
    gpu_matmul_2d(ctx, pool, a, b)
}

/// Internal 2-D MatMul dispatcher; dtype-routed to cuBLAS `gemm` (Float32)
/// or `gemm_fp16_acc32` (Float16).
fn gpu_matmul_2d(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
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

    if a.dtype() != b.dtype() {
        // WS-3.5 Y(3): typed `DtypeMismatch` when either operand is BF16
        // (parallel to the FP16 typed-error stance landed in WS-3 M3.6).
        // Older Float32/Float16-only mismatches keep the legacy
        // `Kernel(format!())` surface for backwards compatibility with
        // existing match-on-message tests.
        if a.dtype() == DType::BFloat16 || b.dtype() == DType::BFloat16 {
            return Err(CudaError::DtypeMismatch {
                expected: a.dtype().name(),
                actual: b.dtype().name(),
            });
        }
        return Err(CudaError::Kernel(format!(
            "MatMul: dtype mismatch (A: {}, B: {})",
            a.dtype().name(),
            b.dtype().name()
        )));
    }

    match a.dtype() {
        DType::Float32 => {
            let mut c = pool.get_tensor_f32(ctx, vec![m, n])?;
            // cuBLAS uses column-major; iconnx tensors are row-major. For row-major
            // C = A @ B, we compute column-major C^T = B^T @ A^T, which gives the
            // same row-major result. Hence the swapped `a`/`b` order at the call
            // site and the `m`/`n` swap here.
            let config = GemmConfig::<f32> {
                transa: Transpose::NoTrans,
                transb: Transpose::NoTrans,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: n as i32,
            };
            ctx.blas()
                .gemm(
                    ctx.garboard_stream(),
                    &config,
                    b.data_f32()?,
                    a.data_f32()?,
                    c.data_f32_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;
            Ok(c)
        }
        DType::Float16 => {
            let mut c = pool.get_tensor_f16(ctx, vec![m, n])?;
            // gemm_fp16_acc32: FP16 IO, FP32 accumulator. alpha/beta are f32.
            // Same row-major↔col-major swap as the f32 path.
            let config = GemmFp16Acc32Config {
                transa: Transpose::NoTrans,
                transb: Transpose::NoTrans,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0_f32,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0_f32,
                ldc: n as i32,
            };
            ctx.blas()
                .gemm_fp16_acc32(
                    ctx.garboard_stream(),
                    &config,
                    b.data_f16()?,
                    a.data_f16()?,
                    c.data_f16_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;
            Ok(c)
        }
        DType::BFloat16 => {
            // WS-3.5 Y(3): garboard `gemm_bf16` — `CUBLAS_COMPUTE_32F`
            // single-flavor (cuBLAS exposes no BF16-accumulator mode;
            // BF16's design assumes a wider FP32 accumulator). alpha/beta
            // are f32 per `GemmBf16Config`. Same row-major↔col-major
            // swap as the f32 / FP16 paths.
            let mut c = pool.get_tensor_bf16(ctx, vec![m, n])?;
            let config = GemmBf16Config {
                transa: Transpose::NoTrans,
                transb: Transpose::NoTrans,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0_f32,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0_f32,
                ldc: n as i32,
            };
            ctx.blas()
                .gemm_bf16(
                    ctx.garboard_stream(),
                    &config,
                    b.data_bf16()?,
                    a.data_bf16()?,
                    c.data_bf16_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;
            Ok(c)
        }
        dt => Err(CudaError::Kernel(format!(
            "MatMul: unsupported dtype {} (Float32 + Float16 + BFloat16 supported; Int* paths route through MatMulInteger)",
            dt.name()
        ))),
    }
}

/// GPU Gemm: C = alpha * A' @ B' + beta * C.
///
/// Generalized matrix multiply with alpha/beta scaling and optional
/// transposes on A and B. Matches the ONNX Gemm operator semantics.
/// Dispatches to cuBLAS `gemm` (Float32) or `gemm_fp16_acc32` (Float16).
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
    assert_eq!(a.ndim(), 2, "Gemm: A must be 2D");
    assert_eq!(b.ndim(), 2, "Gemm: B must be 2D");

    if a.dtype() != b.dtype() {
        if a.dtype() == DType::BFloat16 || b.dtype() == DType::BFloat16 {
            return Err(CudaError::DtypeMismatch {
                expected: a.dtype().name(),
                actual: b.dtype().name(),
            });
        }
        return Err(CudaError::Kernel(format!(
            "Gemm: A/B dtype mismatch ({} vs {})",
            a.dtype().name(),
            b.dtype().name()
        )));
    }
    if let Some(c_tensor) = c {
        if c_tensor.dtype() != a.dtype() {
            if a.dtype() == DType::BFloat16 || c_tensor.dtype() == DType::BFloat16 {
                return Err(CudaError::DtypeMismatch {
                    expected: a.dtype().name(),
                    actual: c_tensor.dtype().name(),
                });
            }
            return Err(CudaError::Kernel(format!(
                "Gemm: C dtype must match A/B (got C={}, A={})",
                c_tensor.dtype().name(),
                a.dtype().name()
            )));
        }
    }

    // Effective dimensions after transpose.
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

    // cuBLAS transpose flags. Note the swap: cuBLAS is col-major, iconnx
    // is row-major — so the transpose applied to row-major A becomes the
    // transpose applied to what cuBLAS sees as its *second* operand.
    let transa = if trans_b {
        Transpose::Trans
    } else {
        Transpose::NoTrans
    };
    let transb = if trans_a {
        Transpose::Trans
    } else {
        Transpose::NoTrans
    };

    // Leading dimensions (in row-major memory layout).
    let lda = b.shape()[1] as i32;
    let ldb = a.shape()[1] as i32;
    let ldc = n as i32;

    match a.dtype() {
        DType::Float32 => {
            let mut output =
                gemm_init_output_f32(ctx, pool, c, beta, m, n)?;
            let config = GemmConfig::<f32> {
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
            ctx.blas()
                .gemm(
                    ctx.garboard_stream(),
                    &config,
                    b.data_f32()?,
                    a.data_f32()?,
                    output.data_f32_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;
            Ok(output)
        }
        DType::Float16 => {
            let mut output =
                gemm_init_output_f16(ctx, pool, c, beta, m, n)?;
            let config = GemmFp16Acc32Config {
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
            ctx.blas()
                .gemm_fp16_acc32(
                    ctx.garboard_stream(),
                    &config,
                    b.data_f16()?,
                    a.data_f16()?,
                    output.data_f16_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;
            Ok(output)
        }
        DType::BFloat16 => {
            // WS-3.5 Y(3): mirrors the FP16 Gemm arm, calling
            // `gemm_bf16` instead of `gemm_fp16_acc32`. alpha/beta are
            // f32 (matching `CUBLAS_COMPUTE_32F`); cuBLAS down-converts
            // to BF16 only at the C write-back.
            let mut output = gemm_init_output_bf16(ctx, pool, c, beta, m, n)?;
            let config = GemmBf16Config {
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
            ctx.blas()
                .gemm_bf16(
                    ctx.garboard_stream(),
                    &config,
                    b.data_bf16()?,
                    a.data_bf16()?,
                    output.data_bf16_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;
            Ok(output)
        }
        dt => Err(CudaError::Kernel(format!(
            "Gemm: unsupported dtype {}",
            dt.name()
        ))),
    }
}

/// ONNX Gemm allows C to have various broadcastable shapes:
/// `[N]`, `[1, N]`, `[M, 1]`, `[M, N]`. Expand to `[M, N]` so cuBLAS
/// can read it directly.
fn broadcast_c<T: Copy>(c_host: &[T], c_shape: &[usize], m: usize, n: usize) -> Result<Vec<T>, CudaError> {
    let mut expanded = Vec::with_capacity(m * n);
    if c_shape == [m, n] {
        expanded.extend_from_slice(c_host);
    } else if c_shape == [n] || c_shape == [1, n] {
        for _ in 0..m {
            expanded.extend_from_slice(c_host);
        }
    } else if c_shape == [m, 1] {
        for row in 0..m {
            for _ in 0..n {
                expanded.push(c_host[row]);
            }
        }
    } else if c_shape == [1] {
        let val = c_host[0];
        expanded.resize(m * n, val);
    } else {
        return Err(CudaError::Kernel(format!(
            "Gemm: C shape {:?} cannot broadcast to [{}, {}]",
            c_shape, m, n
        )));
    }
    Ok(expanded)
}

fn gemm_init_output_f32(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    c: Option<&GpuTensor>,
    beta: f32,
    m: usize,
    n: usize,
) -> Result<GpuTensor, CudaError> {
    if let Some(c_tensor) = c {
        if beta != 0.0 {
            let c_host = c_tensor.to_host_f32(ctx)?;
            let expanded = broadcast_c(&c_host, c_tensor.shape(), m, n)?;
            return GpuTensor::from_host_f32(ctx, &expanded, vec![m, n]);
        }
    }
    pool.get_tensor_f32(ctx, vec![m, n])
}

fn gemm_init_output_f16(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    c: Option<&GpuTensor>,
    beta: f32,
    m: usize,
    n: usize,
) -> Result<GpuTensor, CudaError> {
    if let Some(c_tensor) = c {
        if beta != 0.0 {
            let c_host = c_tensor.to_host_f16(ctx)?;
            let expanded = broadcast_c(&c_host, c_tensor.shape(), m, n)?;
            return GpuTensor::from_host_f16(ctx, &expanded, vec![m, n]);
        }
    }
    pool.get_tensor_f16(ctx, vec![m, n])
}

/// WS-3.5 Y(3): BFloat16 mirror of `gemm_init_output_f16`. ONNX Gemm's
/// `C` broadcasts via the same shape vocabulary; the host detour for
/// `beta != 0.0` is unavoidable because we don't have a GPU-side
/// expand-and-add helper for BF16 yet.
fn gemm_init_output_bf16(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    c: Option<&GpuTensor>,
    beta: f32,
    m: usize,
    n: usize,
) -> Result<GpuTensor, CudaError> {
    if let Some(c_tensor) = c {
        if beta != 0.0 {
            let c_host = c_tensor.to_host_bf16(ctx)?;
            let expanded = broadcast_c(&c_host, c_tensor.shape(), m, n)?;
            return GpuTensor::from_host_bf16(ctx, &expanded, vec![m, n]);
        }
    }
    pool.get_tensor_bf16(ctx, vec![m, n])
}

/// GPU MatMul 2D x 3D: A[M, K] @ B[batch, K, N] = C[batch, M, N].
///
/// Broadcasts the 2D matrix A across all batches of B. The Float32
/// path uses cuBLAS strided batched GEMM with `stride_a = 0`. The
/// Float16 path falls back to a per-batch loop over `gemm_fp16_acc32`
/// because garboard's strided-batched GEMM is sealed to `f32` / `f64`
/// — no FP16 binding yet. The loop is numerically identical; only
/// the batched-throughput optimization is missing. Tracked for a
/// garboard follow-up alongside the perf workstream.
pub fn gpu_matmul_2d_3d(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    assert_eq!(a.ndim(), 2, "MatMul2D3D: A must be 2D, got {}D", a.ndim());
    assert_eq!(b.ndim(), 3, "MatMul2D3D: B must be 3D, got {}D", b.ndim());

    let m = a.shape()[0];
    let k = a.shape()[1];
    let batch = b.shape()[0];
    let k_b = b.shape()[1];
    let n = b.shape()[2];

    assert_eq!(
        k, k_b,
        "MatMul2D3D: A cols ({}) must match B rows ({})",
        k, k_b
    );
    if a.dtype() != b.dtype() {
        if a.dtype() == DType::BFloat16 || b.dtype() == DType::BFloat16 {
            return Err(CudaError::DtypeMismatch {
                expected: a.dtype().name(),
                actual: b.dtype().name(),
            });
        }
        return Err(CudaError::Kernel(format!(
            "MatMul2D3D: dtype mismatch (A: {}, B: {})",
            a.dtype().name(),
            b.dtype().name()
        )));
    }

    match a.dtype() {
        DType::Float32 => {
            let mut c = pool.get_tensor_f32(ctx, vec![batch, m, n])?;

            // Row-major → col-major swap (see gpu_matmul comment). In cuBLAS-land,
            // we swap A↔B, so `stride_a` (cuBLAS) = our stride_b, and vice versa.
            // Here we want to broadcast *our* A over batches, which means the
            // matrix that cuBLAS sees as "B" is broadcast — so its stride is 0.
            let stride_a_ours: c_longlong = 0; // our A is broadcast
            let stride_b_ours = (k * n) as c_longlong;
            let stride_c = (m * n) as c_longlong;

            let config = GemmStridedBatchedConfig::<f32> {
                transa: Transpose::NoTrans,
                transb: Transpose::NoTrans,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: n as i32,
                stride_a: stride_b_ours,
                ldb: k as i32,
                stride_b: stride_a_ours,
                beta: 0.0,
                ldc: n as i32,
                stride_c,
                batch_count: batch as i32,
            };

            ctx.blas()
                .gemm_strided_batched(
                    ctx.garboard_stream(),
                    &config,
                    b.data_f32()?,
                    a.data_f32()?,
                    c.data_f32_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;

            Ok(c)
        }
        DType::Float16 => {
            // Per-batch fallback (see fn doc): split B into [batch] [K, N]
            // slabs via reshape, run gpu_matmul (which dispatches to
            // gemm_fp16_acc32) per batch, copy each result into the [batch,
            // M, N] output.
            //
            // Reshape requires a host detour because we don't have a
            // sub-slice API on GpuTensor today. For Whisper-Tiny encoder
            // shapes (batch=1) this is a no-op; the broader-batch case
            // pays one D2H + H2D round-trip per batch slab. Documented
            // perf gap, not a correctness gap.
            let b_host = b.to_host_f16(ctx)?;
            let mut out_host: Vec<half::f16> = Vec::with_capacity(batch * m * n);
            let slab = k * n;

            for batch_idx in 0..batch {
                let b_slab = &b_host[batch_idx * slab..(batch_idx + 1) * slab];
                let b_slab_gpu = GpuTensor::from_host_f16(ctx, b_slab, vec![k, n])?;
                let c_slab = gpu_matmul_2d(ctx, pool, a, &b_slab_gpu)?;
                let c_slab_host = c_slab.to_host_f16(ctx)?;
                out_host.extend_from_slice(&c_slab_host);
            }
            GpuTensor::from_host_f16(ctx, &out_host, vec![batch, m, n])
        }
        DType::BFloat16 => {
            // WS-3.5 Y(3): per-batch fallback over `gemm_bf16`. garboard
            // does not yet expose `gemm_strided_batched_bf16` — Tracked
            // Follow-Up #2 (cross-repo ask). Numerically identical to a
            // fused strided-batched call; only batched-throughput is
            // suboptimal. For Whisper-Tiny encoder shapes (batch=1) this
            // is a no-op; broader-batch case pays one D2H+H2D round-trip
            // per batch slab. Documented perf gap, not correctness.
            let b_host = b.to_host_bf16(ctx)?;
            let mut out_host: Vec<half::bf16> = Vec::with_capacity(batch * m * n);
            let slab = k * n;

            for batch_idx in 0..batch {
                let b_slab = &b_host[batch_idx * slab..(batch_idx + 1) * slab];
                let b_slab_gpu = GpuTensor::from_host_bf16(ctx, b_slab, vec![k, n])?;
                let c_slab = gpu_matmul_2d(ctx, pool, a, &b_slab_gpu)?;
                let c_slab_host = c_slab.to_host_bf16(ctx)?;
                out_host.extend_from_slice(&c_slab_host);
            }
            GpuTensor::from_host_bf16(ctx, &out_host, vec![batch, m, n])
        }
        dt => Err(CudaError::Kernel(format!(
            "MatMul2D3D: unsupported dtype {}",
            dt.name()
        ))),
    }
}

/// GPU Batched MatMul: C[i] = A[i] @ B[i] for i in 0..batch_size.
///
/// For A[batch, M, K] @ B[batch, K, N] = C[batch, M, N]. Float32 uses
/// strided-batched cuBLAS. Float16 falls back to a per-batch loop over
/// `gemm_fp16_acc32` — see [`gpu_matmul_2d_3d`] for the same upstream-
/// gap rationale (no `gemm_strided_batched_fp16_acc32` in garboard
/// yet). Numerically identical; tracked for a garboard follow-up.
pub fn gpu_batched_matmul(
    ctx: &IconnxCudaContext,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
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
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[2];

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
    if a.dtype() != b.dtype() {
        if a.dtype() == DType::BFloat16 || b.dtype() == DType::BFloat16 {
            return Err(CudaError::DtypeMismatch {
                expected: a.dtype().name(),
                actual: b.dtype().name(),
            });
        }
        return Err(CudaError::Kernel(format!(
            "BatchedMatMul: dtype mismatch (A: {}, B: {})",
            a.dtype().name(),
            b.dtype().name()
        )));
    }

    match a.dtype() {
        DType::Float32 => {
            let mut c = pool.get_tensor_f32(ctx, vec![batch, m, n])?;

            let stride_a_ours = (m * k) as c_longlong;
            let stride_b_ours = (k * n) as c_longlong;
            let stride_c = (m * n) as c_longlong;

            let config = GemmStridedBatchedConfig::<f32> {
                transa: Transpose::NoTrans,
                transb: Transpose::NoTrans,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: n as i32,
                stride_a: stride_b_ours,
                ldb: k as i32,
                stride_b: stride_a_ours,
                beta: 0.0,
                ldc: n as i32,
                stride_c,
                batch_count: batch as i32,
            };

            ctx.blas()
                .gemm_strided_batched(
                    ctx.garboard_stream(),
                    &config,
                    b.data_f32()?,
                    a.data_f32()?,
                    c.data_f32_mut()?,
                )
                .map_err(|e| CudaError::Cublas(e.to_string()))?;

            Ok(c)
        }
        DType::Float16 => {
            // Per-batch fallback over gemm_fp16_acc32 (see fn doc).
            let a_host = a.to_host_f16(ctx)?;
            let b_host = b.to_host_f16(ctx)?;
            let a_slab = m * k;
            let b_slab = k * n;
            let mut out_host: Vec<half::f16> = Vec::with_capacity(batch * m * n);

            for batch_idx in 0..batch {
                let a_data = &a_host[batch_idx * a_slab..(batch_idx + 1) * a_slab];
                let b_data = &b_host[batch_idx * b_slab..(batch_idx + 1) * b_slab];
                let a_gpu = GpuTensor::from_host_f16(ctx, a_data, vec![m, k])?;
                let b_gpu = GpuTensor::from_host_f16(ctx, b_data, vec![k, n])?;
                let c_slab = gpu_matmul_2d(ctx, pool, &a_gpu, &b_gpu)?;
                let c_slab_host = c_slab.to_host_f16(ctx)?;
                out_host.extend_from_slice(&c_slab_host);
            }
            GpuTensor::from_host_f16(ctx, &out_host, vec![batch, m, n])
        }
        DType::BFloat16 => {
            // WS-3.5 Y(3): per-batch fallback over `gemm_bf16` mirroring
            // the FP16 loop above. `gemm_strided_batched_bf16` is
            // Tracked Follow-Up #2 — until then, correctness > perf.
            let a_host = a.to_host_bf16(ctx)?;
            let b_host = b.to_host_bf16(ctx)?;
            let a_slab = m * k;
            let b_slab = k * n;
            let mut out_host: Vec<half::bf16> = Vec::with_capacity(batch * m * n);

            for batch_idx in 0..batch {
                let a_data = &a_host[batch_idx * a_slab..(batch_idx + 1) * a_slab];
                let b_data = &b_host[batch_idx * b_slab..(batch_idx + 1) * b_slab];
                let a_gpu = GpuTensor::from_host_bf16(ctx, a_data, vec![m, k])?;
                let b_gpu = GpuTensor::from_host_bf16(ctx, b_data, vec![k, n])?;
                let c_slab = gpu_matmul_2d(ctx, pool, &a_gpu, &b_gpu)?;
                let c_slab_host = c_slab.to_host_bf16(ctx)?;
                out_host.extend_from_slice(&c_slab_host);
            }
            GpuTensor::from_host_bf16(ctx, &out_host, vec![batch, m, n])
        }
        dt => Err(CudaError::Kernel(format!(
            "BatchedMatMul: unsupported dtype {}",
            dt.name()
        ))),
    }
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

        // A = [[1, 2], [3, 4]] (2x2)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![2, 2]).unwrap();

        // B = [[5, 6], [7, 8]] (2x2)
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![2, 2]).unwrap();

        // A @ B = [[19, 22], [43, 50]]
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

        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![2, 3]).unwrap();

        // B = [[1, 2], [3, 4], [5, 6]] (3x2)
        let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![3, 2]).unwrap();

        // A @ B = [[22, 28], [49, 64]] (2x2)
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
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![2, 2]).unwrap();
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![2, 2]).unwrap();
        // 2.0 * A @ I = [[2, 4], [6, 8]]
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
        let size = 64;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 3) % 10) as f32).collect();
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![size, size]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![size, size]).unwrap();
        let c = gpu_matmul(&ctx, &mut pool, &a, &b).expect("MatMul failed");
        let result = c.to_host_f32(&ctx).expect("Failed to copy result");
        assert_eq!(result.len(), size * size);
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
    fn benchmark_gpu_vs_cpu_matmul() {
        // Just exercises the path — timing isn't asserted.
        let (ctx, mut pool) = setup();
        let size = 128;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32).sin()).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32).cos()).collect();
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![size, size]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![size, size]).unwrap();
        let c = gpu_matmul(&ctx, &mut pool, &a, &b).expect("MatMul failed");
        let _ = c.to_host_f32(&ctx).expect("Failed to copy result");
    }
}
