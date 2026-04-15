#![allow(clippy::too_many_arguments)] // LSTM operations require many parameters

use super::context::{CudaError, IconnxCudaContext};
use super::matmul::gpu_gemm;
use super::memory_pool::GpuMemoryPool;
use super::ops::{gpu_concat, gpu_slice_nd, OpsKernelCache};
use super::tensor::GpuTensor;
/// GPU LSTM (Long Short-Term Memory) using cuBLAS.
///
/// Implements LSTM entirely on GPU:
/// 1. cuBLAS GEMM for matrix multiplications (W @ X and R @ H).
/// 2. Custom CUDA kernels for gate activations (sigmoid, tanh).
/// 3. Fused kernel for gate combinations (fused_lstm_step_kernel).
use garboard::{DeviceSlice, LaunchConfig, Module, Program};

/// Compiled LSTM kernels as a garboard `Module`.
pub struct LstmKernelCache {
    module: Module<'static>,
}

impl LstmKernelCache {
    /// Compile all LSTM kernels and create the cache.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let program = Program::compile_for_device(LSTM_KERNELS, ctx.garboard_device(), &[])
            .map_err(|e| CudaError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        let module = ctx
            .garboard_device()
            .load_module(&program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        for name in KERNEL_NAMES {
            module.function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
        }

        Ok(Self { module })
    }

    /// Access to the underlying module for launch sites.
    pub(crate) fn module(&self) -> &Module<'static> {
        &self.module
    }
}

const KERNEL_NAMES: &[&str] = &[
    "lstm_gates_kernel",
    "lstm_cell_update_kernel",
    "fused_lstm_step_kernel",
];

/// LSTM CUDA kernel source code
const LSTM_KERNELS: &str = r#"
// Compute LSTM gates from pre-computed matrix multiplications
// gates_xw: X @ W^T results [batch, 4*hidden] (iofc order)
// gates_hr: H @ R^T results [batch, 4*hidden] (iofc order)
// bias: biases [8*hidden] (Wb_iofc, Rb_iofc)
// Output: activated gates [batch, 4*hidden]
extern "C" __global__ void lstm_gates_kernel(
    float* gates,           // Output: [batch, 4*hidden]
    const float* gates_xw,  // X @ W^T: [batch, 4*hidden]
    const float* gates_hr,  // H @ R^T: [batch, 4*hidden]
    const float* bias,      // Bias: [8*hidden] (Wb + Rb interleaved by gate)
    size_t batch_size,
    size_t hidden_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * 4 * hidden_size;

    if (idx >= total) return;

    size_t batch = idx / (4 * hidden_size);
    size_t gate_idx = idx % (4 * hidden_size);
    size_t gate = gate_idx / hidden_size;  // 0=i, 1=o, 2=f, 3=c
    size_t h = gate_idx % hidden_size;

    // Combine XW + HR + Wb + Rb
    float xw = gates_xw[batch * 4 * hidden_size + gate_idx];
    float hr = gates_hr[batch * 4 * hidden_size + gate_idx];
    float wb = bias[gate * hidden_size + h];
    float rb = bias[4 * hidden_size + gate * hidden_size + h];

    float val = xw + hr + wb + rb;

    // Apply activation: sigmoid for i,o,f gates; tanh for c gate
    if (gate == 3) {
        // Cell gate: tanh
        gates[idx] = tanhf(val);
    } else {
        // Input, output, forget gates: sigmoid
        gates[idx] = 1.0f / (1.0f + expf(-val));
    }
}

// Update cell state and compute hidden state
// c_new = f * c_old + i * g   (g is the cell gate / candidate)
// h_new = o * tanh(c_new)
extern "C" __global__ void lstm_cell_update_kernel(
    float* h_new,       // Output: new hidden state [batch, hidden]
    float* c_new,       // Output: new cell state [batch, hidden]
    const float* c_old, // Input: previous cell state [batch, hidden]
    const float* gates, // Input: activated gates [batch, 4*hidden] (i,o,f,g order)
    size_t batch_size,
    size_t hidden_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * hidden_size;

    if (idx >= total) return;

    size_t batch = idx / hidden_size;
    size_t h = idx % hidden_size;

    // Extract gate values (iofc order)
    float i = gates[batch * 4 * hidden_size + 0 * hidden_size + h];  // input gate
    float o = gates[batch * 4 * hidden_size + 1 * hidden_size + h];  // output gate
    float f = gates[batch * 4 * hidden_size + 2 * hidden_size + h];  // forget gate
    float g = gates[batch * 4 * hidden_size + 3 * hidden_size + h];  // cell gate (candidate)

    // Update cell state
    float c = f * c_old[idx] + i * g;
    c_new[idx] = c;

    // Compute hidden state
    h_new[idx] = o * tanhf(c);
}

// ============================================================================
// Fused LSTM step: combines gate computation and cell/hidden state update
// Avoids intermediate global memory writes by keeping gate values in registers
// Replaces: lstm_gates_kernel + lstm_cell_update_kernel (2 kernels -> 1)
// ============================================================================

extern "C" __global__ void fused_lstm_step_kernel(
    float* h_new,           // Output: new hidden state [batch, hidden]
    float* c_new,           // Output: new cell state [batch, hidden]
    const float* gates_xw,  // X @ W^T: [batch, 4*hidden] (iofc order)
    const float* gates_hr,  // H @ R^T: [batch, 4*hidden] (iofc order)
    const float* bias,      // Bias: [8*hidden] (Wb_iofc, Rb_iofc)
    const float* c_old,     // Input: previous cell state [batch, hidden]
    size_t batch_size,
    size_t hidden_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * hidden_size;

    if (idx >= total) return;

    size_t batch = idx / hidden_size;
    size_t h = idx % hidden_size;
    size_t base = batch * 4 * hidden_size;

    // Compute all 4 gates (i, o, f, g) in registers
    // Gate order: i=0, o=1, f=2, c(g)=3

    // Input gate (sigmoid)
    float i_xw = gates_xw[base + 0 * hidden_size + h];
    float i_hr = gates_hr[base + 0 * hidden_size + h];
    float i_wb = bias[0 * hidden_size + h];
    float i_rb = bias[4 * hidden_size + 0 * hidden_size + h];
    float i_val = i_xw + i_hr + i_wb + i_rb;
    float i_gate = 1.0f / (1.0f + expf(-i_val));

    // Output gate (sigmoid)
    float o_xw = gates_xw[base + 1 * hidden_size + h];
    float o_hr = gates_hr[base + 1 * hidden_size + h];
    float o_wb = bias[1 * hidden_size + h];
    float o_rb = bias[4 * hidden_size + 1 * hidden_size + h];
    float o_val = o_xw + o_hr + o_wb + o_rb;
    float o_gate = 1.0f / (1.0f + expf(-o_val));

    // Forget gate (sigmoid)
    float f_xw = gates_xw[base + 2 * hidden_size + h];
    float f_hr = gates_hr[base + 2 * hidden_size + h];
    float f_wb = bias[2 * hidden_size + h];
    float f_rb = bias[4 * hidden_size + 2 * hidden_size + h];
    float f_val = f_xw + f_hr + f_wb + f_rb;
    float f_gate = 1.0f / (1.0f + expf(-f_val));

    // Cell gate / candidate (tanh)
    float g_xw = gates_xw[base + 3 * hidden_size + h];
    float g_hr = gates_hr[base + 3 * hidden_size + h];
    float g_wb = bias[3 * hidden_size + h];
    float g_rb = bias[4 * hidden_size + 3 * hidden_size + h];
    float g_val = g_xw + g_hr + g_wb + g_rb;
    float g_gate = tanhf(g_val);

    // Update cell state: c_new = f * c_old + i * g
    float c = f_gate * c_old[idx] + i_gate * g_gate;
    c_new[idx] = c;

    // Compute hidden state: h_new = o * tanh(c_new)
    h_new[idx] = o_gate * tanhf(c);
}
"#;

/// GPU LSTM forward pass
///
/// # Arguments
/// * `x` - Input sequence [seq_len, batch, input_size]
/// * `w` - Input weights [num_directions, 4*hidden_size, input_size]
/// * `r` - Recurrent weights [num_directions, 4*hidden_size, hidden_size]
/// * `b` - Biases [num_directions, 8*hidden_size] (optional, zeros if None)
/// * `h_init` - Initial hidden state [num_directions, batch, hidden_size] (optional)
/// * `c_init` - Initial cell state [num_directions, batch, hidden_size] (optional)
///
/// # Returns
/// Output Y: [seq_len, num_directions, batch, hidden_size]
pub fn gpu_lstm(
    ctx: &IconnxCudaContext,
    lstm_cache: &LstmKernelCache,
    ops_cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    w: &GpuTensor,
    r: &GpuTensor,
    b: Option<&GpuTensor>,
    h_init: Option<&GpuTensor>,
    c_init: Option<&GpuTensor>,
) -> Result<GpuTensor, CudaError> {
    let x_shape = x.shape();
    let w_shape = w.shape();

    // Validate input shapes
    assert!(
        x_shape.len() == 2 || x_shape.len() == 3,
        "X must be 2D or 3D"
    );

    // Handle 2D input by adding batch dimension
    let (seq_len, batch_size, input_size) = if x_shape.len() == 2 {
        (x_shape[0], 1, x_shape[1])
    } else {
        (x_shape[0], x_shape[1], x_shape[2])
    };

    let num_directions = w_shape[0];
    let hidden_size = w_shape[1] / 4;

    assert_eq!(w_shape[2], input_size, "W input dimension mismatch");
    assert!(
        num_directions == 1 || num_directions == 2,
        "num_directions must be 1 or 2"
    );

    // Prepare bias (create zeros if not provided)
    let bias_zeros;
    let bias = if let Some(b) = b {
        b
    } else {
        bias_zeros = pool.get_tensor_f32(ctx, vec![num_directions, 8 * hidden_size])?;
        &bias_zeros
    };

    // Initialize outputs
    let mut outputs: Vec<GpuTensor> = Vec::with_capacity(seq_len);

    // Process each direction
    for dir in 0..num_directions {
        // Extract weights for this direction using GPU-side slicing
        // W: [4*hidden, input] - need to reshape from [num_dirs, 4*hidden, input]
        let w_dir = extract_direction_weights(ctx, ops_cache, pool, w, dir, 4 * hidden_size, input_size)?;
        let r_dir =
            extract_direction_weights(ctx, ops_cache, pool, r, dir, 4 * hidden_size, hidden_size)?;
        let b_dir = extract_direction_bias(ctx, ops_cache, pool, bias, dir, 8 * hidden_size)?;

        // Initialize hidden and cell states
        let mut h = if let Some(h_init) = h_init {
            extract_direction_state(ctx, ops_cache, pool, h_init, dir, batch_size, hidden_size)?
        } else {
            pool.get_tensor_f32(ctx, vec![batch_size, hidden_size])?
        };

        let mut c = if let Some(c_init) = c_init {
            extract_direction_state(ctx, ops_cache, pool, c_init, dir, batch_size, hidden_size)?
        } else {
            pool.get_tensor_f32(ctx, vec![batch_size, hidden_size])?
        };

        // Process timesteps (forward for dir=0, backward for dir=1)
        let time_indices: Vec<usize> = if dir == 0 {
            (0..seq_len).collect()
        } else {
            (0..seq_len).rev().collect()
        };

        let mut dir_outputs: Vec<(usize, GpuTensor)> = Vec::with_capacity(seq_len);

        for t in time_indices {
            // Extract x_t: [batch, input_size] using GPU-side slicing
            let x_t = extract_timestep(ctx, ops_cache, pool, x, t, batch_size, input_size)?;

            // Compute LSTM step on GPU
            let (h_new, c_new) = lstm_step(ctx, lstm_cache, pool, &x_t, &w_dir, &r_dir, &b_dir, &h, &c)?;

            dir_outputs.push((
                t,
                h_new
                    .clone()
                    .reshape(vec![batch_size, hidden_size])
                    .ok_or_else(|| CudaError::Kernel("Failed to reshape h".into()))?,
            ));
            h = h_new;
            c = c_new;
        }

        // Sort outputs by time (for backward direction)
        if dir == 1 {
            dir_outputs.sort_by_key(|(t, _)| *t);
        }

        // Store outputs
        for (_, h_t) in dir_outputs {
            outputs.push(h_t);
        }
    }

    // Combine outputs into [seq_len, num_directions, batch, hidden] using GPU operations
    // outputs is organized as: [fwd_t0, fwd_t1, ..., fwd_t(n-1), bwd_t0, bwd_t1, ..., bwd_t(n-1)]
    // We need to interleave: [t0_fwd, t0_bwd, t1_fwd, t1_bwd, ...]

    // For each direction, stack timesteps to get [seq_len, batch, hidden]
    let mut direction_stacked: Vec<GpuTensor> = Vec::with_capacity(num_directions);

    for dir in 0..num_directions {
        // Get outputs for this direction (already sorted by time)
        let dir_outputs: Vec<&GpuTensor> =
            (0..seq_len).map(|t| &outputs[dir * seq_len + t]).collect();

        // Reshape each from [batch, hidden] to [1, batch, hidden] for concatenation
        let reshaped: Vec<GpuTensor> = dir_outputs
            .iter()
            .map(|t| {
                (*t).clone()
                    .reshape(vec![1, batch_size, hidden_size])
                    .ok_or_else(|| CudaError::Kernel("Failed to reshape output".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Concat along time axis (0) to get [seq_len, batch, hidden]
        let refs: Vec<&GpuTensor> = reshaped.iter().collect();
        let stacked = gpu_concat(ctx, ops_cache, pool, &refs, 0)?;

        // Reshape to [seq_len, 1, batch, hidden] for direction interleaving
        let with_dir = stacked
            .reshape(vec![seq_len, 1, batch_size, hidden_size])
            .ok_or_else(|| CudaError::Kernel("Failed to add direction dim".into()))?;

        direction_stacked.push(with_dir);
    }

    // Concat all directions along axis 1 to get [seq_len, num_directions, batch, hidden]
    let dir_refs: Vec<&GpuTensor> = direction_stacked.iter().collect();
    gpu_concat(ctx, ops_cache, pool, &dir_refs, 1)
}

/// Compute one LSTM timestep on GPU using fused kernel
/// Combines gate computation and cell/hidden state update into a single kernel
/// to avoid intermediate global memory writes
fn lstm_step(
    ctx: &IconnxCudaContext,
    cache: &LstmKernelCache,
    pool: &mut GpuMemoryPool,
    x_t: &GpuTensor, // [batch, input_size]
    w: &GpuTensor,   // [4*hidden, input_size]
    r: &GpuTensor,   // [4*hidden, hidden_size]
    b: &GpuTensor,   // [8*hidden]
    h: &GpuTensor,   // [batch, hidden_size]
    c: &GpuTensor,   // [batch, hidden_size]
) -> Result<(GpuTensor, GpuTensor), CudaError> {
    let batch_size = x_t.shape()[0];
    let hidden_size = h.shape()[1];

    // Step 1: Compute X @ W^T and H @ R^T using cuBLAS
    // x_t: [batch, input] @ w^T: [input, 4*hidden] -> [batch, 4*hidden]
    let gates_xw = gpu_gemm(ctx, pool, x_t, w, None, 1.0, 0.0, false, true)?;
    // h: [batch, hidden] @ r^T: [hidden, 4*hidden] -> [batch, 4*hidden]
    let gates_hr = gpu_gemm(ctx, pool, h, r, None, 1.0, 0.0, false, true)?;

    // Step 2: Fused gate computation and cell/hidden state update
    // This avoids storing intermediate gate values to global memory
    let mut h_new = pool.get_tensor_f32(ctx, vec![batch_size, hidden_size])?;
    let mut c_new = pool.get_tensor_f32(ctx, vec![batch_size, hidden_size])?;

    let total_cells = batch_size * hidden_size;

    // SAFETY: fused_lstm_step_kernel signature is
    // `(float* h_new, float* c_new, const float* gates_xw,
    //   const float* gates_hr, const float* bias, const float* c_prev,
    //   size_t batch_size, size_t hidden_size)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
            usize,
        )>("fused_lstm_step_kernel")
    }
    .map_err(|e| {
        CudaError::Kernel(format!("fused_lstm_step_kernel lookup failed: {}", e))
    })?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(total_cells as u32),
            (
                h_new.data_f32_mut()?,
                c_new.data_f32_mut()?,
                gates_xw.data_f32()?,
                gates_hr.data_f32()?,
                b.data_f32()?,
                c.data_f32()?,
                batch_size,
                hidden_size,
            ),
        )
        .map_err(|e| CudaError::Kernel(format!("fused_lstm_step launch failed: {}", e)))?;

    Ok((h_new, c_new))
}

/// Extract weights for a specific direction from [num_dirs, rows, cols] tensor
/// Uses GPU-side slicing to avoid D2H + H2D round trip
fn extract_direction_weights(
    ctx: &IconnxCudaContext,
    ops_cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    tensor: &GpuTensor,
    dir: usize,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, CudaError> {
    // Slice [dir:dir+1, :, :] then reshape to [rows, cols]
    let sliced = gpu_slice_nd(
        ctx,
        ops_cache,
        pool,
        tensor,
        &[dir as i64, 0, 0],
        &[(dir + 1) as i64, rows as i64, cols as i64],
        Some(&[0, 1, 2]),
        None,
    )?;
    sliced
        .reshape(vec![rows, cols])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape weights".into()))
}

/// Extract bias for a specific direction from [num_dirs, 8*hidden] tensor
/// Uses GPU-side slicing to avoid D2H + H2D round trip
fn extract_direction_bias(
    ctx: &IconnxCudaContext,
    ops_cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    tensor: &GpuTensor,
    dir: usize,
    size: usize,
) -> Result<GpuTensor, CudaError> {
    // Slice [dir:dir+1, :] then reshape to [size]
    let sliced = gpu_slice_nd(
        ctx,
        ops_cache,
        pool,
        tensor,
        &[dir as i64, 0],
        &[(dir + 1) as i64, size as i64],
        Some(&[0, 1]),
        None,
    )?;
    sliced
        .reshape(vec![size])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape bias".into()))
}

/// Extract state for a specific direction from [num_dirs, batch, hidden] tensor
/// Uses GPU-side slicing to avoid D2H + H2D round trip
fn extract_direction_state(
    ctx: &IconnxCudaContext,
    ops_cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    tensor: &GpuTensor,
    dir: usize,
    batch: usize,
    hidden: usize,
) -> Result<GpuTensor, CudaError> {
    // Slice [dir:dir+1, :, :] then reshape to [batch, hidden]
    let sliced = gpu_slice_nd(
        ctx,
        ops_cache,
        pool,
        tensor,
        &[dir as i64, 0, 0],
        &[(dir + 1) as i64, batch as i64, hidden as i64],
        Some(&[0, 1, 2]),
        None,
    )?;
    sliced
        .reshape(vec![batch, hidden])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape state".into()))
}

/// Extract a timestep from input tensor [seq_len, batch, input_size]
/// Uses GPU-side slicing to avoid D2H + H2D round trip
fn extract_timestep(
    ctx: &IconnxCudaContext,
    ops_cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    tensor: &GpuTensor,
    t: usize,
    batch: usize,
    input_size: usize,
) -> Result<GpuTensor, CudaError> {
    let shape = tensor.shape();

    // Handle 2D input [seq_len, input_size] -> [1, input_size]
    if shape.len() == 2 {
        let sliced = gpu_slice_nd(
            ctx,
            ops_cache,
            pool,
            tensor,
            &[t as i64, 0],
            &[(t + 1) as i64, input_size as i64],
            Some(&[0, 1]),
            None,
        )?;
        return sliced
            .reshape(vec![1, input_size])
            .ok_or_else(|| CudaError::Kernel("Failed to reshape 2D timestep".into()));
    }

    // 3D input [seq_len, batch, input_size] -> [batch, input_size]
    let sliced = gpu_slice_nd(
        ctx,
        ops_cache,
        pool,
        tensor,
        &[t as i64, 0, 0],
        &[(t + 1) as i64, batch as i64, input_size as i64],
        Some(&[0, 1, 2]),
        None,
    )?;
    sliced
        .reshape(vec![batch, input_size])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape 3D timestep".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (IconnxCudaContext, LstmKernelCache, OpsKernelCache, GpuMemoryPool) {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let lstm_cache = LstmKernelCache::new(&ctx).expect("Failed to compile LSTM kernels");
        let ops_cache = OpsKernelCache::new(&ctx).expect("Failed to compile ops kernels");
        let pool = GpuMemoryPool::new();
        (ctx, lstm_cache, ops_cache, pool)
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_kernel_compilation() {
        let (ctx, lstm_cache, _ops_cache, _pool) = setup();

        assert!(lstm_cache.module().function("lstm_gates_kernel").is_ok());
        assert!(lstm_cache.module().function("lstm_cell_update_kernel").is_ok());

        println!("All LSTM kernels compiled successfully!");
        drop(ctx);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_lstm_single_step() {
        let (ctx, lstm_cache, ops_cache, mut pool) = setup();

        let batch = 1;
        let input_size = 4;
        let hidden_size = 3;

        // Create simple test data
        // X: [1, 1, 4] - single timestep, batch=1, input=4
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = GpuTensor::from_host_f32(&ctx, &x_data, vec![1, batch, input_size]).unwrap();

        // W: [1, 12, 4] - 4*hidden=12, input=4
        let w_data = vec![0.1f32; 12 * input_size];
        let w =
            GpuTensor::from_host_f32(&ctx, &w_data, vec![1, 4 * hidden_size, input_size]).unwrap();

        // R: [1, 12, 3] - 4*hidden=12, hidden=3
        let r_data = vec![0.1f32; 12 * hidden_size];
        let r =
            GpuTensor::from_host_f32(&ctx, &r_data, vec![1, 4 * hidden_size, hidden_size]).unwrap();

        // B: [1, 24] - 8*hidden=24
        let b_data = vec![0.0f32; 8 * hidden_size];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![1, 8 * hidden_size]).unwrap();

        let output = gpu_lstm(
            &ctx,
            &lstm_cache,
            &ops_cache,
            &mut pool,
            &x,
            &w,
            &r,
            Some(&b),
            None,
            None,
        )
        .unwrap();

        // Output shape: [1, 1, 1, 3] = [seq_len, num_dirs, batch, hidden]
        assert_eq!(output.shape(), &[1, 1, batch, hidden_size]);

        let result = output.to_host_f32(&ctx).unwrap();

        // All values should be non-zero (LSTM processed the input)
        assert!(
            result.iter().all(|&v| v != 0.0),
            "Output should be non-zero"
        );

        // Values should be bounded (tanh output is in [-1, 1])
        assert!(
            result.iter().all(|&v| v.abs() <= 1.0),
            "Hidden state should be bounded"
        );

        println!("LSTM single step test passed! Output: {:?}", result);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_lstm_sequence() {
        let (ctx, lstm_cache, ops_cache, mut pool) = setup();

        let seq_len = 3;
        let batch = 2;
        let input_size = 4;
        let hidden_size = 5;

        // X: [3, 2, 4] - 3 timesteps, batch=2, input=4
        let x_data: Vec<f32> = (0..seq_len * batch * input_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let x = GpuTensor::from_host_f32(&ctx, &x_data, vec![seq_len, batch, input_size]).unwrap();

        // W: [1, 20, 4] - 4*hidden=20, input=4
        let w_data = vec![0.1f32; 4 * hidden_size * input_size];
        let w =
            GpuTensor::from_host_f32(&ctx, &w_data, vec![1, 4 * hidden_size, input_size]).unwrap();

        // R: [1, 20, 5] - 4*hidden=20, hidden=5
        let r_data = vec![0.1f32; 4 * hidden_size * hidden_size];
        let r =
            GpuTensor::from_host_f32(&ctx, &r_data, vec![1, 4 * hidden_size, hidden_size]).unwrap();

        // B: [1, 40] - 8*hidden=40
        let b_data = vec![0.0f32; 8 * hidden_size];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![1, 8 * hidden_size]).unwrap();

        let output = gpu_lstm(
            &ctx,
            &lstm_cache,
            &ops_cache,
            &mut pool,
            &x,
            &w,
            &r,
            Some(&b),
            None,
            None,
        )
        .unwrap();

        // Output shape: [3, 1, 2, 5] = [seq_len, num_dirs, batch, hidden]
        assert_eq!(output.shape(), &[seq_len, 1, batch, hidden_size]);

        let result = output.to_host_f32(&ctx).unwrap();

        // Verify output is valid
        assert!(!result.iter().any(|v| v.is_nan()), "Output contains NaN");
        assert!(
            !result.iter().any(|v| v.is_infinite()),
            "Output contains Inf"
        );

        println!(
            "LSTM sequence test passed! Output shape: {:?}",
            output.shape()
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_lstm_with_initial_states() {
        let (ctx, lstm_cache, ops_cache, mut pool) = setup();

        let seq_len = 2;
        let batch = 1;
        let input_size = 3;
        let hidden_size = 4;

        // X: [2, 1, 3]
        let x_data = vec![1.0f32; seq_len * batch * input_size];
        let x = GpuTensor::from_host_f32(&ctx, &x_data, vec![seq_len, batch, input_size]).unwrap();

        // W: [1, 16, 3]
        let w_data = vec![0.1f32; 4 * hidden_size * input_size];
        let w =
            GpuTensor::from_host_f32(&ctx, &w_data, vec![1, 4 * hidden_size, input_size]).unwrap();

        // R: [1, 16, 4]
        let r_data = vec![0.1f32; 4 * hidden_size * hidden_size];
        let r =
            GpuTensor::from_host_f32(&ctx, &r_data, vec![1, 4 * hidden_size, hidden_size]).unwrap();

        // Initial h: [1, 1, 4] - non-zero initial hidden state
        let h_init_data = vec![0.5f32; batch * hidden_size];
        let h_init =
            GpuTensor::from_host_f32(&ctx, &h_init_data, vec![1, batch, hidden_size]).unwrap();

        // Initial c: [1, 1, 4] - non-zero initial cell state
        let c_init_data = vec![0.3f32; batch * hidden_size];
        let c_init =
            GpuTensor::from_host_f32(&ctx, &c_init_data, vec![1, batch, hidden_size]).unwrap();

        let output = gpu_lstm(
            &ctx,
            &lstm_cache,
            &ops_cache,
            &mut pool,
            &x,
            &w,
            &r,
            None,
            Some(&h_init),
            Some(&c_init),
        )
        .unwrap();

        assert_eq!(output.shape(), &[seq_len, 1, batch, hidden_size]);

        let result = output.to_host_f32(&ctx).unwrap();
        assert!(!result.iter().any(|v| v.is_nan()), "Output contains NaN");

        println!("LSTM with initial states test passed!");
    }
}
