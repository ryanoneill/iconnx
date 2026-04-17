//! cuFFT bindings for GPU-accelerated FFT operations
//!
//! Direct FFI bindings to NVIDIA's cuFFT library for Fast Fourier Transform.
//! Used primarily for STFT (Short-Time Fourier Transform) in audio processing.

// Allow dead code - these bindings will be used when STFT is fully integrated
#![allow(dead_code)]

use std::ffi::c_int;
use std::ptr;

use super::context::{CudaError, IconnxCudaContext};
use super::tensor::GpuTensor;

/// CUDA kernels for STFT operations
const STFT_KERNELS: &str = r#"
extern "C" __global__ void window_frames_kernel(
    float* output,              // [total_frames, fft_size]
    const float* signal,        // [signal_length]
    const float* window,        // [fft_size]
    int signal_length,
    int fft_size,
    int hop_size,
    int chunk_frames,           // Number of frames in this chunk
    int frame_offset,           // Offset of first frame in this chunk
    int output_offset           // Offset in output array (frame_offset * fft_size)
) {
    int local_frame = blockIdx.y;  // Frame within this chunk
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_frame < chunk_frames && sample < fft_size) {
        int global_frame = local_frame + frame_offset;  // Actual frame index
        int signal_idx = global_frame * hop_size + sample;
        float signal_val = (signal_idx < signal_length) ? signal[signal_idx] : 0.0f;
        float window_val = window[sample];
        output[output_offset + local_frame * fft_size + sample] = signal_val * window_val;
    }
}

extern "C" __global__ void unpack_complex_kernel(
    float* output,              // [total_frames, num_bins, 2]
    const float* complex_data,  // CufftComplex stored as interleaved [real, imag]
    int chunk_frames,           // Frames in this chunk
    int num_bins,
    int frame_offset,           // Global frame offset
    int output_offset           // Output offset (frame_offset * num_bins * 2)
) {
    int local_frame = blockIdx.y;
    int bin = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_frame < chunk_frames && bin < num_bins) {
        int global_frame = local_frame + frame_offset;
        int complex_idx = global_frame * num_bins + bin;
        int local_output_idx = (local_frame * num_bins + bin) * 2;

        // CufftComplex is stored as {real, imag} pairs
        output[output_offset + local_output_idx] = complex_data[complex_idx * 2];      // real
        output[output_offset + local_output_idx + 1] = complex_data[complex_idx * 2 + 1];  // imag
    }
}
"#;

const STFT_KERNEL_NAMES: &[&str] = &["window_frames_kernel", "unpack_complex_kernel"];

/// STFT kernel cache as a garboard `Module`.
pub struct StftKernelCache {
    module: garboard::Module<'static>,
}

impl StftKernelCache {
    /// Compile STFT kernels.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let program = garboard::Program::compile_for_device(
            STFT_KERNELS,
            ctx.garboard_device(),
            &[],
        )
        .map_err(|e| CudaError::Kernel(format!("STFT kernel compilation failed: {}", e)))?;

        let module = ctx
            .garboard_device()
            .load_module(&program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        for name in STFT_KERNEL_NAMES {
            module.function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
        }

        Ok(Self { module })
    }

    /// Access to the underlying module for launch sites.
    pub(crate) fn module(&self) -> &garboard::Module<'static> {
        &self.module
    }
}

/// cuFFT result codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CufftResult {
    Success = 0,
    InvalidPlan = 1,
    AllocFailed = 2,
    InvalidType = 3,
    InvalidValue = 4,
    InternalError = 5,
    ExecFailed = 6,
    SetupFailed = 7,
    InvalidSize = 8,
    UnalignedData = 9,
    IncompleteParameterList = 10,
    InvalidDevice = 11,
    ParseError = 12,
    NoWorkspace = 13,
    NotImplemented = 14,
    LicenseError = 15,
    NotSupported = 16,
}

impl CufftResult {
    fn to_result(self) -> Result<(), CudaError> {
        if self == CufftResult::Success {
            Ok(())
        } else {
            Err(CudaError::Kernel(format!("cuFFT error: {:?}", self)))
        }
    }
}

/// cuFFT transform types
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CufftType {
    R2C = 0x2a, // Real to Complex (single precision)
    C2R = 0x2c, // Complex to Real (single precision)
    C2C = 0x29, // Complex to Complex (single precision)
    D2Z = 0x6a, // Real to Complex (double precision)
    Z2D = 0x6c, // Complex to Real (double precision)
    Z2Z = 0x69, // Complex to Complex (double precision)
}

/// cuFFT direction
pub const CUFFT_FORWARD: c_int = -1;
pub const CUFFT_INVERSE: c_int = 1;

/// cuFFT plan handle
pub type CufftHandle = c_int;

/// Complex number for cuFFT (single precision)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CufftComplex {
    pub x: f32, // real
    pub y: f32, // imaginary
}

// `CufftComplex` derives `bytemuck::Pod` + `bytemuck::Zeroable` above,
// which is the trait surface garboard's `DeviceSlice<T>` requires.

// FFI declarations for cuFFT
#[link(name = "cufft")]
extern "C" {
    fn cufftPlan1d(
        plan: *mut CufftHandle,
        nx: c_int,
        type_: CufftType,
        batch: c_int,
    ) -> CufftResult;

    fn cufftPlanMany(
        plan: *mut CufftHandle,
        rank: c_int,
        n: *const c_int,
        inembed: *const c_int,
        istride: c_int,
        idist: c_int,
        onembed: *const c_int,
        ostride: c_int,
        odist: c_int,
        type_: CufftType,
        batch: c_int,
    ) -> CufftResult;

    fn cufftExecR2C(plan: CufftHandle, idata: *mut f32, odata: *mut CufftComplex) -> CufftResult;

    fn cufftExecC2C(
        plan: CufftHandle,
        idata: *mut CufftComplex,
        odata: *mut CufftComplex,
        direction: c_int,
    ) -> CufftResult;

    fn cufftDestroy(plan: CufftHandle) -> CufftResult;

    fn cufftSetStream(plan: CufftHandle, stream: *mut std::ffi::c_void) -> CufftResult;
}

/// Safe wrapper around cuFFT plan
pub struct CufftPlan {
    handle: CufftHandle,
}

impl CufftPlan {
    /// Create a 1D FFT plan for batched real-to-complex transforms
    pub fn new_r2c_1d(fft_size: usize, batch_count: usize) -> Result<Self, CudaError> {
        let mut handle: CufftHandle = 0;
        unsafe {
            cufftPlan1d(
                &mut handle,
                fft_size as c_int,
                CufftType::R2C,
                batch_count as c_int,
            )
            .to_result()?;
        }
        Ok(Self { handle })
    }

    /// Create a batched 1D R2C plan with custom strides
    /// This is used for STFT where frames are stored with a specific stride
    pub fn new_r2c_batched(
        fft_size: usize,
        batch_count: usize,
        input_stride: usize, // Distance between elements within a frame
        input_dist: usize,   // Distance between frames
        output_dist: usize,  // Distance between output frames (typically fft_size/2+1)
    ) -> Result<Self, CudaError> {
        let mut handle: CufftHandle = 0;
        let n = [fft_size as c_int];

        unsafe {
            cufftPlanMany(
                &mut handle,
                1, // rank (1D)
                n.as_ptr(),
                ptr::null(), // inembed (NULL for default)
                input_stride as c_int,
                input_dist as c_int,
                ptr::null(), // onembed (NULL for default)
                1,           // output stride
                output_dist as c_int,
                CufftType::R2C,
                batch_count as c_int,
            )
            .to_result()?;
        }
        Ok(Self { handle })
    }

    /// Execute real-to-complex FFT
    ///
    /// # Safety
    /// - input must contain at least `fft_size * batch_count` f32 values
    /// - output must have space for at least `(fft_size/2+1) * batch_count` complex values
    pub unsafe fn exec_r2c(
        &self,
        input: *mut f32,
        output: *mut CufftComplex,
    ) -> Result<(), CudaError> {
        cufftExecR2C(self.handle, input, output).to_result()
    }

    /// Get the plan handle for advanced operations
    pub fn handle(&self) -> CufftHandle {
        self.handle
    }
}

impl Drop for CufftPlan {
    fn drop(&mut self) {
        unsafe {
            let _ = cufftDestroy(self.handle);
        }
    }
}

/// GPU STFT operator
///
/// Performs Short-Time Fourier Transform on GPU using cuFFT.
pub struct GpuStft {
    plan: CufftPlan,
    fft_size: usize,
    hop_size: usize,
    num_frames: usize,
    onesided_bins: usize,
}

impl GpuStft {
    /// Create a new GPU STFT operator
    ///
    /// # Arguments
    /// * `signal_length` - Length of input signal
    /// * `fft_size` - FFT window size
    /// * `hop_size` - Hop between frames
    pub fn new(signal_length: usize, fft_size: usize, hop_size: usize) -> Result<Self, CudaError> {
        let num_frames = (signal_length - fft_size) / hop_size + 1;
        let onesided_bins = fft_size / 2 + 1;

        // Create batched R2C plan
        let plan = CufftPlan::new_r2c_1d(fft_size, num_frames)?;

        Ok(Self {
            plan,
            fft_size,
            hop_size,
            num_frames,
            onesided_bins,
        })
    }

    /// Get number of output frames
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Get number of frequency bins (onesided)
    pub fn num_bins(&self) -> usize {
        self.onesided_bins
    }
}

/// Execute GPU STFT on a signal
///
/// # Arguments
/// * `ctx` - CUDA context
/// * `cache` - Compiled STFT kernels
/// * `signal` - Input signal tensor [signal_length] (batch not yet supported)
/// * `window` - Window function [fft_size]
/// * `fft_size` - FFT size
/// * `hop_size` - Hop between frames
/// * `onesided` - If true, return only positive frequencies
///
/// # Returns
/// Complex spectrogram [frames, bins, 2] where last dim is [real, imag]
pub fn gpu_stft(
    ctx: &IconnxCudaContext,
    cache: &StftKernelCache,
    signal: &GpuTensor,
    window: &GpuTensor,
    fft_size: usize,
    hop_size: usize,
    onesided: bool,
) -> Result<GpuTensor, CudaError> {
    let signal_shape = signal.shape();

    // Currently only support 1D signal
    if signal_shape.len() != 1 {
        return Err(CudaError::Kernel(format!(
            "Signal must be 1D, got {:?}",
            signal_shape
        )));
    }
    let signal_length = signal_shape[0];

    if window.shape() != [fft_size] {
        return Err(CudaError::Kernel(format!(
            "Window must have shape [{}], got {:?}",
            fft_size,
            window.shape()
        )));
    }

    let num_frames = (signal_length.saturating_sub(fft_size)) / hop_size + 1;
    if num_frames == 0 {
        return Err(CudaError::Kernel(
            "Signal too short for given fft_size".to_string(),
        ));
    }

    let output_bins = if onesided { fft_size / 2 + 1 } else { fft_size };

    // Step 1: Extract and window frames
    // Output shape: [num_frames, fft_size]
    let mut windowed_frames = GpuTensor::zeros_f32(ctx, vec![num_frames, fft_size])?;

    // Launch config: grid (ceil(fft_size/256), num_frames), block (256).
    // CUDA grid Y dimension limit is 65535, so we chunk if num_frames > 65535.
    let threads_per_block = 256u32;
    let blocks_x = (fft_size as u32).div_ceil(threads_per_block);

    const MAX_FRAMES_PER_LAUNCH: usize = 65535;

    // SAFETY: window_frames_kernel signature is
    // `(float* out, const float* signal, const float* window, int signal_length,
    //   int fft_size, int hop_size, int chunk_frames, int frame_offset,
    //   int chunk_offset_elements)`.
    let window_kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, f32>,
            i32, i32, i32, i32, i32, i32,
        )>("window_frames_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("window_frames_kernel lookup failed: {}", e)))?;

    // Process frames in chunks to avoid exceeding CUDA grid limits.
    let mut frame_offset = 0usize;
    while frame_offset < num_frames {
        let chunk_frames = (num_frames - frame_offset).min(MAX_FRAMES_PER_LAUNCH);

        let config = garboard::LaunchConfig {
            grid_dim: (blocks_x, chunk_frames as u32, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let chunk_offset_elements = frame_offset * fft_size;

        window_kernel
            .launch(
                ctx.garboard_stream(),
                &config,
                (
                    windowed_frames.data_f32_mut()?,
                    signal.data_f32()?,
                    window.data_f32()?,
                    signal_length as i32,
                    fft_size as i32,
                    hop_size as i32,
                    chunk_frames as i32,
                    frame_offset as i32,
                    chunk_offset_elements as i32,
                ),
            )
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "window_frames launch failed (chunk at offset {}): {}",
                    frame_offset, e
                ))
            })?;

        frame_offset += chunk_frames;
    }

    // Step 2: Execute cuFFT (R2C transform on all frames)
    let plan = CufftPlan::new_r2c_1d(fft_size, num_frames)?;

    // Allocate output buffer for complex FFT results
    // R2C produces num_frames * (fft_size/2+1) complex values
    let complex_output_size = num_frames * (fft_size / 2 + 1);
    let fft_output = ctx.alloc_zeros::<CufftComplex>(complex_output_size)?;

    unsafe {
        // Raw-pointer FFI: garboard's `as_raw_device_ptr` returns the
        // underlying CUdeviceptr. cuFFT's `exec_r2c` runs synchronously
        // with respect to the default stream (implicit via the context),
        // so no separate stream-sync guard is required here.
        let input_ptr = windowed_frames.data_f32_mut()?.as_raw_device_ptr();
        let output_ptr = fft_output.as_raw_device_ptr();
        plan.exec_r2c(input_ptr as *mut f32, output_ptr as *mut CufftComplex)?;
    }

    // Step 3: Unpack complex results to [frames, bins, 2] format.
    let mut output = GpuTensor::zeros_f32(ctx, vec![num_frames, output_bins, 2])?;

    let bins_blocks = (output_bins as u32).div_ceil(threads_per_block);

    // SAFETY: unpack_complex_kernel signature is
    // `(float* out, const CufftComplex* fft_out, int chunk_frames,
    //   int output_bins, int frame_offset, int output_offset_elements)`.
    let unpack_kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, CufftComplex>,
            i32, i32, i32, i32,
        )>("unpack_complex_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("unpack_complex_kernel lookup failed: {}", e)))?;

    // Process frames in chunks to avoid exceeding CUDA grid limits.
    let mut frame_offset = 0usize;
    while frame_offset < num_frames {
        let chunk_frames = (num_frames - frame_offset).min(MAX_FRAMES_PER_LAUNCH);

        let unpack_config = garboard::LaunchConfig {
            grid_dim: (bins_blocks, chunk_frames as u32, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let output_offset_elements = frame_offset * output_bins * 2;

        unpack_kernel
            .launch(
                ctx.garboard_stream(),
                &unpack_config,
                (
                    output.data_f32_mut()?,
                    &fft_output,
                    chunk_frames as i32,
                    output_bins as i32,
                    frame_offset as i32,
                    output_offset_elements as i32,
                ),
            )
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "unpack_complex launch failed (chunk at offset {}): {}",
                    frame_offset, e
                ))
            })?;

        frame_offset += chunk_frames;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU with cuFFT"]
    fn test_cufft_plan_creation() {
        // Test that we can create a cuFFT plan
        let plan = CufftPlan::new_r2c_1d(256, 1);
        assert!(
            plan.is_ok(),
            "Failed to create cuFFT plan: {:?}",
            plan.err()
        );
        println!("cuFFT plan created successfully!");
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuFFT"]
    fn test_simple_fft() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        let fft_size = 8;
        let plan = CufftPlan::new_r2c_1d(fft_size, 1).expect("Failed to create plan");

        // Create input: simple impulse at position 0
        let input_data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let input_gpu = ctx.htod(&input_data).expect("Failed to upload input");

        // Allocate output (R2C produces N/2+1 complex values)
        let output_size = fft_size / 2 + 1;
        let output_gpu = ctx
            .alloc_zeros::<CufftComplex>(output_size)
            .expect("Failed to allocate output");

        // Execute FFT. Raw-pointer FFI to cuFFT; the default stream
        // synchronization is implicit here.
        unsafe {
            let input_ptr = input_gpu.as_raw_device_ptr();
            let output_ptr = output_gpu.as_raw_device_ptr();
            plan.exec_r2c(input_ptr as *mut f32, output_ptr as *mut CufftComplex)
                .expect("FFT execution failed");
        }

        // Copy back results via garboard's device-to-host.
        let result: Vec<CufftComplex> = ctx
            .dtoh(&output_gpu)
            .expect("Failed to download result");

        // For an impulse at position 0, FFT should give all 1s (DC + positive freqs)
        println!("FFT of impulse:");
        for (i, c) in result.iter().enumerate() {
            println!("  bin {}: {} + {}i", i, c.x, c.y);
            // All values should be approximately 1.0 + 0.0i
            assert!((c.x - 1.0).abs() < 1e-5, "Real part should be 1.0");
            assert!(c.y.abs() < 1e-5, "Imaginary part should be 0.0");
        }

        println!("Simple FFT test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuFFT"]
    fn test_stft_kernel_compilation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = StftKernelCache::new(&ctx).expect("Failed to compile STFT kernels");

        assert!(cache.module().function("window_frames_kernel").is_ok());
        assert!(cache.module().function("unpack_complex_kernel").is_ok());

        println!("STFT kernels compiled successfully!");
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuFFT"]
    fn test_gpu_stft() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = StftKernelCache::new(&ctx).expect("Failed to compile STFT kernels");

        // Create a simple test signal: impulse followed by zeros
        // Signal length = 32, fft_size = 8, hop_size = 4
        // This gives us (32 - 8) / 4 + 1 = 7 frames
        let fft_size = 8;
        let hop_size = 4;
        let signal_length = 32;

        let mut signal_data = vec![0.0f32; signal_length];
        signal_data[0] = 1.0; // Impulse at start

        // Create rectangular window (all 1s for simplicity)
        let window_data = vec![1.0f32; fft_size];

        let signal = GpuTensor::from_host_f32(&ctx, &signal_data, vec![signal_length])
            .expect("Failed to create signal tensor");
        let window = GpuTensor::from_host_f32(&ctx, &window_data, vec![fft_size])
            .expect("Failed to create window tensor");

        // Run STFT
        let result = gpu_stft(&ctx, &cache, &signal, &window, fft_size, hop_size, true)
            .expect("STFT failed");

        // Check output shape
        let expected_frames = (signal_length - fft_size) / hop_size + 1;
        let expected_bins = fft_size / 2 + 1; // onesided
        assert_eq!(result.shape(), &[expected_frames, expected_bins, 2]);

        // Get results
        let output = result
            .to_host_f32(&ctx)
            .expect("Failed to copy result to host");

        println!("STFT output shape: {:?}", result.shape());
        println!("Frame 0 spectrum:");
        for bin in 0..expected_bins {
            let real = output[bin * 2];
            let imag = output[bin * 2 + 1];
            println!("  bin {}: {} + {}i", bin, real, imag);
        }

        // First frame contains the impulse, so FFT should give constant magnitude
        // For rectangular window and impulse at position 0:
        // FFT should give 1.0 + 0.0i for all bins
        let first_frame_real = output[0]; // bin 0, real
        let first_frame_imag = output[1]; // bin 0, imag
        assert!(
            (first_frame_real - 1.0).abs() < 1e-4,
            "DC component should be 1.0, got {}",
            first_frame_real
        );
        assert!(
            first_frame_imag.abs() < 1e-4,
            "DC component should have 0 imaginary, got {}",
            first_frame_imag
        );

        println!("GPU STFT test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuFFT"]
    fn test_stft_hann_window() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = StftKernelCache::new(&ctx).expect("Failed to compile STFT kernels");

        let fft_size = 256;
        let hop_size = 128;

        // Generate a simple sinusoid at 440 Hz (assuming 16kHz sample rate)
        // Period = 16000 / 440 ≈ 36.36 samples
        let sample_rate = 16000.0f32;
        let frequency = 440.0f32;
        let signal_length = 1024;

        let signal_data: Vec<f32> = (0..signal_length)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        // Create Hann window
        let window_data: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos())
            })
            .collect();

        let signal = GpuTensor::from_host_f32(&ctx, &signal_data, vec![signal_length])
            .expect("Failed to create signal tensor");
        let window = GpuTensor::from_host_f32(&ctx, &window_data, vec![fft_size])
            .expect("Failed to create window tensor");

        // Run STFT
        let result = gpu_stft(&ctx, &cache, &signal, &window, fft_size, hop_size, true)
            .expect("STFT failed");

        let num_frames = (signal_length - fft_size) / hop_size + 1;
        let num_bins = fft_size / 2 + 1;
        assert_eq!(result.shape(), &[num_frames, num_bins, 2]);

        let output = result
            .to_host_f32(&ctx)
            .expect("Failed to copy result to host");

        // Find the bin with maximum magnitude in the first frame
        let mut max_magnitude = 0.0f32;
        let mut max_bin = 0;
        for bin in 0..num_bins {
            let idx = bin * 2;
            let real = output[idx];
            let imag = output[idx + 1];
            let magnitude = (real * real + imag * imag).sqrt();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                max_bin = bin;
            }
        }

        // The peak should be near the expected frequency bin
        // bin = frequency * fft_size / sample_rate
        let expected_bin = (frequency * fft_size as f32 / sample_rate).round() as usize;

        println!("STFT with Hann window:");
        println!(
            "  Expected peak bin: {} (frequency: {} Hz)",
            expected_bin, frequency
        );
        println!(
            "  Actual peak bin: {} (magnitude: {})",
            max_bin, max_magnitude
        );
        println!("  Number of frames: {}", num_frames);

        // Allow some tolerance for spectral leakage
        assert!(
            (max_bin as i32 - expected_bin as i32).abs() <= 1,
            "Peak should be near bin {}, got {}",
            expected_bin,
            max_bin
        );

        println!("STFT Hann window test passed!");
    }
}
