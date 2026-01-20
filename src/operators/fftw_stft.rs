/// FFTW-based STFT operator for Iconnx
///
/// Uses FFTW3 (via fftw crate) to compute STFT.
/// FFTW is the gold-standard FFT library and may produce results
/// closer to Intel MKL (used by ORT) than rustfft.
///
/// The FFTW library won the 1999 J. H. Wilkinson Prize for Numerical Software.
use anyhow::{Context, Result};
use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan64};
use fftw::types::{Flag, Sign};
use std::f64::consts::PI;

/// Compute STFT using FFTW's implementation
///
/// # Arguments
/// * `signal` - Input signal as f32 slice, shape [signal_length]
/// * `window` - Window function as f32 slice, shape [frame_length]
/// * `frame_length` - FFT size
/// * `hop_length` - Hop length between frames
/// * `onesided` - If true, return only positive frequencies (N/2+1 bins)
///
/// # Returns
/// STFT result as Vec<f32> in shape [1, num_frames, fft_bins, 2]
/// where last dimension is [real, imag]
pub fn fftw_stft(
    signal: &[f32],
    window: &[f32],
    frame_length: usize,
    hop_length: usize,
    onesided: bool,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let signal_length = signal.len();

    // Validate inputs
    if window.len() != frame_length {
        anyhow::bail!(
            "Window length {} doesn't match frame_length {}",
            window.len(),
            frame_length
        );
    }
    if signal_length < frame_length {
        anyhow::bail!(
            "Signal length {} is shorter than frame_length {}",
            signal_length,
            frame_length
        );
    }

    // Calculate number of frames
    let num_frames = (signal_length - frame_length) / hop_length + 1;

    // FFT output bins
    let fft_bins = if onesided {
        frame_length / 2 + 1
    } else {
        frame_length
    };

    // Create FFTW plan (using f64 for precision)
    // FFTW_MEASURE takes more time to plan but produces faster execution
    // For STFT, we use the same FFT size many times, so MEASURE is worth it
    let mut plan: C2CPlan64 = C2CPlan::aligned(
        &[frame_length],
        Sign::Forward,
        Flag::MEASURE | Flag::DESTROYINPUT,
    )
    .context("Failed to create FFTW plan")?;

    // Allocate aligned buffers for FFTW
    let mut input_buffer: AlignedVec<fftw::types::c64> = AlignedVec::new(frame_length);
    let mut output_buffer: AlignedVec<fftw::types::c64> = AlignedVec::new(frame_length);

    // Output: [1, num_frames, fft_bins, 2] per ONNX spec
    let mut output = vec![0.0f32; num_frames * fft_bins * 2];

    // Process each frame
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        // Apply window and convert to complex f64
        for i in 0..frame_length {
            let sample = signal[start + i] as f64;
            let windowed = sample * (window[i] as f64);
            input_buffer[i] = fftw::types::c64::new(windowed, 0.0);
        }

        // Execute FFT
        plan.c2c(&mut input_buffer, &mut output_buffer)
            .context("FFTW execution failed")?;

        // Extract spectrum and convert back to f32
        for bin in 0..fft_bins {
            let complex_val = output_buffer[bin];
            let output_idx = (frame_idx * fft_bins + bin) * 2;
            output[output_idx] = complex_val.re as f32;
            output[output_idx + 1] = complex_val.im as f32;
        }
    }

    let shape = vec![1, num_frames, fft_bins, 2];
    Ok((output, shape))
}

/// Generate Hann window for testing
#[allow(dead_code)]
pub fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (length - 1) as f64).cos()) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fftw_stft_basic() {
        // Simple test with a sine wave
        let signal_length = 4096;
        let frame_length = 2048;
        let hop_length = 512;

        // Generate a simple sine wave
        let signal: Vec<f32> = (0..signal_length)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();

        // Hann window
        let window = hann_window(frame_length);

        let result = fftw_stft(&signal, &window, frame_length, hop_length, true);
        assert!(result.is_ok(), "STFT failed: {:?}", result);

        let (data, shape) = result.unwrap();
        assert_eq!(shape.len(), 4, "Expected 4D output");
        assert_eq!(shape[0], 1, "Batch size should be 1");
        assert_eq!(shape[3], 2, "Last dim should be 2 (real, imag)");

        // Verify expected number of frames
        let expected_frames = (signal_length - frame_length) / hop_length + 1;
        assert_eq!(shape[1], expected_frames, "Wrong number of frames");

        // Verify expected fft bins (onesided)
        let expected_bins = frame_length / 2 + 1;
        assert_eq!(shape[2], expected_bins, "Wrong number of fft bins");

        println!("FFTW STFT output shape: {:?}", shape);
        println!("FFTW STFT output size: {}", data.len());
    }

    #[test]
    fn test_fftw_vs_rustfft() {
        // Compare FFTW output to rustfft for the same input
        use rustfft::num_complex::Complex;
        use rustfft::FftPlanner;

        let signal_length = 4096;
        let frame_length = 2048;
        let hop_length = 512;

        // Generate test signal
        let signal: Vec<f32> = (0..signal_length)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();

        let window = hann_window(frame_length);

        // Get FFTW result
        let (fftw_data, _fftw_shape) =
            fftw_stft(&signal, &window, frame_length, hop_length, true).expect("FFTW STFT failed");

        // Compute rustfft result for first frame
        let mut planner: FftPlanner<f64> = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_length);

        let windowed: Vec<Complex<f64>> = signal[0..frame_length]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| Complex::new((*s * *w) as f64, 0.0))
            .collect();

        let mut rustfft_buffer = windowed;
        fft.process(&mut rustfft_buffer);

        // Compare first frame
        let fft_bins = frame_length / 2 + 1;
        let mut max_diff = 0.0f32;
        let mut correlation_sum = 0.0f64;
        let mut fftw_sum_sq = 0.0f64;
        let mut rustfft_sum_sq = 0.0f64;

        for bin in 0..fft_bins {
            let fftw_real = fftw_data[bin * 2];
            let fftw_imag = fftw_data[bin * 2 + 1];
            let rustfft_real = rustfft_buffer[bin].re as f32;
            let rustfft_imag = rustfft_buffer[bin].im as f32;

            let diff_real = (fftw_real - rustfft_real).abs();
            let diff_imag = (fftw_imag - rustfft_imag).abs();
            max_diff = max_diff.max(diff_real).max(diff_imag);

            correlation_sum += (fftw_real as f64) * (rustfft_real as f64)
                + (fftw_imag as f64) * (rustfft_imag as f64);
            fftw_sum_sq += (fftw_real as f64).powi(2) + (fftw_imag as f64).powi(2);
            rustfft_sum_sq += (rustfft_real as f64).powi(2) + (rustfft_imag as f64).powi(2);
        }

        let correlation = correlation_sum / (fftw_sum_sq.sqrt() * rustfft_sum_sq.sqrt());

        println!("FFTW vs rustfft first frame comparison:");
        println!("  Max difference: {:.6e}", max_diff);
        println!("  Correlation: {:.6}", correlation);

        // FFTW and rustfft should be very close (both are correct FFT implementations)
        assert!(
            max_diff < 1e-4,
            "FFTW and rustfft should produce very similar results, max_diff: {}",
            max_diff
        );
        assert!(
            correlation > 0.9999,
            "FFTW and rustfft should be highly correlated, correlation: {}",
            correlation
        );
    }
}
