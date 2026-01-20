/// STFT operator for Iconnx
///
/// Short-Time Fourier Transform
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__STFT.html
///
/// Note: Uses f64 precision internally for FFT computation to improve
/// numerical precision, then converts to f32 for output.
use crate::tensor::Tensor;
use ndarray::{Array, Axis};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// STFT operator - Short-Time Fourier Transform
pub struct Stft;

impl Stft {
    /// Forward pass: STFT computation
    ///
    /// # Arguments
    /// * `inputs` - [signal, frame_step, window (optional), frame_length (optional)]
    ///   - signal: [batch_size, signal_length, 1] or [batch_size, signal_length, 2]
    ///   - frame_step: scalar (hop length between frames)
    ///   - window: optional [window_length] window function
    ///   - frame_length: optional scalar (FFT size, defaults to window length)
    ///
    /// # Returns
    /// Complex spectrogram [batch_size, frames, dft_unique_bins, 2]
    ///   - frames = (signal_length - frame_length) / frame_step + 1
    ///   - dft_unique_bins = frame_length / 2 + 1 (onesided FFT)
    ///   - Last dimension: [real, imaginary]
    ///
    /// # Attributes
    /// - onesided: If 1 (default), return only positive frequencies (N/2+1 bins).
    ///   If 0, return full N frequency bins.
    ///
    /// # Notes
    /// - If no window provided, uses rectangular window (all ones)
    /// - Supports mono signals (channel=1) only
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 2 && inputs.len() <= 4,
            "STFT requires 2-4 inputs: signal, frame_step, [window], [frame_length]"
        );

        // Parse inputs
        let signal = inputs[0].to_array();
        let frame_step = Self::extract_scalar_i64(&inputs[1]);

        // Get window and frame_length
        let (window_opt, frame_length) = match inputs.len() {
            2 => (None, None),
            3 => {
                // Could be window or frame_length
                // Check shape: if 1D array, it's window; if scalar, it's frame_length
                let third = &inputs[2];
                if third.shape().len() == 1 && third.shape()[0] > 1 {
                    // Window
                    (Some(third.to_array()), None)
                } else {
                    // frame_length
                    (None, Some(Self::extract_scalar_i64(third)))
                }
            }
            4 => {
                // Both window and frame_length
                (
                    Some(inputs[2].to_array()),
                    Some(Self::extract_scalar_i64(&inputs[3])),
                )
            }
            _ => unreachable!(),
        };

        // Handle both 2D [batch, signal_length] and 3D [batch, signal_length, channels] signals
        let signal_shape = signal.shape();
        let (batch_size, signal_length, signal_3d) = if signal_shape.len() == 2 {
            // 2D signal: add channel dimension
            let signal_expanded = signal.to_owned().insert_axis(ndarray::Axis(2));
            (signal_shape[0], signal_shape[1], signal_expanded)
        } else if signal_shape.len() == 3 {
            assert_eq!(
                signal_shape[2], 1,
                "Only mono signals (channels=1) supported, got {} channels",
                signal_shape[2]
            );
            (signal_shape[0], signal_shape[1], signal.to_owned())
        } else {
            panic!(
                "Signal must be 2D [batch, length] or 3D [batch, length, channels], got {:?}",
                signal_shape
            );
        };

        let signal = signal_3d;

        // Determine frame_length
        let frame_len = if let Some(fl) = frame_length {
            fl as usize
        } else if let Some(w) = window_opt {
            w.shape()[0]
        } else {
            panic!("Must provide either window or frame_length");
        };

        // Create window if not provided
        let window = if let Some(w) = window_opt {
            w.as_slice().unwrap().to_vec()
        } else {
            vec![1.0; frame_len]
        };

        assert_eq!(
            window.len(),
            frame_len,
            "Window length must match frame_length"
        );

        // Calculate number of frames
        let hop_size = frame_step as usize;
        assert!(
            signal_length >= frame_len,
            "Signal length ({}) must be >= frame_length ({})",
            signal_length,
            frame_len
        );
        let num_frames = (signal_length - frame_len) / hop_size + 1;

        // Get onesided attribute (ONNX default is 1)
        let onesided = attributes.get_int("onesided").unwrap_or(1) == 1;

        // FFT setup - use f64 precision internally to preserve small values
        let dft_bins = if onesided {
            frame_len / 2 + 1 // Only positive frequencies
        } else {
            frame_len // Full spectrum
        };
        let mut planner: FftPlanner<f64> = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_len);

        // Output: [batch, frames, dft_bins, 2] per ONNX spec
        let mut output = Array::zeros((batch_size, num_frames, dft_bins, 2));

        // Process each batch
        for b in 0..batch_size {
            // Extract signal for this batch: [batch, signal_length, channels] -> [signal_length]
            // Index batch (axis 0), then extract all signal_length values, then take channel 0 (axis 2)
            let signal_2d = signal.index_axis(Axis(0), b); // Shape: [signal_length, channels]
            let signal_slice: Vec<f32> = signal_2d
                .index_axis(Axis(1), 0) // Take channel 0
                .iter()
                .copied()
                .collect();

            // Extract and process each frame
            for frame_idx in 0..num_frames {
                let start = frame_idx * hop_size;
                let end = start + frame_len;

                // Extract frame
                let frame: Vec<f32> = signal_slice[start..end].to_vec();

                // Apply window and convert to f64 for FFT precision
                let windowed: Vec<Complex<f64>> = frame
                    .iter()
                    .zip(window.iter())
                    .map(|(s, w)| Complex::new((s * w) as f64, 0.0))
                    .collect();

                // Compute FFT in f64 precision
                let mut fft_buffer = windowed;
                fft.process(&mut fft_buffer);

                // Extract spectrum and convert back to f32
                // (onesided: N/2+1 bins, twosided: N bins)
                for (bin, &complex_val) in fft_buffer.iter().take(dft_bins).enumerate() {
                    output[[b, frame_idx, bin, 0]] = complex_val.re as f32;
                    output[[b, frame_idx, bin, 1]] = complex_val.im as f32;
                }
            }
        }

        Tensor::from_array(output.into_dyn())
    }

    /// Extract scalar int64 value from tensor
    fn extract_scalar_i64(tensor: &Tensor) -> i64 {
        // Handle both int64 and float32 scalar tensors
        if tensor.is_int64() {
            let vals = tensor.as_slice_i64();
            assert!(!vals.is_empty(), "Expected non-empty tensor");
            vals[0]
        } else if tensor.is_int32() {
            let vals = tensor.as_slice_i32();
            assert!(!vals.is_empty(), "Expected non-empty tensor");
            vals[0] as i64
        } else {
            // Float32
            let vals = tensor.as_slice();
            assert!(!vals.is_empty(), "Expected non-empty tensor");
            vals[0] as i64
        }
    }
}
