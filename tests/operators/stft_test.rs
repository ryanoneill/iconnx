use iconnx::attributes::NodeAttributes;
use iconnx::operators::stft::Stft;
/// STFT operator tests
///
/// Tests Short-Time Fourier Transform implementation against ONNX spec
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__STFT.html
use iconnx::tensor::Tensor;

/// Test: Simple STFT with small signal
///
/// Verifies:
/// - Basic FFT computation on overlapping frames
/// - Output shape [batch, frames, dft_unique_bins, 2]
/// - Complex output (real + imaginary)
#[test]
fn test_stft_simple() {
    // Simple sine wave signal
    // [1, 16, 1] - batch=1, signal_length=16, channels=1
    let signal_data: Vec<f32> = (0..16)
        .map(|i| (i as f32 * std::f32::consts::PI / 4.0).sin())
        .collect();
    let signal = Tensor::from_vec(signal_data, vec![1, 16, 1]);

    // frame_step = 8 (hop length)
    let frame_step = Tensor::from_vec(vec![8.0], vec![]);

    // No window (will use rectangular window)
    // frame_length = 8 (FFT size)
    let frame_length = Tensor::from_vec(vec![8.0], vec![]);

    let inputs = vec![signal, frame_step, frame_length];
    let output = Stft::forward(&inputs, &NodeAttributes::new());

    // Expected shape: [1, frames, dft_unique_bins, 2]
    // frames = (signal_length - frame_length) / frame_step + 1 = (16 - 8) / 8 + 1 = 2
    // dft_unique_bins = frame_length / 2 + 1 = 8 / 2 + 1 = 5
    let shape = output.shape();
    assert_eq!(shape, &[1, 2, 5, 2], "Output shape should be [1, 2, 5, 2]");

    // Verify output contains real and imaginary parts
    let data = output.to_array();
    assert!(data.len() > 0, "Output should contain data");

    // For a sine wave, DC component should be near zero
    // but some frequency bins should have non-zero values
    let frame0_dc_real = data[[0, 0, 0, 0]];
    let frame0_dc_imag = data[[0, 0, 0, 1]];
    println!(
        "Frame 0, DC: real={}, imag={}",
        frame0_dc_real, frame0_dc_imag
    );

    // Check that at least some frequency bin has significant energy
    let mut has_energy = false;
    for bin in 0..5 {
        let real = data[[0, 0, bin, 0]];
        let imag = data[[0, 0, bin, 1]];
        let magnitude = (real * real + imag * imag).sqrt();
        if magnitude > 0.5 {
            has_energy = true;
            break;
        }
    }
    assert!(
        has_energy,
        "STFT should produce non-zero frequency components"
    );
}

/// Test: STFT with Hann window
///
/// Verifies:
/// - Window function application
/// - Output with windowing vs rectangular
#[test]
fn test_stft_with_window() {
    // Signal: constant amplitude
    let signal_data: Vec<f32> = vec![1.0; 16];
    let signal = Tensor::from_vec(signal_data, vec![1, 16, 1]);

    // frame_step = 8
    let frame_step = Tensor::from_vec(vec![8.0], vec![]);

    // Hann window for frame_length = 8
    // Hann: 0.5 - 0.5 * cos(2π * n / (N-1))
    let hann: Vec<f32> = (0..8)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / 7.0).cos())
        .collect();
    let window = Tensor::from_vec(hann, vec![8]);

    // frame_length = 8
    let frame_length = Tensor::from_vec(vec![8.0], vec![]);

    let inputs = vec![signal, frame_step, window, frame_length];
    let output = Stft::forward(&inputs, &NodeAttributes::new());

    // Expected shape: [1, 2, 5, 2]
    let shape = output.shape();
    assert_eq!(shape, &[1, 2, 5, 2]);

    let data = output.to_array();

    // With Hann window on constant signal, DC component should be non-zero
    // (Hann window sums to N/2)
    let frame0_dc_real = data[[0, 0, 0, 0]];
    let frame0_dc_imag = data[[0, 0, 0, 1]];

    println!(
        "Frame 0 DC (windowed): real={}, imag={}",
        frame0_dc_real, frame0_dc_imag
    );

    // DC should be approximately 3.5 (sum of Hann window for N=8 is ~4, but normalized)
    // For constant signal through Hann window, DC = sum of window values
    assert!(
        (frame0_dc_real - 3.5).abs() < 0.5,
        "DC real should be ~3.5, got {}",
        frame0_dc_real
    );
    assert!(
        frame0_dc_imag.abs() < 0.1,
        "DC imaginary should be near zero"
    );
}

/// Test: STFT output shape calculation
///
/// Verifies:
/// - Correct number of frames
/// - Correct number of frequency bins (onesided FFT)
/// - Batch dimension preserved
#[test]
fn test_stft_output_shape() {
    let test_cases = vec![
        // (signal_length, frame_length, frame_step, expected_frames)
        (32, 16, 8, 3),   // (32-16)/8 + 1 = 3
        (64, 32, 16, 3),  // (64-32)/16 + 1 = 3
        (100, 20, 10, 9), // (100-20)/10 + 1 = 9
    ];

    for (signal_len, frame_len, frame_step, expected_frames) in test_cases {
        let signal_data: Vec<f32> = vec![0.5; signal_len];
        let signal = Tensor::from_vec(signal_data, vec![1, signal_len, 1]);

        let step = Tensor::from_vec(vec![frame_step as f32], vec![]);
        let length = Tensor::from_vec(vec![frame_len as f32], vec![]);

        let inputs = vec![signal, step, length];
        let output = Stft::forward(&inputs, &NodeAttributes::new());

        // dft_unique_bins = frame_length / 2 + 1
        let expected_bins = frame_len / 2 + 1;

        let shape = output.shape();
        assert_eq!(
            shape,
            &[1, expected_frames, expected_bins, 2],
            "signal_len={}, frame_len={}, frame_step={}",
            signal_len,
            frame_len,
            frame_step
        );
    }
}

/// Test: STFT with onesided=1 (explicit, same as default)
#[test]
fn test_stft_onesided_true() {
    let signal_data: Vec<f32> = vec![1.0; 16];
    let signal = Tensor::from_vec(signal_data, vec![1, 16, 1]);
    let frame_step = Tensor::from_vec(vec![8.0], vec![]);
    let frame_length = Tensor::from_vec(vec![8.0], vec![]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("onesided".to_string(), 1);

    let inputs = vec![signal, frame_step, frame_length];
    let output = Stft::forward(&inputs, &attrs);

    // onesided=1: dft_bins = 8/2 + 1 = 5
    assert_eq!(output.shape(), &[1, 2, 5, 2]);
}

/// Test: STFT with onesided=0 (full spectrum)
#[test]
fn test_stft_onesided_false() {
    let signal_data: Vec<f32> = vec![1.0; 16];
    let signal = Tensor::from_vec(signal_data, vec![1, 16, 1]);
    let frame_step = Tensor::from_vec(vec![8.0], vec![]);
    let frame_length = Tensor::from_vec(vec![8.0], vec![]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("onesided".to_string(), 0);

    let inputs = vec![signal, frame_step, frame_length];
    let output = Stft::forward(&inputs, &attrs);

    // onesided=0: dft_bins = 8 (full spectrum)
    assert_eq!(output.shape(), &[1, 2, 8, 2]);
}
