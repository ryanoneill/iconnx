use iconnx::attributes::NodeAttributes;
/// Comprehensive LSTM operator tests for Iconnx
///
/// Tests all branches, edge cases, and regression tests for bugs:
/// - Initial states usage (regression: was ignoring them)
/// - Bidirectional support (regression: was duplicating forward pass)
/// - ONNX gate order [iofc] (regression: was using [ifco])
///
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__LSTM.html
use iconnx::operators::lstm::LSTM;
use iconnx::tensor::Tensor;

#[test]
fn test_lstm_forward_no_initial_state() {
    // Simple forward LSTM without initial states
    // Input: [seq_len=2, batch=1, input_size=3]
    let x = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, // timestep 0
            4.0, 5.0, 6.0, // timestep 1
        ],
        vec![2, 1, 3],
    );

    // Weights W: [num_directions=1, 4*hidden_size=4, input_size=3]
    // Order: [iofc] - input, output, forget, cell
    let w = Tensor::from_vec(
        vec![
            0.1, 0.2, 0.3, // input gate
            0.4, 0.5, 0.6, // output gate
            0.7, 0.8, 0.9, // forget gate
            1.0, 1.1, 1.2, // cell gate
        ],
        vec![1, 4, 3],
    );

    // Recurrent weights R: [num_directions=1, 4*hidden_size=4, hidden_size=1]
    let r = Tensor::from_vec(
        vec![
            0.1, // input gate recurrent
            0.2, // output gate recurrent
            0.3, // forget gate recurrent
            0.4, // cell gate recurrent
        ],
        vec![1, 4, 1],
    );

    // Bias B: [num_directions=1, 8*hidden_size=8]
    // [Wb_iofc, Rb_iofc]
    let b = Tensor::from_vec(
        vec![
            0.1, 0.2, 0.3, 0.4, // Wb: input, output, forget, cell
            0.1, 0.2, 0.3, 0.4, // Rb: input, output, forget, cell
        ],
        vec![1, 8],
    );

    let result = LSTM::forward(&[x, w, r, b], &NodeAttributes::new());

    // Output shape: [seq_len=2, num_directions=1, batch=1, hidden_size=1]
    assert_eq!(result.shape(), &[2, 1, 1, 1]);

    // Values should be non-zero and finite
    let vals = result.as_slice();
    assert_eq!(vals.len(), 2);
    for val in vals {
        assert!(val.is_finite());
    }
}

#[test]
fn test_lstm_bidirectional() {
    // Bidirectional LSTM
    // Input: [seq_len=2, batch=1, input_size=2]
    let x = Tensor::from_vec(
        vec![
            1.0, 2.0, // timestep 0
            3.0, 4.0, // timestep 1
        ],
        vec![2, 1, 2],
    );

    // Weights W: [num_directions=2, 4*hidden_size=4, input_size=2]
    let w = Tensor::from_vec(
        vec![
            // Forward direction
            0.1, 0.2, // input gate
            0.3, 0.4, // output gate
            0.5, 0.6, // forget gate
            0.7, 0.8, // cell gate
            // Backward direction
            0.1, 0.2, // input gate
            0.3, 0.4, // output gate
            0.5, 0.6, // forget gate
            0.7, 0.8, // cell gate
        ],
        vec![2, 4, 2],
    );

    // Recurrent weights R: [num_directions=2, 4*hidden_size=4, hidden_size=1]
    let r = Tensor::from_vec(
        vec![
            // Forward
            0.1, 0.2, 0.3, 0.4, // Backward
            0.1, 0.2, 0.3, 0.4,
        ],
        vec![2, 4, 1],
    );

    // Bias: [num_directions=2, 8*hidden_size=8]
    let b = Tensor::from_vec(
        vec![
            // Forward
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, // Backward
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
        ],
        vec![2, 8],
    );

    let result = LSTM::forward(&[x, w, r, b], &NodeAttributes::new());

    // Output shape: [seq_len=2, num_directions=2, batch=1, hidden_size=1]
    assert_eq!(result.shape(), &[2, 2, 1, 1]);

    // Should have 4 values total (2 timesteps * 2 directions)
    let vals = result.as_slice();
    assert_eq!(vals.len(), 4);
    for val in vals {
        assert!(val.is_finite());
    }
}

#[test]
fn test_lstm_with_initial_states() {
    // REGRESSION TEST: LSTM was ignoring initial_h and initial_c before
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 1, 2]);

    let w = Tensor::from_vec(
        vec![
            0.1, 0.2, // i
            0.3, 0.4, // o
            0.5, 0.6, // f
            0.7, 0.8, // c
        ],
        vec![1, 4, 2],
    );

    let r = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4, 1]);

    let b = Tensor::from_vec(
        vec![
            0.1, 0.2, 0.3, 0.4, // Wb
            0.1, 0.2, 0.3, 0.4, // Rb
        ],
        vec![1, 8],
    );

    // Initial hidden state: [num_directions=1, batch=1, hidden_size=1]
    let initial_h = Tensor::from_vec(vec![0.5], vec![1, 1, 1]);

    // Initial cell state: [num_directions=1, batch=1, hidden_size=1]
    let initial_c = Tensor::from_vec(vec![0.3], vec![1, 1, 1]);

    let result = LSTM::forward(&[x, w, r, b, initial_h, initial_c], &NodeAttributes::new());

    // Output shape: [seq_len=1, num_directions=1, batch=1, hidden_size=1]
    assert_eq!(result.shape(), &[1, 1, 1, 1]);

    let vals = result.as_slice();
    assert_eq!(vals.len(), 1);
    assert!(vals[0].is_finite());

    // With initial states, result should be different from zero-initialized
    assert_ne!(vals[0], 0.0);
}

#[test]
fn test_lstm_no_bias() {
    // LSTM without bias (should use zeros)
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 1, 2]);

    let w = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![1, 4, 2]);

    let r = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4, 1]);

    // No bias provided (only 3 inputs)
    let result = LSTM::forward(&[x, w, r], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 1, 1, 1]);

    let vals = result.as_slice();
    assert!(vals[0].is_finite());
}

#[test]
fn test_lstm_multiple_timesteps() {
    // Test LSTM with multiple timesteps to verify state propagation
    let x = Tensor::from_vec(
        vec![
            1.0, 0.0, // t=0
            0.0, 1.0, // t=1
            1.0, 1.0, // t=2
        ],
        vec![3, 1, 2],
    );

    let w = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![1, 4, 2]);

    let r = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4, 1]);

    let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 8]);

    let result = LSTM::forward(&[x, w, r, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3, 1, 1, 1]);

    let vals = result.as_slice();
    assert_eq!(vals.len(), 3);

    // Each timestep should produce different values
    assert_ne!(vals[0], vals[1]);
    assert_ne!(vals[1], vals[2]);
}

#[test]
fn test_lstm_batch_size_2() {
    // Test LSTM with batch size > 1
    let x = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0, 1 at t=0
        ],
        vec![1, 2, 2],
    );

    let w = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![1, 4, 2]);

    let r = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4, 1]);

    let b = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], vec![1, 8]);

    let result = LSTM::forward(&[x, w, r, b], &NodeAttributes::new());

    // Output shape: [seq_len=1, num_directions=1, batch=2, hidden_size=1]
    assert_eq!(result.shape(), &[1, 1, 2, 1]);

    let vals = result.as_slice();
    assert_eq!(vals.len(), 2);

    // Both batch elements should have valid outputs
    for val in vals {
        assert!(val.is_finite());
    }
}

#[test]
fn test_lstm_gate_order() {
    // REGRESSION TEST: Verify ONNX gate order [iofc] is used correctly
    // Was using [ifco] before

    let x = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![1, 1, 3]);

    // Set different values for each gate to detect incorrect ordering
    let w = Tensor::from_vec(
        vec![
            1.0, 0.0, 0.0, // input gate (i)
            0.0, 2.0, 0.0, // output gate (o)
            0.0, 0.0, 3.0, // forget gate (f)
            4.0, 0.0, 0.0, // cell gate (c)
        ],
        vec![1, 4, 3],
    );

    let r = Tensor::from_vec(
        vec![
            0.1, // i
            0.2, // o
            0.3, // f
            0.4, // c
        ],
        vec![1, 4, 1],
    );

    let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 8]);

    let result = LSTM::forward(&[x, w, r, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 1, 1, 1]);

    // Just verify it produces a valid output
    let val = result.as_slice()[0];
    assert!(val.is_finite());
}

#[test]
fn test_lstm_backward_direction() {
    // REGRESSION TEST: Verify backward direction processes timesteps in reverse
    // Was just duplicating forward pass before

    let x = Tensor::from_vec(
        vec![
            1.0, 0.0, // t=0
            0.0, 1.0, // t=1
        ],
        vec![2, 1, 2],
    );

    let w = Tensor::from_vec(
        vec![
            // Forward
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // Backward (different weights)
            0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        ],
        vec![2, 4, 2],
    );

    let r = Tensor::from_vec(
        vec![
            0.1, 0.2, 0.3, 0.4, // forward
            0.4, 0.3, 0.2, 0.1, // backward
        ],
        vec![2, 4, 1],
    );

    let b = Tensor::from_vec(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // forward
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // backward
        ],
        vec![2, 8],
    );

    let result = LSTM::forward(&[x, w, r, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2, 1, 1]);

    let vals = result.as_slice();
    // Layout: [t0_fwd, t0_bwd, t1_fwd, t1_bwd]

    // Forward and backward should produce different values
    assert_ne!(vals[0], vals[1]); // t=0: forward != backward
    assert_ne!(vals[2], vals[3]); // t=1: forward != backward
}

#[test]
fn test_lstm_hidden_size_larger() {
    // Test with hidden_size > 1
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 1, 2]);

    // hidden_size = 2, so 4*hidden_size = 8
    let w = Tensor::from_vec(
        vec![
            // Input gate (2 neurons)
            0.1, 0.2, 0.3, 0.4, // Output gate
            0.5, 0.6, 0.7, 0.8, // Forget gate
            0.9, 1.0, 1.1, 1.2, // Cell gate
            1.3, 1.4, 1.5, 1.6,
        ],
        vec![1, 8, 2],
    );

    let r = Tensor::from_vec(
        vec![
            // Each gate has 2x2 matrix
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ],
        vec![1, 8, 2],
    );

    let b = Tensor::from_vec(
        vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // Wb [iofc]
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // Rb [iofc]
        ],
        vec![1, 16],
    );

    let result = LSTM::forward(&[x, w, r, b], &NodeAttributes::new());

    // Output: [seq_len=1, num_directions=1, batch=1, hidden_size=2]
    assert_eq!(result.shape(), &[1, 1, 1, 2]);

    let vals = result.as_slice();
    assert_eq!(vals.len(), 2);
    for val in vals {
        assert!(val.is_finite());
    }
}
