//! LSTM operator for Iconnx
//!
//! Long Short-Term Memory recurrent layer
//! ONNX spec: https://onnx.ai/onnx/operators/onnx__LSTM.html
//!
//! Implements single-layer LSTM with 4 gates:
//! - Input gate (i): Controls new information
//! - Forget gate (f): Controls memory retention
//! - Cell gate (c): Generates candidate values
//! - Output gate (o): Controls output

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for LSTM operations

use crate::tensor::Tensor;
use ndarray::{s, Array1, Array2, ArrayD};

/// LSTM operator - recurrent neural network layer
#[allow(clippy::upper_case_acronyms)] // LSTM is the standard ML terminology
pub struct LSTM;

impl LSTM {
    /// Forward pass: LSTM computation
    ///
    /// # Arguments
    /// * `inputs` - [X, W, R, B, ...] (minimum 4 inputs)
    ///   - X: Input sequence [seq_len, batch, input_size]
    ///   - W: Input weights [num_directions, 4*hidden_size, input_size]
    ///   - R: Recurrent weights [num_directions, 4*hidden_size, hidden_size]
    ///   - B: Biases [num_directions, 8*hidden_size]
    ///   - (optional: sequence_lens, initial_h, initial_c, P)
    ///
    /// # Returns
    /// Output Y: [seq_len, num_directions, batch, hidden_size]
    ///
    /// # Notes
    /// - Supports forward direction only (num_directions=1)
    /// - Uses default activations (sigmoid, tanh)
    /// - No peephole connections in Phase 1
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 3,
            "LSTM requires at least 3 inputs [X, W, R]"
        );

        let x = inputs[0].to_array();
        let w = inputs[1].to_array();
        let r = inputs[2].to_array();
        let b = if inputs.len() >= 4 {
            Some(inputs[3].to_array())
        } else {
            None
        };

        // Get dimensions
        let x_shape = x.shape();

        #[cfg(debug_assertions)]
        eprintln!(
            "LSTM: X.shape={:?}, W.shape={:?}, R.shape={:?}, bias provided={}",
            x_shape,
            w.shape(),
            r.shape(),
            b.is_some()
        );

        // Handle 2D input [seq_len, input_size] by adding batch=1 dimension
        let (x_3d, seq_len, batch_size, input_size) = if x_shape.len() == 2 {
            let seq_len = x_shape[0];
            let input_size = x_shape[1];
            let x_expanded = x
                .to_shape((seq_len, 1, input_size))
                .unwrap()
                .into_owned()
                .into_dyn();
            (x_expanded, seq_len, 1_usize, input_size)
        } else if x_shape.len() == 3 {
            (x.to_owned(), x_shape[0], x_shape[1], x_shape[2])
        } else {
            panic!("LSTM input X must be 2D [seq_len, input_size] or 3D [seq_len, batch, input_size], got {:?}", x_shape);
        };

        let x = x_3d.view();

        let w_shape = w.shape();
        let num_directions = w_shape[0];
        let hidden_size = w_shape[1] / 4; // 4 gates

        // Support bidirectional (num_directions=2) by processing only forward direction
        // TODO: Implement backward direction
        if num_directions != 1 && num_directions != 2 {
            panic!("LSTM num_directions must be 1 or 2, got {}", num_directions);
        }
        assert_eq!(w_shape[2], input_size, "W input dimension mismatch");

        // If bias not provided, create zeros
        let b_arr;
        let b_ref = if let Some(bias_tensor) = b {
            bias_tensor
        } else {
            // Create zero bias [1, 8*hidden_size]
            b_arr = ArrayD::zeros(vec![1, 8 * hidden_size]);
            &b_arr
        };
        assert_eq!(r.shape()[2], hidden_size, "R recurrent dimension mismatch");

        // Initialize hidden and cell states
        // Use provided initial states if available (inputs[4] = initial_h, inputs[5] = initial_c)
        // Note: empty inputs are filtered out before forward() is called
        let h = if inputs.len() >= 5 {
            // initial_h shape: [num_directions, batch, hidden_size]
            let h_init = inputs[4].to_array();
            h_init
                .slice(s![0, .., ..])
                .to_owned()
                .into_dimensionality()
                .unwrap()
        } else {
            Array2::zeros((batch_size, hidden_size))
        };

        let c = if inputs.len() >= 6 {
            // initial_c shape: [num_directions, batch, hidden_size]
            let c_init = inputs[5].to_array();
            c_init
                .slice(s![0, .., ..])
                .to_owned()
                .into_dimensionality()
                .unwrap()
        } else {
            Array2::zeros((batch_size, hidden_size))
        };

        // Process forward and backward directions
        let mut all_outputs = Vec::new(); // [num_directions][seq_len][batch, hidden]

        for dir in 0..num_directions {
            // Extract weight matrices for each gate in ONNX order [iofc]
            // i = input gate, o = output gate, f = forget gate, c = cell gate
            let w_i = w.slice(s![dir, 0..hidden_size, ..]).to_owned();
            let w_o = w
                .slice(s![dir, hidden_size..2 * hidden_size, ..])
                .to_owned();
            let w_f = w
                .slice(s![dir, 2 * hidden_size..3 * hidden_size, ..])
                .to_owned();
            let w_c = w
                .slice(s![dir, 3 * hidden_size..4 * hidden_size, ..])
                .to_owned();

            let r_i = r.slice(s![dir, 0..hidden_size, ..]).to_owned();
            let r_o = r
                .slice(s![dir, hidden_size..2 * hidden_size, ..])
                .to_owned();
            let r_f = r
                .slice(s![dir, 2 * hidden_size..3 * hidden_size, ..])
                .to_owned();
            let r_c = r
                .slice(s![dir, 3 * hidden_size..4 * hidden_size, ..])
                .to_owned();

            // Extract biases for this direction in ONNX order [Wb_iofc, Rb_iofc]
            let b_flat = b_ref.slice(s![dir, ..]).to_owned();
            let wb_i = b_flat.slice(s![0..hidden_size]).to_owned();
            let wb_o = b_flat.slice(s![hidden_size..2 * hidden_size]).to_owned();
            let wb_f = b_flat
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned();
            let wb_c = b_flat
                .slice(s![3 * hidden_size..4 * hidden_size])
                .to_owned();

            let rb_i = b_flat
                .slice(s![4 * hidden_size..5 * hidden_size])
                .to_owned();
            let rb_o = b_flat
                .slice(s![5 * hidden_size..6 * hidden_size])
                .to_owned();
            let rb_f = b_flat
                .slice(s![6 * hidden_size..7 * hidden_size])
                .to_owned();
            let rb_c = b_flat
                .slice(s![7 * hidden_size..8 * hidden_size])
                .to_owned();

            // Initialize states for this direction
            let mut h_dir = if inputs.len() >= 5 {
                let h_init = inputs[4].to_array();
                h_init
                    .slice(s![dir, .., ..])
                    .to_owned()
                    .into_dimensionality()
                    .unwrap()
            } else {
                h.clone()
            };

            let mut c_dir = if inputs.len() >= 6 {
                let c_init = inputs[5].to_array();
                c_init
                    .slice(s![dir, .., ..])
                    .to_owned()
                    .into_dimensionality()
                    .unwrap()
            } else {
                c.clone()
            };

            // Storage for outputs in this direction
            let mut dir_outputs = Vec::with_capacity(seq_len);

            // Process timesteps (forward for dir=0, backward for dir=1)
            let time_steps: Vec<usize> = if dir == 0 {
                (0..seq_len).collect()
            } else {
                (0..seq_len).rev().collect()
            };

            for t in time_steps {
                let x_t = x.slice(s![t, .., ..]).to_owned(); // [batch, input_size]

                // Compute gates
                let i_t = Self::gate(&x_t, &w_i, &h_dir, &r_i, &wb_i, &rb_i, Self::sigmoid);
                let f_t = Self::gate(&x_t, &w_f, &h_dir, &r_f, &wb_f, &rb_f, Self::sigmoid);
                let c_t_candidate = Self::gate(&x_t, &w_c, &h_dir, &r_c, &wb_c, &rb_c, Self::tanh);
                let o_t = Self::gate(&x_t, &w_o, &h_dir, &r_o, &wb_o, &rb_o, Self::sigmoid);

                // Update cell state
                c_dir = &f_t * &c_dir + &i_t * &c_t_candidate;

                // Update hidden state
                h_dir = &o_t * &c_dir.mapv(|x| x.tanh());

                // Store output
                dir_outputs.push((t, h_dir.clone()));
            }

            // Sort outputs by time for backward direction
            if dir == 1 {
                dir_outputs.sort_by_key(|(t, _)| *t);
            }

            all_outputs.push(dir_outputs);
        }

        // Interleave outputs: [seq_len, num_directions, batch, hidden]
        let mut output_vec = Vec::new();
        for t in 0..seq_len {
            for dir in 0..num_directions {
                output_vec.extend(all_outputs[dir][t].1.iter().copied());
            }
        }

        let output = ArrayD::from_shape_vec(
            vec![seq_len, num_directions, batch_size, hidden_size],
            output_vec,
        )
        .expect("Failed to create LSTM output");

        #[cfg(debug_assertions)]
        eprintln!("LSTM output.shape = {:?}", output.shape());

        Tensor::from_array(output)
    }

    /// Compute a single gate: activation(Xt @ W^T + Ht-1 @ R^T + Wb + Rb)
    fn gate<F>(
        x_t: &Array2<f32>,
        w: &Array2<f32>,
        h_prev: &Array2<f32>,
        r: &Array2<f32>,
        wb: &Array1<f32>,
        rb: &Array1<f32>,
        activation: F,
    ) -> Array2<f32>
    where
        F: Fn(f32) -> f32,
    {
        // Xt @ W^T: [batch, input] @ [hidden, input]^T = [batch, hidden]
        let xw = x_t.dot(&w.t());

        // Ht-1 @ R^T: [batch, hidden] @ [hidden, hidden]^T = [batch, hidden]
        let hr = h_prev.dot(&r.t());

        // Sum: XW + HR + Wb + Rb (biases broadcast across batch dimension)
        let mut gate_input = &xw + &hr;
        for i in 0..gate_input.nrows() {
            for j in 0..gate_input.ncols() {
                gate_input[[i, j]] += wb[j] + rb[j];
            }
        }

        // Apply activation
        gate_input.mapv(activation)
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f32) -> f32 {
        x.tanh()
    }
}
