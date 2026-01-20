/// Gemm operator for Iconnx
///
/// Implements generalized matrix multiply: alpha*A'@B' + beta*C
/// where A' = transpose(A) if transA else A
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Gemm.html
use crate::tensor::Tensor;

/// Gemm operator - generalized matrix multiplication
pub struct Gemm;

impl Gemm {
    /// Forward pass: alpha*A'@B' + beta*C
    ///
    /// # Arguments
    /// * `inputs` - 2-3 inputs: [A, B, C?]
    ///   - A: 2D matrix
    ///   - B: 2D matrix
    ///   - C: Optional 2D matrix for bias
    /// * `attributes` - ONNX attributes:
    ///   - alpha: Scalar multiplier for A@B (default 1.0)
    ///   - beta: Scalar multiplier for C (default 1.0)
    ///   - transA: Whether to transpose A (default 0)
    ///   - transB: Whether to transpose B (default 0)
    ///
    /// # Returns
    /// Tensor containing alpha*A'@B' + beta*C
    ///
    /// # Panics
    /// Panics if A or B are not 2D or shapes incompatible
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 2 && inputs.len() <= 3,
            "Gemm requires 2-3 inputs [A, B, C?], got {}",
            inputs.len()
        );

        let a = inputs[0].to_array();
        let b = inputs[1].to_array();

        assert_eq!(a.ndim(), 2, "A must be 2D, got {}D", a.ndim());
        assert_eq!(b.ndim(), 2, "B must be 2D, got {}D", b.ndim());

        // Get attributes (ONNX defaults)
        let alpha = attributes.get_float("alpha").unwrap_or(1.0);
        let beta = attributes.get_float("beta").unwrap_or(1.0);
        let trans_a = attributes.get_int("transA").unwrap_or(0) != 0;
        let trans_b = attributes.get_int("transB").unwrap_or(0) != 0;

        #[cfg(debug_assertions)]
        {}

        // Convert to 2D
        let a_2d = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        // Apply transposes if needed
        let a_final = if trans_a {
            a_2d.t().to_owned()
        } else {
            a_2d.to_owned()
        };

        let b_final = if trans_b {
            b_2d.t().to_owned()
        } else {
            b_2d.to_owned()
        };

        // Compute A' @ B'
        let ab = a_final.dot(&b_final);

        // Scale by alpha
        let mut result = ab.mapv(|x| x * alpha);

        // Add beta * C if present
        if inputs.len() == 3 {
            let c = inputs[2].to_array();

            // C can be 1D or 2D (ONNX allows 1D bias that broadcasts)
            if c.ndim() == 1 {
                // 1D bias - broadcast to each row
                let scaled_c = c.mapv(|x| x * beta);
                // Convert result to dynamic, add, convert back
                let result_dyn = result.into_dyn();
                let sum = result_dyn + &scaled_c;
                result = sum.into_dimensionality::<ndarray::Ix2>().unwrap();
            } else if c.ndim() == 2 {
                // 2D bias - element-wise addition
                let c_2d = c.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                result += &c_2d.mapv(|x| x * beta);
            } else {
                panic!("C must be 1D or 2D, got {}D", c.ndim());
            }
        }

        Tensor::from_array(result.into_dyn())
    }
}
