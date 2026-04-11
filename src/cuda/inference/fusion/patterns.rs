//! Type definitions for fused operator patterns.
//!
//! These types are populated by the `detection` module and consumed by
//! the executor's `execute_fused_pattern` method. They deliberately
//! carry only string names (not tensor references) so the detection
//! pass can run without touching any GPU state.

/// Types of fused kernel patterns we can detect and optimize.
///
/// These types are `pub` so test-only re-exports through
/// `cuda::inference::testing` can return them. The containing
/// `fusion` module itself is `pub(super)`, so external code can only
/// reach these types via the intended re-export path.
#[derive(Clone, Debug)]
pub enum FusedPattern {
    /// Add -> Sqrt -> Div: y / sqrt(x + eps)
    /// Stores: (numerator_input, variance_input, eps_input, output_name)
    DivRsqrt {
        numerator_input: String,
        variance_input: String,
        eps_input: String,
        output_name: String,
    },
    /// Add -> Mul -> Add: (x + a) * b + c
    /// Stores: (x, a, b, c, output_name)
    AddMulAdd {
        x_input: String,
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Full GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Replaces 8 ops: Pow -> Mul -> Add -> Mul -> Tanh -> Add -> Mul -> Mul
    Gelu {
        x_input: String,
        output_name: String,
    },
    /// Mul -> Add: y = a * b + c
    /// Simple multiply-add pattern for scalar operations
    MulAdd {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Add -> Mul: y = (a + b) * c
    /// Simple add-multiply pattern for scalar operations
    AddMul {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Sub -> Mul: y = (a - b) * c
    /// Simple sub-multiply pattern
    SubMul {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Div -> Mul: y = (a / b) * c
    /// Simple div-multiply pattern
    DivMul {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
}

/// Information about a detected fused pattern
#[derive(Clone, Debug)]
pub struct FusedPatternInfo {
    /// The pattern type and its inputs
    pub pattern: FusedPattern,
    /// Node names that are part of this pattern (to skip during execution)
    pub nodes_to_skip: Vec<String>,
    /// The first node name (head) that triggers execution of the fused kernel
    pub head_node: String,
}
