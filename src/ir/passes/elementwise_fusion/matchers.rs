//! Chain-matchers for the 8 specialized `FusedPattern` variants.
//!
//! Each matcher checks a chain of node indices (produced by the chain
//! grower in the parent module) against the exact op sequence + attribute
//! constraints of one variant. Matchers return `Some(MatchResult)` on a
//! successful match, `None` otherwise. The caller tries them in
//! longest-first order so a longer, more-specific pattern claims its
//! chain before a shorter one could grab a suffix.
//!
//! Predicate logic mirrors `cuda::inference::fusion::detection` but
//! operates chain-first rather than per-op-first: a single chain is
//! examined once per head candidate, so matchers are linear in the chain
//! length. Static-value predicates draw from `OptimizableGraph`'s CPU-
//! side initializer map and from `Constant`-node outputs (via the
//! caller-provided producers map).

use std::collections::HashMap;

use crate::cuda::inference::fusion::FusedPattern;
use crate::ir::graph::GraphNode;
use crate::tensor::Tensor;

/// Result of matching a chain against one of the 8 specialized variants.
/// `extra_skip_nodes` lists node names outside the chain that should also
/// be marked as skipped (e.g. GELU's parallel `Mul(x, 0.5)` branch).
pub(super) struct MatchResult {
    pub pattern: FusedPattern,
    pub extra_skip_nodes: Vec<String>,
}

impl MatchResult {
    fn new(pattern: FusedPattern) -> Self {
        Self {
            pattern,
            extra_skip_nodes: Vec::new(),
        }
    }
    fn with_extra(pattern: FusedPattern, extra: Vec<String>) -> Self {
        Self {
            pattern,
            extra_skip_nodes: extra,
        }
    }
}

/// Try to match `chain` against the 8 existing `FusedPattern` variants.
/// Returns `Some(MatchResult)` if an exact shape/op-sequence match is
/// found, `None` otherwise (caller falls back to `GeneralChain` when the
/// chain is pure-unary, or drops it entirely when it contains a binary
/// op).
pub(super) fn match_known_pattern(
    chain: &[usize],
    nodes: &[GraphNode],
    producers: &HashMap<&str, usize>,
    initializers: &HashMap<String, Tensor>,
) -> Option<MatchResult> {
    // Longest first so they claim their chain before shorter patterns
    // could snap up a suffix.
    if let Some(m) = match_gelu(chain, nodes, producers, initializers) {
        return Some(m);
    }
    if let Some(m) = match_mul_sin_pow_mul_add(chain, nodes, initializers) {
        return Some(m);
    }
    if let Some(m) = match_add_mul_add(chain, nodes, initializers, producers) {
        return Some(m);
    }
    if let Some(m) = match_div_rsqrt(chain, nodes) {
        return Some(m);
    }
    if let Some(m) = match_mul_add(chain, nodes, initializers) {
        return Some(m);
    }
    if let Some(m) = match_add_mul(chain, nodes, initializers) {
        return Some(m);
    }
    if let Some(m) = match_sub_mul(chain, nodes, initializers) {
        return Some(m);
    }
    if let Some(m) = match_div_mul(chain, nodes, initializers) {
        return Some(m);
    }
    None
}

/// Read a scalar Float32 initializer; returns None if not present,
/// wrong dtype, or not length-1.
fn read_scalar_f32(name: &str, initializers: &HashMap<String, Tensor>) -> Option<f32> {
    let tensor = initializers.get(name)?;
    match tensor {
        Tensor::Float32(_) if tensor.len() == 1 => tensor.as_slice().first().copied(),
        _ => None,
    }
}

/// Is this tensor name produced by a `Constant` node (as opposed to an
/// initializer)? Used for the "static value" check in AddMulAdd.
fn is_constant_node_output(
    name: &str,
    nodes: &[GraphNode],
    producers: &HashMap<&str, usize>,
) -> bool {
    producers
        .get(name)
        .map(|&idx| nodes[idx].op_type == "Constant")
        .unwrap_or(false)
}

/// Is this tensor name static (initializer OR Constant-node output)?
fn is_static(
    name: &str,
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
    producers: &HashMap<&str, usize>,
) -> bool {
    initializers.contains_key(name) || is_constant_node_output(name, nodes, producers)
}

/// DivRsqrt: Add -> Sqrt -> Div, where Sqrt's output is Div's divisor
/// (input[1]). Represents `numerator / sqrt(variance + eps)`.
fn match_div_rsqrt(chain: &[usize], nodes: &[GraphNode]) -> Option<MatchResult> {
    if chain.len() != 3 {
        return None;
    }
    let add = &nodes[chain[0]];
    let sqrt = &nodes[chain[1]];
    let div = &nodes[chain[2]];
    if add.op_type != "Add" || sqrt.op_type != "Sqrt" || div.op_type != "Div" {
        return None;
    }
    if add.inputs.len() < 2 || div.inputs.len() < 2 {
        return None;
    }
    // Div's divisor (input[1]) must be Sqrt's output (input[0] is the numerator).
    let sqrt_output = sqrt.outputs.first()?;
    if &div.inputs[1] != sqrt_output {
        return None;
    }
    Some(MatchResult::new(FusedPattern::DivRsqrt {
        numerator_input: div.inputs[0].clone(),
        variance_input: add.inputs[0].clone(),
        eps_input: add.inputs[1].clone(),
        output_name: div.outputs[0].clone(),
    }))
}

/// AddMulAdd: Add -> Mul -> Add — affine transform `(x + a) * b + c`.
/// Only matched when `b` and `c` are static (initializers or Constant
/// node outputs). The dynamic case is handled by the legacy
/// `detect_fusion` path (which wires up `dynamic_candidates`).
fn match_add_mul_add(
    chain: &[usize],
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
    producers: &HashMap<&str, usize>,
) -> Option<MatchResult> {
    if chain.len() != 3 {
        return None;
    }
    let add1 = &nodes[chain[0]];
    let mul = &nodes[chain[1]];
    let add2 = &nodes[chain[2]];
    if add1.op_type != "Add" || mul.op_type != "Mul" || add2.op_type != "Add" {
        return None;
    }
    if add1.inputs.len() < 2 || mul.inputs.len() < 2 || add2.inputs.len() < 2 {
        return None;
    }

    let add1_output = add1.outputs.first()?;
    let mul_output = mul.outputs.first()?;

    // Identify x / a / b / c as in detection.rs.
    let (x_input, a_input) = (add1.inputs[0].clone(), add1.inputs[1].clone());
    let b_input = if &mul.inputs[0] == add1_output {
        mul.inputs[1].clone()
    } else {
        mul.inputs[0].clone()
    };
    let c_input = if &add2.inputs[0] == mul_output {
        add2.inputs[1].clone()
    } else {
        add2.inputs[0].clone()
    };

    if !is_static(&b_input, nodes, initializers, producers) {
        return None;
    }
    if !is_static(&c_input, nodes, initializers, producers) {
        return None;
    }

    Some(MatchResult::new(FusedPattern::AddMulAdd {
        x_input,
        a_input,
        b_input,
        c_input,
        output_name: add2.outputs[0].clone(),
    }))
}

/// GELU: Pow -> Mul -> Add -> Mul -> Tanh -> Add -> Mul (7-op chain),
/// plus the parallel `Mul(x, 0.5)` branch that feeds the final Mul.
///
/// The chain grower produces 7 in-chain nodes; this matcher additionally
/// identifies the parallel `Mul(x, 0.5)` node so it can be marked for
/// skip alongside the chain interior.
fn match_gelu(
    chain: &[usize],
    nodes: &[GraphNode],
    producers: &HashMap<&str, usize>,
    initializers: &HashMap<String, Tensor>,
) -> Option<MatchResult> {
    if chain.len() != 7 {
        return None;
    }
    let pow_n = &nodes[chain[0]];
    let mul1 = &nodes[chain[1]];
    let add1 = &nodes[chain[2]];
    let mul2 = &nodes[chain[3]];
    let tanh_n = &nodes[chain[4]];
    let add2 = &nodes[chain[5]];
    let mul3 = &nodes[chain[6]];
    if pow_n.op_type != "Pow"
        || mul1.op_type != "Mul"
        || add1.op_type != "Add"
        || mul2.op_type != "Mul"
        || tanh_n.op_type != "Tanh"
        || add2.op_type != "Add"
        || mul3.op_type != "Mul"
    {
        return None;
    }
    if pow_n.inputs.len() < 2 {
        return None;
    }

    // Exponent must be exactly 3.0 (the canonical GELU form).
    let exp_input = &pow_n.inputs[1];
    let exp3 = read_scalar_f32(exp_input, initializers)
        .map(|v| (v - 3.0).abs() < 1e-6)
        .unwrap_or(false);
    if !exp3 {
        return None;
    }

    // x is pow's base (input[0]); add1 must consume x as one of its
    // inputs (the other being mul1's output).
    let x_input = pow_n.inputs[0].clone();
    if !add1.inputs.contains(&x_input) {
        return None;
    }

    // Identify mul3's OTHER input (not add2's output); it must come
    // from a Mul(x, _) node that the chain grower left unclaimed.
    let add2_output = add2.outputs.first()?;
    let other_input = if &mul3.inputs[0] == add2_output {
        &mul3.inputs[1]
    } else {
        &mul3.inputs[0]
    };
    let mul_half_idx = *producers.get(other_input.as_str())?;
    let mul_half = &nodes[mul_half_idx];
    if mul_half.op_type != "Mul" {
        return None;
    }
    if !mul_half.inputs.contains(&x_input) {
        return None;
    }

    Some(MatchResult::with_extra(
        FusedPattern::Gelu {
            x_input,
            output_name: mul3.outputs[0].clone(),
        },
        vec![mul_half.name.clone()],
    ))
}

/// MulSinPowMulAdd: Mul -> Sin -> Pow -> Mul -> Add — vocoder periodic
/// activation `sin(x * w0)^p * w1 + b`. Exponent `p` must be a scalar
/// integer initializer.
fn match_mul_sin_pow_mul_add(
    chain: &[usize],
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
) -> Option<MatchResult> {
    if chain.len() != 5 {
        return None;
    }
    let head_mul = &nodes[chain[0]];
    let sin_n = &nodes[chain[1]];
    let pow_n = &nodes[chain[2]];
    let tail_mul = &nodes[chain[3]];
    let bias_add = &nodes[chain[4]];
    if head_mul.op_type != "Mul"
        || sin_n.op_type != "Sin"
        || pow_n.op_type != "Pow"
        || tail_mul.op_type != "Mul"
        || bias_add.op_type != "Add"
    {
        return None;
    }
    if head_mul.inputs.len() < 2
        || pow_n.inputs.len() < 2
        || tail_mul.inputs.len() < 2
        || bias_add.inputs.len() < 2
    {
        return None;
    }

    // p must be a scalar integer (positive or negative, fractional
    // exponents aren't supported by the integer-squaring kernel).
    let p_input = &pow_n.inputs[1];
    let p_is_int = read_scalar_f32(p_input, initializers)
        .map(|v| v.fract() == 0.0)
        .unwrap_or(false);
    if !p_is_int {
        return None;
    }

    let (x_input, w0_input) = (head_mul.inputs[0].clone(), head_mul.inputs[1].clone());
    let pow_output = pow_n.outputs.first()?;
    let w1_input = if &tail_mul.inputs[0] == pow_output {
        tail_mul.inputs[1].clone()
    } else {
        tail_mul.inputs[0].clone()
    };
    let tail_mul_output = tail_mul.outputs.first()?;
    let b_input = if &bias_add.inputs[0] == tail_mul_output {
        bias_add.inputs[1].clone()
    } else {
        bias_add.inputs[0].clone()
    };

    Some(MatchResult::new(FusedPattern::MulSinPowMulAdd {
        x_input,
        w0_input,
        w1_input,
        b_input,
        p_input: p_input.clone(),
        output_name: bias_add.outputs[0].clone(),
    }))
}

/// MulAdd: Mul -> Add — `y = a * b + c`. Matched only when `c` is an
/// initializer (not just any static value). The fused kernel supports
/// two shape layouts: all-same-shape, or `b` and `c` broadcast along
/// `a`'s last dim (`[last_dim]`, `[1, last_dim]`, …). Shape compat is
/// validated at dispatch time in `gpu_fused_mul_add`; the matcher
/// accepts any shape and lets the wrapper reject incompatible cases.
fn match_mul_add(
    chain: &[usize],
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
) -> Option<MatchResult> {
    if chain.len() != 2 {
        return None;
    }
    let mul = &nodes[chain[0]];
    let add = &nodes[chain[1]];
    if mul.op_type != "Mul" || add.op_type != "Add" {
        return None;
    }
    if mul.inputs.len() < 2 || add.inputs.len() < 2 {
        return None;
    }
    let mul_output = mul.outputs.first()?;
    let c_input = if &add.inputs[0] == mul_output {
        add.inputs[1].clone()
    } else {
        add.inputs[0].clone()
    };
    if !initializers.contains_key(&c_input) {
        return None;
    }
    Some(MatchResult::new(FusedPattern::MulAdd {
        a_input: mul.inputs[0].clone(),
        b_input: mul.inputs[1].clone(),
        c_input,
        output_name: add.outputs[0].clone(),
    }))
}

/// AddMul: Add -> Mul — `y = (a + b) * c`. Matched only when `c` is an
/// initializer.
fn match_add_mul(
    chain: &[usize],
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
) -> Option<MatchResult> {
    if chain.len() != 2 {
        return None;
    }
    let add = &nodes[chain[0]];
    let mul = &nodes[chain[1]];
    if add.op_type != "Add" || mul.op_type != "Mul" {
        return None;
    }
    if add.inputs.len() < 2 || mul.inputs.len() < 2 {
        return None;
    }
    let add_output = add.outputs.first()?;
    let c_input = if &mul.inputs[0] == add_output {
        mul.inputs[1].clone()
    } else {
        mul.inputs[0].clone()
    };
    if !initializers.contains_key(&c_input) {
        return None;
    }
    Some(MatchResult::new(FusedPattern::AddMul {
        a_input: add.inputs[0].clone(),
        b_input: add.inputs[1].clone(),
        c_input,
        output_name: mul.outputs[0].clone(),
    }))
}

/// SubMul: Sub -> Mul — `y = (a - b) * c`. Matched only when `c` is an
/// initializer.
fn match_sub_mul(
    chain: &[usize],
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
) -> Option<MatchResult> {
    if chain.len() != 2 {
        return None;
    }
    let sub = &nodes[chain[0]];
    let mul = &nodes[chain[1]];
    if sub.op_type != "Sub" || mul.op_type != "Mul" {
        return None;
    }
    if sub.inputs.len() < 2 || mul.inputs.len() < 2 {
        return None;
    }
    let sub_output = sub.outputs.first()?;
    let c_input = if &mul.inputs[0] == sub_output {
        mul.inputs[1].clone()
    } else {
        mul.inputs[0].clone()
    };
    if !initializers.contains_key(&c_input) {
        return None;
    }
    Some(MatchResult::new(FusedPattern::SubMul {
        a_input: sub.inputs[0].clone(),
        b_input: sub.inputs[1].clone(),
        c_input,
        output_name: mul.outputs[0].clone(),
    }))
}

/// DivMul: Div -> Mul — `y = (a / b) * c`. Matched only when `c` is an
/// initializer.
fn match_div_mul(
    chain: &[usize],
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
) -> Option<MatchResult> {
    if chain.len() != 2 {
        return None;
    }
    let div = &nodes[chain[0]];
    let mul = &nodes[chain[1]];
    if div.op_type != "Div" || mul.op_type != "Mul" {
        return None;
    }
    if div.inputs.len() < 2 || mul.inputs.len() < 2 {
        return None;
    }
    let div_output = div.outputs.first()?;
    let c_input = if &mul.inputs[0] == div_output {
        mul.inputs[1].clone()
    } else {
        mul.inputs[0].clone()
    };
    if !initializers.contains_key(&c_input) {
        return None;
    }
    Some(MatchResult::new(FusedPattern::DivMul {
        a_input: div.inputs[0].clone(),
        b_input: div.inputs[1].clone(),
        c_input,
        output_name: mul.outputs[0].clone(),
    }))
}
