//! Graph walker that identifies fused operator patterns before inference.
//!
//! Detection is a free function (not a method on the executor) so it can
//! be tested with a fabricated node list and a dummy weights map —
//! no CUDA context required for the walker logic itself, though a
//! context is needed for the GELU walker which reads a constant scalar
//! exponent via `to_host_f32`.

use std::collections::{HashMap, HashSet};

use super::patterns::{FusedPattern, FusedPatternInfo};
use crate::cuda::context::IconnxCudaContext;
use crate::cuda::tensor::GpuTensor;
use crate::ir::graph::GraphNode;
use crate::tensor::Tensor;

/// Detect all fuseable operator patterns in the given node list.
///
/// Returns a 3-tuple of:
/// - `HashMap<String, FusedPatternInfo>` — static patterns keyed by head
///   node name (unconditionally fusible; their interior nodes are added to
///   the skip set)
/// - `HashSet<String>` — node names to skip during execution because they
///   are interior to a static pattern
/// - `HashMap<String, FusedPatternInfo>` — dynamic candidates (AddMulAdd
///   chains where b/c are runtime-computed). These are attempted at
///   execution time with a shape-check fallback; their interior nodes are
///   NOT added to the skip set so they remain executable on the fallback
///   path.
///
/// The `ctx` parameter is needed only to read constant scalar values
/// from weight tensors during exponent validation (GELU's `Pow` exponent
/// must be exactly 3.0). It is not used for any GPU computation during
/// detection.
pub(crate) fn detect_fused_patterns(
    nodes: &[GraphNode],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (
    HashMap<String, FusedPatternInfo>,
    HashSet<String>,
    HashMap<String, FusedPatternInfo>,
) {
    detect_with_scalar_lookup(
        nodes,
        |name| {
            weights
                .get(name)
                .filter(|t| t.len() == 1)
                .and_then(|t| t.to_host_f32(ctx).ok())
        },
        |name| weights.contains_key(name),
        |name| weights.get(name).map(|t| t.shape().to_vec()),
    )
}

/// CPU-side variant used by `ir::passes::fusion`. Identical pattern-matching
/// logic to `detect_fused_patterns` but sources scalar values from CPU
/// `Tensor` initializers instead of round-tripping them through the GPU via
/// `to_host_f32(ctx)`. Eliminates the D2H copy during fusion detection.
pub(crate) fn detect_fused_patterns_from_cpu(
    nodes: &[GraphNode],
    initializers: &HashMap<String, Tensor>,
) -> (
    HashMap<String, FusedPatternInfo>,
    HashSet<String>,
    HashMap<String, FusedPatternInfo>,
) {
    detect_with_scalar_lookup(
        nodes,
        |name| {
            initializers.get(name).and_then(|t| match t {
                Tensor::Float32(_) if t.len() == 1 => Some(t.as_slice()),
                _ => None,
            })
        },
        |name| initializers.contains_key(name),
        |name| initializers.get(name).map(|t| t.shape().to_vec()),
    )
}

/// Shared core of the fusion detector, parameterised by scalar-lookup and
/// presence-check closures so the same pattern-matching logic can serve
/// both the GPU entry point (`detect_fused_patterns`, reads scalars via
/// `to_host_f32`) and the CPU entry point (`detect_fused_patterns_from_cpu`,
/// reads scalars directly from CPU `Tensor` initializers).
///
/// `read_f32(name)` returns `Some(vec)` when `name` refers to a length-1
/// float tensor whose value has been brought to host memory (for GPU: via
/// `to_host_f32`; for CPU: via `Tensor::as_slice`), or `None` otherwise
/// (name not registered, wrong dtype, not scalar, or GPU D2H failure).
///
/// `is_initializer(name)` returns `true` iff `name` is registered as an
/// initializer (regardless of dtype). The detector uses this for static-
/// value predicates that don't need to inspect the value itself.
///
/// `static_shape(name)` returns the shape of an initializer/static tensor
/// when known. The MulAdd walker uses this to verify that the static-side
/// `b` shares the bias `c`'s shape — the only contract `gpu_fused_mul_add`
/// can honor. WS-4 M4.7: without this check the matcher claimed any
/// Mul→Add chain whose `c` was an initializer, and DistilBERT-INT8's
/// per-tensor scale × per-channel bias chain panicked at dispatch.
fn detect_with_scalar_lookup(
    nodes: &[GraphNode],
    read_f32: impl Fn(&str) -> Option<Vec<f32>>,
    is_initializer: impl Fn(&str) -> bool,
    static_shape: impl Fn(&str) -> Option<Vec<usize>>,
) -> (
    HashMap<String, FusedPatternInfo>,
    HashSet<String>,
    HashMap<String, FusedPatternInfo>,
) {
    // Build lookup maps
    let output_to_node: HashMap<&str, &GraphNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();

    // Count uses of each tensor output (for single-use check)
    let mut output_uses: HashMap<&str, usize> = HashMap::new();
    for node in nodes {
        for input in &node.inputs {
            *output_uses.entry(input.as_str()).or_insert(0) += 1;
        }
    }

    let mut fused_patterns: HashMap<String, FusedPatternInfo> = HashMap::new();
    let mut nodes_to_skip: HashSet<String> = HashSet::new();
    let mut dynamic_candidates: HashMap<String, FusedPatternInfo> = HashMap::new();
    // Head nodes of dynamic candidates must not be claimed by later (shorter)
    // walkers as static patterns. Keeping a dedicated set avoids adding heads
    // to nodes_to_skip (which would prevent them from executing on the
    // fallback path).
    let mut dynamic_heads: HashSet<String> = HashSet::new();
    // Interior nodes of dynamic candidates must also not be claimed by later
    // walkers as heads or interiors of static patterns. Like dynamic_heads,
    // these are NOT added to nodes_to_skip so the fallback path can execute
    // them independently.
    let mut dynamic_interiors: HashSet<String> = HashSet::new();

    // Predicate: is this tensor name a "static" value (known at graph-build
    // time)? Matches either (a) an initializer registered via
    // add_initializer (present in `weights`), or (b) the output of a
    // `Constant` node added earlier in topological order (looked up via
    // `output_to_node[name].op_type`).
    //
    // Used by the AddMulAdd walker's semantic-intent check: fuse only
    // affine transforms where b (scale) and c (bias) are static values.
    // Extends the original "initializer-only" check — Cycle 3 confirmed
    // Kokoro's affine scales/biases come from Constant Float32 nodes, not
    // initializers, so the walker was missing all 51 real instances.
    //
    // Not recognized as static: Cast-of-Constant, Unsqueeze-of-Constant,
    // or any other chain. A broader shape/value inference pass would be
    // needed for those.
    let is_static = |name: &str| -> bool {
        is_initializer(name)
            || output_to_node
                .get(name)
                .map(|n| n.op_type == "Constant")
                .unwrap_or(false)
    };

    // Detect Add -> Sqrt -> Div pattern (normalization)
    // Pattern: numerator / sqrt(variance + eps)
    // Now supports broadcasting when variance has trailing 1 dimensions
    for node in nodes {
        if node.op_type == "Add" {
            let add_output = &node.outputs[0];
            // Check if Add output is single-use
            if output_uses.get(add_output.as_str()) != Some(&1) {
                continue;
            }

            // Find Sqrt that uses this Add's output
            if let Some(sqrt_node) = output_to_node
                .iter()
                .find(|(_, n)| n.op_type == "Sqrt" && n.inputs.contains(add_output))
                .map(|(_, n)| *n)
            {
                let sqrt_output = &sqrt_node.outputs[0];
                // Check if Sqrt output is single-use
                if output_uses.get(sqrt_output.as_str()) != Some(&1) {
                    continue;
                }

                // Find Div where Sqrt output is the divisor (second input)
                if let Some(div_node) = output_to_node
                    .iter()
                    .find(|(_, n)| {
                        n.op_type == "Div"
                            && n.inputs.len() >= 2
                            && &n.inputs[1] == sqrt_output
                    })
                    .map(|(_, n)| *n)
                {
                    // Found pattern: numerator / sqrt(variance + eps)
                    let pattern_info = FusedPatternInfo {
                        pattern: FusedPattern::DivRsqrt {
                            numerator_input: div_node.inputs[0].clone(),
                            variance_input: node.inputs[0].clone(),
                            eps_input: node.inputs[1].clone(),
                            output_name: div_node.outputs[0].clone(),
                        },
                        nodes_to_skip: vec![sqrt_node.name.clone(), div_node.name.clone()],
                        head_node: node.name.clone(),
                    };

                    // Register the pattern
                    for skip_name in &pattern_info.nodes_to_skip {
                        nodes_to_skip.insert(skip_name.clone());
                    }
                    fused_patterns.insert(node.name.clone(), pattern_info);
                }
            }
        }
    }

    // Detect Add -> Mul -> Add pattern (affine transform: (x + a) * b + c)
    for node in nodes {
        if node.op_type == "Add" {
            let add1_output = &node.outputs[0];
            // Check if first Add output is single-use
            if output_uses.get(add1_output.as_str()) != Some(&1) {
                continue;
            }
            // Skip if this node is already part of another pattern
            if nodes_to_skip.contains(&node.name) {
                continue;
            }

            // Find Mul that uses this Add's output
            if let Some(mul_node) = output_to_node
                .iter()
                .find(|(_, n)| n.op_type == "Mul" && n.inputs.contains(add1_output))
                .map(|(_, n)| *n)
            {
                let mul_output = &mul_node.outputs[0];
                // Check if Mul output is single-use
                if output_uses.get(mul_output.as_str()) != Some(&1) {
                    continue;
                }
                // Skip if Mul is already part of another pattern
                if nodes_to_skip.contains(&mul_node.name) {
                    continue;
                }

                // Find second Add that uses Mul output
                if let Some(add2_node) = output_to_node
                    .iter()
                    .find(|(_, n)| n.op_type == "Add" && n.inputs.contains(mul_output))
                    .map(|(_, n)| *n)
                {
                    // Skip if second Add is already part of another pattern
                    if nodes_to_skip.contains(&add2_node.name) {
                        continue;
                    }

                    // Identify which input to Add1 is the main operand (x) and which is the addend (a)
                    let (x_input, a_input) = (node.inputs[0].clone(), node.inputs[1].clone());

                    // Identify the multiplier (b) - the input to Mul that isn't the Add1 output
                    let b_input = if &mul_node.inputs[0] == add1_output {
                        mul_node.inputs[1].clone()
                    } else {
                        mul_node.inputs[0].clone()
                    };

                    // Identify the final addend (c) - the input to Add2 that isn't the Mul output
                    let c_input = if &add2_node.inputs[0] == mul_output {
                        add2_node.inputs[1].clone()
                    } else {
                        add2_node.inputs[0].clone()
                    };

                    // Only fuse statically if b and c are static (initializers
                    // or Constant node outputs). This preserves the affine-
                    // transform semantic — gamma/beta are baked into the
                    // model — while recognizing that modern ONNX exports
                    // emit these as Constant Float32 nodes rather than
                    // initializers. See the `is_static` closure above.
                    if !is_static(&b_input) || !is_static(&c_input) {
                        // Non-static b or c: register as dynamic candidate.
                        // The executor will attempt fusion at runtime and
                        // fall back on shape mismatch.
                        let pattern_info = FusedPatternInfo {
                            pattern: FusedPattern::AddMulAdd {
                                x_input,
                                a_input,
                                b_input,
                                c_input,
                                output_name: add2_node.outputs[0].clone(),
                            },
                            nodes_to_skip: vec![
                                mul_node.name.clone(),
                                add2_node.name.clone(),
                            ],
                            head_node: node.name.clone(),
                        };
                        // Do NOT add interior nodes to nodes_to_skip —
                        // they must remain executable for the fallback path.
                        // Track the head and interiors so later (shorter)
                        // walkers cannot re-claim them as static patterns.
                        dynamic_heads.insert(node.name.clone());
                        for interior in &pattern_info.nodes_to_skip {
                            dynamic_interiors.insert(interior.clone());
                        }
                        dynamic_candidates.insert(node.name.clone(), pattern_info);
                        continue;
                    }

                    let pattern_info = FusedPatternInfo {
                        pattern: FusedPattern::AddMulAdd {
                            x_input,
                            a_input,
                            b_input,
                            c_input,
                            output_name: add2_node.outputs[0].clone(),
                        },
                        nodes_to_skip: vec![mul_node.name.clone(), add2_node.name.clone()],
                        head_node: node.name.clone(),
                    };

                    // Register the pattern
                    for skip_name in &pattern_info.nodes_to_skip {
                        nodes_to_skip.insert(skip_name.clone());
                    }
                    fused_patterns.insert(node.name.clone(), pattern_info);
                }
            }
        }
    }

    // Detect full GELU pattern:
    // x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Pattern: Pow(x,3) -> Mul(coeff) -> Add(x) -> Mul(sqrt2pi) -> Tanh -> Add(1) -> Mul(x*0.5)
    // Plus parallel path: Mul(x, 0.5)
    for node in nodes {
        if node.op_type == "Pow" {
            let pow_output = &node.outputs[0];
            // Skip if already part of another pattern
            if nodes_to_skip.contains(&node.name) {
                continue;
            }

            // Check if exponent is 3 (for GELU pattern)
            let exp_input = &node.inputs[1];
            let exp_is_three = read_f32(exp_input)
                .map(|data| (data[0] - 3.0).abs() < 1e-6)
                .unwrap_or(false);

            if !exp_is_three {
                continue;
            }

            // The x input is the base of Pow (first input)
            let x_input = node.inputs[0].clone();

            // Step 1: Find Mul(pow, 0.044715 coeff)
            let mul1 = output_to_node
                .iter()
                .find(|(_, n)| n.op_type == "Mul" && n.inputs.contains(pow_output))
                .map(|(_, n)| *n);
            let Some(mul1_node) = mul1 else { continue };
            let mul1_output = &mul1_node.outputs[0];

            // Step 2: Find Add(x, mul1) - the inner sum x + x^3 * coeff
            let add1 = output_to_node
                .iter()
                .find(|(_, n)| {
                    n.op_type == "Add"
                        && n.inputs.contains(mul1_output)
                        && n.inputs.contains(&x_input)
                })
                .map(|(_, n)| *n);
            let Some(add1_node) = add1 else { continue };
            let add1_output = &add1_node.outputs[0];

            // Step 3: Find Mul(add1, sqrt(2/pi)) - scaling for tanh
            let mul2 = output_to_node
                .iter()
                .find(|(_, n)| n.op_type == "Mul" && n.inputs.contains(add1_output))
                .map(|(_, n)| *n);
            let Some(mul2_node) = mul2 else { continue };
            let mul2_output = &mul2_node.outputs[0];

            // Step 4: Find Tanh
            let tanh = output_to_node
                .iter()
                .find(|(_, n)| n.op_type == "Tanh" && n.inputs.contains(mul2_output))
                .map(|(_, n)| *n);
            let Some(tanh_node) = tanh else { continue };
            let tanh_output = &tanh_node.outputs[0];

            // Step 5: Find Add(tanh, 1) - the 1 + tanh(...) part
            let add2 = output_to_node
                .iter()
                .find(|(_, n)| n.op_type == "Add" && n.inputs.contains(tanh_output))
                .map(|(_, n)| *n);
            let Some(add2_node) = add2 else { continue };
            let add2_output = &add2_node.outputs[0];

            // Step 6: Find Mul(x*0.5, add2) - the final multiply
            // This Mul should take add2_output and the output of a Mul(x, 0.5)
            let mul3 = output_to_node
                .iter()
                .find(|(_, n)| n.op_type == "Mul" && n.inputs.contains(add2_output))
                .map(|(_, n)| *n);
            let Some(mul3_node) = mul3 else { continue };

            // Find the other input to mul3 (should be x * 0.5)
            let mul3_other_input = if &mul3_node.inputs[0] == add2_output {
                &mul3_node.inputs[1]
            } else {
                &mul3_node.inputs[0]
            };

            // This other input should come from a Mul(x, 0.5)
            let mul_half_node = match output_to_node.get(mul3_other_input.as_str()) {
                Some(n) if n.op_type == "Mul" && n.inputs.contains(&x_input) => *n,
                _ => continue,
            };

            // Verify all intermediate outputs are single-use (safe to fuse)
            let all_single_use = [
                pow_output,
                mul1_output,
                add1_output,
                mul2_output,
                tanh_output,
                add2_output,
                &mul_half_node.outputs[0],
            ]
            .iter()
            .all(|out| output_uses.get(out.as_str()) == Some(&1));

            if !all_single_use {
                continue;
            }

            // Register the full GELU pattern
            let nodes_in_pattern = vec![
                mul1_node.name.clone(),
                add1_node.name.clone(),
                mul2_node.name.clone(),
                tanh_node.name.clone(),
                add2_node.name.clone(),
                mul3_node.name.clone(),
                mul_half_node.name.clone(),
            ];

            // Skip if any node is already part of another pattern
            if nodes_in_pattern.iter().any(|n| nodes_to_skip.contains(n)) {
                continue;
            }

            let pattern_info = FusedPatternInfo {
                pattern: FusedPattern::Gelu {
                    x_input,
                    output_name: mul3_node.outputs[0].clone(),
                },
                nodes_to_skip: nodes_in_pattern.clone(),
                head_node: node.name.clone(),
            };

            // Register the pattern
            for skip_name in &nodes_in_pattern {
                nodes_to_skip.insert(skip_name.clone());
            }
            fused_patterns.insert(node.name.clone(), pattern_info);
        }
    }

    // Detect Mul -> Sin -> Pow -> Mul -> Add pattern
    // (vocoder periodic activation: y = sin(x * w0)^p * w1 + b).
    //
    // NOTE: This walker runs BEFORE the shorter MulAdd/AddMul/SubMul/DivMul
    // walkers because a longer, more specific pattern must take priority
    // over a shorter one that would otherwise claim its suffix. In
    // particular, when w0/w1/b are all weights (the realistic Kokoro
    // case), the MulAdd walker would otherwise grab `tail_mul + bias_add`
    // as `a*b+c` before MulSinPowMulAdd had a chance to claim them.
    //
    // Requirements:
    //  - All intermediate outputs (t0, t1, t2, t3) must be single-use.
    //  - No node in the chain may already belong to another pattern.
    //  - The Pow exponent must be a scalar integer weight constant.
    for node in nodes {
        if node.op_type != "Mul" {
            continue;
        }
        if nodes_to_skip.contains(&node.name) || dynamic_heads.contains(&node.name) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }

        let head_mul_output = &node.outputs[0];
        if output_uses.get(head_mul_output.as_str()) != Some(&1) {
            continue;
        }

        // Step 1: Sin consuming head Mul's output.
        let sin_node = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Sin"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.contains(head_mul_output)
            })
            .map(|(_, n)| *n);
        let Some(sin_node) = sin_node else { continue };
        let sin_output = &sin_node.outputs[0];
        if output_uses.get(sin_output.as_str()) != Some(&1) {
            continue;
        }

        // Step 2: Pow consuming Sin's output; exponent must be a scalar
        // integer constant in the weights map.
        let pow_node = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Pow"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.len() >= 2
                    && n.inputs[0] == *sin_output
            })
            .map(|(_, n)| *n);
        let Some(pow_node) = pow_node else { continue };
        let pow_output = &pow_node.outputs[0];
        if output_uses.get(pow_output.as_str()) != Some(&1) {
            continue;
        }
        let p_input = &pow_node.inputs[1];
        let p_is_integer_scalar_weight = read_f32(p_input)
            .map(|data| data[0].fract() == 0.0)
            .unwrap_or(false);
        if !p_is_integer_scalar_weight {
            continue;
        }

        // Step 3: Tail Mul consuming Pow's output. Its other operand is w1.
        let tail_mul = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Mul"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.contains(pow_output)
            })
            .map(|(_, n)| *n);
        let Some(tail_mul_node) = tail_mul else { continue };
        let tail_mul_output = &tail_mul_node.outputs[0];
        if output_uses.get(tail_mul_output.as_str()) != Some(&1) {
            continue;
        }
        let w1_input = if tail_mul_node.inputs[0] == *pow_output {
            tail_mul_node.inputs[1].clone()
        } else {
            tail_mul_node.inputs[0].clone()
        };

        // Step 4: Add consuming tail Mul's output. Other operand is b.
        let bias_add = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Add"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.contains(tail_mul_output)
            })
            .map(|(_, n)| *n);
        let Some(bias_add_node) = bias_add else { continue };
        let b_input = if bias_add_node.inputs[0] == *tail_mul_output {
            bias_add_node.inputs[1].clone()
        } else {
            bias_add_node.inputs[0].clone()
        };

        // Identify x and w0 (the two inputs to the head Mul).
        let (x_input, w0_input) = (node.inputs[0].clone(), node.inputs[1].clone());

        // Commit: register the pattern and mark interior nodes as skip.
        let pattern_info = FusedPatternInfo {
            pattern: FusedPattern::MulSinPowMulAdd {
                x_input,
                w0_input,
                w1_input,
                b_input,
                p_input: p_input.clone(),
                output_name: bias_add_node.outputs[0].clone(),
            },
            nodes_to_skip: vec![
                sin_node.name.clone(),
                pow_node.name.clone(),
                tail_mul_node.name.clone(),
                bias_add_node.name.clone(),
            ],
            head_node: node.name.clone(),
        };

        for skip_name in &pattern_info.nodes_to_skip {
            nodes_to_skip.insert(skip_name.clone());
        }
        fused_patterns.insert(node.name.clone(), pattern_info);
    }

    // Detect Mul -> Add pattern (simple multiply-add: a * b + c).
    //
    // The fused kernel `gpu_fused_mul_add` supports two layouts and BOTH
    // require `b.shape == c.shape`:
    //   1. All-same-shape: `a.shape == b.shape == c.shape`.
    //   2. Per-channel broadcast: `b.shape == c.shape && b.numel == a.last`.
    //
    // The walker therefore (a) requires `c` to be a static initializer
    // with a known shape, and (b) demands one of the two `Mul` inputs be
    // a static initializer of the same shape as `c` — that one is `b`
    // (the bias-paired scale); the other is the dynamic activation `a`.
    // When neither qualifies, fusion is unsafe (Path 1 needs runtime
    // shape agreement we can't verify; Path 2 needs static `b`); the
    // unfused Mul + Add then fall back to separate elementwise ops,
    // which broadcast correctly via the standard binary kernels.
    for node in nodes {
        if node.op_type == "Mul"
            && !nodes_to_skip.contains(&node.name)
            && !dynamic_heads.contains(&node.name)
        {
            let mul_output = &node.outputs[0];
            // Check if Mul output is single-use
            if output_uses.get(mul_output.as_str()) != Some(&1) {
                continue;
            }

            // Find Add that uses this Mul's output
            if let Some(add_node) = nodes.iter().find(|n| {
                n.op_type == "Add"
                    && n.inputs.contains(mul_output)
                    && !nodes_to_skip.contains(&n.name)
                    && !dynamic_interiors.contains(&n.name)
            }) {
                // Get the other Add input (c)
                let c_input = add_node
                    .inputs
                    .iter()
                    .find(|i| i.as_str() != mul_output.as_str())
                    .cloned()
                    .unwrap_or_default();

                let Some(c_shape) = static_shape(&c_input) else {
                    continue;
                };

                // Identify which Mul input is the static side matching
                // c's shape; that one is `b`. Refuse if neither
                // qualifies — wrapper has no path for that case.
                let in0 = &node.inputs[0];
                let in1 = &node.inputs[1];
                let in0_matches = static_shape(in0).is_some_and(|s| s == c_shape);
                let in1_matches = static_shape(in1).is_some_and(|s| s == c_shape);
                let (a_input, b_input) = if in1_matches {
                    (in0.clone(), in1.clone())
                } else if in0_matches {
                    (in1.clone(), in0.clone())
                } else {
                    continue;
                };

                let pattern_info = FusedPatternInfo {
                    pattern: FusedPattern::MulAdd {
                        a_input,
                        b_input,
                        c_input,
                        output_name: add_node.outputs[0].clone(),
                    },
                    nodes_to_skip: vec![add_node.name.clone()],
                    head_node: node.name.clone(),
                };

                // Register the pattern
                nodes_to_skip.insert(add_node.name.clone());
                fused_patterns.insert(node.name.clone(), pattern_info);
            }
        }
    }

    // Detect Add -> Mul pattern (simple add-multiply: (a + b) * c)
    // Complement pattern to Mul-Add
    // Only fuse when c is a weight (constant), ensuring it's available at execution time
    for node in nodes {
        if node.op_type == "Add"
            && !nodes_to_skip.contains(&node.name)
            && !dynamic_heads.contains(&node.name)
        {
            let add_output = &node.outputs[0];
            // Check if Add output is single-use
            if output_uses.get(add_output.as_str()) != Some(&1) {
                continue;
            }

            // Find Mul that uses this Add's output
            if let Some(mul_node) = nodes.iter().find(|n| {
                n.op_type == "Mul"
                    && n.inputs.contains(add_output)
                    && !nodes_to_skip.contains(&n.name)
                    && !dynamic_interiors.contains(&n.name)
            }) {
                // Get the other Mul input (c)
                let c_input = mul_node
                    .inputs
                    .iter()
                    .find(|i| i.as_str() != add_output.as_str())
                    .cloned()
                    .unwrap_or_default();

                // Only fuse if c is a weight (constant) to ensure shape compatibility
                // Fused kernels don't support broadcasting
                if !is_initializer(&c_input) {
                    continue;
                }

                let pattern_info = FusedPatternInfo {
                    pattern: FusedPattern::AddMul {
                        a_input: node.inputs[0].clone(),
                        b_input: node.inputs[1].clone(),
                        c_input,
                        output_name: mul_node.outputs[0].clone(),
                    },
                    nodes_to_skip: vec![mul_node.name.clone()],
                    head_node: node.name.clone(),
                };

                // Register the pattern
                nodes_to_skip.insert(mul_node.name.clone());
                fused_patterns.insert(node.name.clone(), pattern_info);
            }
        }
    }

    // Detect Sub -> Mul pattern (simple sub-multiply: (a - b) * c)
    // Similar to Add -> Mul but with subtraction
    for node in nodes {
        if node.op_type == "Sub"
            && !nodes_to_skip.contains(&node.name)
            && !dynamic_heads.contains(&node.name)
        {
            let sub_output = &node.outputs[0];
            // Check if Sub output is single-use
            if output_uses.get(sub_output.as_str()) != Some(&1) {
                continue;
            }

            // Find Mul that uses this Sub's output
            if let Some(mul_node) = nodes.iter().find(|n| {
                n.op_type == "Mul"
                    && n.inputs.contains(sub_output)
                    && !nodes_to_skip.contains(&n.name)
                    && !dynamic_interiors.contains(&n.name)
            }) {
                // Get the other Mul input (c)
                let c_input = mul_node
                    .inputs
                    .iter()
                    .find(|i| i.as_str() != sub_output.as_str())
                    .cloned()
                    .unwrap_or_default();

                // Only fuse if c is a weight (constant) to ensure shape compatibility
                // Fused kernels don't support broadcasting
                if !is_initializer(&c_input) {
                    continue;
                }

                let pattern_info = FusedPatternInfo {
                    pattern: FusedPattern::SubMul {
                        a_input: node.inputs[0].clone(),
                        b_input: node.inputs[1].clone(),
                        c_input,
                        output_name: mul_node.outputs[0].clone(),
                    },
                    nodes_to_skip: vec![mul_node.name.clone()],
                    head_node: node.name.clone(),
                };

                // Register the pattern
                nodes_to_skip.insert(mul_node.name.clone());
                fused_patterns.insert(node.name.clone(), pattern_info);
            }
        }
    }

    // Detect Div -> Mul pattern (simple div-multiply: (a / b) * c)
    // Similar to Add -> Mul but with division
    for node in nodes {
        if node.op_type == "Div"
            && !nodes_to_skip.contains(&node.name)
            && !dynamic_heads.contains(&node.name)
        {
            let div_output = &node.outputs[0];
            // Check if Div output is single-use
            if output_uses.get(div_output.as_str()) != Some(&1) {
                continue;
            }

            // Find Mul that uses this Div's output
            if let Some(mul_node) = nodes.iter().find(|n| {
                n.op_type == "Mul"
                    && n.inputs.contains(div_output)
                    && !nodes_to_skip.contains(&n.name)
                    && !dynamic_interiors.contains(&n.name)
            }) {
                // Get the other Mul input (c)
                let c_input = mul_node
                    .inputs
                    .iter()
                    .find(|i| i.as_str() != div_output.as_str())
                    .cloned()
                    .unwrap_or_default();

                // Only fuse if c is a weight (constant) to ensure shape compatibility
                // Fused kernels don't support broadcasting
                if !is_initializer(&c_input) {
                    continue;
                }

                let pattern_info = FusedPatternInfo {
                    pattern: FusedPattern::DivMul {
                        a_input: node.inputs[0].clone(),
                        b_input: node.inputs[1].clone(),
                        c_input,
                        output_name: mul_node.outputs[0].clone(),
                    },
                    nodes_to_skip: vec![mul_node.name.clone()],
                    head_node: node.name.clone(),
                };

                // Register the pattern
                nodes_to_skip.insert(mul_node.name.clone());
                fused_patterns.insert(node.name.clone(), pattern_info);
            }
        }
    }

    // NOTE: Transpose absorption into MatMul (trans_a/trans_b flags) is a potential
    // optimization but requires deeper changes to the run() execution loop.
    // Deferred for Phase 4.

    (fused_patterns, nodes_to_skip, dynamic_candidates)
}
