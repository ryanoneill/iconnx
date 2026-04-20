//! NVRTC kernel source generation for `FusedPattern::GeneralChain`.
//!
//! Produces a CUDA kernel that applies a sequence of elementwise unary
//! ops per element. The kernel takes one input tensor and one output
//! tensor (same shape); all attribute-driven ops (LeakyRelu alpha, Clip
//! min/max) bake their scalars into the generated source rather than
//! taking a runtime parameter, so each unique chain_signature compiles
//! once and is cached forever.
//!
//! Binary ops are deliberately out of scope: the `elementwise_fusion`
//! pass only ever calls this module for chains that pass its
//! pure-unary check, so a binary op reaching `emit_op` would be a
//! programmer error and the generated source will contain a `UNHANDLED`
//! comment that breaks NVRTC compilation — failing loudly at the
//! compile cache line rather than silently producing wrong results.

use crate::attributes::NodeAttributes;

/// A generated CUDA kernel — source text plus the `extern "C"` entry
/// name the dispatcher looks up on the compiled module.
pub struct GeneratedKernel {
    pub source: String,
    pub entry_name: String,
}

/// Generate a kernel for a single chain. The `chain_signature` keys the
/// kernel cache (identical signatures reuse the same compile); `ops`
/// supplies the op-type + attribute sequence the kernel body applies in
/// order.
pub fn generate(chain_signature: &str, ops: &[(String, NodeAttributes)]) -> GeneratedKernel {
    let mut body = String::new();
    body.push_str("    float v = x[i];\n");
    for (op_type, attrs) in ops {
        body.push_str(&emit_op(op_type, attrs));
    }
    body.push_str("    y[i] = v;\n");

    let entry_name = sanitize_entry_name(chain_signature);

    let source = format!(
        r#"extern "C" __global__ void {name}(float* y, const float* x, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{body}
}}"#,
        name = entry_name,
        body = body
    );

    GeneratedKernel { source, entry_name }
}

/// Sanitize a chain_signature into a valid C identifier. Must be a pure
/// function (two identical signatures always produce identical names) or
/// the compile cache breaks.
fn sanitize_entry_name(chain_signature: &str) -> String {
    let mut s = String::from("general_chain_");
    for c in chain_signature.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c);
        } else {
            // Any non-alphanumeric character (|, (, ), =, ., ,, -, etc.)
            // collapses to an underscore. Collisions between e.g.
            // `LeakyRelu(alpha=0.2)` and `LeakyRelu(alpha=0,2)` are not
            // possible in practice because the signature comes from a
            // controlled format string in the chain builder.
            s.push('_');
        }
    }
    s
}

/// Emit a single op's kernel code. Attribute-driven scalars are baked
/// into the source so the kernel takes no extra runtime parameters.
fn emit_op(op_type: &str, attrs: &NodeAttributes) -> String {
    match op_type {
        "Sqrt" => "    v = sqrtf(v);\n".into(),
        "Exp" => "    v = expf(v);\n".into(),
        "Erf" => "    v = erff(v);\n".into(),
        "Sin" => "    v = sinf(v);\n".into(),
        "Cos" => "    v = cosf(v);\n".into(),
        "Tanh" => "    v = tanhf(v);\n".into(),
        "Sigmoid" => "    v = 1.0f / (1.0f + expf(-v));\n".into(),
        "Atan" => "    v = atanf(v);\n".into(),
        "Floor" => "    v = floorf(v);\n".into(),
        "Round" => "    v = roundf(v);\n".into(),
        "LeakyRelu" => {
            let alpha = attrs.get_float("alpha").unwrap_or(0.01);
            format!(
                "    v = v >= 0.0f ? v : {alpha:?}f * v;\n",
                alpha = alpha
            )
        }
        "Clip" => {
            let min = attrs.get_float("min").unwrap_or(f32::NEG_INFINITY);
            let max = attrs.get_float("max").unwrap_or(f32::INFINITY);
            // Guard against NEG_INFINITY / INFINITY which Rust's Debug
            // formatter prints as "-inf" / "inf" — not legal CUDA. Emit
            // CUDA's own infinity macros (math_constants.h is auto-
            // included by garboard's compile_for_device).
            let min_str = if min == f32::NEG_INFINITY {
                "-CUDART_INF_F".to_string()
            } else {
                format!("{:?}f", min)
            };
            let max_str = if max == f32::INFINITY {
                "CUDART_INF_F".to_string()
            } else {
                format!("{:?}f", max)
            };
            format!(
                "    v = fminf(fmaxf(v, {min}), {max});\n",
                min = min_str,
                max = max_str
            )
        }
        // If a binary op or an unknown op reaches here, emit an
        // UNHANDLED comment — garboard's NVRTC compile will then fail on
        // the undefined-behavior emit, surfacing the bug immediately
        // instead of silently producing wrong results.
        _ => format!(
            "    // UNHANDLED {op_type}_NOT_EMITTED_BY_CODEGEN — chain should have rejected this\n#error \"UNHANDLED op {op_type} in GeneralChain codegen\"\n"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_sqrt_exp_chain() {
        let ops = vec![
            ("Sqrt".to_string(), NodeAttributes::new()),
            ("Exp".to_string(), NodeAttributes::new()),
        ];
        let k = generate("Sqrt|Exp", &ops);
        assert!(k.source.contains("sqrtf(v)"));
        assert!(k.source.contains("expf(v)"));
        assert!(k.entry_name.starts_with("general_chain_"));
    }

    #[test]
    fn leaky_relu_emits_alpha() {
        let mut attrs = NodeAttributes::new();
        attrs.add_float("alpha".into(), 0.2);
        let ops = vec![("LeakyRelu".to_string(), attrs)];
        let k = generate("LeakyRelu(alpha=0.2)", &ops);
        // Rust's Debug format for 0.2_f32 is "0.2"; we just need to see
        // the value appear somewhere in the source body.
        assert!(
            k.source.contains("0.2"),
            "expected alpha=0.2 in emitted source:\n{}",
            k.source
        );
        assert!(k.source.contains("0.0f"));
    }

    #[test]
    fn clip_emits_min_max() {
        let mut attrs = NodeAttributes::new();
        attrs.add_float("min".into(), -1.0);
        attrs.add_float("max".into(), 1.0);
        let ops = vec![("Clip".to_string(), attrs)];
        let k = generate("Clip(min=-1,max=1)", &ops);
        assert!(k.source.contains("fminf"));
        assert!(k.source.contains("fmaxf"));
    }

    #[test]
    fn clip_defaults_emit_cuda_infinity() {
        // A Clip with no min/max should emit CUDART_INF_F, not the Rust
        // debug "inf" which NVRTC would reject.
        let ops = vec![("Clip".to_string(), NodeAttributes::new())];
        let k = generate("Clip()", &ops);
        assert!(
            k.source.contains("CUDART_INF_F"),
            "expected CUDA infinity macro in:\n{}",
            k.source
        );
        assert!(!k.source.contains("inff"), "Rust's 'inf' must not leak");
    }

    #[test]
    fn sanitize_entry_name_is_stable() {
        // Two identical signatures must produce identical names —
        // otherwise the kernel cache can't coalesce compiles.
        let a = sanitize_entry_name("Sqrt|Exp");
        let b = sanitize_entry_name("Sqrt|Exp");
        assert_eq!(a, b);
        // And the name is a valid C identifier.
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    }

    #[test]
    fn full_source_has_extern_c_entry() {
        let ops = vec![("Sqrt".to_string(), NodeAttributes::new())];
        let k = generate("Sqrt", &ops);
        assert!(k.source.contains("extern \"C\" __global__"));
        assert!(k.source.contains(&k.entry_name));
    }

    #[test]
    fn emit_op_produces_erff_for_erf() {
        let attrs = NodeAttributes::new();
        let src = emit_op("Erf", &attrs);
        assert_eq!(src, "    v = erff(v);\n");
    }
}
