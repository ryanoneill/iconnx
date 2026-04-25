//! Graph execution engine for Iconnx
//!
//! Executes ONNX computation graphs using implemented operators

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for graph operations

use crate::operators;
use crate::operators::{
    add, and, atan, cast, clip, concat, constant, constant_of_shape, conv, conv_transpose, cos,
    cumsum, div, equal, erf, exp, expand, floor, gather, gemm, greater, greater_or_equal,
    layer_normalization, leaky_relu, less, lstm, matmul, mul, nonzero, pad, pow, range,
    reduce_mean, reduce_sum, relu, reshape, resize, round, scatter_nd, shape, sigmoid, sin, slice,
    softmax, sqrt, squeeze, stft, sub, tanh, transpose, unsqueeze, where_op,
};
use crate::tensor::Tensor;
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

/// A node in the execution graph
#[derive(Clone, Debug)]
struct ExecutionNode {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: crate::attributes::NodeAttributes,
}

/// Graph executor - runs computation graphs
pub struct GraphExecutor {
    /// All execution nodes
    nodes: Vec<ExecutionNode>,

    /// Initializers (constants/weights)
    initializers: HashMap<String, Tensor>,
}

impl GraphExecutor {
    /// Create new graph executor
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            initializers: HashMap::new(),
        }
    }

    /// Add an initializer (constant or weight tensor)
    pub fn add_initializer(&mut self, name: String, tensor: Tensor) {
        self.initializers.insert(name, tensor);
    }

    /// Add a computation node
    pub fn add_node(&mut self, name: &str, op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) {
        self.add_node_with_attributes(
            name,
            op_type,
            inputs,
            outputs,
            crate::attributes::NodeAttributes::new(),
        );
    }

    /// Add a computation node with attributes
    pub fn add_node_with_attributes(
        &mut self,
        name: &str,
        op_type: &str,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        attributes: crate::attributes::NodeAttributes,
    ) {
        self.nodes.push(ExecutionNode {
            name: name.to_string(),
            op_type: op_type.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes,
        });
    }

    /// Execute the graph
    ///
    /// # Arguments
    /// * `inputs` - Map of input names to tensors
    /// * `output_names` - Names of outputs to compute
    ///
    /// # Returns
    /// Map of output names to tensors
    pub fn run(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<HashMap<String, Tensor>> {
        self.run_with_overrides(inputs, output_names, HashMap::new())
    }

    /// Execute the graph with tensor overrides
    ///
    /// This allows injecting pre-computed tensors at specific node outputs,
    /// useful for debugging and hybrid execution (e.g., using ORT's STFT output).
    ///
    /// # Arguments
    /// * `inputs` - Map of input names to tensors
    /// * `output_names` - Names of outputs to compute
    /// * `overrides` - Map of output names to pre-computed tensors (skip computing these nodes)
    ///
    /// # Returns
    /// Map of output names to tensors
    pub fn run_with_overrides(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
        overrides: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Storage for intermediate tensors
        let mut values: HashMap<String, Tensor> = HashMap::new();

        // Add inputs
        for (name, tensor) in inputs {
            values.insert(name, tensor);
        }

        // Add initializers
        for (name, tensor) in &self.initializers {
            values.insert(name.clone(), tensor.clone());
        }

        // Add overrides (these take precedence and will cause node execution to be skipped)
        for (name, tensor) in overrides {
            values.insert(name, tensor);
        }

        // Topologically sort nodes
        let sorted_nodes = self.topological_sort()?;

        // Execute nodes in order
        for node in sorted_nodes {
            // Skip nodes whose outputs are already provided (via overrides)
            if node.outputs.iter().any(|out| values.contains_key(out)) {
                continue;
            }

            // Special handling for Constant nodes - extract value from attributes
            if node.op_type == "Constant" {
                if let Some(value_tensor) = node.attributes.resolve_constant_value() {
                    // Constant node produces a tensor from whichever value-
                    // carrying attribute is present (see
                    // NodeAttributes::resolve_constant_value for the full
                    // ONNX spec's accepted set: value, value_int(s),
                    // value_float(s)).
                    assert_eq!(node.outputs.len(), 1, "Constant should have 1 output");
                    values.insert(node.outputs[0].clone(), value_tensor);
                    continue; // Skip normal execution
                } else {
                    anyhow::bail!(
                        "Constant node '{}' missing all value-carrying attributes \
                         (value / value_int / value_ints / value_float / value_floats)",
                        node.name
                    );
                }
            }

            // Get input tensors
            // Empty names indicate optional inputs - try to get them, but don't fail if missing
            let input_tensors: Vec<Tensor> = node
                .inputs
                .iter()
                .filter_map(|name| {
                    if name.is_empty() {
                        // Optional input not provided - skip it
                        None
                    } else {
                        // Required input - must exist
                        Some(values.get(name).cloned().with_context(|| {
                            format!("Missing input '{}' for node '{}'", name, node.name)
                        }))
                    }
                })
                .collect::<Result<Vec<_>>>()?;

            // Split is the only true multi-output op iconnx supports today.
            // Other multi-output ops (LSTM) are aliased — single tensor
            // under multiple names — and are handled via the single-output
            // path below with the LSTM-style Arc-share fallback.
            if node.op_type == "Split" {
                let split_outputs = operators::split::Split::forward(
                    &input_tensors,
                    &node.attributes,
                    node.outputs.len(),
                );
                for (name, tensor) in node.outputs.iter().zip(split_outputs.into_iter()) {
                    if !name.is_empty() {
                        values.insert(name.clone(), tensor);
                    }
                }
                continue;
            }

            // Execute operator with attributes
            let output = self
                .execute_operator(&node.op_type, &input_tensors, &node.attributes)
                .with_context(|| {
                    format!(
                        "Failed to execute node '{}' (op_type: {})",
                        node.name, node.op_type
                    )
                })?;

            #[cfg(debug_assertions)]
            {
                // Log every 100th node to track progress
                static COUNTER: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count.is_multiple_of(100) {
                    eprintln!("Executed {} nodes...", count);
                }

                // Log nodes around errors
                if (395..=420).contains(&count) || (445..=455).contains(&count) {
                    eprintln!(
                        "Node {}: {} -> {} (shape: {:?})",
                        count,
                        node.op_type,
                        node.outputs[0],
                        output.shape()
                    );
                }
            }

            // Store outputs
            // Most nodes have 1 output; aliased multi-output (LSTM) shares
            // one tensor under multiple names. Genuine multi-output ops
            // are handled above by the per-op-type branch (e.g. Split).
            if node.outputs.len() == 1 {
                values.insert(node.outputs[0].clone(), output);
            } else {
                // Aliased multi-output node — store the shared output under
                // each name (LSTM Y/Y_h/Y_c precedent).
                for output_name in &node.outputs {
                    if !output_name.is_empty() {
                        values.insert(output_name.clone(), output.clone());
                    }
                }
            }
        }

        // Collect requested outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            let tensor = values
                .get(name)
                .cloned()
                .with_context(|| format!("Output '{}' not found", name))?;
            outputs.insert(name.to_string(), tensor);
        }

        Ok(outputs)
    }

    /// Topologically sort nodes for execution
    fn topological_sort(&self) -> Result<Vec<ExecutionNode>> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        // Build dependency map
        let node_map: HashMap<String, &ExecutionNode> =
            self.nodes.iter().map(|n| (n.name.clone(), n)).collect();

        fn visit(
            node: &ExecutionNode,
            node_map: &HashMap<String, &ExecutionNode>,
            visited: &mut HashSet<String>,
            temp_mark: &mut HashSet<String>,
            sorted: &mut Vec<ExecutionNode>,
            available_values: &HashSet<String>,
        ) -> Result<()> {
            if visited.contains(&node.name) {
                return Ok(());
            }

            if temp_mark.contains(&node.name) {
                anyhow::bail!("Cycle detected in graph at node '{}'", node.name);
            }

            temp_mark.insert(node.name.clone());

            // Visit dependencies (nodes that produce our inputs)
            for input in &node.inputs {
                // Skip if input is an initializer or graph input
                if available_values.contains(input) {
                    continue;
                }

                // Find node that produces this input
                if let Some(dep_node) = node_map.values().find(|n| n.outputs.contains(input)) {
                    visit(
                        dep_node,
                        node_map,
                        visited,
                        temp_mark,
                        sorted,
                        available_values,
                    )?;
                }
            }

            temp_mark.remove(&node.name);
            visited.insert(node.name.clone());
            sorted.push(node.clone());

            Ok(())
        }

        // Available values = initializers
        let available: HashSet<String> = self.initializers.keys().cloned().collect();

        // Visit all nodes
        for node in &self.nodes {
            if !visited.contains(&node.name) {
                visit(
                    node,
                    &node_map,
                    &mut visited,
                    &mut temp_mark,
                    &mut sorted,
                    &available,
                )?;
            }
        }

        Ok(sorted)
    }

    /// Execute a single operator
    fn execute_operator(
        &self,
        op_type: &str,
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Result<Tensor> {
        match op_type {
            "Add" => Ok(add::Add::forward(inputs, attributes)),
            "And" => Ok(and::And::forward(inputs, attributes)),
            "Atan" => Ok(atan::Atan::forward(inputs, attributes)),
            "Cast" => Ok(cast::Cast::forward(inputs, attributes)),
            "Clip" => Ok(clip::Clip::forward(inputs, attributes)),
            "Concat" => Ok(concat::Concat::forward(inputs, attributes)),
            "Constant" => Ok(constant::Constant::forward(inputs, attributes)),
            "ConstantOfShape" => Ok(constant_of_shape::ConstantOfShape::forward(
                inputs, attributes,
            )),
            "Conv" => Ok(conv::Conv::forward(inputs, attributes)),
            "ConvTranspose" => Ok(conv_transpose::ConvTranspose::forward(inputs, attributes)),
            "Cos" => Ok(cos::Cos::forward(inputs, attributes)),
            "CumSum" => Ok(cumsum::CumSum::forward(inputs, attributes)),
            "Div" => Ok(div::Div::forward(inputs, attributes)),
            "Equal" => Ok(equal::Equal::forward(inputs, attributes)),
            "Exp" => Ok(exp::Exp::forward(inputs, attributes)),
            "Expand" => Ok(expand::Expand::forward(inputs, attributes)),
            "Floor" => Ok(floor::Floor::forward(inputs, attributes)),
            "Gather" => Ok(gather::Gather::forward(inputs, attributes)),
            "Gemm" => Ok(gemm::Gemm::forward(inputs, attributes)),
            "Greater" => Ok(greater::Greater::forward(inputs, attributes)),
            "GreaterOrEqual" => Ok(greater_or_equal::GreaterOrEqual::forward(
                inputs, attributes,
            )),
            "LayerNormalization" => Ok(layer_normalization::LayerNormalization::forward(
                inputs, attributes,
            )),
            "LeakyRelu" => Ok(leaky_relu::LeakyRelu::forward(inputs, attributes)),
            "Less" => Ok(less::Less::forward(inputs, attributes)),
            "LSTM" => Ok(lstm::LSTM::forward(inputs, attributes)),
            "MatMul" => Ok(matmul::MatMul::forward(inputs, attributes)),
            "Mul" => Ok(mul::Mul::forward(inputs, attributes)),
            "NonZero" => Ok(nonzero::NonZero::forward(inputs, attributes)),
            "Pad" => Ok(pad::Pad::forward(inputs, attributes)),
            "Pow" => Ok(pow::Pow::forward(inputs, attributes)),
            "Range" => Ok(range::Range::forward(inputs, attributes)),
            "ReduceMean" => Ok(reduce_mean::ReduceMean::forward(inputs, attributes)),
            "ReduceSum" => Ok(reduce_sum::ReduceSum::forward(inputs, attributes)),
            "Resize" => Ok(resize::Resize::forward(inputs, attributes)),
            "Reshape" => Ok(reshape::Reshape::forward(inputs, attributes)),
            "Round" => Ok(round::Round::forward(inputs, attributes)),
            "ScatterND" => Ok(scatter_nd::ScatterND::forward(inputs, attributes)),
            "Shape" => Ok(shape::Shape::forward(inputs, attributes)),
            "Sigmoid" => Ok(sigmoid::Sigmoid::forward(inputs, attributes)),
            "Sin" => Ok(sin::Sin::forward(inputs, attributes)),
            "Slice" => Ok(slice::Slice::forward(inputs, attributes)),
            "Relu" => Ok(relu::Relu::forward(inputs, attributes)),
            "Softmax" => Ok(softmax::Softmax::forward(inputs, attributes)),
            "Sqrt" => Ok(sqrt::Sqrt::forward(inputs, attributes)),
            "Squeeze" => Ok(squeeze::Squeeze::forward(inputs, attributes)),
            "STFT" => Ok(stft::Stft::forward(inputs, attributes)),
            "Sub" => Ok(sub::Sub::forward(inputs, attributes)),
            "Erf" => Ok(erf::Erf::forward(inputs, attributes)),
            "Tanh" => Ok(tanh::Tanh::forward(inputs, attributes)),
            "Transpose" => Ok(transpose::Transpose::forward(inputs, attributes)),
            "Unsqueeze" => Ok(unsqueeze::Unsqueeze::forward(inputs, attributes)),
            "Where" => Ok(where_op::Where::forward(inputs, attributes)),
            _ => anyhow::bail!("Unsupported operator: {}", op_type),
        }
    }

    /// Execute the graph with NaN detection (for debugging)
    ///
    /// Logs the first node that produces NaN values
    pub fn run_with_nan_detection(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<HashMap<String, Tensor>> {
        // Storage for intermediate tensors
        let mut values: HashMap<String, Tensor> = HashMap::new();

        // Add inputs
        for (name, tensor) in inputs {
            values.insert(name, tensor);
        }

        // Add initializers
        for (name, tensor) in &self.initializers {
            values.insert(name.clone(), tensor.clone());
        }

        // Topologically sort nodes
        let sorted_nodes = self.topological_sort()?;

        let mut first_nan_node: Option<(usize, String, String)> = None;

        // Execute nodes in order
        for (idx, node) in sorted_nodes.iter().enumerate() {
            // Get node output name for logging (used in debug traces)
            let _name = node.outputs.first().map(|s| s.as_str()).unwrap_or("");
            // Special handling for Constant nodes
            if node.op_type == "Constant" {
                if let Some(value_tensor) = node.attributes.resolve_constant_value() {
                    assert_eq!(node.outputs.len(), 1, "Constant should have 1 output");
                    values.insert(node.outputs[0].clone(), value_tensor);
                    continue;
                } else {
                    anyhow::bail!(
                        "Constant node '{}' missing all value-carrying attributes \
                         (value / value_int / value_ints / value_float / value_floats)",
                        node.name
                    );
                }
            }

            // Get input tensors
            let input_tensors: Vec<Tensor> = node
                .inputs
                .iter()
                .filter_map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        Some(values.get(name).cloned().with_context(|| {
                            format!("Missing input '{}' for node '{}'", name, node.name)
                        }))
                    }
                })
                .collect::<Result<Vec<_>>>()?;

            // Split: genuine multi-output. Bypass execute_operator's
            // single-output signature and store each slice independently.
            if node.op_type == "Split" {
                let split_outputs = operators::split::Split::forward(
                    &input_tensors,
                    &node.attributes,
                    node.outputs.len(),
                );
                for (name, tensor) in node.outputs.iter().zip(split_outputs.into_iter()) {
                    if !name.is_empty() {
                        values.insert(name.clone(), tensor);
                    }
                }
                continue;
            }

            // Execute operator
            let output = self
                .execute_operator(&node.op_type, &input_tensors, &node.attributes)
                .with_context(|| {
                    format!(
                        "Failed to execute node '{}' (op_type: {})",
                        node.name, node.op_type
                    )
                })?;

            // Check for NaN in output (only for float32 tensors)
            if first_nan_node.is_none() {
                if let Tensor::Float32(_) = &output {
                    let data = output.as_slice();
                    let has_nan = data.iter().any(|x| x.is_nan());
                    if has_nan {
                        first_nan_node = Some((idx, node.op_type.clone(), node.name.clone()));

                        // Log details about the problematic node
                        eprintln!("\n=== FIRST NaN DETECTED at node {} ===", idx);
                        eprintln!("  Op: {}", node.op_type);
                        eprintln!("  Name: {}", node.name);
                        eprintln!("  Inputs: {:?}", node.inputs);
                        eprintln!("  Outputs: {:?}", node.outputs);
                        eprintln!("  Output shape: {:?}", output.shape());

                        // Log input tensor info
                        for (i, (input_name, input_tensor)) in
                            node.inputs.iter().zip(input_tensors.iter()).enumerate()
                        {
                            if let Tensor::Float32(_) = input_tensor {
                                let input_data = input_tensor.as_slice();
                                let has_nan = input_data.iter().any(|x| x.is_nan());
                                let has_inf = input_data.iter().any(|x| x.is_infinite());
                                let min = input_data.iter().cloned().fold(f32::INFINITY, f32::min);
                                let max =
                                    input_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                eprintln!(
                                    "  Input {}: name={}, shape={:?}, has_nan={}, has_inf={}, range=[{:.6}, {:.6}]",
                                    i, input_name, input_tensor.shape(), has_nan, has_inf, min, max
                                );
                            } else {
                                eprintln!(
                                    "  Input {}: name={}, shape={:?}, dtype=non-float32",
                                    i,
                                    input_name,
                                    input_tensor.shape()
                                );
                            }
                        }

                        let nan_count = data.iter().filter(|x| x.is_nan()).count();
                        eprintln!("  Output NaN count: {} / {}", nan_count, data.len());

                        // For Div operations, check for 0/0 situations
                        if node.op_type == "Div" && input_tensors.len() == 2 {
                            if let (Tensor::Float32(_), Tensor::Float32(_)) =
                                (&input_tensors[0], &input_tensors[1])
                            {
                                let num_data = input_tensors[0].as_slice();
                                let denom_data = input_tensors[1].as_slice();
                                let min_len = num_data.len().min(denom_data.len());

                                let mut zero_zero_count = 0;
                                let mut nonzero_zero_count = 0;
                                let mut near_zero_denom_count = 0;

                                for i in 0..min_len {
                                    let n = num_data[i];
                                    let d = denom_data[i];
                                    if d == 0.0 && n == 0.0 {
                                        zero_zero_count += 1;
                                    } else if d == 0.0 && n != 0.0 {
                                        nonzero_zero_count += 1;
                                    }
                                    if d.abs() < 1e-10 {
                                        near_zero_denom_count += 1;
                                    }
                                }

                                eprintln!("  Div analysis:");
                                eprintln!("    0/0 cases: {}", zero_zero_count);
                                eprintln!("    x/0 (non-zero/zero): {}", nonzero_zero_count);
                                eprintln!(
                                    "    Near-zero denominators (<1e-10): {}",
                                    near_zero_denom_count
                                );

                                // Show sample of zero denominators
                                let mut zero_indices: Vec<usize> = Vec::new();
                                for i in 0..min_len.min(10000) {
                                    if denom_data[i] == 0.0 {
                                        zero_indices.push(i);
                                        if zero_indices.len() >= 10 {
                                            break;
                                        }
                                    }
                                }
                                if !zero_indices.is_empty() {
                                    eprintln!(
                                        "    First zero denominator indices: {:?}",
                                        zero_indices
                                    );
                                    for &i in &zero_indices[..3.min(zero_indices.len())] {
                                        eprintln!(
                                            "      idx {}: num={}, denom={}",
                                            i, num_data[i], denom_data[i]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Store outputs
            if node.outputs.len() == 1 {
                values.insert(node.outputs[0].clone(), output);
            } else {
                for output_name in &node.outputs {
                    if !output_name.is_empty() {
                        values.insert(output_name.clone(), output.clone());
                    }
                }
            }
        }

        if let Some((idx, op, name)) = first_nan_node {
            eprintln!(
                "\n=== Summary: First NaN at node {} ({}) name={} ===\n",
                idx, op, name
            );
        }

        // Collect requested outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            let tensor = values
                .get(name)
                .cloned()
                .with_context(|| format!("Output '{}' not found", name))?;
            outputs.insert(name.to_string(), tensor);
        }

        Ok(outputs)
    }
}

impl Default for GraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}
