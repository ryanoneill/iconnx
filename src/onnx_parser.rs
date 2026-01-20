/// ONNX model parser for Iconnx inference framework
///
/// Parses ONNX protobuf files to extract:
/// - Model computation graph
/// - Weight tensors (initializers)
/// - Input/output specifications
/// - Operator attributes
///
/// Uses prost for protobuf decoding.
use anyhow::{Context, Result};
use prost::Message;
use std::path::Path;

/// Generated ONNX protobuf types
mod onnx_proto {
    #![allow(clippy::all, clippy::pedantic)]
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

/// ONNX model parser
pub struct OnnxParser;

impl OnnxParser {
    /// Parse an ONNX model file
    ///
    /// TDD Implementation - First test cycle
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<OnnxModel> {
        // 1. Read file bytes
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("Failed to read ONNX file: {:?}", path.as_ref()))?;

        // 2. Decode protobuf
        let proto =
            onnx_proto::ModelProto::decode(&bytes[..]).context("Failed to decode ONNX protobuf")?;

        // 3. Extract graph
        let _graph = proto
            .graph
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ONNX model has no graph"))?;

        // 4. Return model
        Ok(OnnxModel { proto })
    }
}

/// Parsed ONNX model
///
/// Contains the parsed ONNX protobuf model
pub struct OnnxModel {
    #[allow(dead_code)]
    proto: onnx_proto::ModelProto,
}

impl OnnxModel {
    /// Get model inputs
    ///
    /// Returns the names of all graph inputs
    pub fn inputs(&self) -> Vec<String> {
        self.proto
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .input
                    .iter()
                    .filter_map(|input| input.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get model outputs
    ///
    /// Returns the names of all graph outputs
    pub fn outputs(&self) -> Vec<String> {
        self.proto
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .output
                    .iter()
                    .filter_map(|output| output.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List names of all weight tensors (initializers)
    ///
    /// Returns the names of all weight tensors stored in graph.initializer
    pub fn list_weight_names(&self) -> Vec<String> {
        self.proto
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .initializer
                    .iter()
                    .filter_map(|tensor| tensor.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extract weight tensors from initializers
    ///
    /// Returns a map of tensor name -> Tensor
    /// Handles Float32, Int64, Int32 dtypes from ONNX
    pub fn extract_weights(
        &self,
    ) -> std::collections::HashMap<String, crate::tensor::Tensor> {
        use std::collections::HashMap;

        let mut weights = HashMap::new();

        if let Some(graph) = self.proto.graph.as_ref() {
            for tensor_proto in &graph.initializer {
                let name = tensor_proto.name.clone().unwrap_or_default();

                // Get shape
                let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();
                let expected_len: usize = if shape.is_empty() {
                    1
                } else {
                    shape.iter().product()
                };

                // Determine dtype from ONNX data_type field
                // https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
                // 1 = FLOAT, 6 = INT32, 7 = INT64
                let data_type = tensor_proto.data_type.unwrap_or(1); // Default to FLOAT

                let tensor_opt = match data_type {
                    1 => {
                        // FLOAT (f32)
                        let data: Vec<f32> = if !tensor_proto.float_data.is_empty() {
                            tensor_proto.float_data.clone()
                        } else if let Some(raw) = &tensor_proto.raw_data {
                            if !raw.is_empty() {
                                raw.chunks_exact(4)
                                    .map(|chunk| {
                                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                    })
                                    .collect()
                            } else {
                                vec![]
                            }
                        } else {
                            vec![]
                        };

                        if !data.is_empty() {
                            let final_shape =
                                Self::correct_shape(&name, &shape, expected_len, data.len());
                            Some(crate::tensor::Tensor::from_vec_f32(
                                data,
                                final_shape,
                            ))
                        } else {
                            None
                        }
                    }
                    7 => {
                        // INT64 (i64)
                        let data: Vec<i64> = if !tensor_proto.int64_data.is_empty() {
                            tensor_proto.int64_data.clone()
                        } else if let Some(raw) = &tensor_proto.raw_data {
                            if !raw.is_empty() {
                                raw.chunks_exact(8)
                                    .map(|chunk| {
                                        i64::from_le_bytes([
                                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4],
                                            chunk[5], chunk[6], chunk[7],
                                        ])
                                    })
                                    .collect()
                            } else {
                                vec![]
                            }
                        } else {
                            vec![]
                        };

                        if !data.is_empty() {
                            let final_shape =
                                Self::correct_shape(&name, &shape, expected_len, data.len());
                            Some(crate::tensor::Tensor::from_vec_i64(
                                data,
                                final_shape,
                            ))
                        } else {
                            None
                        }
                    }
                    6 => {
                        // INT32 (i32)
                        let data: Vec<i32> = if !tensor_proto.int32_data.is_empty() {
                            tensor_proto.int32_data.clone()
                        } else if let Some(raw) = &tensor_proto.raw_data {
                            if !raw.is_empty() {
                                raw.chunks_exact(4)
                                    .map(|chunk| {
                                        i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                    })
                                    .collect()
                            } else {
                                vec![]
                            }
                        } else {
                            vec![]
                        };

                        if !data.is_empty() {
                            let final_shape =
                                Self::correct_shape(&name, &shape, expected_len, data.len());
                            Some(crate::tensor::Tensor::from_vec_i32(
                                data,
                                final_shape,
                            ))
                        } else {
                            None
                        }
                    }
                    _ => {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Warning: Unsupported data_type {} for '{}', skipping",
                            data_type, name
                        );
                        None
                    }
                };

                if let Some(tensor) = tensor_opt {
                    weights.insert(name, tensor);
                }
            }
        }

        weights
    }

    /// Helper: Correct shape mismatches between ONNX metadata and actual data
    fn correct_shape(
        name: &str,
        shape: &[usize],
        expected_len: usize,
        actual_len: usize,
    ) -> Vec<usize> {
        // Suppress warning when debug_assertions is off
        let _ = name;
        if actual_len == expected_len {
            shape.to_vec()
        } else {
            #[cfg(debug_assertions)]
            eprintln!(
                "Info: Correcting shape for '{}': {} elements, ONNX says {:?} (expects {}), using [{}]",
                name, actual_len, shape, expected_len, actual_len
            );
            vec![actual_len]
        }
    }

    /// List all unique operators used
    ///
    /// Returns a sorted list of unique operator types used in the computation graph
    pub fn list_unique_operators(&self) -> Vec<String> {
        use std::collections::HashSet;

        let mut operators = HashSet::new();

        if let Some(graph) = self.proto.graph.as_ref() {
            for node in &graph.node {
                if let Some(op_type) = &node.op_type {
                    operators.insert(op_type.clone());
                }
            }
        }

        let mut result: Vec<String> = operators.into_iter().collect();
        result.sort();
        result
    }

    /// Get computation graph
    ///
    /// Returns the computation graph with all nodes and their attributes
    pub fn computation_graph(&self) -> ComputationGraph {
        let nodes = self
            .proto
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .node
                    .iter()
                    .map(|node| {
                        let attributes = Self::parse_attributes(&node.attribute);
                        GraphNode {
                            op_type: node.op_type.clone().unwrap_or_default(),
                            inputs: node.input.clone(),
                            outputs: node.output.clone(),
                            attributes,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        ComputationGraph { nodes }
    }

    /// Parse attributes from ONNX AttributeProto
    fn parse_attributes(
        attrs: &[onnx_proto::AttributeProto],
    ) -> crate::attributes::NodeAttributes {
        use crate::attributes::NodeAttributes;

        let mut node_attrs = NodeAttributes::new();

        for attr in attrs {
            let name = attr.name.clone().unwrap_or_default();

            // Integer attribute
            if let Some(i) = attr.i {
                node_attrs.add_int(name.clone(), i);
            }

            // Integer list
            if !attr.ints.is_empty() {
                node_attrs.add_ints(name.clone(), attr.ints.clone());
            }

            // Float attribute
            if let Some(f) = attr.f {
                node_attrs.add_float(name.clone(), f);
            }

            // Float list
            if !attr.floats.is_empty() {
                node_attrs.add_floats(name.clone(), attr.floats.clone());
            }

            // String attribute
            if let Some(s) = &attr.s {
                if let Ok(string) = String::from_utf8(s.clone()) {
                    node_attrs.add_string(name.clone(), string);
                }
            }

            // Tensor attribute
            if let Some(tensor_proto) = &attr.t {
                if let Ok(tensor) = Self::parse_tensor_proto(tensor_proto) {
                    node_attrs.add_tensor(name.clone(), tensor);
                }
            }
        }

        node_attrs
    }

    /// Parse a TensorProto into a Tensor
    fn parse_tensor_proto(
        tensor_proto: &onnx_proto::TensorProto,
    ) -> Result<crate::tensor::Tensor, String> {
        let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();

        let data: Vec<f32> = if !tensor_proto.float_data.is_empty() {
            tensor_proto.float_data.clone()
        } else if let Some(raw) = &tensor_proto.raw_data {
            if !raw.is_empty() {
                raw.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            } else {
                return Err("Empty raw_data".to_string());
            }
        } else {
            return Err("No float data".to_string());
        };

        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if data.len() != expected_len {
            return Err(format!("Shape mismatch: {} vs {:?}", data.len(), shape));
        }

        Ok(crate::tensor::Tensor::from_vec(data, shape))
    }
}

/// Computation graph structure
///
/// Contains all computation nodes in execution order
pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
}

impl ComputationGraph {
    /// Get all nodes in execution order
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Get number of nodes in graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// A node in the computation graph
///
/// Represents a single operation in the ONNX graph
#[derive(Clone)]
pub struct GraphNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: crate::attributes::NodeAttributes,
}
