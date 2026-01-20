/// ONNX node attribute handling
///
/// Parses and provides access to ONNX node attributes (parameters)
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Parsed node attributes
#[derive(Debug, Clone, Default)]
pub struct NodeAttributes {
    /// Integer attributes
    pub ints: HashMap<String, Vec<i64>>,
    /// Float attributes
    pub floats: HashMap<String, Vec<f32>>,
    /// String attributes
    pub strings: HashMap<String, String>,
    /// Tensor attributes
    pub tensors: HashMap<String, Tensor>,
}

impl NodeAttributes {
    /// Create empty attributes
    pub fn new() -> Self {
        Self::default()
    }

    /// Get integer attribute (single value)
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.ints.get(name).and_then(|v| v.first().copied())
    }

    /// Get integer list attribute
    pub fn get_ints(&self, name: &str) -> Option<&[i64]> {
        self.ints.get(name).map(|v| v.as_slice())
    }

    /// Get float attribute (single value)
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.floats.get(name).and_then(|v| v.first().copied())
    }

    /// Get float list attribute
    pub fn get_floats(&self, name: &str) -> Option<&[f32]> {
        self.floats.get(name).map(|v| v.as_slice())
    }

    /// Get string attribute
    pub fn get_string(&self, name: &str) -> Option<&str> {
        self.strings.get(name).map(|s| s.as_str())
    }

    /// Get tensor attribute
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Add integer attribute
    pub fn add_int(&mut self, name: String, value: i64) {
        self.ints.insert(name, vec![value]);
    }

    /// Add integer list attribute
    pub fn add_ints(&mut self, name: String, values: Vec<i64>) {
        self.ints.insert(name, values);
    }

    /// Add float attribute
    pub fn add_float(&mut self, name: String, value: f32) {
        self.floats.insert(name, vec![value]);
    }

    /// Add float list attribute
    pub fn add_floats(&mut self, name: String, values: Vec<f32>) {
        self.floats.insert(name, values);
    }

    /// Add string attribute
    pub fn add_string(&mut self, name: String, value: String) {
        self.strings.insert(name, value);
    }

    /// Add tensor attribute
    pub fn add_tensor(&mut self, name: String, tensor: Tensor) {
        self.tensors.insert(name, tensor);
    }
}
