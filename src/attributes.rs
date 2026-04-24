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

    /// Resolve a Constant node's value from whichever value-carrying
    /// attribute is present. ONNX's Constant op allows the value to
    /// live in `value`, `value_float`, `value_floats`, `value_int`,
    /// `value_ints`, `value_string`, `value_strings`, or
    /// `sparse_value`. Callers that previously special-cased only
    /// `get_tensor("value")` miss graphs whose exporters emit one of
    /// the scalar/vector alternatives — e.g. Whisper's optimum-cli
    /// export uses `value_int` / `value_ints` for attribute-like
    /// axes/indices constants.
    ///
    /// Priority order matches the spec: `value` (full tensor) first,
    /// then the typed scalar/vector variants. Returns `None` only if
    /// all supported attributes are missing. `value_string(s)` and
    /// `sparse_value` return `None` — iconnx's runtime doesn't yet
    /// carry string tensors or sparse tensors in any op dispatch, so
    /// promoting those to initializers would only defer the error
    /// to a downstream mismatch.
    pub fn resolve_constant_value(&self) -> Option<Tensor> {
        // Full tensor — the common case, covers all dtypes.
        if let Some(t) = self.get_tensor("value") {
            return Some(t.clone());
        }

        // Scalar int → rank-0 Int64 tensor (shape []).
        if let Some(n) = self.get_int("value_int") {
            return Some(Tensor::from_vec_i64(vec![n], vec![]));
        }

        // 1-D int vector → rank-1 Int64 tensor.
        if let Some(ns) = self.get_ints("value_ints") {
            let len = ns.len();
            return Some(Tensor::from_vec_i64(ns.to_vec(), vec![len]));
        }

        // Scalar float → rank-0 Float32 tensor.
        if let Some(f) = self.get_float("value_float") {
            return Some(Tensor::from_vec_f32(vec![f], vec![]));
        }

        // 1-D float vector → rank-1 Float32 tensor.
        if let Some(fs) = self.get_floats("value_floats") {
            let len = fs.len();
            return Some(Tensor::from_vec_f32(fs.to_vec(), vec![len]));
        }

        // value_string(s) and sparse_value intentionally unsupported.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_constant_value_prefers_tensor_attribute() {
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".to_string(),
            Tensor::from_vec_i64(vec![1, 2, 3], vec![3]),
        );
        // Also set value_int to confirm priority — tensor wins.
        attrs.add_int("value_int".to_string(), 99);

        let t = attrs.resolve_constant_value().expect("value takes priority");
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.to_array_i64().as_slice().unwrap(), &[1i64, 2, 3]);
    }

    #[test]
    fn resolve_constant_value_reads_value_int() {
        let mut attrs = NodeAttributes::new();
        attrs.add_int("value_int".to_string(), 42);
        let t = attrs.resolve_constant_value().expect("value_int present");
        // ONNX scalar → rank-0 tensor.
        assert_eq!(t.shape(), &[] as &[usize]);
        assert_eq!(t.to_array_i64().as_slice().unwrap(), &[42i64]);
    }

    #[test]
    fn resolve_constant_value_reads_value_ints() {
        let mut attrs = NodeAttributes::new();
        attrs.add_ints("value_ints".to_string(), vec![10, 20, 30, 40]);
        let t = attrs.resolve_constant_value().expect("value_ints present");
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.to_array_i64().as_slice().unwrap(), &[10i64, 20, 30, 40]);
    }

    #[test]
    fn resolve_constant_value_reads_value_float() {
        let mut attrs = NodeAttributes::new();
        attrs.add_float("value_float".to_string(), 2.5);
        let t = attrs.resolve_constant_value().expect("value_float present");
        assert_eq!(t.shape(), &[] as &[usize]);
        let vals = t.to_array();
        assert!((vals.as_slice().unwrap()[0] - 2.5_f32).abs() < 1e-6);
    }

    #[test]
    fn resolve_constant_value_reads_value_floats() {
        let mut attrs = NodeAttributes::new();
        attrs.add_floats("value_floats".to_string(), vec![1.0, 2.0, 3.0]);
        let t = attrs.resolve_constant_value().expect("value_floats present");
        assert_eq!(t.shape(), &[3]);
        let vals = t.to_array();
        assert_eq!(vals.as_slice().unwrap(), &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn resolve_constant_value_returns_none_when_all_missing() {
        let attrs = NodeAttributes::new();
        assert!(attrs.resolve_constant_value().is_none());
    }

    #[test]
    fn resolve_constant_value_int_wins_over_float_when_both_present() {
        // Exporters shouldn't emit both, but if they do, value_int's
        // priority (per our spec-inspired ordering) is higher than
        // value_float. This test locks that contract.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("value_int".to_string(), 7);
        attrs.add_float("value_float".to_string(), 2.5);
        let t = attrs.resolve_constant_value().unwrap();
        assert_eq!(t.to_array_i64().as_slice().unwrap(), &[7i64]);
    }
}
