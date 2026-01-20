/// Cast operator for Iconnx
///
/// Type conversion operator
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Cast.html
use crate::tensor::Tensor;

/// Cast operator - type conversion
pub struct Cast;

// ONNX TensorProto DataType enum values
const ONNX_FLOAT: i64 = 1;
const ONNX_INT32: i64 = 6;
const ONNX_INT64: i64 = 7;
const ONNX_BOOL: i64 = 9;

impl Cast {
    /// Forward pass: cast to different type
    ///
    /// # Arguments
    /// * `inputs` - [data_tensor]
    /// * `attributes` - Contains "to" attribute specifying target dtype
    ///
    /// # Returns
    /// Tensor with converted dtype
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Cast requires exactly 1 input");

        let to_dtype = attributes
            .get_int("to")
            .expect("Cast requires 'to' attribute");

        match to_dtype {
            ONNX_FLOAT => {
                // Convert to Float32
                match &inputs[0] {
                    Tensor::Float32(_) => inputs[0].clone(),
                    Tensor::Int64(data) => {
                        let float_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(float_data, data.shape().to_vec())
                    }
                    Tensor::Int32(data) => {
                        let float_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(float_data, data.shape().to_vec())
                    }
                    _ => panic!("Cast from {} to Float32 not implemented", inputs[0].dtype()),
                }
            }
            ONNX_INT64 => {
                // Convert to Int64
                match &inputs[0] {
                    Tensor::Int64(_) => inputs[0].clone(),
                    Tensor::Float32(data) => {
                        let int_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                        Tensor::from_vec_i64(int_data, data.shape().to_vec())
                    }
                    Tensor::Int32(data) => {
                        let int_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                        Tensor::from_vec_i64(int_data, data.shape().to_vec())
                    }
                    _ => panic!("Cast from {} to Int64 not implemented", inputs[0].dtype()),
                }
            }
            ONNX_INT32 => {
                // Convert to Int32
                match &inputs[0] {
                    Tensor::Int32(_) => inputs[0].clone(),
                    Tensor::Float32(data) => {
                        let int_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                        Tensor::from_vec_i32(int_data, data.shape().to_vec())
                    }
                    Tensor::Int64(data) => {
                        let int_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                        Tensor::from_vec_i32(int_data, data.shape().to_vec())
                    }
                    _ => panic!("Cast from {} to Int32 not implemented", inputs[0].dtype()),
                }
            }
            ONNX_BOOL => {
                // Convert to Bool (represented as Float32 with 0.0/1.0 values)
                match &inputs[0] {
                    Tensor::Float32(data) => {
                        let bool_data: Vec<f32> = data
                            .iter()
                            .map(|&x| if x != 0.0 { 1.0 } else { 0.0 })
                            .collect();
                        Tensor::from_vec(bool_data, data.shape().to_vec())
                    }
                    Tensor::Int64(data) => {
                        let bool_data: Vec<f32> = data
                            .iter()
                            .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                            .collect();
                        Tensor::from_vec(bool_data, data.shape().to_vec())
                    }
                    Tensor::Int32(data) => {
                        let bool_data: Vec<f32> = data
                            .iter()
                            .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                            .collect();
                        Tensor::from_vec(bool_data, data.shape().to_vec())
                    }
                    _ => panic!("Cast from {} to Bool not implemented", inputs[0].dtype()),
                }
            }
            _ => panic!("Cast to dtype {} not implemented", to_dtype),
        }
    }
}
