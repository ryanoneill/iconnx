//! Build script for iconnx
//!
//! Compiles ONNX protobuf schema for model parsing.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile ONNX protobuf schema
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"])?;

    Ok(())
}
