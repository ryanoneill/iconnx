# iconnx

High-performance ONNX inference engine with CUDA GPU acceleration, written in pure Rust.

## Features

- **ONNX Opset 17+ support** - Modern operators for transformer models
- **CUDA GPU acceleration** - Native cuBLAS/cuDNN integration via cudarc
- **Concurrent inference** - Multiple streams share weights for optimal GPU utilization
- **CPU fallback** - Full CPU implementation for testing and portability
- **Pure Rust** - No C++ dependencies, just Rust + CUDA

## Quick Start

```rust
use iconnx::{OnnxParser, GpuGraphExecutor, Tensor};

// Load ONNX model
let model = OnnxParser::parse_file("model.onnx")?;

// Create GPU executor
let mut executor = GpuGraphExecutor::new()?;

// Add model weights
for (name, tensor) in model.extract_weights() {
    executor.add_initializer(name, &tensor)?;
}

// Build computation graph
for node in model.computation_graph().nodes() {
    executor.add_node(
        node.name(),
        node.op_type(),
        node.inputs(),
        node.outputs(),
        node.attributes().clone(),
    );
}

// Run inference
let inputs = HashMap::from([
    ("input".to_string(), Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![1, 3])),
]);
let outputs = executor.run(inputs, &["output"])?;
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
iconnx = "0.1"
```

### CUDA Requirements

For GPU acceleration (enabled by default), you need:

- CUDA Toolkit 12.0+
- cuBLAS and cuDNN libraries

To build without CUDA:

```toml
[dependencies]
iconnx = { version = "0.1", default-features = false }
```

## Supported Operators

iconnx supports ONNX Opset 17+ operators including:

### Arithmetic
- Add, Sub, Mul, Div, Pow, Sqrt, Exp

### Linear Algebra
- MatMul, Gemm

### Activations
- Sigmoid, Tanh, Relu, LeakyRelu, Softmax

### Normalization
- LayerNormalization

### Shape Operations
- Reshape, Transpose, Squeeze, Unsqueeze, Concat, Slice, Gather, ScatterND

### Reductions
- ReduceSum, ReduceMean

### Convolutions
- Conv, ConvTranspose

### Recurrent
- LSTM

### Signal Processing
- STFT (Short-Time Fourier Transform)

### Comparison & Logic
- Equal, Greater, Less, GreaterOrEqual, Where, And

### Type Operations
- Cast, Expand, Pad, Resize, NonZero

## Architecture

```
┌─────────────────────────────────────────┐
│           OnnxParser                    │  ← Parse .onnx files
│  - Load model weights                   │
│  - Extract computation graph            │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│       GpuGraphExecutor (CUDA)           │  ← GPU inference
│  - Shared weights via Arc               │
│  - Per-inference CUDA streams           │
│  - cuBLAS for matrix operations         │
│  - Custom CUDA kernels                  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│       GraphExecutor (CPU)               │  ← CPU fallback
│  - Pure ndarray implementation          │
│  - Used for testing/validation          │
└─────────────────────────────────────────┘
```

## Performance

iconnx is designed for high-throughput inference:

- **Memory pooling** - Avoid allocation fragmentation
- **Kernel caching** - Compiled CUDA kernels are reused
- **Kernel fusion** - MulAdd, DivMul patterns fused
- **Concurrent streams** - Multiple inferences share weights

## Testing

iconnx is thoroughly tested against ONNX Runtime:

```bash
# Run all tests
cargo test

# Run with CUDA tests (requires GPU)
cargo test --release

# Run specific operator tests
cargo test operators::
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and add tests for new functionality.
