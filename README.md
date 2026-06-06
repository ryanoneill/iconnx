# iconnx

High-performance ONNX inference engine with CUDA GPU acceleration, written in pure Rust.

## Features

- **ONNX opset 11–20 support** - Modern operators for transformer models
- **CUDA GPU acceleration** - Native cuBLAS/cuDNN integration via **garboard**
- **Pure Rust** - No C++ dependencies, just Rust + CUDA

`GpuGraphExecutor` is `!Sync` — use one executor per thread.

## Quick Start

```rust
use std::collections::HashMap;
use iconnx::{OnnxParser, GpuGraphExecutor, Tensor};

let model = OnnxParser::parse_file("model.onnx")?;
let executor = GpuGraphExecutor::from_model(&model)?; // !Sync: one executor per thread

let mut inputs = HashMap::new();
inputs.insert("input".to_string(), Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![1, 3]));
let outputs = executor.run(inputs, vec!["output"])?;
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

iconnx supports ONNX opset 11–20 operators including:

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
│  - Weights loaded via from_model()      │
│  - Per-inference CUDA streams           │
│  - cuBLAS/cuDNN via garboard            │
│  - Custom CUDA kernels                  │
└─────────────────────────────────────────┘
```

## Performance

iconnx is designed for high-throughput inference:

- **Memory pooling** - Avoid allocation fragmentation
- **Kernel caching** - Compiled CUDA kernels are reused
- **Kernel fusion** - MulAdd, DivMul patterns fused

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
