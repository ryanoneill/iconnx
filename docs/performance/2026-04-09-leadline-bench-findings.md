# iconnx Performance Investigation: Leadline Bench Findings

**Date:** 2026-04-09
**Hardware:** NVIDIA GeForce RTX 4070 Laptop (8GB VRAM), WSL2
**Model:** Kokoro v1.0 TTS (ONNX, ~325MB weights)
**Tool:** leadline-bench comparison binary (`rust-ai-explorations` repo)
**Baseline:** ORT 2.0.0-rc.12 with CUDA EP, Level3 graph optimization

---

## Executive Summary

iconnx is **~5x slower than ORT** on the Kokoro TTS model at short sequence lengths (seq_len=5). The root cause is **kernel launch overhead** from ~2,400 individual CUDA kernel dispatches per inference, compared to ORT's aggressively fused graph that likely executes far fewer kernels.

This is not a fundamental algorithmic problem. ORT's advantage comes from its graph optimizer (Level3), which fuses elementwise chains, eliminates shape-only ops, and uses cuDNN's fused LSTM. iconnx already fuses DivRsqrt and GELU patterns. The gap can be closed by extending iconnx's fusion pass and eliminating unnecessary kernel launches.

---

## Raw Data

### 10-iteration comparison at seq_len=5

```
iconnx: avg 186.96 ms, min 135.20 ms, max 221.29 ms (46% variance)
ORT:    avg  34.73 ms, min  27.07 ms, max  40.06 ms (37% variance)
Speedup: 0.21x (ORT is 5.4x faster)
```

### iconnx operator breakdown (from last profiled iteration)

Top operators by total time:

| Operator | Total ms | Calls | Avg us/call | Notes |
|---|---|---|---|---|
| Conv | 21.19 | 88 | 240.8 | cuDNN, heaviest per-call |
| Mul | 20.84 | 267 | 78.0 | Elementwise, fusible |
| Add | 19.97 | 352 | 56.7 | Elementwise, fusible |
| LSTM | 9.37 | 6 | 1562.0 | Custom kernel, cuDNN has fused version |
| Fused_DivRsqrt | 7.30 | 65 | 112.4 | Already fused (good) |
| Transpose | 7.02 | 89 | 78.9 | Memory-bound, some eliminable |
| Sub | 6.63 | 68 | 97.5 | Elementwise, fusible |
| Reshape | 6.56 | 61 | 107.5 | Should be zero-cost (view change) |
| Gemm | 6.18 | 73 | 84.6 | cuBLAS |
| Slice | 5.75 | 170 | 33.8 | Many calls, small per-call |
| Pow | 5.34 | 50 | 106.7 | Elementwise, fusible |

Full breakdown: 51 distinct operator types, 2,400+ total kernel launches.

### Key observations

1. **Elementwise ops (Mul, Add, Sub, Pow) = ~62ms total, ~740 launches.** These are individually cheap but collectively dominate. ORT fuses chains of elementwise ops into single kernels.

2. **Reshape at 6.56ms is wrong.** Reshape should be a zero-cost view change with no data movement. 61 calls at 107 us/call suggests iconnx is either copying data or launching unnecessary kernels for reshape.

3. **Shape/Unsqueeze/Squeeze = ~2ms, ~270 launches.** These are pure metadata ops. ORT eliminates them during graph optimization. They should never launch a GPU kernel.

4. **LSTM at 1.5ms/call.** cuDNN's `cudnnRNNForward` fuses the entire LSTM cell (4 gate matmuls + activations + cell state update) into one kernel. iconnx likely implements LSTM as a sequence of individual ops.

5. **High variance (135-221ms).** The 86ms spread suggests non-deterministic behavior — likely CUDA stream scheduling or memory allocation jitter. ORT's variance is tighter (27-40ms).

6. **Fused_DivRsqrt and Fused_Gelu work.** These fused patterns are 7.3ms and 0.23ms respectively. The fusion infrastructure exists and is effective — it just needs more patterns.

---

## Diagnosis: Why ORT is Faster

### 1. Graph optimization (the big one)

ORT Level3 applies these optimizations before any kernel launches:
- **Constant folding** — pre-compute anything that depends only on weights/constants
- **Shape inference + elimination** — remove Shape, Unsqueeze, Squeeze, Reshape nodes entirely
- **Elementwise fusion** — chains like `Add -> Mul -> Add` become single kernels
- **LSTM fusion** — entire LSTM cell becomes one cuDNN call
- **Dead node elimination** — remove ops whose outputs aren't consumed
- **Layout optimization** — avoid unnecessary Transpose by choosing NCHW/NHWC at graph level

iconnx has pattern-based fusion (DivRsqrt, Gelu, AddMulAdd, MulAdd, etc.) but doesn't do graph-level optimization. Every ONNX node becomes a kernel launch.

### 2. Kernel launch overhead

Each CUDA kernel launch has ~10-20 us of overhead (driver API call, command buffer submission, synchronization). With 2,400 launches:
- Estimated launch overhead: 24-48ms (15-30% of total time)
- This is pure waste — the GPU isn't doing any math during launch overhead

ORT minimizes launches by fusing ops. Where iconnx launches 5 kernels for a normalized-multiply pattern, ORT launches 1.

### 3. Zero-cost operations that aren't zero-cost

Reshape (6.56ms), Shape (1.0ms), Unsqueeze (0.24ms), Squeeze (0.03ms) should all be metadata-only operations with no GPU kernel. Combined they're ~8ms of unnecessary work.

---

## Recommended Optimization Targets

Ordered by estimated impact:

### Priority 1: Eliminate shape-only ops (~8ms, 395 launches eliminated)

**Reshape, Shape, Unsqueeze, Squeeze** should not launch kernels. These ops change tensor metadata (shape, strides) without moving data. 

- **Reshape:** Return a new `GpuTensor` with different shape pointing to the same device memory. No `cudaMemcpy`, no kernel.
- **Shape:** Return the shape as a CPU-side `Vec<i64>`. No GPU involvement.
- **Unsqueeze/Squeeze:** Same as Reshape — metadata-only view change.

**Expected savings:** ~8ms and 395 fewer kernel launches.

### Priority 2: Broader elementwise fusion (~30ms potential)

The existing fusion infrastructure (DivRsqrt, Gelu patterns) works. Extend it to cover the most common elementwise chains in the Kokoro graph:

- **Add+Mul chains** (already have AddMulAdd but may not be matching all instances)
- **Sub+Mul, Mul+Add, Add+Sub chains** (352 Add calls + 267 Mul calls + 68 Sub calls suggest many unfused chains)
- **Pow+Mul patterns** (50 Pow calls, often followed by Mul)

Approach: Run the graph through a topological pass that identifies chains of elementwise-only ops and generates fused NVRTC kernels for them. This is what ORT's "NchwcTransformer" and "GeluFusion" passes do.

**Expected savings:** Could reduce ~740 elementwise launches to ~100-200 fused launches, saving ~15-30ms.

### Priority 3: cuDNN LSTM (~5ms potential)

Replace the current multi-kernel LSTM implementation with cuDNN's `cudnnRNNForward`. This fuses the 4 gate matmuls, sigmoid/tanh activations, and cell state update into a single highly-optimized kernel.

iconnx already has a `CudnnHandle` — the infrastructure is there.

**Expected savings:** 6 LSTM calls at 1.5ms -> ~0.3ms each = ~7ms saved.

### Priority 4: Constant folding (variable savings)

Pre-compute any subgraph that depends only on weights and constants during graph construction (before the first `run()`). This eliminates ConstantOfShape (0.79ms), some Gather ops on weight tensors, and Range/CumSum nodes that produce fixed sequences.

**Expected savings:** ~2-5ms and fewer launches.

### Priority 5: Reduce Transpose calls (~7ms)

89 Transpose calls at 79 us/call. Many of these may be avoidable if the preceding op can output in the desired layout directly. This requires layout-aware operator implementations (e.g., Conv with NHWC output when the next op expects NHWC).

This is a deeper change — ORT does it by tracking "preferred layout" through the graph.

---

## How to Reproduce and Measure

### Setup

The comparison tool lives in `~/workspace/ryanoneill/rust-ai-explorations`:

```bash
cd ~/workspace/ryanoneill/rust-ai-explorations

# Single comparison
cargo run -p leadline-bench --bin compare -- \
  ~/workspace/ryanoneill/claudio/kokoro-v1.0.onnx \
  --iterations 10 --output-dir /tmp

# Sequence length sweep (manual, until --sweep is added)
for n in 5 20 50 100 200; do
  cargo run -p leadline-bench --bin compare -- \
    ~/workspace/ryanoneill/claudio/kokoro-v1.0.onnx \
    --seq-len $n --iterations 3 --output-dir /tmp
done
```

Requires `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64` (set in `~/.zshrc`).

### Output

Produces `.leadline` files (JSON header + protobuf payload) readable by the leadline library:
- `/tmp/iconnx.leadline` — per-operator spans with call counts, total/avg microseconds
- `/tmp/ort.leadline` — envelope timing (total inference duration)

### After making changes to iconnx

1. Rebuild: `cd ~/workspace/ryanoneill/iconnx && cargo build`
2. Re-run comparison: the leadline-bench crate uses iconnx as a git dep, so you'll need to update the dep or temporarily switch to a path dep
3. Compare the operator breakdown — look for eliminated launches and reduced total time
4. The "Speedup" ratio is the headline metric: target >1.0x (iconnx faster than ORT)

### Switching leadline-bench to a local iconnx path dep (for development)

In `~/workspace/ryanoneill/rust-ai-explorations/leadline-bench/Cargo.toml`, temporarily change:
```toml
# From:
iconnx = { git = "https://github.com/ryanoneill/iconnx.git", branch = "main" }
# To:
iconnx = { path = "../../iconnx" }
```

This lets you iterate on iconnx and immediately re-run the comparison without pushing to GitHub.

---

## Architecture Notes for iconnx Optimization

### Where fusion happens today

`src/cuda/inference/mod.rs` lines 253-321 define `FusedPattern` variants:
- `DivRsqrt` — Add -> Sqrt -> Div (65 instances in Kokoro)
- `Gelu` — 8-op chain (12 instances)
- `AddMulAdd`, `MulAdd`, `AddMul`, `SubMul`, `DivMul` — binary chains

Pattern detection runs on first inference call (`patterns_detected` flag). Fused patterns are executed as single NVRTC kernels; their constituent nodes are skipped.

### Where to add new optimizations

1. **Zero-cost reshape:** In `execute_gpu_operator()` (line 1381+), the Reshape/Shape/Unsqueeze/Squeeze cases should return the input tensor with modified metadata instead of launching a kernel.

2. **New fusion patterns:** Add variants to `FusedPattern` enum, add detection logic in the pattern detection pass, add NVRTC kernel generation in the fused kernel cache.

3. **cuDNN LSTM:** Replace the per-gate LSTM implementation with `cudnnRNNForward` using the existing `cudnn_handle`.

### iconnx's profiling API

```rust
// Constructor-time profiling (kernel compilation, cuDNN init)
let executor = GpuGraphExecutor::new_with_profiling(true)?;

// Per-inference profiling
let (outputs, profile) = executor.run_with_profiling(inputs, vec!["audio"])?;
// profile.op_times: BTreeMap<String, (total_us, call_count)>
// profile.input_upload_us, .weight_ref_us, .topo_sort_us, etc.
profile.print_summary();

// Memory pool analysis
let (allocations, hits, hit_rate) = executor.pool_stats();
let top_patterns = executor.pool_allocation_analysis(10);
```

---

## Next Steps

1. **Sequence length sweep** — run at 5/20/50/100/200 tokens to confirm the overhead-vs-compute hypothesis. If the ORT gap narrows at longer sequences, launch overhead is confirmed as the primary bottleneck.

2. **Zero-cost reshape** — quickest win, well-defined scope, ~8ms savings.

3. **Broader elementwise fusion** — biggest potential impact (~30ms) but more engineering effort. Start by logging which elementwise chains exist in the Kokoro graph that aren't currently fused.

4. **cuDNN LSTM** — medium effort, significant savings for this specific model.
