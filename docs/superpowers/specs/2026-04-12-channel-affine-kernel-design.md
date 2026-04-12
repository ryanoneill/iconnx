# Cycle 5: Per-Channel Affine Kernel — Design Spec

**Date:** 2026-04-12
**Predecessor:** Cycles 4 + 4.5 (dynamic AddMulAdd infrastructure + scheduling edges)
**Branch:** `cycle5/channel-affine-kernel`

---

## Goal

Make the 73 dynamic AddMulAdd fusions on Kokoro actually fire at runtime by adding a `fused_channel_affine_kernel` that matches the actual tensor shapes discovered in the Cycle 4.5 diagnostic.

## The actual Kokoro shape pattern

Cycle 4.5's diagnostic revealed that every Add→Mul→Add chain in Kokoro has:

```
x = [1, C, 1]   — per-channel scale (the walker's "x_input", from Add1 input[0])
a = []           — empty (Add1 is identity)
b = [1, C, T]   — full-shape normalized input (the walker's "b_input", from Div)
c = [1, C, 1]   — per-channel bias (the walker's "c_input", from Slice)
```

The math is: `out[i] = scale[i/T] * input[i] + bias[i/T]` — a per-channel affine transform.

## Kernel

```cuda
extern "C" __global__ void fused_channel_affine_kernel(
    float* out,
    const float* input,   // [1, C, T] — full shape
    const float* scale,   // [1, C, 1] — per-channel
    const float* bias,    // [1, C, 1] — per-channel
    size_t n,             // C * T
    size_t T              // trailing dim size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t ch = i / T;
        out[i] = scale[ch] * input[i] + bias[ch];
    }
}
```

## Wrapper dispatch

In `gpu_fused_add_mul_add`, add a new dispatch path BEFORE the FusionShapeMismatch fallback:

**Path 3 (new): Per-channel affine — x and c are `[1,C,1]`, b is `[1,C,T]`, a is empty or scalar.**

Detection logic:
1. `b.shape()` has 3 dims: `[1, C, T]` with `T > 1`
2. `x.shape() == [1, C, 1]` (same C as b, trailing dim 1)
3. `c.shape() == [1, C, 1]` (same as x)
4. `a` is either empty (len 0) or scalar (len 1) — treated as identity

If matched: dispatch `fused_channel_affine_kernel` with `input=b, scale=x, bias=c, n=C*T, T=T`.

The output shape is `b.shape()` (the full `[1, C, T]`).

## Changes

| File | Change |
|---|---|
| `src/cuda/kernels/fused/mod.rs` | Add kernel source + register name |
| `src/cuda/kernels/fused/wrappers.rs` | Add Path 3 detection in `gpu_fused_add_mul_add` |
| `src/cuda/kernels/fused/mod.rs` (tests) | 1 unit test for the kernel |
| `tests/kokoro_gpu_fusion_test.rs` | Verify `Fused_AddMulAdd` appears in op_times |

## Scope

- Add the kernel + dispatch path + 1 GPU test
- Verify on Kokoro: `Fused_AddMulAdd` appears in op_times with ~70 calls
- No changes to walker, executor, or other fusion patterns
- No changes to the existing broadcast-c kernel (it stays for future models that match its pattern)

## Success criteria

1. `Fused_AddMulAdd` appears in op_times on Kokoro with a positive call count
2. `Add` call count drops from 304 to ~234 (304 - 70 = ~234, since each fusion removes 2 Add calls)
3. `Mul` call count drops from 171 to ~101 (171 - 70 = ~101)
4. All existing tests pass
5. Clippy clean
