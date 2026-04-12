# Cycle 4.5: Dynamic AddMulAdd Scheduling Edges — Design Spec

**Date:** 2026-04-11
**Predecessor:** Cycle 4 (dynamic AddMulAdd infrastructure)
**Branch:** `cycle4.5/addmuladd-scheduling-edges`

---

## Goal

Make Cycle 4's dynamic AddMulAdd fusion actually fire at runtime on Kokoro by adding scheduling edges to the topological sort. Currently, 73 dynamic candidates are detected but 0 fuse because the head node (Add1) executes before the Div/Slice nodes that produce b and c. Scheduling edges force the topo sort to place b/c producers before Add1, making all 4 inputs available at the head position so the existing Cycle 4 dispatch fires.

## Problem

The `Add → Mul → Add` chain:
- Add1(x, a) → t0
- Mul(t0, b) → t1   where b = Div_1_output
- Add2(t1, c) → y   where c = Slice_1_output

Add1's native inputs are (x, a). It has no graph-level dependency on b or c. The topological sort freely places Div and Slice after Add1. At execution time, the dynamic dispatch at Add1 can't find b/c in `values` → input-readiness check fails → falls through to unfused.

## Solution

For each dynamic AddMulAdd candidate, add artificial scheduling edges that make Add1 depend on the outputs `b` and `c`. This forces the topo sort to place the nodes producing b (Div) and c (Slice) before Add1. The cycle-safety check is a formality — Div/Slice produce b/c which feed into Mul/Add2, not back into Add1 — but is included as defense-in-depth.

After reordering, the existing Cycle 4 head-triggered dispatch fires: b and c are in `values`, shapes are checked, broadcast-c kernel dispatches, interior nodes (Mul, Add2) are added to `dynamically_fused` and skipped. Net result: 3 kernel launches → 1 fused kernel launch per chain.

## Changes

### 1. Walker: collect scheduling dependencies

When the walker registers a dynamic candidate, it also records the tensor names that must be available before the head can dispatch: `b_input` and `c_input`. These are returned alongside the dynamic candidates as a `Vec<(String, String)>` of (output_name, head_node_name) pairs — "output_name must be produced before head_node_name executes."

### 2. Executor: store and apply scheduling edges

The executor stores the scheduling edges from the walker. Before running the topological sort, it injects these edges into the dependency graph.

### 3. Topological sort: incorporate extra edges

The `topological_sort` method (or the sort function it calls) accepts an optional set of extra edges. For each edge `(output_name, dependent_node_name)`:
- Find the node that produces `output_name`
- Add an artificial dependency: `dependent_node_name` depends on that producer node
- Increment `dependent_node_name`'s in-degree accordingly

If adding an edge would create a cycle (detected by the edge's target already being an ancestor of the edge's source in the dependency graph), skip that edge and don't register the candidate. This is the cycle-safety check.

### 4. No changes to execution dispatch

Cycle 4's dispatch code handles everything once the topo sort places b/c producers first. The input-readiness check passes, the fused kernel dispatches, interior nodes are skipped via `dynamically_fused`.

## Testing

1. **Kokoro integration:** Change `pattern_counts.add_mul_add >= 40` to verify AND check that `Fused_AddMulAdd` appears in `op_times` with a positive call count (proving runtime fusion fires).
2. **Unit test:** Synthetic graph where b/c producers are topologically after the head without scheduling edges, verifying they fire WITH scheduling edges.

## Scope

- Add scheduling edges for dynamic AddMulAdd candidates only
- Cycle-safety check (defense-in-depth)
- No changes to the broadcast-c kernel, wrapper, or FusedPattern types
- No changes to other fusion patterns
- Topological sort change is additive (extra edges parameter, backward compatible)

## Expected impact

73 dynamic AddMulAdd fusions on Kokoro → 73 × 2 = 146 kernel launches eliminated per inference. Each fused launch replaces 3 ops (Add, Mul, Add) with 1 fused kernel. Memory bandwidth savings: 6 loads + 3 stores → 4 loads + 1 store per chain (per-chain ~44% bandwidth reduction for element-wise tensors).
