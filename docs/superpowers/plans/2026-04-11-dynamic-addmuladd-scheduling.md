# Cycle 4.5: Dynamic AddMulAdd Scheduling Edges — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add scheduling edges to the topological sort so Cycle 4's dynamic AddMulAdd dispatch fires at runtime on Kokoro.

**Architecture:** The executor's `compute_topological_sort` reads `self.dynamic_candidates`, extracts b/c input names for each AddMulAdd candidate, and treats them as extra dependencies of the head node. The DFS visit function visits these extra deps alongside the node's native inputs, forcing b/c producers to sort before the head. Cache invalidation in `detect_fused_patterns` ensures the topo sort is recomputed after scheduling edges are known.

**Tech Stack:** Rust, existing iconnx topo sort infrastructure. No new crate dependencies.

---

## Task 1: Add scheduling edges to the topological sort

**Files:**
- Modify: `src/cuda/inference/mod.rs:644-660` (detect method — cache invalidation), `:4776-4860` (compute_topological_sort — extra deps)

- [ ] **Step 1: Compute extra scheduling deps from dynamic_candidates**

In `compute_topological_sort`, after building `output_producers` and before the `visit` function definition, add:

```rust
        // Scheduling edges: for each dynamic AddMulAdd candidate, the head
        // node must execute after the nodes producing b and c. This forces
        // the topo sort to place Div/Slice before Add1.
        let mut extra_deps: HashMap<String, Vec<String>> = HashMap::new();
        for (head_name, info) in self.dynamic_candidates.borrow().iter() {
            if let FusedPattern::AddMulAdd {
                b_input, c_input, ..
            } = &info.pattern
            {
                let mut deps = Vec::new();
                if !b_input.is_empty() && !self.weights.contains_key(b_input) {
                    deps.push(b_input.clone());
                }
                if !c_input.is_empty() && !self.weights.contains_key(c_input) {
                    deps.push(c_input.clone());
                }
                if !deps.is_empty() {
                    extra_deps.insert(head_name.clone(), deps);
                }
            }
        }
```

- [ ] **Step 2: Pass extra_deps to the visit function**

Add `extra_deps: &HashMap<String, Vec<String>>` as a parameter to the inner `visit` function. Thread it through all recursive calls.

- [ ] **Step 3: Visit extra deps in the visit function**

After the existing `for input in &node.inputs` loop, add:

```rust
            // Visit extra scheduling dependencies (fusion-motivated edges)
            if let Some(extra_inputs) = extra_deps.get(&node.name) {
                for input in extra_inputs {
                    if let Some(producer_name) = output_producers.get(input) {
                        if let Some(producer_node) = node_map.get(*producer_name) {
                            visit(
                                producer_node,
                                node_map,
                                output_producers,
                                visited,
                                temp_mark,
                                sorted,
                                weights,
                                extra_deps,
                            )?;
                        }
                    }
                }
            }
```

The cycle-safety check is already handled by `temp_mark` — if visiting an extra dep would create a cycle, the existing `temp_mark.contains(&node.name)` check catches it and returns an error.

- [ ] **Step 4: Invalidate topo sort cache when dynamic candidates change**

In `detect_fused_patterns` (around line 644-660), after `*self.dynamic_candidates.borrow_mut() = dynamic;`, add:

```rust
        // Invalidate topo sort cache — scheduling edges from dynamic
        // candidates may change the execution order.
        if !dynamic.is_empty() {
            *self.sorted_nodes_cache.borrow_mut() = None;
        }
```

Wait — at this point `dynamic` has already been moved into `self.dynamic_candidates`. Check the moved value. Actually, use `self.dynamic_candidates.borrow().is_empty()` instead:

```rust
        if !self.dynamic_candidates.borrow().is_empty() {
            *self.sorted_nodes_cache.borrow_mut() = None;
        }
```

- [ ] **Step 5: Build, test, clippy, commit**

```bash
cargo build --features cuda --tests
cargo test --features cuda 2>&1 | grep -E "^test result"
cargo clippy --features cuda --tests -- -D warnings
git add src/cuda/inference/mod.rs
git commit -S -m "$(cat <<'EOF'
feat: add scheduling edges for dynamic AddMulAdd topo sort ordering

For each dynamic AddMulAdd candidate, the head node gains
artificial dependencies on the tensors b and c (produced by Div
and Slice nodes). This forces the topological sort to place those
producers before the head, making all 4 inputs available at
dispatch time so the existing Cycle 4 head-triggered fusion fires.

Topo sort cache is invalidated when dynamic candidates are
populated to ensure the new edges take effect.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Verify Kokoro runtime fusion + results doc

**Files:**
- Modify: `tests/kokoro_gpu_fusion_test.rs` (if assertion needs updating)
- Create: `docs/performance/2026-04-11-cycle4-dynamic-addmuladd-results.md`

- [ ] **Step 1: Run Kokoro integration test**

```bash
ln -sf /home/ryano/workspace/ryanoneill/claudio/kokoro-v1.0.onnx kokoro-v1.0.onnx
cargo test --features cuda --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | tee /tmp/cycle4.5-kokoro-output.txt | tail -40
```

Expected: `Fused_AddMulAdd` appears in op_times with ~70 calls. `AddMulAdd: 73` in pattern counts.

- [ ] **Step 2: Clean up + commit any test changes + write results doc**

```bash
rm -f kokoro-v1.0.onnx
```

Write `docs/performance/2026-04-11-cycle4-dynamic-addmuladd-results.md` covering Cycles 4 and 4.5 together.

- [ ] **Step 3: Commit signed**
