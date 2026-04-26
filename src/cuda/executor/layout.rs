//! Layout / shape / metadata op dispatch.
//!
//! Default bucket for every op not claimed by another dispatcher — includes
//! Reshape, Transpose, Concat, Slice, Unsqueeze, Squeeze, Shape, Gather,
//! Cast, Pad, Resize, Range, Where, Expand, ConstantOfShape, ScatterND,
//! NonZero, CumSum, Identity.
//!
//! Takes the full [`ExecutionPlan`] because layout ops consume precomputed
//! Slice/Reshape/Unsqueeze/Squeeze parameters (via `op.precomputed`) and
//! may read `plan.static_i64` for initializer-sourced i64 tensors.
//!
//! Dispatch target parity with today's
//! `execute_gpu_operator_with_precomputed` in `src/cuda/inference/mod.rs`:
//! same per-op `gpu_*` wrappers, same attribute defaults, same fallback
//! semantics (precomputed → static cache → D2H). Kernel wrappers stay in
//! their source modules; this file owns only the dispatch match.

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for tensor ops (matches inference/mod.rs).

use std::borrow::Cow;
use std::collections::HashMap;

use crate::cuda::inference::{compute_broadcast_shape, maybe_expand};
use crate::cuda::ops::{
    gpu_cast, gpu_concat, gpu_constant_of_shape_direct, gpu_constant_of_shape_direct_i64, gpu_copy,
    gpu_cumsum, gpu_dequantize_linear, gpu_expand, gpu_gather_from_gpu, gpu_matmul_integer,
    gpu_max_pool_2d,
    gpu_nonzero, gpu_pad, gpu_range, gpu_range_i64, gpu_resize, gpu_scatter_nd, gpu_shape,
    gpu_slice_nd, gpu_transpose_2d, gpu_transpose_nd, gpu_where, ResizeCoordMode, ResizeMode,
    ResizeNearestMode,
};
use crate::cuda::tensor::DType;
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::{ExecutionPlan, PlannedOp};
use crate::tensor::Tensor;

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch a single layout/shape/metadata op to its `gpu_*` wrapper.
    ///
    /// Matches on `op.node.op_type`. Each arm mirrors the corresponding
    /// arm in today's `execute_gpu_operator_with_precomputed`: same kernel
    /// calls, same attribute reads, same precomputed/cache fallback order.
    /// Unknown op_types return a Kernel error.
    pub(crate) fn dispatch_layout(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<GpuTensor, CudaError> {
        let weights = &plan.weights;
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let input_names = &op.node.inputs;
        let attributes = &op.node.attributes;
        let precomputed = &op.precomputed;
        let mut pool = self.memory_pool.borrow_mut();

        match op.node.op_type.as_str() {
            // --- Where with tri-input broadcasting -----------------------
            "Where" => {
                let (condition, x, y) = (inputs[0], inputs[1], inputs[2]);

                // WS-3 M3.5 sub-B: Where consumes native Bool. Comparison
                // ops produce Bool natively now, so the common path is
                // already-Bool. For non-Bool conditions (e.g., a Cast→Bool
                // that hasn't been lowered, or an Int32/Int64 mask passed
                // by the graph author), insert a Cast→Bool. Cow elides the
                // clone in the common case.
                let condition: Cow<'_, GpuTensor> = if condition.dtype() != DType::Bool {
                    Cow::Owned(gpu_cast(
                        &self.ctx,
                        &self.ops_kernels,
                        condition,
                        DType::Bool,
                    )?)
                } else {
                    Cow::Borrowed(condition)
                };

                if condition.shape() == x.shape() && x.shape() == y.shape() {
                    return gpu_where(&self.ctx, &self.ops_kernels, &mut pool, &condition, x, y);
                }

                let shape_cx =
                    compute_broadcast_shape(condition.shape(), x.shape()).ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Where: condition {:?} and x {:?} are not broadcastable",
                            condition.shape(),
                            x.shape()
                        ))
                    })?;
                let broadcast_shape =
                    compute_broadcast_shape(&shape_cx, y.shape()).ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Where: intermediate {:?} and y {:?} are not broadcastable",
                            shape_cx,
                            y.shape()
                        ))
                    })?;

                let cond_expanded = maybe_expand(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    &condition,
                    broadcast_shape.clone(),
                )?;
                let x_expanded = maybe_expand(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    x,
                    broadcast_shape.clone(),
                )?;
                let y_expanded = maybe_expand(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    y,
                    broadcast_shape,
                )?;

                gpu_where(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    &cond_expanded,
                    &x_expanded,
                    &y_expanded,
                )
            }

            // --- Shape: produces Int64, cache result for downstream ops --
            "Shape" => {
                let shape_values: Vec<i64> = inputs[0].shape().iter().map(|&d| d as i64).collect();
                let result = gpu_shape(&self.ctx, inputs[0])?;
                self.cache_i64(&result, shape_values);
                Ok(result)
            }

            // --- Cast: dtype routing via TensorProto enum -----------------
            "Cast" => {
                let target_type = attributes
                    .get_int("to")
                    .ok_or_else(|| CudaError::Kernel("Cast: missing 'to' attribute".into()))?;
                let target_dtype = match target_type {
                    1 => DType::Float32,  // FLOAT
                    6 => DType::Int32,    // INT32
                    7 => DType::Int64,    // INT64
                    9 => DType::Bool,     // BOOL (M3.5 sub-B: native Bool, was Int32 0/1)
                    10 => DType::Float16, // FLOAT16 (M3.5 sub-A)
                    _ => {
                        return Err(CudaError::Kernel(format!(
                            "Cast: unsupported target type {}",
                            target_type
                        )));
                    }
                };
                gpu_cast(&self.ctx, &self.ops_kernels, inputs[0], target_dtype)
            }

            // --- ConstantOfShape: fill tensor of given shape with value ---
            //
            // ONNX spec: the `value` attribute (a scalar tensor) determines
            // both the fill value AND the output dtype. When absent, the
            // spec default is an f32 zero. YOLOS-tiny's ViT embedding
            // interpolation produces ConstantOfShape nodes with an Int64
            // `value` (its input is derived from Shape → …, which is Int64
            // by spec); pre-fix that path hit `Tensor::as_slice()`'s
            // "called on non-Float32 tensor" panic at tensor.rs:296.
            "ConstantOfShape" => {
                let shape_name = input_names.first().map(|s| s.as_str()).unwrap_or("");
                let target_shape: Vec<usize> = self
                    .get_or_fetch_i64(shape_name, inputs[0], plan)?
                    .iter()
                    .map(|&d| d as usize)
                    .collect();

                match attributes.get_tensor("value") {
                    None => {
                        // Spec default: fill with f32 0.
                        gpu_constant_of_shape_direct(
                            &self.ctx,
                            &self.ops_kernels,
                            &mut pool,
                            target_shape,
                            0.0f32,
                        )
                    }
                    Some(Tensor::Float32(_)) => {
                        let value = attributes
                            .get_tensor("value")
                            .map(|t| t.as_slice()[0])
                            .unwrap_or(0.0f32);
                        gpu_constant_of_shape_direct(
                            &self.ctx,
                            &self.ops_kernels,
                            &mut pool,
                            target_shape,
                            value,
                        )
                    }
                    Some(Tensor::Int64(_)) => {
                        let value = attributes
                            .get_tensor("value")
                            .map(|t| t.as_slice_i64()[0])
                            .unwrap_or(0i64);
                        gpu_constant_of_shape_direct_i64(
                            &self.ctx,
                            &self.ops_kernels,
                            &mut pool,
                            target_shape,
                            value,
                        )
                    }
                    Some(other) => Err(CudaError::Kernel(format!(
                        "ConstantOfShape: unsupported `value` dtype {} \
                         (supported: Float32, Int64)",
                        other.dtype()
                    ))),
                }
            }

            // --- Transpose with perm-length repair -----------------------
            "Transpose" => {
                let shape = inputs[0].shape();
                let ndim = shape.len();

                let mut perm: Vec<i64> = if let Some(perm_attr) = attributes.get_ints("perm") {
                    perm_attr.to_vec()
                } else {
                    (0..ndim as i64).rev().collect()
                };

                // Handle perm length mismatch (dynamic shapes / static perm
                // for varying tensor shapes).
                if perm.len() > ndim {
                    perm.retain(|&d| (d as usize) < ndim || (d < 0 && (ndim as i64 + d) >= 0));
                    if perm.len() != ndim {
                        perm = (0..ndim as i64).rev().collect();
                    }
                } else if perm.len() < ndim {
                    for i in perm.len()..ndim {
                        perm.push(i as i64);
                    }
                }

                // 2D fast path (Float32 only); general N-D for everything else.
                if shape.len() == 2 && perm == [1, 0] && inputs[0].dtype() == DType::Float32 {
                    gpu_transpose_2d(&self.ctx, &self.ops_kernels, &mut pool, inputs[0])
                } else {
                    gpu_transpose_nd(&self.ctx, &self.ops_kernels, &mut pool, inputs[0], &perm)
                }
            }

            // --- Gather with transpose-front-gather-transpose-back -------
            "Gather" => {
                let axis_raw = attributes.get_int("axis").unwrap_or(0);
                let ndim = inputs[0].shape().len() as i64;
                let axis = if axis_raw < 0 {
                    (ndim + axis_raw) as usize
                } else {
                    axis_raw as usize
                };
                let indices_shape = inputs[1].shape().to_vec();

                if axis == 0 {
                    gpu_gather_from_gpu(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        inputs[0],
                        inputs[1],
                    )
                } else {
                    let input_shape = inputs[0].shape();
                    let mut perm: Vec<i64> = (0..ndim).collect();
                    perm.remove(axis);
                    perm.insert(0, axis as i64);

                    let transposed = gpu_transpose_nd(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        inputs[0],
                        &perm,
                    )?;

                    let gathered = gpu_gather_from_gpu(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        &transposed,
                        inputs[1],
                    )?;

                    let gathered_shape = gathered.shape();
                    if indices_shape.is_empty() {
                        // Scalar indices: axis dim is removed. Inverse
                        // permutation is already encoded in the gathered
                        // shape; returning gathered directly matches today's
                        // behavior.
                        let mut inv_perm: Vec<i64> = Vec::with_capacity(gathered_shape.len());
                        for (new_pos, &old_pos) in perm.iter().skip(1).enumerate() {
                            let target = if old_pos > axis as i64 {
                                old_pos - 1
                            } else {
                                old_pos
                            };
                            inv_perm.push(target);
                            let _ = new_pos;
                        }
                        Ok(gathered)
                    } else {
                        let indices_dims = indices_shape.len();
                        let remaining_dims = input_shape.len() - 1;

                        let total_dims = indices_dims + remaining_dims;
                        let mut inv_perm = vec![0i64; total_dims];
                        for i in 0..axis {
                            inv_perm[i] = (indices_dims + i) as i64;
                        }
                        for i in 0..indices_dims {
                            inv_perm[axis + i] = i as i64;
                        }
                        for i in axis..remaining_dims {
                            inv_perm[indices_dims + i] = (indices_dims + i) as i64;
                        }

                        gpu_transpose_nd(
                            &self.ctx,
                            &self.ops_kernels,
                            &mut pool,
                            &gathered,
                            &inv_perm,
                        )
                    }
                }
            }

            // --- Slice with precomputed-first param resolution -----------
            "Slice" => {
                if inputs.len() < 3 {
                    return Err(CudaError::Kernel(
                        "Slice requires at least 3 inputs: data, starts, ends".into(),
                    ));
                }

                let slice_pre = precomputed.slice.as_ref();

                let starts = if let Some(precomp) =
                    slice_pre.and_then(|p| p.starts.as_ref())
                {
                    precomp.clone()
                } else {
                    let starts_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(starts_name, inputs[1], plan)?
                };

                let ends =
                    if let Some(precomp) = slice_pre.and_then(|p| p.ends.as_ref()) {
                        precomp.clone()
                    } else {
                        let ends_name = input_names.get(2).map(|s| s.as_str()).unwrap_or("");
                        self.get_or_fetch_i64(ends_name, inputs[2], plan)?
                    };

                if std::env::var("DEBUG_SLICE").is_ok() {
                    let axes_dbg = if inputs.len() > 3 && !inputs[3].is_empty() {
                        let name = input_names.get(3).map(|s| s.as_str()).unwrap_or("");
                        self.get_or_fetch_i64(name, inputs[3], plan).ok()
                    } else {
                        None
                    };
                    let steps_dbg = if inputs.len() > 4 && !inputs[4].is_empty() {
                        let name = input_names.get(4).map(|s| s.as_str()).unwrap_or("");
                        self.get_or_fetch_i64(name, inputs[4], plan).ok()
                    } else {
                        None
                    };
                    eprintln!(
                        "DEBUG Slice: input_shape={:?} starts={:?} ends={:?} axes={:?} steps={:?}",
                        inputs[0].shape(),
                        starts,
                        ends,
                        axes_dbg,
                        steps_dbg
                    );
                }

                // Broadcast / truncate starts ↔ ends to a common length.
                let starts_len = starts.len();
                let ends_len = ends.len();
                let (starts, ends) = if starts_len != ends_len {
                    if starts_len == 1 && ends_len > 1 {
                        (vec![starts[0]; ends_len], ends)
                    } else if ends_len == 1 && starts_len > 1 {
                        (starts, vec![ends[0]; starts_len])
                    } else {
                        let len = starts_len.min(ends_len);
                        (starts[..len].to_vec(), ends[..len].to_vec())
                    }
                } else {
                    (starts, ends)
                };

                let num_slices = starts.len();

                let axes = if let Some(precomp) = slice_pre.and_then(|p| p.axes.as_ref()) {
                    if precomp.len() == num_slices {
                        Some(precomp.clone())
                    } else {
                        None
                    }
                } else if inputs.len() > 3 && !inputs[3].is_empty() {
                    let axes_name = input_names.get(3).map(|s| s.as_str()).unwrap_or("");
                    let axes_data = self.get_or_fetch_i64(axes_name, inputs[3], plan)?;
                    if axes_data.len() == num_slices {
                        Some(axes_data)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let steps = if let Some(precomp) = slice_pre.and_then(|p| p.steps.as_ref()) {
                    if precomp.len() == num_slices {
                        Some(precomp.clone())
                    } else {
                        None
                    }
                } else if inputs.len() > 4 && !inputs[4].is_empty() {
                    let steps_name = input_names.get(4).map(|s| s.as_str()).unwrap_or("");
                    let steps_data = self.get_or_fetch_i64(steps_name, inputs[4], plan)?;
                    if steps_data.len() == num_slices {
                        Some(steps_data)
                    } else {
                        None
                    }
                } else {
                    None
                };

                gpu_slice_nd(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &starts,
                    &ends,
                    axes.as_deref(),
                    steps.as_deref(),
                )
            }

            // --- MaxPool 2-D ---------------------------------------------
            //
            // Resolves attributes into 2-D pooling parameters via the
            // CPU operator's helper (single source of truth) and
            // launches the NVRTC `max_pool_2d_kernel`. ResNet-50 stem
            // uses kernel=3, stride=2, pads=[1,1,1,1].
            "MaxPool" => {
                let p = crate::operators::max_pool::MaxPool::params_2d(attributes)
                    .map_err(CudaError::Kernel)?;
                if op.node.outputs.len() > 1 {
                    return Err(CudaError::Kernel(
                        "MaxPool: optional Indices output not supported".into(),
                    ));
                }
                gpu_max_pool_2d(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    p.kernel_h,
                    p.kernel_w,
                    p.stride_h,
                    p.stride_w,
                    p.pad_top,
                    p.pad_left,
                    p.pad_bottom,
                    p.pad_right,
                    p.dilation_h,
                    p.dilation_w,
                    p.ceil_mode,
                )
            }

            // --- Flatten: pure shape relabel into 2-D --------------------
            //
            // axis=1 by default; negative values count from the rear.
            // Output shape is `[product(0..axis), product(axis..ndim)]`.
            // No data movement — same memcpy-class semantics as Reshape,
            // implemented as a `GpuTensor::reshape` clone of the input.
            "Flatten" => {
                let input = inputs[0];
                let input_shape = input.shape();
                let ndim = input_shape.len();
                let axis_attr = attributes.get_int("axis").unwrap_or(1);
                let axis = if axis_attr < 0 {
                    ((ndim as i64 + axis_attr).max(0) as usize).min(ndim)
                } else {
                    (axis_attr as usize).min(ndim)
                };
                let outer: usize = input_shape[..axis].iter().product::<usize>().max(1);
                let inner: usize = input_shape[axis..].iter().product::<usize>().max(1);
                let target = vec![outer, inner];
                input.clone().reshape(target.clone()).ok_or_else(|| {
                    CudaError::Kernel(format!(
                        "Flatten: element count mismatch. Input shape {:?} cannot reshape to {:?}",
                        input_shape, target
                    ))
                })
            }

            // --- Reshape with 0/-1 resolution ----------------------------
            "Reshape" => {
                let input_shape = inputs[0].shape();
                let input_len = inputs[0].len();

                let shape_data: Vec<i64> = if let Some(precomp) = precomputed.reshape_shape.as_ref()
                {
                    precomp.clone()
                } else if let Some(shape_attr) = attributes.get_ints("shape") {
                    shape_attr.to_vec()
                } else if inputs.len() > 1 {
                    let shape_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(shape_name, inputs[1], plan)?
                } else {
                    return Err(CudaError::Kernel(
                        "Reshape requires shape attribute or second input".into(),
                    ));
                };

                // ONNX Reshape semantics: 0 = copy from input, -1 = infer.
                let mut new_shape: Vec<usize> = Vec::with_capacity(shape_data.len());
                let mut neg_one_idx: Option<usize> = None;
                let mut known_product = 1usize;

                for (i, &dim) in shape_data.iter().enumerate() {
                    let resolved = if dim == 0 {
                        if i < input_shape.len() {
                            input_shape[i]
                        } else {
                            1
                        }
                    } else if dim == -1 {
                        neg_one_idx = Some(i);
                        1
                    } else {
                        dim as usize
                    };
                    if dim != -1 {
                        known_product *= resolved;
                    }
                    new_shape.push(resolved);
                }

                if let Some(idx) = neg_one_idx {
                    if let Some(quot) = input_len.checked_div(known_product) {
                        new_shape[idx] = quot;
                    }
                }

                inputs[0].clone().reshape(new_shape.clone()).ok_or_else(|| {
                    let new_len: usize = new_shape.iter().product();
                    CudaError::Kernel(format!(
                        "Reshape: element count mismatch. Input has {} elements (shape {:?}), target shape {:?} has {} elements. Raw shape_data: {:?}",
                        input_len, input_shape, new_shape, new_len, shape_data
                    ))
                })
            }

            // --- Squeeze: removes dims of size 1 -------------------------
            "Squeeze" => {
                let axes: Option<Vec<i64>> = if let Some(precomp) =
                    precomputed.squeeze_axes.as_ref()
                {
                    Some(precomp.clone())
                } else if let Some(axes_attr) = attributes.get_ints("axes") {
                    Some(axes_attr.to_vec())
                } else if inputs.len() > 1 {
                    let axes_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    Some(self.get_or_fetch_i64(axes_name, inputs[1], plan)?)
                } else {
                    None
                };

                let old_shape = inputs[0].shape();
                let new_shape: Vec<usize> = if let Some(axes) = axes {
                    let ndim = old_shape.len() as i64;
                    let axes_set: std::collections::HashSet<usize> = axes
                        .iter()
                        .map(|&a| if a < 0 { (ndim + a) as usize } else { a as usize })
                        .collect();
                    old_shape
                        .iter()
                        .enumerate()
                        .filter(|(i, &dim)| !(dim == 1 && axes_set.contains(i)))
                        .map(|(_, &dim)| dim)
                        .collect()
                } else {
                    old_shape.iter().copied().filter(|&d| d != 1).collect()
                };
                inputs[0]
                    .clone()
                    .reshape(new_shape)
                    .ok_or_else(|| CudaError::Kernel("Squeeze: element count mismatch".into()))
            }

            // --- Unsqueeze: inserts dims of size 1 -----------------------
            "Unsqueeze" => {
                let axes: Vec<i64> = if let Some(precomp) = precomputed.unsqueeze_axes.as_ref() {
                    precomp.clone()
                } else if let Some(axes_attr) = attributes.get_ints("axes") {
                    axes_attr.to_vec()
                } else if inputs.len() > 1 {
                    let axes_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(axes_name, inputs[1], plan)?
                } else {
                    return Err(CudaError::Kernel(
                        "Unsqueeze requires 'axes' attribute or second input tensor".into(),
                    ));
                };

                let old_shape = inputs[0].shape();
                let new_ndim = old_shape.len() + axes.len();
                let mut new_shape = vec![1usize; new_ndim];

                let mut axes_sorted: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (new_ndim as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                axes_sorted.sort();

                let mut old_idx = 0;
                for i in 0..new_ndim {
                    if !axes_sorted.contains(&i) {
                        new_shape[i] = old_shape[old_idx];
                        old_idx += 1;
                    }
                }

                inputs[0]
                    .clone()
                    .reshape(new_shape)
                    .ok_or_else(|| CudaError::Kernel("Unsqueeze: element count mismatch".into()))
            }

            // --- Identity: GPU copy --------------------------------------
            "Identity" => gpu_copy(&self.ctx, &self.ops_kernels, &mut pool, inputs[0]),

            // --- Concat / Expand ----------------------------------------
            "Concat" => {
                let axis = attributes.get_int("axis").unwrap_or(0);
                gpu_concat(&self.ctx, &self.ops_kernels, &mut pool, &inputs, axis)
            }
            "Expand" => {
                let out_shape: Vec<usize> = if let Some(shape_attr) = attributes.get_ints("shape") {
                    shape_attr.iter().map(|&x| x as usize).collect()
                } else if inputs.len() > 1 {
                    let shape_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(shape_name, inputs[1], plan)?
                        .iter()
                        .map(|&x| x as usize)
                        .collect()
                } else {
                    return Err(CudaError::Kernel(
                        "Expand requires shape attribute or second input".into(),
                    ));
                };
                gpu_expand(&self.ctx, &self.ops_kernels, &mut pool, inputs[0], out_shape)
            }

            // --- Pad with constant/reflect/edge modes --------------------
            "Pad" => {
                let pads: Vec<i64> = if let Some(attr_pads) = attributes.get_ints("pads") {
                    attr_pads.to_vec()
                } else if inputs.len() >= 2 {
                    let pads_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(pads_name, inputs[1], plan)?
                } else {
                    return Err(CudaError::Kernel(
                        "Pad requires 'pads' attribute or input".into(),
                    ));
                };
                let value = if inputs.len() > 2 {
                    let constant_data = inputs[2].to_host_f32(&self.ctx)?;
                    constant_data.first().copied().unwrap_or(0.0)
                } else {
                    attributes.get_float("value").unwrap_or(0.0)
                };
                let mode = attributes.get_string("mode").unwrap_or("constant");
                gpu_pad(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &pads,
                    value,
                    mode,
                )
            }

            // --- Resize with coord/nearest/interp modes ------------------
            //
            // ONNX Resize opset 13+: inputs = [X, roi, scales, sizes].
            // Exactly one of `scales` / `sizes` is populated; the other
            // is a 0-element placeholder. Older opsets (10/11) take
            // [X, scales] (2 inputs). When sizes is provided, we derive
            // `scales[i] = sizes[i] / input.shape[i]` so the kernel
            // path stays unchanged (all interp math is driven by scale
            // factors; `round(input_shape * derived_scale)` recovers
            // `sizes` exactly within f32 ulp).
            "Resize" => {
                let inp_shape = inputs[0].shape();
                let ndim = inp_shape.len();

                // Locate scales / sizes inputs by arity. iconnx preserves
                // empty-name placeholders as 0-element tensors (WS-1 M1.2
                // zero-element tensor fix), so the 4-input opset-13 shape
                // is always visible here when the graph uses it.
                let (scales_idx, sizes_idx) = match inputs.len() {
                    2 => (Some(1), None),          // [X, scales]
                    3 => (Some(2), None),          // [X, roi, scales]
                    4 => (Some(2), Some(3)),       // [X, roi, scales, sizes]
                    n => {
                        return Err(CudaError::Kernel(format!(
                            "Resize: expected 2-4 inputs, got {}",
                            n
                        )));
                    }
                };

                let scales_from_input: Vec<f32> = match scales_idx {
                    Some(i) => inputs[i].to_host_f32(&self.ctx)?,
                    None => Vec::new(),
                };
                let sizes_from_input: Vec<i64> = match sizes_idx {
                    Some(i) => inputs[i].to_host_i64(&self.ctx)?,
                    None => Vec::new(),
                };

                // ONNX spec: exactly one of scales / sizes matches ndim;
                // the other must be empty. Fail-fast on any deviation
                // rather than silently picking one.
                let scales: Vec<f32> = match (scales_from_input.len(), sizes_from_input.len()) {
                    (s, 0) if s == ndim => scales_from_input,
                    (0, z) if z == ndim => inp_shape
                        .iter()
                        .zip(sizes_from_input.iter())
                        .map(|(&dim, &size)| size as f32 / dim as f32)
                        .collect(),
                    (0, 0) => {
                        return Err(CudaError::Kernel(
                            "Resize: both scales and sizes are empty; \
                             exactly one must be provided per ONNX spec"
                                .into(),
                        ));
                    }
                    (s, z) if s > 0 && z > 0 => {
                        return Err(CudaError::Kernel(format!(
                            "Resize: both scales (len {}) and sizes (len {}) \
                             are non-empty; exactly one must be per ONNX spec",
                            s, z
                        )));
                    }
                    (s, z) => {
                        return Err(CudaError::Kernel(format!(
                            "Resize: neither scales (len {}) nor sizes (len {}) \
                             matches input ndim {}",
                            s, z, ndim
                        )));
                    }
                };

                if std::env::var("DEBUG_RESIZE").is_ok() {
                    let out_shape: Vec<usize> = inp_shape
                        .iter()
                        .zip(scales.iter())
                        .map(|(&dim, &scale)| (dim as f32 * scale).round() as usize)
                        .collect();
                    eprintln!(
                        "DEBUG Resize: input={:?} scales={:?} -> output={:?}",
                        inp_shape, scales, out_shape
                    );
                }

                let coord_mode_str = attributes
                    .get_string("coordinate_transformation_mode")
                    .unwrap_or("half_pixel");
                let coord_mode = match coord_mode_str {
                    "asymmetric" => ResizeCoordMode::Asymmetric,
                    _ => ResizeCoordMode::HalfPixel,
                };

                let nearest_mode_str = attributes
                    .get_string("nearest_mode")
                    .unwrap_or("round_prefer_floor");
                let nearest_mode = match nearest_mode_str {
                    "floor" => ResizeNearestMode::Floor,
                    _ => ResizeNearestMode::RoundPreferFloor,
                };

                let mode_str = attributes.get_string("mode").unwrap_or("nearest");
                let mode = match mode_str {
                    "linear" => ResizeMode::Linear,
                    _ => ResizeMode::Nearest,
                };

                gpu_resize(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &scales,
                    coord_mode,
                    nearest_mode,
                    mode,
                )
            }

            // --- NonZero -------------------------------------------------
            "NonZero" => gpu_nonzero(&self.ctx, inputs[0]),

            // --- ScatterND -----------------------------------------------
            "ScatterND" => {
                if inputs.len() < 3 {
                    return Err(CudaError::Kernel(
                        "ScatterND requires 3 inputs: data, indices, updates".into(),
                    ));
                }
                let indices_host = inputs[1].to_host_i64(&self.ctx)?;
                let indices_shape = inputs[1].shape().to_vec();
                gpu_scatter_nd(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &indices_host,
                    inputs[2],
                    &indices_shape,
                )
            }

            // --- Range (Float32 and Int64) -------------------------------
            "Range" => {
                if inputs.len() != 3 {
                    return Err(CudaError::Kernel(format!(
                        "Range requires 3 inputs, got {}",
                        inputs.len()
                    )));
                }
                let dtype = inputs[0].dtype();
                match dtype {
                    DType::Float32 => {
                        let start_data = inputs[0].to_host_f32(&self.ctx)?;
                        let limit_data = inputs[1].to_host_f32(&self.ctx)?;
                        let delta_data = inputs[2].to_host_f32(&self.ctx)?;
                        let start = start_data[0];
                        let limit = limit_data[0];
                        let delta = delta_data[0];
                        let n = ((limit - start) / delta).ceil().max(0.0) as usize;

                        if std::env::var("DEBUG_SHAPES").is_ok() {
                            eprintln!(
                                "DEBUG_RANGE_F32: start={}, limit={}, delta={}, n={}",
                                start, limit, delta, n
                            );
                        }

                        gpu_range(&self.ctx, &self.ops_kernels, start, delta, n)
                    }
                    DType::Int64 => {
                        let start_name = input_names.first().map(|s| s.as_str()).unwrap_or("");
                        let start = self.get_or_fetch_i64(start_name, inputs[0], plan)?[0];
                        let limit_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                        let limit = self.get_or_fetch_i64(limit_name, inputs[1], plan)?[0];
                        let delta_name = input_names.get(2).map(|s| s.as_str()).unwrap_or("");
                        let delta = self.get_or_fetch_i64(delta_name, inputs[2], plan)?[0];

                        if std::env::var("DEBUG_SHAPES").is_ok() {
                            eprintln!(
                                "DEBUG_RANGE_I64: start={}, limit={}, delta={}",
                                start, limit, delta
                            );
                        }

                        let n = if delta == 0 {
                            return Err(CudaError::Kernel("Range: delta cannot be 0".into()));
                        } else if delta > 0 {
                            if limit <= start {
                                0
                            } else {
                                let diff = limit - start;
                                ((diff + delta - 1) / delta) as usize
                            }
                        } else if limit >= start {
                            0
                        } else {
                            let diff = start - limit;
                            let abs_delta = -delta;
                            ((diff + abs_delta - 1) / abs_delta) as usize
                        };
                        gpu_range_i64(&self.ctx, &self.ops_kernels, start, delta, n)
                    }
                    _ => Err(CudaError::Kernel(format!(
                        "Range: unsupported dtype {}",
                        dtype.name()
                    ))),
                }
            }

            // --- CumSum --------------------------------------------------
            "CumSum" => {
                let exclusive = attributes.get_int("exclusive").unwrap_or(0) != 0;
                let reverse = attributes.get_int("reverse").unwrap_or(0) != 0;

                let axis = if inputs.len() > 1 {
                    let axis_tensor = inputs[1];
                    if axis_tensor.is_i64() {
                        axis_tensor.to_host_i64(&self.ctx)?[0]
                    } else if axis_tensor.is_i32() {
                        axis_tensor.to_host_i32(&self.ctx)?[0] as i64
                    } else {
                        return Err(CudaError::Kernel(
                            "CumSum: axis must be int64 or int32".to_string(),
                        ));
                    }
                } else {
                    -1
                };

                gpu_cumsum(
                    &self.ctx,
                    &self.ops_kernels,
                    inputs[0],
                    axis,
                    exclusive,
                    reverse,
                )
            }

            // --- DequantizeLinear (WS-4 M4.4) ----------------------------
            //
            // y = (x - zero_point) * scale; output is FP32. `scale` is FP32
            // (scalar = per-tensor; 1-D = per-axis). `zero_point` (input 2)
            // is optional and shares dtype with `x`. `axis` defaults to 1
            // and is ignored when `scale` is scalar.
            "DequantizeLinear" => {
                if inputs.len() < 2 || inputs.len() > 3 {
                    return Err(CudaError::Kernel(format!(
                        "DequantizeLinear requires 2 or 3 inputs, got {}",
                        inputs.len()
                    )));
                }
                let x = inputs[0];
                let scale = inputs[1];
                let zp = inputs.get(2).copied();

                // Resolve `axis` — default 1, normalize negatives modulo
                // ndim. Only meaningful in the per-axis case; the wrapper
                // ignores it when `scale` is scalar.
                let raw_axis = attributes.get_int("axis").unwrap_or(1);
                let ndim = x.ndim() as i64;
                let resolved = if raw_axis < 0 {
                    raw_axis + ndim
                } else {
                    raw_axis
                };
                if !(0..ndim).contains(&resolved) {
                    return Err(CudaError::Kernel(format!(
                        "DequantizeLinear: axis {} out of range for ndim {}",
                        raw_axis, ndim
                    )));
                }
                let axis = resolved as usize;

                gpu_dequantize_linear(&self.ctx, &self.ops_kernels, &mut pool, x, scale, zp, axis)
            }

            // --- MatMulInteger (WS-4 M4.6 + M4.7 batched) ----------------
            //
            // out = matmul((a - a_zp), (b - b_zp)), output Int32. The
            // backing kernel is 2-D only; `a` is `[M, K]`, `b` is `[K, N]`;
            // each operand is Int8 or UInt8 (4 sign-combination kernel
            // variants). `a_zero_point` (input 2) and `b_zero_point` (input
            // 3) are optional and must be scalar (rank 0 or single-element
            // 1-D). Per-axis zp is rejected — DistilBERT does not exercise
            // that path and the M4.6 GPU kernel ABI is scalar-zp only.
            //
            // M4.7 surfaced 3-D × 2-D inputs from DistilBERT-INT8
            // (`[B, M, K] × [K, N] → [B, M, N]`). Mirrors the regular
            // `MatMul` executor's 3-D × 2-D handling in
            // `src/cuda/executor/matmul.rs`: reshape A `[B, M, K]` →
            // `[B*M, K]` (metadata-only on `GpuTensor`), run the existing
            // 2-D kernel, reshape result `[B*M, N]` → `[B, M, N]`. No new
            // kernel, no new math — just dimension flatten/restore at the
            // wrapper boundary.
            //
            // Routed through `dispatch_layout` rather than `dispatch_matmul`
            // because every other quantization op (DequantizeLinear,
            // DynamicQuantizeLinear) lives here, and the wrapper boundary
            // is `gpu_matmul_integer` in `src/cuda/ops/quantize.rs` —
            // consistent with the rest of the WS-4 op family.
            "MatMulInteger" => {
                if inputs.len() < 2 || inputs.len() > 4 {
                    return Err(CudaError::Kernel(format!(
                        "MatMulInteger requires 2, 3, or 4 inputs, got {}",
                        inputs.len()
                    )));
                }
                let a = inputs[0];
                let b = inputs[1];
                let a_zp = inputs.get(2).copied();
                let b_zp = inputs.get(3).copied();

                let a_ndim = a.ndim();
                let b_ndim = b.ndim();
                match (a_ndim, b_ndim) {
                    (2, 2) => gpu_matmul_integer(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        a,
                        b,
                        a_zp,
                        b_zp,
                    ),
                    (3, 2) => {
                        // [B, M, K] × [K, N] → [B, M, N]. Reshape A to
                        // [B*M, K], 2-D matmul, reshape result back.
                        let a_shape = a.shape();
                        let b_shape = b.shape();
                        let batch = a_shape[0];
                        let m = a_shape[1];
                        let k = a_shape[2];
                        let n = b_shape[1];

                        let a_reshaped =
                            a.clone().reshape(vec![batch * m, k]).ok_or_else(|| {
                                CudaError::Kernel(
                                    "MatMulInteger: failed to reshape A to 2-D".into(),
                                )
                            })?;

                        let result_2d = gpu_matmul_integer(
                            &self.ctx,
                            &self.ops_kernels,
                            &mut pool,
                            &a_reshaped,
                            b,
                            a_zp,
                            b_zp,
                        )?;

                        result_2d.reshape(vec![batch, m, n]).ok_or_else(|| {
                            CudaError::Kernel(
                                "MatMulInteger: failed to reshape result to 3-D".into(),
                            )
                        })
                    }
                    _ => Err(CudaError::Kernel(format!(
                        "MatMulInteger: unsupported ranks A={}D, B={}D \
                         (M4.7 supports 2×2 and 3×2)",
                        a_ndim, b_ndim
                    ))),
                }
            }

            other => Err(CudaError::Kernel(format!(
                "dispatch_layout called with non-layout op '{}'",
                other
            ))),
        }
    }

    /// Get Int64 values by name, falling back to D2H transfer if not cached.
    ///
    /// Lookup order mirrors today's `Inference::get_or_fetch_i64`:
    /// 1. `plan.static_i64` — per-plan cache populated at lowering time
    ///    from initializers and Constant-node values.
    /// 2. D2H transfer — reads the tensor contents from the device.
    ///
    /// Intentionally does NOT cache D2H results: dynamic values (e.g. Shape
    /// op outputs with varying sequence lengths) would produce stale data
    /// across runs. The per-run `i64_cache` (keyed by device pointer) is
    /// populated only by `Shape` via `cache_i64` and consumed implicitly
    /// through ownership of `GpuTensor` identity — matching today's code.
    fn get_or_fetch_i64(
        &self,
        name: &str,
        tensor: &GpuTensor,
        plan: &ExecutionPlan,
    ) -> Result<Vec<i64>, CudaError> {
        if let Some(cached) = plan.static_i64.get(name) {
            return Ok(cached.clone());
        }
        tensor.to_host_i64(&self.ctx)
    }

    /// Cache Int64 values for a tensor by device pointer (used within a
    /// single run). Written by the `Shape` arm to short-circuit downstream
    /// Reshape / Unsqueeze / Squeeze D2H reads — parity with today's
    /// `Inference::cache_i64`.
    fn cache_i64(&self, tensor: &GpuTensor, values: Vec<i64>) {
        let ptr = tensor.device_ptr(&self.ctx);
        self.i64_cache.borrow_mut().insert(ptr, values);
    }

    /// Dispatch ONNX Split: produce N distinct output tensors from one
    /// input. Split is the first true multi-output op iconnx supports —
    /// the LSTM "multi-output" case is single-tensor-aliased — so this
    /// returns `Vec<GpuTensor>`, one entry per `node.outputs` slot.
    ///
    /// Resolves split sizes by the ONNX-spec priority: (1) explicit
    /// `split` Int64 input at slot 1 if non-empty; (2) `num_outputs`
    /// attribute (opset 18+); (3) the count of `node.outputs`. Sizes are
    /// then handed to `gpu_split` which performs N parallel
    /// `gpu_slice_nd` operations along the resolved axis.
    pub(crate) fn dispatch_split(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<Vec<GpuTensor>, CudaError> {
        let weights = &plan.weights;
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let attributes = &op.node.attributes;

        if inputs.is_empty() {
            return Err(CudaError::Kernel("Split: missing input tensor".into()));
        }
        let input = inputs[0];
        let ndim = input.ndim();
        let axis_attr = attributes.get_int("axis").unwrap_or(0);
        let axis = if axis_attr < 0 {
            (ndim as i64 + axis_attr).max(0) as usize
        } else {
            (axis_attr as usize).min(ndim.saturating_sub(1))
        };
        let axis_size = input.shape()[axis];

        // ONNX-spec resolution order for split sizes.
        let sizes: Vec<usize> = if let Some(split_tensor) = inputs.get(1) {
            // Explicit split input. Empty placeholder (length 0) means
            // "fall through to num_outputs / hint" per the same
            // empty-placeholder convention as Resize's scales/sizes.
            let raw = split_tensor.to_host_i64(&self.ctx)?;
            if !raw.is_empty() {
                raw.into_iter().map(|s| s as usize).collect()
            } else {
                self.split_sizes_from_count_hint(attributes, axis_size, op)
            }
        } else {
            self.split_sizes_from_count_hint(attributes, axis_size, op)
        };

        let total: usize = sizes.iter().sum();
        if total != axis_size {
            return Err(CudaError::Kernel(format!(
                "Split: split sizes sum {} != input axis {} size {}",
                total, axis, axis_size
            )));
        }

        let mut pool = self.memory_pool.borrow_mut();
        crate::cuda::ops::gpu_split(
            &self.ctx,
            &self.ops_kernels,
            &mut pool,
            input,
            axis as i64,
            &sizes,
        )
    }

    /// GPU dispatch for `DynamicQuantizeLinear` (WS-4 M4.5). Multi-output
    /// op like `Split` — returns three tensors `(y, y_scale, y_zero_point)`
    /// stored under their own names by `store_outputs`'s distinct-multi
    /// path. Routed through `dispatch_op_multi`, never single-output
    /// `dispatch_op`.
    pub(crate) fn dispatch_dynamic_quantize_linear(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<Vec<GpuTensor>, CudaError> {
        let weights = &plan.weights;
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        if inputs.is_empty() {
            return Err(CudaError::Kernel(
                "DynamicQuantizeLinear: missing input tensor".into(),
            ));
        }
        // Host-quantize MVP — wrapper neither launches a kernel nor touches
        // the pool, so we don't borrow either. See gpu_dynamic_quantize_linear
        // for the f64-parity rationale.
        crate::cuda::ops::gpu_dynamic_quantize_linear(&self.ctx, inputs[0])
    }

    /// Resolve split sizes when no explicit `split` input was provided.
    /// Falls back to `num_outputs` (opset 18+) then to the number of
    /// declared graph outputs. Even split with the remainder appended to
    /// the last chunk — matches ONNX Reference op semantics.
    fn split_sizes_from_count_hint(
        &self,
        attributes: &crate::attributes::NodeAttributes,
        axis_size: usize,
        op: &PlannedOp,
    ) -> Vec<usize> {
        let n = attributes
            .get_int("num_outputs")
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                op.node
                    .outputs
                    .iter()
                    .filter(|name| !name.is_empty())
                    .count()
            })
            .max(1);
        let chunk = axis_size / n;
        let remainder = axis_size % n;
        let mut sizes = vec![chunk; n];
        if let Some(last) = sizes.last_mut() {
            *last += remainder;
        }
        sizes
    }
}
