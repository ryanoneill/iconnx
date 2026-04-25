//! GPU Memory Pool for efficient tensor allocation.
//!
//! Reduces allocation overhead by reusing GPU memory blocks. Tracks freed
//! memory by (dtype, size) and returns pooled memory when available,
//! falling back to fresh allocation when not.
//!
//! Pooled buffers are stored as garboard `DeviceSlice<'static, T>`.

use std::collections::HashMap;
use std::sync::Arc;

use garboard::DeviceSlice;

use super::context::{CudaError, IconnxCudaContext};
use super::tensor::{GpuTensor, TypedSlice};

/// Compute size class for pool bucket lookup.
///
/// Rounds up to power-of-2 boundaries to improve hit rates. Similar-sized
/// allocations share the same bucket, reducing fragmentation.
///
/// Important: Both GET and RETURN must use this consistently:
/// - GET allocates `size_class(len)` elements, not `len`.
/// - RETURN uses `size_class(slice.len())` as bucket key.
///
/// Note: We only apply size-class bucketing for large allocations (>= 4096)
/// to avoid issues with small tensor operations that may be sensitive to
/// exact sizes. For large allocations, the overhead percentage is small
/// anyway.
#[inline]
fn size_class(len: usize) -> usize {
    // Use power-of-2 bucketing for allocations >= 4096 elements. This
    // improves pool hit rates by grouping similar-sized allocations.
    // Small allocations use exact matching to avoid excessive memory
    // overhead.
    if len < 4096 {
        len
    } else {
        len.next_power_of_two()
    }
}

/// GPU memory pool for reusing allocations.
///
/// Maintains separate pools for each data type (f32, i64, i32). Memory is
/// keyed by size-class bucket (power-of-2 rounding) for better reuse.
pub struct GpuMemoryPool {
    /// Pool of f32 slices, keyed by size-class bucket.
    f32_pool: HashMap<usize, Vec<DeviceSlice<'static, f32>>>,
    /// Pool of i64 slices, keyed by size-class bucket.
    i64_pool: HashMap<usize, Vec<DeviceSlice<'static, i64>>>,
    /// Pool of i32 slices, keyed by size-class bucket.
    i32_pool: HashMap<usize, Vec<DeviceSlice<'static, i32>>>,
    /// Pool of i8 slices, keyed by size-class bucket.
    i8_pool: HashMap<usize, Vec<DeviceSlice<'static, i8>>>,
    /// Pool of u8 slices, keyed by size-class bucket.
    u8_pool: HashMap<usize, Vec<DeviceSlice<'static, u8>>>,
    /// Number of successful pool hits.
    hits: usize,
    /// Number of pool misses (fresh allocations).
    misses: usize,
    /// Maximum slices to keep per size bucket (prevents unbounded growth).
    max_per_bucket: usize,
    /// Diagnostic: count of allocations per size (for analysis).
    alloc_counts: HashMap<usize, usize>,
    /// Diagnostic: count of returns per size.
    return_counts: HashMap<usize, usize>,
    /// Diagnostic: count of failed returns (Arc refcount > 1).
    failed_returns: usize,
}

impl GpuMemoryPool {
    /// Create a new empty memory pool.
    pub fn new() -> Self {
        Self {
            f32_pool: HashMap::new(),
            i64_pool: HashMap::new(),
            i32_pool: HashMap::new(),
            i8_pool: HashMap::new(),
            u8_pool: HashMap::new(),
            hits: 0,
            misses: 0,
            max_per_bucket: 8, // Keep up to 8 slices per size.
            alloc_counts: HashMap::new(),
            return_counts: HashMap::new(),
            failed_returns: 0,
        }
    }

    /// Get or allocate a Float32 `DeviceSlice`.
    ///
    /// Returns a pooled slice if available, otherwise allocates fresh.
    /// Note: The returned slice may be larger than `len` due to
    /// size-class bucketing. Pooled slices are zeroed before being
    /// returned to ensure consistent behavior.
    pub fn get_f32(
        &mut self,
        ctx: &IconnxCudaContext,
        len: usize,
    ) -> Result<DeviceSlice<'static, f32>, CudaError> {
        let bucket = size_class(len);
        *self.alloc_counts.entry(len).or_insert(0) += 1;
        if let Some(slices) = self.f32_pool.get_mut(&bucket) {
            // Find a slice that has at least `len` elements. This
            // handles the case where different sizes hash to the same
            // bucket.
            if let Some(idx) = slices.iter().position(|s| s.len() >= len) {
                let mut slice = slices.swap_remove(idx);
                self.hits += 1;
                // Zero the slice to ensure consistent behavior with
                // fresh allocations.
                ctx.zero_f32(&mut slice)?;
                return Ok(slice);
            }
        }
        self.misses += 1;
        // Allocate bucket size to ensure consistency with return_f32.
        ctx.alloc_zeros::<f32>(bucket)
    }

    /// Get or allocate a Float32 `GpuTensor` (convenience method).
    pub fn get_tensor_f32(
        &mut self,
        ctx: &IconnxCudaContext,
        shape: Vec<usize>,
    ) -> Result<GpuTensor, CudaError> {
        let len: usize = shape.iter().product();
        let slice = self.get_f32(ctx, len)?;
        Ok(GpuTensor::Float32 {
            data: Arc::new(slice),
            shape,
        })
    }

    /// Get or allocate an Int64 `DeviceSlice`.
    ///
    /// Note: The returned slice may be larger than `len` due to
    /// size-class bucketing. Pooled slices are zeroed before being
    /// returned to ensure consistent behavior.
    pub fn get_i64(
        &mut self,
        ctx: &IconnxCudaContext,
        len: usize,
    ) -> Result<DeviceSlice<'static, i64>, CudaError> {
        let bucket = size_class(len);
        if let Some(slices) = self.i64_pool.get_mut(&bucket) {
            // Find a slice that has at least `len` elements.
            if let Some(idx) = slices.iter().position(|s| s.len() >= len) {
                let mut slice = slices.swap_remove(idx);
                self.hits += 1;
                // Zero the slice to ensure consistent behavior with
                // fresh allocations.
                ctx.zero_i64(&mut slice)?;
                // Sync to ensure zeroing completes before use.
                ctx.sync()?;
                return Ok(slice);
            }
        }
        self.misses += 1;
        // Allocate bucket size to ensure consistency with return_i64.
        ctx.alloc_zeros::<i64>(bucket)
    }

    /// Get or allocate an Int64 `GpuTensor` (convenience method).
    pub fn get_tensor_i64(
        &mut self,
        ctx: &IconnxCudaContext,
        shape: Vec<usize>,
    ) -> Result<GpuTensor, CudaError> {
        let len: usize = shape.iter().product();
        let slice = self.get_i64(ctx, len)?;
        Ok(GpuTensor::Int64 {
            data: Arc::new(slice),
            shape,
        })
    }

    /// Get or allocate an Int32 `DeviceSlice`.
    ///
    /// Note: The returned slice may be larger than `len` due to
    /// size-class bucketing. Pooled slices are zeroed before being
    /// returned to ensure consistent behavior.
    pub fn get_i32(
        &mut self,
        ctx: &IconnxCudaContext,
        len: usize,
    ) -> Result<DeviceSlice<'static, i32>, CudaError> {
        let bucket = size_class(len);
        if let Some(slices) = self.i32_pool.get_mut(&bucket) {
            // Find a slice that has at least `len` elements.
            if let Some(idx) = slices.iter().position(|s| s.len() >= len) {
                let mut slice = slices.swap_remove(idx);
                self.hits += 1;
                // Zero the slice to ensure consistent behavior with
                // fresh allocations.
                ctx.zero_i32(&mut slice)?;
                return Ok(slice);
            }
        }
        self.misses += 1;
        // Allocate bucket size to ensure consistency with return_i32.
        ctx.alloc_zeros::<i32>(bucket)
    }

    /// Get or allocate an Int32 `GpuTensor` (convenience method).
    pub fn get_tensor_i32(
        &mut self,
        ctx: &IconnxCudaContext,
        shape: Vec<usize>,
    ) -> Result<GpuTensor, CudaError> {
        let len: usize = shape.iter().product();
        let slice = self.get_i32(ctx, len)?;
        Ok(GpuTensor::Int32 {
            data: Arc::new(slice),
            shape,
        })
    }

    /// Get or allocate an Int8 `DeviceSlice`.
    ///
    /// Note: The returned slice may be larger than `len` due to
    /// size-class bucketing. Pooled slices are zeroed before being
    /// returned to ensure consistent behavior.
    pub fn get_i8(
        &mut self,
        ctx: &IconnxCudaContext,
        len: usize,
    ) -> Result<DeviceSlice<'static, i8>, CudaError> {
        let bucket = size_class(len);
        if let Some(slices) = self.i8_pool.get_mut(&bucket) {
            // Find a slice that has at least `len` elements.
            if let Some(idx) = slices.iter().position(|s| s.len() >= len) {
                let mut slice = slices.swap_remove(idx);
                self.hits += 1;
                // Zero the slice to ensure consistent behavior with
                // fresh allocations.
                ctx.zero_i8(&mut slice)?;
                return Ok(slice);
            }
        }
        self.misses += 1;
        // Allocate bucket size to ensure consistency with return_i8.
        ctx.alloc_zeros::<i8>(bucket)
    }

    /// Get or allocate an Int8 `GpuTensor` (convenience method).
    pub fn get_tensor_i8(
        &mut self,
        ctx: &IconnxCudaContext,
        shape: Vec<usize>,
    ) -> Result<GpuTensor, CudaError> {
        let len: usize = shape.iter().product();
        let slice = self.get_i8(ctx, len)?;
        Ok(GpuTensor::Int8 {
            data: Arc::new(slice),
            shape,
        })
    }

    /// Get or allocate a UInt8 `DeviceSlice`.
    ///
    /// Note: The returned slice may be larger than `len` due to
    /// size-class bucketing. Pooled slices are zeroed before being
    /// returned to ensure consistent behavior.
    pub fn get_u8(
        &mut self,
        ctx: &IconnxCudaContext,
        len: usize,
    ) -> Result<DeviceSlice<'static, u8>, CudaError> {
        let bucket = size_class(len);
        if let Some(slices) = self.u8_pool.get_mut(&bucket) {
            // Find a slice that has at least `len` elements.
            if let Some(idx) = slices.iter().position(|s| s.len() >= len) {
                let mut slice = slices.swap_remove(idx);
                self.hits += 1;
                // Zero the slice to ensure consistent behavior with
                // fresh allocations.
                ctx.zero_u8(&mut slice)?;
                return Ok(slice);
            }
        }
        self.misses += 1;
        // Allocate bucket size to ensure consistency with return_u8.
        ctx.alloc_zeros::<u8>(bucket)
    }

    /// Get or allocate a UInt8 `GpuTensor` (convenience method).
    pub fn get_tensor_u8(
        &mut self,
        ctx: &IconnxCudaContext,
        shape: Vec<usize>,
    ) -> Result<GpuTensor, CudaError> {
        let len: usize = shape.iter().product();
        let slice = self.get_u8(ctx, len)?;
        Ok(GpuTensor::UInt8 {
            data: Arc::new(slice),
            shape,
        })
    }

    /// Return a Float32 slice to the pool.
    pub fn return_f32(&mut self, slice: DeviceSlice<'static, f32>) {
        let len = slice.len();
        *self.return_counts.entry(len).or_insert(0) += 1;
        // Use size_class for bucket key to match get_f32.
        let bucket_key = size_class(len);
        let bucket = self.f32_pool.entry(bucket_key).or_default();
        if bucket.len() < self.max_per_bucket {
            bucket.push(slice);
        }
        // If bucket is full, slice is dropped (freed).
    }

    /// Return an Int64 slice to the pool.
    pub fn return_i64(&mut self, slice: DeviceSlice<'static, i64>) {
        let len = slice.len();
        // Use size_class for bucket key to match get_i64.
        let bucket_key = size_class(len);
        let bucket = self.i64_pool.entry(bucket_key).or_default();
        if bucket.len() < self.max_per_bucket {
            bucket.push(slice);
        }
    }

    /// Return an Int32 slice to the pool.
    pub fn return_i32(&mut self, slice: DeviceSlice<'static, i32>) {
        let len = slice.len();
        // Use size_class for bucket key to match get_i32.
        let bucket_key = size_class(len);
        let bucket = self.i32_pool.entry(bucket_key).or_default();
        if bucket.len() < self.max_per_bucket {
            bucket.push(slice);
        }
    }

    /// Return an Int8 slice to the pool.
    pub fn return_i8(&mut self, slice: DeviceSlice<'static, i8>) {
        let len = slice.len();
        // Use size_class for bucket key to match get_i8.
        let bucket_key = size_class(len);
        let bucket = self.i8_pool.entry(bucket_key).or_default();
        if bucket.len() < self.max_per_bucket {
            bucket.push(slice);
        }
    }

    /// Return a UInt8 slice to the pool.
    pub fn return_u8(&mut self, slice: DeviceSlice<'static, u8>) {
        let len = slice.len();
        // Use size_class for bucket key to match get_u8.
        let bucket_key = size_class(len);
        let bucket = self.u8_pool.entry(bucket_key).or_default();
        if bucket.len() < self.max_per_bucket {
            bucket.push(slice);
        }
    }

    /// Return a `GpuTensor` to the pool (extracts slice if possible).
    ///
    /// Returns true if the tensor was successfully returned to the pool.
    /// Returns false if the tensor couldn't be returned (e.g., shared Arc).
    pub fn return_tensor(&mut self, tensor: GpuTensor) -> bool {
        match tensor.into_slice() {
            Some(TypedSlice::Float32(slice)) => {
                self.return_f32(slice);
                true
            }
            Some(TypedSlice::Int64(slice)) => {
                self.return_i64(slice);
                true
            }
            Some(TypedSlice::Int32(slice)) => {
                self.return_i32(slice);
                true
            }
            Some(TypedSlice::Int8(slice)) => {
                self.return_i8(slice);
                true
            }
            Some(TypedSlice::UInt8(slice)) => {
                self.return_u8(slice);
                true
            }
            None => {
                self.failed_returns += 1;
                false // Couldn't extract (shared Arc).
            }
        }
    }

    /// Get pool statistics: (hits, misses).
    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    /// Get hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Get total number of pooled slices.
    pub fn pooled_count(&self) -> usize {
        self.f32_pool.values().map(|v| v.len()).sum::<usize>()
            + self.i64_pool.values().map(|v| v.len()).sum::<usize>()
            + self.i32_pool.values().map(|v| v.len()).sum::<usize>()
            + self.i8_pool.values().map(|v| v.len()).sum::<usize>()
            + self.u8_pool.values().map(|v| v.len()).sum::<usize>()
    }

    /// Clear all pooled memory (frees GPU memory).
    pub fn clear(&mut self) {
        self.f32_pool.clear();
        self.i64_pool.clear();
        self.i32_pool.clear();
        self.i8_pool.clear();
        self.u8_pool.clear();
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.alloc_counts.clear();
        self.return_counts.clear();
        self.failed_returns = 0;
    }

    /// Get number of failed tensor returns (Arc refcount > 1).
    pub fn failed_returns(&self) -> usize {
        self.failed_returns
    }

    /// Get allocation analysis: top N sizes by allocation count.
    /// Returns Vec of (size, alloc_count, return_count, reuse_rate).
    pub fn allocation_analysis(&self, top_n: usize) -> Vec<(usize, usize, usize, f64)> {
        let mut sizes: Vec<_> = self.alloc_counts.iter().collect();
        sizes.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending.

        sizes
            .into_iter()
            .take(top_n)
            .map(|(&size, &alloc_count)| {
                let return_count = *self.return_counts.get(&size).unwrap_or(&0);
                let reuse_rate = if alloc_count > 0 {
                    (return_count as f64 / alloc_count as f64) * 100.0
                } else {
                    0.0
                };
                (size, alloc_count, return_count, reuse_rate)
            })
            .collect()
    }

    /// Get sizes that are allocated but never returned (optimization
    /// opportunities).
    pub fn unreturned_sizes(&self) -> Vec<(usize, usize)> {
        self.alloc_counts
            .iter()
            .filter(|(&size, _)| !self.return_counts.contains_key(&size))
            .map(|(&size, &count)| (size, count))
            .collect()
    }

    /// Get unique size count (measure of allocation diversity).
    pub fn unique_sizes(&self) -> usize {
        self.alloc_counts.len()
    }

    /// Pre-allocate common sizes to warm the pool.
    ///
    /// This improves first-inference performance by having memory ready
    /// before it's needed. Call this after creating the executor but
    /// before running inference.
    ///
    /// The sizes are based on empirical analysis of Kokoro TTS model
    /// allocations. Note: With size-class pooling, these sizes are
    /// rounded up to power-of-2 buckets. `count` specifies how many
    /// slices to pre-allocate per size (default 2).
    pub fn warm(&mut self, ctx: &IconnxCudaContext, count: usize) -> Result<(), CudaError> {
        // Common sizes observed in Kokoro TTS model (high allocation
        // frequency). These will be rounded up to size-class buckets
        // (128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536).
        const COMMON_SIZES: &[usize] = &[
            128,   // 118 allocations, 81% reuse -> bucket 128.
            256,   // 534 allocations, 27% reuse -> bucket 256.
            512,   // 67 allocations, 115% reuse -> bucket 512.
            1024,  // 566 allocations, 5% reuse -> bucket 1024.
            3584,  // 48 allocations, 83% reuse -> bucket 4096.
            5376,  // 339 allocations, 75% reuse -> bucket 8192.
            14336, // 50 allocations, 48% reuse -> bucket 16384.
            28672, // 126 allocations, 69% reuse -> bucket 32768.
            57344, // 63 allocations, 87% reuse -> bucket 65536.
        ];

        for &size in COMMON_SIZES {
            // Allocate bucket size for consistency with get_f32.
            let bucket = size_class(size);
            for _ in 0..count {
                let slice = ctx.alloc_zeros::<f32>(bucket)?;
                self.return_f32(slice);
            }
        }

        Ok(())
    }

    /// Pre-allocate with custom sizes.
    ///
    /// Use this when you know the specific allocation pattern for your
    /// model. `sizes` is a list of (element_count, count) pairs. Note:
    /// Sizes are rounded up to size-class buckets.
    pub fn warm_sizes(
        &mut self,
        ctx: &IconnxCudaContext,
        sizes: &[(usize, usize)],
    ) -> Result<(), CudaError> {
        for &(size, count) in sizes {
            // Allocate bucket size for consistency with get_f32.
            let bucket = size_class(size);
            for _ in 0..count {
                let slice = ctx.alloc_zeros::<f32>(bucket)?;
                self.return_f32(slice);
            }
        }
        Ok(())
    }
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_pool_basic() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let mut pool = GpuMemoryPool::new();

        // First allocation should be a miss.
        let slice1 = pool.get_f32(&ctx, 1000).unwrap();
        assert_eq!(pool.stats(), (0, 1)); // 0 hits, 1 miss.

        // Return to pool.
        pool.return_f32(slice1);
        assert_eq!(pool.pooled_count(), 1);

        // Second allocation should hit.
        let _slice2 = pool.get_f32(&ctx, 1000).unwrap();
        assert_eq!(pool.stats(), (1, 1)); // 1 hit, 1 miss.

        println!("Pool basic test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_pool_different_sizes() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let mut pool = GpuMemoryPool::new();

        // Allocate different sizes.
        let s1 = pool.get_f32(&ctx, 100).unwrap();
        let s2 = pool.get_f32(&ctx, 200).unwrap();
        let s3 = pool.get_f32(&ctx, 100).unwrap(); // Same as s1.

        pool.return_f32(s1);
        pool.return_f32(s2);
        pool.return_f32(s3);

        assert_eq!(pool.pooled_count(), 3);

        // Get size 100 - should hit.
        let _s4 = pool.get_f32(&ctx, 100).unwrap();
        assert_eq!(pool.stats().0, 1); // 1 hit.

        // Get size 300 - should miss.
        let _s5 = pool.get_f32(&ctx, 300).unwrap();
        assert_eq!(pool.stats(), (1, 4)); // 1 hit, 4 misses.

        println!("Pool different sizes test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_pool_tensor_convenience() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let mut pool = GpuMemoryPool::new();

        // Allocate tensor.
        let tensor = pool.get_tensor_f32(&ctx, vec![10, 20]).unwrap();
        assert_eq!(tensor.shape(), &[10, 20]);
        assert_eq!(tensor.len(), 200);

        // Return tensor.
        assert!(pool.return_tensor(tensor));
        assert_eq!(pool.pooled_count(), 1);

        println!("Pool tensor convenience test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_pool_warming() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let mut pool = GpuMemoryPool::new();

        // Warm the pool with 2 slices per size.
        pool.warm(&ctx, 2).expect("Warming failed");

        // Should have 9 sizes * 2 slices = 18 pooled slices.
        assert_eq!(pool.pooled_count(), 18);

        // Now allocate sizes that were pre-warmed - should hit.
        let _s1 = pool.get_f32(&ctx, 1024).unwrap();
        let _s2 = pool.get_f32(&ctx, 256).unwrap();
        let _s3 = pool.get_f32(&ctx, 512).unwrap();

        // All three should be hits (from warmed pool).
        assert_eq!(pool.stats(), (3, 0)); // 3 hits, 0 misses.

        println!("Pool warming test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_pool_warm_custom_sizes() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let mut pool = GpuMemoryPool::new();

        // Warm with custom sizes.
        pool.warm_sizes(&ctx, &[(100, 3), (200, 2)])
            .expect("Custom warming failed");

        // Should have 3 + 2 = 5 pooled slices.
        assert_eq!(pool.pooled_count(), 5);

        // Allocate from warmed sizes.
        let _s1 = pool.get_f32(&ctx, 100).unwrap();
        let _s2 = pool.get_f32(&ctx, 100).unwrap();
        let _s3 = pool.get_f32(&ctx, 200).unwrap();

        assert_eq!(pool.stats(), (3, 0)); // All hits.

        println!("Pool custom warming test passed!");
    }
}
