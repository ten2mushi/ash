//! SIMD-optimized batch query implementations.
//!
//! Provides platform-specific SIMD implementations with automatic fallback to scalar code.
//! The API is the same regardless of which implementation is used.
//!
//! # Architecture Support
//!
//! - **x86-64 with AVX2**: Processes 8 points per iteration using 256-bit vectors
//! - **ARM64 with NEON**: Processes 4 points per iteration using 128-bit vectors
//! - **Fallback**: Optimized scalar implementation for other platforms
//!
//! # Performance Characteristics
//!
//! The SIMD implementations provide significant speedups for batch queries:
//!
//! | Platform | Batch Size | Speedup vs Scalar |
//! |----------|------------|-------------------|
//! | AVX2     | 1000       | ~4-6x             |
//! | NEON     | 1000       | ~3-4x             |
//!
//! The speedup comes from:
//! 1. Vectorized trilinear weight computation
//! 2. SIMD FMA (fused multiply-add) for weighted sums
//! 3. Better cache utilization from chunked processing

mod scalar;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

// Re-export scalar for use in platform-specific modules and fallback
#[allow(unused_imports)]
pub(crate) use scalar::query_batch_scalar;

// Re-export the best available implementation
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use avx2::query_batch_avx2 as query_batch_native;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::query_batch_neon_gather as query_batch_native;

// Fallback to scalar if no SIMD available
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
pub use scalar::query_batch_scalar as query_batch_native;

use ash_core::Point3;

/// Result of a batch query operation.
///
/// Contains SDF values and validity masks for multiple query points.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// SDF values for each query point.
    /// Invalid entries contain `f32::NAN`.
    pub values: Vec<f32>,

    /// Validity mask: `true` if the corresponding value is valid.
    pub valid_mask: Vec<bool>,
}

impl BatchResult {
    /// Create a new batch result with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            valid_mask: vec![false; capacity],
        }
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the result is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the number of valid results.
    pub fn num_valid(&self) -> usize {
        self.valid_mask.iter().filter(|&&v| v).count()
    }

    /// Get the value at index, returning `None` if invalid.
    pub fn get(&self, index: usize) -> Option<f32> {
        if index < self.valid_mask.len() && self.valid_mask[index] {
            Some(self.values[index])
        } else {
            None
        }
    }

    /// Iterate over (index, value) pairs for valid entries only.
    pub fn valid_entries(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.values
            .iter()
            .enumerate()
            .zip(self.valid_mask.iter())
            .filter_map(|((idx, &val), &valid)| if valid { Some((idx, val)) } else { None })
    }
}

/// Batch result with gradients.
#[derive(Debug, Clone)]
pub struct BatchResultWithGradients {
    /// SDF values for each query point.
    pub values: Vec<f32>,

    /// Gradients for each query point (world-space).
    /// Invalid entries contain `[f32::NAN; 3]`.
    pub gradients: Vec<[f32; 3]>,

    /// Validity mask.
    pub valid_mask: Vec<bool>,
}

impl BatchResultWithGradients {
    /// Create a new batch result with gradients.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            gradients: Vec::with_capacity(capacity),
            valid_mask: vec![false; capacity],
        }
    }

    /// Get value and gradient at index, returning `None` if invalid.
    pub fn get(&self, index: usize) -> Option<(f32, [f32; 3])> {
        if index < self.valid_mask.len() && self.valid_mask[index] {
            Some((self.values[index], self.gradients[index]))
        } else {
            None
        }
    }
}

/// Query multiple points in batch.
///
/// This function automatically selects the best available implementation:
/// - AVX2 on x86-64 with AVX2 support (8 points per iteration)
/// - NEON on ARM64 (4 points per iteration)
/// - Scalar fallback otherwise
///
/// # Arguments
/// * `grid` - The SDF grid to query
/// * `points` - Points to query
///
/// # Returns
/// Batch result with SDF values and validity masks.
///
/// # Performance
/// - ~3-6x faster than sequential queries for large batches with SIMD
/// - Processes 8 points per iteration (AVX2) or 4 (NEON)
/// - Falls back to optimized scalar for small batches (<16 points)
pub fn query_batch(grid: &crate::SparseDenseGrid, points: &[Point3]) -> BatchResult {
    query_batch_native(grid, points)
}

/// Query multiple points with gradients in batch.
///
/// Like `query_batch` but also computes analytical gradients.
/// Currently uses scalar implementation for gradients.
pub fn query_batch_with_gradients(
    grid: &crate::SparseDenseGrid,
    points: &[Point3],
) -> BatchResultWithGradients {
    scalar::query_batch_with_gradients_scalar(grid, points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_result_operations() {
        let mut result = BatchResult::with_capacity(3);
        result.values = vec![1.0, 2.0, 3.0];
        result.valid_mask = vec![true, false, true];

        assert_eq!(result.len(), 3);
        assert_eq!(result.num_valid(), 2);
        assert_eq!(result.get(0), Some(1.0));
        assert_eq!(result.get(1), None);
        assert_eq!(result.get(2), Some(3.0));
    }

    #[test]
    fn test_valid_entries_iterator() {
        let mut result = BatchResult::with_capacity(5);
        result.values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        result.valid_mask = vec![true, false, true, false, true];

        let valid: Vec<_> = result.valid_entries().collect();
        assert_eq!(valid, vec![(0, 1.0), (2, 3.0), (4, 5.0)]);
    }
}
