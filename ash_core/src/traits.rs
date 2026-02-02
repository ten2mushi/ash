//! Traits for storage abstraction in ash_core.
//!
//! These traits allow downstream crates (ash_rs, ash_io, neural_ash) to implement their own
//! storage backends while sharing the core mathematical algorithms.

use crate::types::{BlockCoord, CellCoord};

/// Trait for types that provide feature values at cell corners.
///
/// N = number of f32 values per cell (1 for SDF-only, 4 for SDF+semantic, etc.)
///
/// This trait abstracts over different storage implementations:
/// - `ash_io`: Uses `Vec<f32>` or similar for SIMD-optimized inference
/// - `neural_ash`: Uses Burn `Tensor` storage for autodiff training
///
/// Implementors provide access to the feature values at corners of cells in the sparse-dense grid.
pub trait CellValueProvider<const N: usize> {
    /// Get the feature values at a specific corner of a cell.
    ///
    /// # Arguments
    /// * `block` - The block coordinates in the sparse grid
    /// * `cell` - The cell coordinates within the block
    /// * `corner` - The corner offset (0-1 for each axis)
    ///
    /// # Returns
    /// * `Some(values)` - The N feature values at the corner
    /// * `None` - If the block doesn't exist or the corner is inaccessible
    fn get_corner_values(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
    ) -> Option<[f32; N]>;

    /// The dimension of the dense grid within each block (cells per axis).
    /// For example, 8 means each block contains an 8x8x8 grid of cells.
    fn grid_dim(&self) -> u32;

    /// The size of each cell in world units.
    fn cell_size(&self) -> f32;

    /// The size of each block in world units.
    /// Default implementation: grid_dim * cell_size
    #[inline]
    fn block_size(&self) -> f32 {
        self.grid_dim() as f32 * self.cell_size()
    }
}

/// Convenience trait for SDF-only grids (N=1).
///
/// Provides a simpler API for the common case of single-feature (SDF) grids.
pub trait SdfProvider: CellValueProvider<1> {
    /// Get the SDF value at a specific corner of a cell.
    #[inline]
    fn get_corner_value(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
    ) -> Option<f32> {
        self.get_corner_values(block, cell, corner).map(|[v]| v)
    }
}

/// Blanket implementation of SdfProvider for any CellValueProvider<1>.
impl<T: CellValueProvider<1>> SdfProvider for T {}

/// Trait for accumulating gradients during backpropagation.
///
/// This enables analytical gradient computation for training without full autodiff overhead.
/// The `neural_ash` crate implements this to efficiently backpropagate gradients through
/// trilinear interpolation to the underlying embeddings.
pub trait GradientAccumulator<const N: usize> {
    /// Accumulate a gradient contribution to a specific corner's embedding.
    ///
    /// During backpropagation, the upstream gradient is distributed to each corner
    /// based on the interpolation weights:
    /// `∂L/∂corner_value[i] = weight * upstream_grad[i]`
    ///
    /// # Arguments
    /// * `block` - The block coordinates
    /// * `cell` - The cell coordinates within the block
    /// * `corner` - The corner offset (0-1 for each axis)
    /// * `weight` - The interpolation weight for this corner
    /// * `upstream_grad` - The gradient flowing back from the loss (N features)
    fn accumulate_gradient(
        &mut self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
        weight: f32,
        upstream_grad: [f32; N],
    );
}

/// Convert a corner index (0-7) to a corner offset tuple (i, j, k).
///
/// The corner ordering follows the standard marching cubes convention:
/// ```text
/// Corner:  0      1      2      3      4      5      6      7
/// Offset: (0,0,0)(1,0,0)(1,1,0)(0,1,0)(0,0,1)(1,0,1)(1,1,1)(0,1,1)
/// ```
///
/// This maps corners in a Z-order pattern on each face:
/// - Corners 0-3: z=0 plane
/// - Corners 4-7: z=1 plane
#[inline]
pub const fn corner_from_index(idx: usize) -> (u32, u32, u32) {
    // Using lookup table for clarity and const-correctness
    const CORNERS: [(u32, u32, u32); 8] = [
        (0, 0, 0), // 0
        (1, 0, 0), // 1
        (1, 1, 0), // 2
        (0, 1, 0), // 3
        (0, 0, 1), // 4
        (1, 0, 1), // 5
        (1, 1, 1), // 6
        (0, 1, 1), // 7
    ];
    CORNERS[idx & 7] // Mask to prevent out-of-bounds
}

/// Convert a corner offset tuple (i, j, k) to a corner index (0-7).
///
/// This is the inverse of `corner_from_index`.
#[inline]
pub const fn index_from_corner(corner: (u32, u32, u32)) -> usize {
    // Index = x + 2*y + 4*z (with wrapping around the standard MC corner order)
    // But MC uses a different order, so we use a lookup
    // (0,0,0)->0, (1,0,0)->1, (1,1,0)->2, (0,1,0)->3
    // (0,0,1)->4, (1,0,1)->5, (1,1,1)->6, (0,1,1)->7
    match (corner.0 & 1, corner.1 & 1, corner.2 & 1) {
        (0, 0, 0) => 0,
        (1, 0, 0) => 1,
        (1, 1, 0) => 2,
        (0, 1, 0) => 3,
        (0, 0, 1) => 4,
        (1, 0, 1) => 5,
        (1, 1, 1) => 6,
        (0, 1, 1) => 7,
        _ => unreachable!(), // Can't happen due to & 1 masking
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corner_from_index() {
        assert_eq!(corner_from_index(0), (0, 0, 0));
        assert_eq!(corner_from_index(1), (1, 0, 0));
        assert_eq!(corner_from_index(2), (1, 1, 0));
        assert_eq!(corner_from_index(3), (0, 1, 0));
        assert_eq!(corner_from_index(4), (0, 0, 1));
        assert_eq!(corner_from_index(5), (1, 0, 1));
        assert_eq!(corner_from_index(6), (1, 1, 1));
        assert_eq!(corner_from_index(7), (0, 1, 1));
    }

    #[test]
    fn test_index_from_corner() {
        assert_eq!(index_from_corner((0, 0, 0)), 0);
        assert_eq!(index_from_corner((1, 0, 0)), 1);
        assert_eq!(index_from_corner((1, 1, 0)), 2);
        assert_eq!(index_from_corner((0, 1, 0)), 3);
        assert_eq!(index_from_corner((0, 0, 1)), 4);
        assert_eq!(index_from_corner((1, 0, 1)), 5);
        assert_eq!(index_from_corner((1, 1, 1)), 6);
        assert_eq!(index_from_corner((0, 1, 1)), 7);
    }

    #[test]
    fn test_corner_index_roundtrip() {
        for i in 0..8 {
            let corner = corner_from_index(i);
            let back = index_from_corner(corner);
            assert_eq!(i, back, "Roundtrip failed for index {}", i);
        }
    }

    /// Mock provider for testing with N=1
    struct MockProvider1 {
        grid_dim: u32,
        cell_size: f32,
    }

    impl CellValueProvider<1> for MockProvider1 {
        fn get_corner_values(
            &self,
            _block: BlockCoord,
            _cell: CellCoord,
            _corner: (u32, u32, u32),
        ) -> Option<[f32; 1]> {
            Some([1.0])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    /// Mock provider for testing with N=4
    struct MockProvider4 {
        grid_dim: u32,
        cell_size: f32,
    }

    impl CellValueProvider<4> for MockProvider4 {
        fn get_corner_values(
            &self,
            _block: BlockCoord,
            _cell: CellCoord,
            _corner: (u32, u32, u32),
        ) -> Option<[f32; 4]> {
            Some([1.0, 2.0, 3.0, 4.0])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    #[test]
    fn test_block_size_default() {
        let provider = MockProvider1 {
            grid_dim: 8,
            cell_size: 0.1,
        };
        assert!((provider.block_size() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_sdf_provider_convenience() {
        let provider = MockProvider1 {
            grid_dim: 8,
            cell_size: 0.1,
        };

        // Test SdfProvider convenience method
        let value = provider.get_corner_value(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
        );
        assert_eq!(value, Some(1.0));
    }

    #[test]
    fn test_multi_feature_provider() {
        let provider = MockProvider4 {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let values = provider.get_corner_values(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
        );
        assert_eq!(values, Some([1.0, 2.0, 3.0, 4.0]));
    }
}
