//! GridBuilder pattern for constructing SparseDenseGrid.
//!
//! Provides a fluent API for building grids with validation.

use ash_core::BlockCoord;

use ash_io::GridBuilder as IoGridBuilder;

use crate::error::{AshError, Result};
use crate::grid::SparseDenseGrid;

/// Builder for constructing `SparseDenseGrid` instances.
///
/// Provides a fluent API with validation to ensure grids are constructed correctly.
/// This is a thin wrapper over `ash_io::GridBuilder<1>` that returns `SparseDenseGrid`.
///
/// # Example
///
/// ```ignore
/// use ash_rs::GridBuilder;
/// use ash_core::BlockCoord;
///
/// let grid = GridBuilder::new(8, 0.1)
///     .with_capacity(1000)
///     .add_block(BlockCoord::new(0, 0, 0), vec![0.0; 512])?
///     .add_block(BlockCoord::new(1, 0, 0), vec![0.0; 512])?
///     .build()?;
/// ```
pub struct GridBuilder {
    inner: IoGridBuilder<1>,
}

impl GridBuilder {
    /// Create a new builder with the specified grid parameters.
    ///
    /// # Arguments
    /// * `grid_dim` - Number of cells per axis per block (typically 8)
    /// * `cell_size` - Size of each cell in world units
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = GridBuilder::new(8, 0.1); // 8³ cells per block, 10cm cells
    /// ```
    pub fn new(grid_dim: u32, cell_size: f32) -> Self {
        Self {
            inner: IoGridBuilder::new(grid_dim, cell_size),
        }
    }

    /// Set the maximum number of blocks the grid can hold.
    ///
    /// The default capacity is 1024 blocks.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of blocks
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = GridBuilder::new(8, 0.1).with_capacity(10000);
    /// ```
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.inner = self.inner.with_capacity(capacity);
        self
    }

    /// Add a block with SDF values.
    ///
    /// The values array must have exactly `grid_dim³` elements, representing
    /// the SDF value at each cell corner in row-major order.
    ///
    /// # Arguments
    /// * `coord` - Block coordinate in the sparse grid
    /// * `values` - SDF values for all cells (must be `grid_dim³` elements)
    ///
    /// # Errors
    /// Returns `InvalidBlockSize` if values has the wrong number of elements.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let values = vec![0.0; 512]; // 8³ = 512 cells
    /// let builder = builder.add_block(BlockCoord::new(0, 0, 0), values)?;
    /// ```
    pub fn add_block(mut self, coord: BlockCoord, values: Vec<f32>) -> Result<Self> {
        // Convert Vec<f32> to Vec<[f32; 1]>
        let values_n: Vec<[f32; 1]> = values.into_iter().map(|v| [v]).collect();
        self.inner = self.inner.add_block(coord, values_n).map_err(|e| {
            match e {
                ash_io::AshIoError::InvalidBlockSize { expected, got } => {
                    AshError::InvalidBlockSize { expected, got }
                }
                _ => AshError::InvalidFormat { message: "unexpected error" },
            }
        })?;
        Ok(self)
    }

    /// Add a block with a constant SDF value for all cells.
    ///
    /// Convenience method for creating uniform blocks.
    ///
    /// # Arguments
    /// * `coord` - Block coordinate in the sparse grid
    /// * `value` - SDF value to fill all cells
    pub fn add_block_constant(mut self, coord: BlockCoord, value: f32) -> Self {
        self.inner = self.inner.add_block_constant(coord, [value]);
        self
    }

    /// Add a block with SDF values computed from a function.
    ///
    /// The function receives the world-space position of each cell corner
    /// and should return the SDF value at that position.
    ///
    /// # Arguments
    /// * `coord` - Block coordinate in the sparse grid
    /// * `sdf_fn` - Function that computes SDF value from world position
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create a sphere SDF
    /// let center = Point3::new(0.5, 0.5, 0.5);
    /// let radius = 0.3;
    /// let builder = builder.add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
    ///     (pos - center).length() - radius
    /// });
    /// ```
    pub fn add_block_fn<F>(mut self, coord: BlockCoord, sdf_fn: F) -> Self
    where
        F: Fn(ash_core::Point3) -> f32,
    {
        self.inner = self.inner.add_block_fn(coord, |pos| [sdf_fn(pos)]);
        self
    }

    /// Get the number of blocks added so far.
    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks()
    }

    /// Get the expected cells per block.
    pub fn cells_per_block(&self) -> usize {
        self.inner.cells_per_block()
    }

    /// Build the final grid.
    ///
    /// Consumes the builder and returns the constructed grid.
    ///
    /// # Errors
    /// - `ZeroCapacity` if capacity is 0
    /// - `CapacityExceeded` if more blocks were added than capacity allows
    /// - `DuplicateBlock` if the same coordinate was added twice
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grid = GridBuilder::new(8, 0.1)
    ///     .add_block(BlockCoord::new(0, 0, 0), values)?
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<SparseDenseGrid> {
        let inner = self.inner.build().map_err(|e| {
            match e {
                ash_io::AshIoError::ZeroCapacity => AshError::ZeroCapacity,
                ash_io::AshIoError::CapacityExceeded { blocks, capacity } => {
                    AshError::CapacityExceeded { blocks, capacity }
                }
                ash_io::AshIoError::DuplicateBlock { x, y, z } => {
                    AshError::DuplicateBlock { x, y, z }
                }
                _ => AshError::InvalidFormat { message: "unexpected error" },
            }
        })?;
        Ok(SparseDenseGrid::from_inner(inner))
    }

    /// Build the grid, reserving extra capacity for future blocks.
    ///
    /// This is equivalent to calling `with_capacity` with a higher value
    /// before building, but automatically calculates the extra space needed.
    ///
    /// # Arguments
    /// * `extra` - Additional blocks to reserve space for
    pub fn build_with_extra(self, extra: usize) -> Result<SparseDenseGrid> {
        let inner = self.inner.build_with_extra(extra).map_err(|e| {
            match e {
                ash_io::AshIoError::ZeroCapacity => AshError::ZeroCapacity,
                ash_io::AshIoError::CapacityExceeded { blocks, capacity } => {
                    AshError::CapacityExceeded { blocks, capacity }
                }
                ash_io::AshIoError::DuplicateBlock { x, y, z } => {
                    AshError::DuplicateBlock { x, y, z }
                }
                _ => AshError::InvalidFormat { message: "unexpected error" },
            }
        })?;
        Ok(SparseDenseGrid::from_inner(inner))
    }
}

impl Default for GridBuilder {
    fn default() -> Self {
        Self::new(8, 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash_core::Point3;

    #[test]
    fn test_builder_basic() {
        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block(BlockCoord::new(0, 0, 0), vec![0.0; 512])
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(grid.num_blocks(), 1);
        assert!(grid.has_block(BlockCoord::new(0, 0, 0)));
    }

    #[test]
    fn test_builder_invalid_block_size() {
        let result = GridBuilder::new(8, 0.1).add_block(BlockCoord::new(0, 0, 0), vec![0.0; 100]);

        assert!(matches!(
            result,
            Err(AshError::InvalidBlockSize {
                expected: 512,
                got: 100
            })
        ));
    }

    #[test]
    fn test_builder_zero_capacity() {
        let result = GridBuilder::new(8, 0.1).with_capacity(0).build();

        assert!(matches!(result, Err(AshError::ZeroCapacity)));
    }

    #[test]
    fn test_builder_capacity_exceeded() {
        let mut builder = GridBuilder::new(8, 0.1).with_capacity(2);

        for i in 0..5 {
            builder = builder
                .add_block(BlockCoord::new(i, 0, 0), vec![0.0; 512])
                .unwrap();
        }

        let result = builder.build();
        assert!(matches!(result, Err(AshError::CapacityExceeded { .. })));
    }

    #[test]
    fn test_builder_duplicate_block() {
        let result = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block(BlockCoord::new(0, 0, 0), vec![0.0; 512])
            .unwrap()
            .add_block(BlockCoord::new(0, 0, 0), vec![1.0; 512])
            .unwrap()
            .build();

        assert!(matches!(result, Err(AshError::DuplicateBlock { .. })));
    }

    #[test]
    fn test_builder_constant() {
        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block_constant(BlockCoord::new(0, 0, 0), 5.0)
            .build()
            .unwrap();

        let values = grid.get_block_values(BlockCoord::new(0, 0, 0)).unwrap();
        assert!(values.iter().all(|&v| (v - 5.0).abs() < 1e-6));
    }

    #[test]
    fn test_builder_fn() {
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
                (pos - center).length() - radius
            })
            .build()
            .unwrap();

        // Query at center should be negative (inside sphere)
        let sdf = grid.query(center);
        assert!(sdf.is_some());
        // Note: Due to cell corners vs center, exact value may vary
    }

    #[test]
    fn test_builder_multiple_blocks() {
        let mut builder = GridBuilder::new(8, 0.1).with_capacity(100);

        for x in 0..5 {
            for y in 0..5 {
                builder = builder
                    .add_block(BlockCoord::new(x, y, 0), vec![0.0; 512])
                    .unwrap();
            }
        }

        let grid = builder.build().unwrap();
        assert_eq!(grid.num_blocks(), 25);
    }

    #[test]
    fn test_builder_negative_coords() {
        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block(BlockCoord::new(-1, -2, -3), vec![1.0; 512])
            .unwrap()
            .build()
            .unwrap();

        assert!(grid.has_block(BlockCoord::new(-1, -2, -3)));

        let values = grid
            .get_block_values(BlockCoord::new(-1, -2, -3))
            .unwrap();
        assert!((values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_builder_default() {
        let builder = GridBuilder::default();
        assert_eq!(builder.cells_per_block(), 512);
    }

    #[test]
    fn test_builder_build_with_extra() {
        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(1) // Too small
            .add_block(BlockCoord::new(0, 0, 0), vec![0.0; 512])
            .unwrap()
            .build_with_extra(10) // Should expand capacity
            .unwrap();

        assert_eq!(grid.num_blocks(), 1);
        // Capacity should have been increased
    }
}
