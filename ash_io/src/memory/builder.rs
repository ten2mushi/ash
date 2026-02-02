//! GridBuilder<N> pattern for constructing InMemoryGrid<N>.
//!
//! Provides a fluent API for building grids with validation.

use ash_core::{BlockCoord, Point3};

use super::grid::InMemoryGrid;
use crate::config::GridConfig;
use crate::error::{AshIoError, Result};

/// Builder for constructing `InMemoryGrid<N>` instances.
///
/// Provides a fluent API with validation to ensure grids are constructed correctly.
///
/// # Type Parameter
///
/// `N` is the number of f32 features per cell.
///
/// # Example
///
/// ```ignore
/// use ash_io::{GridBuilder, BlockCoord};
///
/// // SDF-only grid (N=1)
/// let grid = GridBuilder::<1>::new(8, 0.1)
///     .with_capacity(1000)
///     .add_block(BlockCoord::new(0, 0, 0), vec![[0.0]; 512])?
///     .build()?;
///
/// // Multi-feature grid (N=4)
/// let grid = GridBuilder::<4>::new(8, 0.1)
///     .with_capacity(1000)
///     .add_block(BlockCoord::new(0, 0, 0), vec![[0.0, 0.1, 0.2, 0.3]; 512])?
///     .build()?;
/// ```
pub struct GridBuilder<const N: usize> {
    grid_dim: u32,
    cell_size: f32,
    capacity: usize,
    blocks: Vec<(BlockCoord, Vec<[f32; N]>)>,
}

impl<const N: usize> GridBuilder<N> {
    /// Create a new builder with the specified grid parameters.
    ///
    /// # Arguments
    /// * `grid_dim` - Number of cells per axis per block (typically 8)
    /// * `cell_size` - Size of each cell in world units
    pub fn new(grid_dim: u32, cell_size: f32) -> Self {
        Self {
            grid_dim,
            cell_size,
            capacity: 1024,
            blocks: Vec::new(),
        }
    }

    /// Set the maximum number of blocks the grid can hold.
    ///
    /// The default capacity is 1024 blocks.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Add a block with N-feature values.
    ///
    /// The values array must have exactly `grid_dim³` elements.
    ///
    /// # Arguments
    /// * `coord` - Block coordinate in the sparse grid
    /// * `values` - Feature values for all cells (must be `grid_dim³` elements)
    ///
    /// # Errors
    /// Returns `InvalidBlockSize` if values has the wrong number of elements.
    pub fn add_block(mut self, coord: BlockCoord, values: Vec<[f32; N]>) -> Result<Self> {
        let expected = (self.grid_dim as usize).pow(3);
        if values.len() != expected {
            return Err(AshIoError::InvalidBlockSize {
                expected,
                got: values.len(),
            });
        }
        self.blocks.push((coord, values));
        Ok(self)
    }

    /// Add a block with a constant value for all features and cells.
    ///
    /// # Arguments
    /// * `coord` - Block coordinate in the sparse grid
    /// * `values` - Feature values to fill all cells
    pub fn add_block_constant(mut self, coord: BlockCoord, values: [f32; N]) -> Self {
        let cells = (self.grid_dim as usize).pow(3);
        self.blocks.push((coord, vec![values; cells]));
        self
    }

    /// Add a block with feature values computed from a function.
    ///
    /// The function receives the world-space position of each cell corner
    /// and should return all N feature values at that position.
    ///
    /// # Arguments
    /// * `coord` - Block coordinate in the sparse grid
    /// * `feature_fn` - Function that computes N features from world position
    pub fn add_block_fn<F>(mut self, coord: BlockCoord, feature_fn: F) -> Self
    where
        F: Fn(Point3) -> [f32; N],
    {
        let dim = self.grid_dim as usize;
        let cells = dim * dim * dim;
        let block_size = self.cell_size * self.grid_dim as f32;

        let mut values = Vec::with_capacity(cells);

        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let pos = Point3::new(
                        coord.x as f32 * block_size + x as f32 * self.cell_size,
                        coord.y as f32 * block_size + y as f32 * self.cell_size,
                        coord.z as f32 * block_size + z as f32 * self.cell_size,
                    );
                    values.push(feature_fn(pos));
                }
            }
        }

        self.blocks.push((coord, values));
        self
    }

    /// Get the number of blocks added so far.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the expected cells per block.
    pub fn cells_per_block(&self) -> usize {
        (self.grid_dim as usize).pow(3)
    }

    /// Build the final grid.
    ///
    /// Consumes the builder and returns the constructed grid.
    ///
    /// # Errors
    /// - `ZeroCapacity` if capacity is 0
    /// - `CapacityExceeded` if more blocks were added than capacity allows
    /// - `DuplicateBlock` if the same coordinate was added twice
    pub fn build(self) -> Result<InMemoryGrid<N>> {
        // Validate capacity
        if self.capacity == 0 {
            return Err(AshIoError::ZeroCapacity);
        }

        let num_blocks = self.blocks.len();
        if num_blocks > self.capacity {
            return Err(AshIoError::CapacityExceeded {
                blocks: num_blocks,
                capacity: self.capacity,
            });
        }

        let config = GridConfig::new(self.grid_dim, self.cell_size, self.capacity);
        let mut grid = InMemoryGrid::new(config);

        // Insert all blocks
        for (coord, values) in self.blocks {
            // Allocate block in storage
            let block_idx = grid
                .storage
                .allocate_block(coord)
                .ok_or(AshIoError::CapacityExceeded {
                    blocks: num_blocks,
                    capacity: self.capacity,
                })?;

            // Insert into hash map
            grid.block_map.insert(coord, block_idx)?;

            // Copy values
            grid.storage.set_block_values(block_idx, &values);
        }

        Ok(grid)
    }

    /// Build the grid, reserving extra capacity for future blocks.
    pub fn build_with_extra(mut self, extra: usize) -> Result<InMemoryGrid<N>> {
        let needed = self.blocks.len().saturating_add(extra);
        if needed > self.capacity {
            self.capacity = needed;
        }
        self.build()
    }
}

impl<const N: usize> Default for GridBuilder<N> {
    fn default() -> Self {
        Self::new(8, 0.1)
    }
}

// Convenience type aliases
/// Builder for SDF-only grids.
pub type SdfGridBuilder = GridBuilder<1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let grid: InMemoryGrid<1> = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block(BlockCoord::new(0, 0, 0), vec![[0.0]; 512])
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(grid.num_blocks(), 1);
        assert!(grid.has_block(BlockCoord::new(0, 0, 0)));
    }

    #[test]
    fn test_builder_multi_feature() {
        let grid: InMemoryGrid<4> = GridBuilder::new(4, 0.1)
            .with_capacity(10)
            .add_block(BlockCoord::new(0, 0, 0), vec![[0.1, 0.2, 0.3, 0.4]; 64])
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(grid.num_blocks(), 1);
    }

    #[test]
    fn test_builder_invalid_block_size() {
        let result: Result<GridBuilder<1>> = GridBuilder::new(8, 0.1)
            .add_block(BlockCoord::new(0, 0, 0), vec![[0.0]; 100]);

        assert!(matches!(
            result,
            Err(AshIoError::InvalidBlockSize {
                expected: 512,
                got: 100
            })
        ));
    }

    #[test]
    fn test_builder_zero_capacity() {
        let result: Result<InMemoryGrid<1>> = GridBuilder::new(8, 0.1).with_capacity(0).build();
        assert!(matches!(result, Err(AshIoError::ZeroCapacity)));
    }

    #[test]
    fn test_builder_capacity_exceeded() {
        let mut builder: GridBuilder<1> = GridBuilder::new(8, 0.1).with_capacity(2);

        for i in 0..5 {
            builder = builder
                .add_block(BlockCoord::new(i, 0, 0), vec![[0.0]; 512])
                .unwrap();
        }

        let result = builder.build();
        assert!(matches!(result, Err(AshIoError::CapacityExceeded { .. })));
    }

    #[test]
    fn test_builder_duplicate_block() {
        let result: Result<InMemoryGrid<1>> = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block(BlockCoord::new(0, 0, 0), vec![[0.0]; 512])
            .unwrap()
            .add_block(BlockCoord::new(0, 0, 0), vec![[1.0]; 512])
            .unwrap()
            .build();

        assert!(matches!(result, Err(AshIoError::DuplicateBlock { .. })));
    }

    #[test]
    fn test_builder_constant() {
        let grid: InMemoryGrid<2> = GridBuilder::new(4, 0.1)
            .with_capacity(10)
            .add_block_constant(BlockCoord::new(0, 0, 0), [5.0, 10.0])
            .build()
            .unwrap();

        // Query a point and verify values
        let point = Point3::new(0.15, 0.15, 0.15);
        let values = grid.query(point).unwrap();
        assert!((values[0] - 5.0).abs() < 1e-3);
        assert!((values[1] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_builder_fn() {
        let center = Point3::new(0.2, 0.2, 0.2);
        let radius = 0.15;

        let grid: InMemoryGrid<1> = GridBuilder::new(4, 0.1)
            .with_capacity(10)
            .add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
                [(pos - center).length() - radius]
            })
            .build()
            .unwrap();

        // Query at center should be negative (inside sphere)
        let sdf = grid.query_sdf(center);
        assert!(sdf.is_some());
    }

    #[test]
    fn test_builder_multiple_blocks() {
        let mut builder: GridBuilder<1> = GridBuilder::new(8, 0.1).with_capacity(100);

        for x in 0..5 {
            for y in 0..5 {
                builder = builder
                    .add_block(BlockCoord::new(x, y, 0), vec![[0.0]; 512])
                    .unwrap();
            }
        }

        let grid = builder.build().unwrap();
        assert_eq!(grid.num_blocks(), 25);
    }

    #[test]
    fn test_builder_negative_coords() {
        let grid: InMemoryGrid<1> = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block(BlockCoord::new(-1, -2, -3), vec![[1.0]; 512])
            .unwrap()
            .build()
            .unwrap();

        assert!(grid.has_block(BlockCoord::new(-1, -2, -3)));
    }

    #[test]
    fn test_builder_default() {
        let builder: GridBuilder<1> = GridBuilder::default();
        assert_eq!(builder.grid_dim, 8);
        assert!((builder.cell_size - 0.1).abs() < 1e-6);
    }
}
