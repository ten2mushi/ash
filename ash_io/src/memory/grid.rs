//! InMemoryGrid<N> - Generic N-feature grid implementation.
//!
//! This is the primary container for multi-feature SDF data, implementing the
//! `CellValueProvider<N>` trait from ash_core for seamless integration.

use ash_core::{
    decompose_point, resolve_corner, trilinear_interpolate, trilinear_with_gradient,
    BlockCoord, CellCoord, CellValueProvider, InterpolationResult, Point3, UNTRAINED_SENTINEL,
};

use super::block_map::BlockMap;
use super::storage::BlockStorage;
use crate::config::GridConfig;

/// In-memory grid with N f32 features per cell.
///
/// This structure provides:
/// - O(1) block lookup via lock-free hash map
/// - < 100ns single-point queries (N=1)
/// - Lock-free concurrent read access
/// - Memory-efficient SoA storage
///
/// # Type Parameter
///
/// `N` is the number of f32 features per cell:
/// - `N=1`: SDF-only grid (most common)
/// - `N=4`: SDF + 3-channel semantic features
/// - `N=8`: SDF + higher-dimensional features
///
/// # Architecture
///
/// The grid uses a two-level hierarchy:
/// 1. **Sparse level**: Hash map from block coordinates to block indices
/// 2. **Dense level**: SoA arrays of feature values for each block
pub struct InMemoryGrid<const N: usize> {
    /// Block coordinate → index lookup.
    pub(crate) block_map: BlockMap,

    /// SoA storage for all block data.
    pub(crate) storage: BlockStorage<N>,

    /// Grid configuration (immutable after construction).
    pub(crate) config: GridConfig,
}

impl<const N: usize> InMemoryGrid<N> {
    /// Create a new grid with the given configuration.
    ///
    /// This is typically called by `GridBuilder::build()`.
    pub(crate) fn new(config: GridConfig) -> Self {
        // Use 2x capacity for hash map to maintain ~50% load factor
        let map_capacity = config.capacity.saturating_mul(2).max(16);

        Self {
            block_map: BlockMap::with_capacity(map_capacity),
            storage: BlockStorage::new(&config),
            config,
        }
    }

    /// Get the grid configuration.
    #[inline]
    pub fn config(&self) -> &GridConfig {
        &self.config
    }

    /// Get the number of allocated blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.storage.num_blocks()
    }

    /// Get a reference to the storage (for advanced use cases).
    #[inline]
    pub fn storage(&self) -> &BlockStorage<N> {
        &self.storage
    }

    /// Get the maximum capacity (number of blocks).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Check if a block exists at the given coordinate.
    #[inline]
    pub fn has_block(&self, coord: BlockCoord) -> bool {
        self.block_map.contains(coord)
    }

    /// Get the index of a block, if it exists.
    #[inline]
    pub fn get_block_index(&self, coord: BlockCoord) -> Option<usize> {
        self.block_map.get(coord)
    }

    /// Get the SDF values (first feature) for a block.
    ///
    /// Returns `None` if the block doesn't exist.
    #[inline]
    pub fn get_block_sdf_values(&self, coord: BlockCoord) -> Option<&[f32]> {
        let idx = self.block_map.get(coord)?;
        Some(self.storage.block_sdf_values(idx))
    }

    /// Get a specific feature array for a block.
    #[inline]
    pub fn get_block_feature_values(&self, coord: BlockCoord, feature_idx: usize) -> Option<&[f32]> {
        let idx = self.block_map.get(coord)?;
        Some(self.storage.block_feature_values(idx, feature_idx))
    }

    /// Iterate over all allocated block coordinates.
    ///
    /// The iteration order is by block index, not by spatial position.
    pub fn block_coords(&self) -> impl Iterator<Item = BlockCoord> + '_ {
        (0..self.storage.num_blocks()).map(|i| self.storage.get_coord(i))
    }

    /// Get all allocated block coordinates as a vector.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn block_coords_vec(&self) -> Vec<BlockCoord> {
        self.block_coords().collect()
    }

    /// Iterate over allocated blocks with direct storage access.
    ///
    /// This is more efficient than hash table iteration because it directly
    /// traverses the storage array without any hash lookups.
    ///
    /// # Returns
    /// Iterator yielding `(BlockCoord, block_index)` pairs.
    #[inline]
    pub fn iter_active(&self) -> impl Iterator<Item = (BlockCoord, usize)> + '_ {
        (0..self.storage.num_blocks()).map(move |idx| (self.storage.get_coord(idx), idx))
    }

    /// Parallel iterator over active blocks.
    ///
    /// Uses Rayon for parallel iteration, which is useful for processing
    /// multiple blocks concurrently during mesh extraction or SDF updates.
    ///
    /// # Returns
    /// Parallel iterator yielding `(BlockCoord, block_index)` pairs.
    #[cfg(feature = "rayon")]
    pub fn par_iter_active(
        &self,
    ) -> impl rayon::prelude::ParallelIterator<Item = (BlockCoord, usize)> + '_ {
        use rayon::prelude::*;
        (0..self.storage.num_blocks())
            .into_par_iter()
            .map(move |idx| (self.storage.get_coord(idx), idx))
    }

    /// Get the coordinate and SDF values for a block by index.
    ///
    /// This is a convenience method for active iteration.
    #[inline]
    pub fn get_active_block(&self, block_idx: usize) -> Option<(BlockCoord, &[f32])> {
        if block_idx >= self.storage.num_blocks() {
            return None;
        }
        let coord = self.storage.get_coord(block_idx);
        let values = self.storage.block_sdf_values(block_idx);
        Some((coord, values))
    }
}

// -----------------------------------------------------------------------------
// CellValueProvider Implementation
// -----------------------------------------------------------------------------

impl<const N: usize> CellValueProvider<N> for InMemoryGrid<N> {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
    ) -> Option<[f32; N]> {
        // Resolve corner that may cross block boundary
        let (resolved_block, resolved_cell) =
            resolve_corner(block, cell, corner, self.config.grid_dim);

        // Look up block index
        let block_idx = self.block_map.get(resolved_block)?;

        // Compute flat cell index
        let cell_idx = resolved_cell.flat_index(self.config.grid_dim);

        // Get all N feature values
        let values = self.storage.get_values(block_idx, cell_idx);

        // Check for untrained sentinel in first feature (SDF)
        if values[0] >= UNTRAINED_SENTINEL * 0.5 {
            return None;
        }

        Some(values)
    }

    #[inline]
    fn grid_dim(&self) -> u32 {
        self.config.grid_dim
    }

    #[inline]
    fn cell_size(&self) -> f32 {
        self.config.cell_size
    }
}

// -----------------------------------------------------------------------------
// Query Methods
// -----------------------------------------------------------------------------

impl<const N: usize> InMemoryGrid<N> {
    /// Query all N feature values at a world-space point.
    ///
    /// Returns `None` if the point is in an unallocated or untrained region.
    ///
    /// # Performance
    /// Target: < 100ns on x86-64 for N=1 (warm cache)
    #[inline]
    pub fn query(&self, point: Point3) -> Option<[f32; N]> {
        let (block, cell, local) =
            decompose_point(point, self.config.cell_size, self.config.grid_dim);

        let result: InterpolationResult<N> = trilinear_interpolate(self, block, cell, local)?;
        Some(result.values)
    }

    /// Query all N feature values and their gradients at a world-space point.
    ///
    /// Returns `None` if the point is in an unallocated region.
    ///
    /// # Returns
    /// `Some((values, gradients))` where:
    /// - `values`: `[f32; N]` interpolated feature values
    /// - `gradients`: `[[f32; 3]; N]` world-space gradient for each feature
    #[inline]
    pub fn query_with_gradient(&self, point: Point3) -> Option<([f32; N], [[f32; 3]; N])> {
        let (block, cell, local) =
            decompose_point(point, self.config.cell_size, self.config.grid_dim);

        let (result, cell_grads): (InterpolationResult<N>, [[f32; 3]; N]) =
            trilinear_with_gradient(self, block, cell, local)?;

        // Convert gradients from cell-space to world-space
        let inv_cell = 1.0 / self.config.cell_size;
        let mut world_grads = [[0.0f32; 3]; N];
        for i in 0..N {
            world_grads[i] = [
                cell_grads[i][0] * inv_cell,
                cell_grads[i][1] * inv_cell,
                cell_grads[i][2] * inv_cell,
            ];
        }

        Some((result.values, world_grads))
    }

    /// Batch query for multiple points.
    ///
    /// Returns a vector of Option values, one per input point.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn query_batch(&self, points: &[Point3]) -> Vec<Option<[f32; N]>> {
        points.iter().map(|&p| self.query(p)).collect()
    }
}

// SDF-specific methods for N=1 grids
impl InMemoryGrid<1> {
    /// Query the SDF value at a world-space point (convenience for N=1).
    #[inline]
    pub fn query_sdf(&self, point: Point3) -> Option<f32> {
        self.query(point).map(|[v]| v)
    }

    /// Fast collision check - the most common use case.
    ///
    /// Returns `true` if the SDF value at the point is less than the threshold.
    /// Unknown regions (unallocated blocks) are treated as safe (returns `false`).
    #[inline]
    pub fn in_collision(&self, point: Point3, threshold: f32) -> bool {
        self.query(point)
            .map(|[sdf]| sdf < threshold)
            .unwrap_or(false)
    }

    /// Get the surface normal at a point (normalized gradient).
    pub fn surface_normal(&self, point: Point3) -> Option<Point3> {
        let (_, grads) = self.query_with_gradient(point)?;
        let grad = grads[0];
        let len_sq = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];

        if len_sq < 1e-16 {
            return None;
        }

        let len = libm::sqrtf(len_sq);
        Some(Point3::new(grad[0] / len, grad[1] / len, grad[2] / len))
    }
}

// Safety: InMemoryGrid is safe to share across threads for reading
unsafe impl<const N: usize> Sync for InMemoryGrid<N> {}
unsafe impl<const N: usize> Send for InMemoryGrid<N> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::GridBuilder;

    fn make_sdf_grid() -> InMemoryGrid<1> {
        let dim = 8;
        let cells_per_block = dim * dim * dim;

        let mut builder: GridBuilder<1> = GridBuilder::new(dim as u32, 0.1).with_capacity(8);

        // Add blocks around the origin with sphere SDF
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    let coord = BlockCoord::new(bx, by, bz);
                    let mut values = vec![[0.0f32]; cells_per_block];

                    for z in 0..dim {
                        for y in 0..dim {
                            for x in 0..dim {
                                let cell_idx = x + y * dim + z * dim * dim;
                                let pos = Point3::new(
                                    bx as f32 * 0.8 + x as f32 * 0.1,
                                    by as f32 * 0.8 + y as f32 * 0.1,
                                    bz as f32 * 0.8 + z as f32 * 0.1,
                                );
                                let diff = pos - center;
                                values[cell_idx] = [diff.length() - radius];
                            }
                        }
                    }

                    builder = builder.add_block(coord, values).unwrap();
                }
            }
        }

        builder.build().unwrap()
    }

    fn make_multi_feature_grid() -> InMemoryGrid<4> {
        let dim = 4;
        let cells_per_block = dim * dim * dim;

        let mut builder: GridBuilder<4> = GridBuilder::new(dim as u32, 0.1).with_capacity(1);

        let coord = BlockCoord::new(0, 0, 0);
        let mut values = vec![[0.0f32; 4]; cells_per_block];

        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let cell_idx = x + y * dim + z * dim * dim;
                    // SDF: linear function
                    // Features 1-3: position components
                    values[cell_idx] = [
                        (x + y + z) as f32 * 0.1, // SDF
                        x as f32 * 0.1,           // Feature 1
                        y as f32 * 0.1,           // Feature 2
                        z as f32 * 0.1,           // Feature 3
                    ];
                }
            }
        }

        builder = builder.add_block(coord, values).unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn test_grid_creation() {
        let grid = make_sdf_grid();
        assert_eq!(grid.num_blocks(), 8);
        assert_eq!(grid.config().grid_dim, 8);
    }

    #[test]
    fn test_sdf_query() {
        let grid = make_sdf_grid();

        // Query at center of sphere (should be negative)
        let center = Point3::new(0.4, 0.4, 0.4);
        let result = grid.query(center);
        assert!(result.is_some());
        assert!(result.unwrap()[0] < 0.0, "Center should be inside sphere");
    }

    #[test]
    fn test_sdf_convenience_methods() {
        let grid = make_sdf_grid();

        let center = Point3::new(0.4, 0.4, 0.4);
        let sdf = grid.query_sdf(center);
        assert!(sdf.is_some());
        assert!(sdf.unwrap() < 0.0);

        assert!(grid.in_collision(center, 0.0));
    }

    #[test]
    fn test_multi_feature_query() {
        let grid = make_multi_feature_grid();

        let point = Point3::new(0.15, 0.15, 0.15);
        let result = grid.query(point);
        assert!(result.is_some());

        let values = result.unwrap();
        // All 4 features should have reasonable values
        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
        assert!(values[2].is_finite());
        assert!(values[3].is_finite());
    }

    #[test]
    fn test_gradient() {
        let grid = make_sdf_grid();

        let point = Point3::new(0.7, 0.4, 0.4);
        let result = grid.query_with_gradient(point);
        assert!(result.is_some());

        let (_, grads) = result.unwrap();
        // Gradient should point mostly in +X direction
        assert!(grads[0][0] > 0.5, "Gradient X should be positive: {:?}", grads[0]);
    }

    #[test]
    fn test_surface_normal() {
        let grid = make_sdf_grid();

        let point = Point3::new(0.7, 0.4, 0.4);
        let normal = grid.surface_normal(point);
        assert!(normal.is_some());

        let n = normal.unwrap();
        let len = n.length();
        assert!((len - 1.0).abs() < 0.01, "Normal should be unit length: {}", len);
        assert!(n.x > 0.5, "Normal should point +X: {:?}", n);
    }

    #[test]
    fn test_unknown_region() {
        let grid = make_sdf_grid();
        let unknown = Point3::new(-1.0, -1.0, -1.0);
        assert!(grid.query(unknown).is_none());
    }

    #[test]
    fn iter_active_empty_grid() {
        let builder: GridBuilder<1> = GridBuilder::new(8, 0.1).with_capacity(10);
        let grid = builder.build().unwrap();

        let active: Vec<_> = grid.iter_active().collect();
        assert!(active.is_empty());
    }

    #[test]
    fn iter_active_visits_all() {
        let grid = make_sdf_grid();
        let active: Vec<_> = grid.iter_active().collect();

        // Should visit all 8 blocks
        assert_eq!(active.len(), 8);

        // Each index should be unique
        let indices: std::collections::HashSet<_> = active.iter().map(|(_, idx)| *idx).collect();
        assert_eq!(indices.len(), 8);

        // All indices should be valid
        for (coord, idx) in &active {
            assert!(*idx < grid.num_blocks());
            // Coordinate should match what's stored
            assert_eq!(grid.storage.get_coord(*idx), *coord);
        }
    }

    #[test]
    fn get_active_block_works() {
        let grid = make_sdf_grid();

        // Valid index
        let result = grid.get_active_block(0);
        assert!(result.is_some());
        let (coord, values) = result.unwrap();
        assert_eq!(values.len(), 512); // 8³ cells
        assert!(grid.has_block(coord));

        // Invalid index
        assert!(grid.get_active_block(100).is_none());
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn par_iter_active_same_results() {
        use rayon::prelude::*;
        use std::collections::HashSet;

        let grid = make_sdf_grid();

        // Sequential
        let sequential: HashSet<_> = grid.iter_active().collect();

        // Parallel
        let parallel: HashSet<_> = grid.par_iter_active().collect();

        // Should have same results
        assert_eq!(sequential, parallel);
    }
}
