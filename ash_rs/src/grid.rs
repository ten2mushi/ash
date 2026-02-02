//! SparseDenseGrid implementation.
//!
//! A thin wrapper over `ash_io::InMemoryGrid<1>` providing SDF-specific
//! functionality and backward compatibility.

use ash_core::{
    BlockCoord, CellCoord, CellValueProvider, Point3,
};

use ash_io::{InMemoryGrid, GridConfig};

/// High-performance sparse-dense SDF grid.
///
/// This structure provides:
/// - O(1) block lookup via lock-free hash map
/// - < 100ns single-point queries
/// - Lock-free concurrent read access
/// - Memory-efficient SoA storage
///
/// # Architecture
///
/// The grid uses a two-level hierarchy:
/// 1. **Sparse level**: Hash map from block coordinates to block indices
/// 2. **Dense level**: Contiguous array of SDF values for each block
///
/// # Example
///
/// ```ignore
/// use ash_rs::{GridBuilder, SparseDenseGrid};
/// use ash_core::Point3;
///
/// let grid = GridBuilder::new(8, 0.1)
///     .with_capacity(1000)
///     .add_block(BlockCoord::new(0, 0, 0), vec![[0.0]; 512])?
///     .build()?;
///
/// let sdf_value = grid.query(Point3::new(0.4, 0.4, 0.4));
/// ```
pub struct SparseDenseGrid {
    /// Inner N=1 grid from ash_io.
    inner: InMemoryGrid<1>,
}

impl SparseDenseGrid {
    /// Create a new grid from an InMemoryGrid<1>.
    pub fn from_inner(inner: InMemoryGrid<1>) -> Self {
        Self { inner }
    }

    /// Get the inner grid reference.
    #[inline]
    pub fn inner(&self) -> &InMemoryGrid<1> {
        &self.inner
    }

    /// Consume self and return the inner grid.
    #[inline]
    pub fn into_inner(self) -> InMemoryGrid<1> {
        self.inner
    }

    /// Get the grid configuration.
    #[inline]
    pub fn config(&self) -> &GridConfig {
        self.inner.config()
    }

    /// Get the number of allocated blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks()
    }

    /// Get the maximum capacity (number of blocks).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Check if a block exists at the given coordinate.
    #[inline]
    pub fn has_block(&self, coord: BlockCoord) -> bool {
        self.inner.has_block(coord)
    }

    /// Get the index of a block, if it exists.
    #[inline]
    pub fn get_block_index(&self, coord: BlockCoord) -> Option<usize> {
        self.inner.get_block_index(coord)
    }

    /// Get a reference to the SDF values for a block.
    ///
    /// Returns `None` if the block doesn't exist.
    #[inline]
    pub fn get_block_values(&self, coord: BlockCoord) -> Option<&[f32]> {
        self.inner.get_block_sdf_values(coord)
    }

    /// Iterate over all allocated block coordinates.
    ///
    /// The iteration order is by block index, not by spatial position.
    pub fn block_coords(&self) -> impl Iterator<Item = BlockCoord> + '_ {
        self.inner.block_coords()
    }

    /// Get all allocated block coordinates as a vector.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn block_coords_vec(&self) -> Vec<BlockCoord> {
        self.inner.block_coords_vec()
    }
}

// -----------------------------------------------------------------------------
// CellValueProvider Implementation
// -----------------------------------------------------------------------------

impl CellValueProvider<1> for SparseDenseGrid {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
    ) -> Option<[f32; 1]> {
        self.inner.get_corner_values(block, cell, corner)
    }

    #[inline]
    fn grid_dim(&self) -> u32 {
        self.inner.grid_dim()
    }

    #[inline]
    fn cell_size(&self) -> f32 {
        self.inner.cell_size()
    }
}

// -----------------------------------------------------------------------------
// Query Methods
// -----------------------------------------------------------------------------

impl SparseDenseGrid {
    /// Query the SDF value at a world-space point.
    ///
    /// Returns `None` if the point is in an unallocated or untrained region.
    ///
    /// # Performance
    /// Target: < 100ns on x86-64 (warm cache)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sdf = grid.query(Point3::new(0.5, 0.5, 0.5));
    /// if let Some(distance) = sdf {
    ///     println!("Distance to surface: {}", distance);
    /// }
    /// ```
    #[inline]
    pub fn query(&self, point: Point3) -> Option<f32> {
        self.inner.query_sdf(point)
    }

    /// Query the SDF value and analytical gradient at a world-space point.
    ///
    /// The gradient points in the direction of increasing SDF values (outward
    /// from surfaces). This is useful for:
    /// - Computing surface normals
    /// - Motion planning (gradient descent toward/away from obstacles)
    /// - Eikonal loss computation during training
    ///
    /// # Performance
    /// Target: < 200ns on x86-64 (warm cache)
    ///
    /// # Returns
    /// `Some((value, [dx, dy, dz]))` - The SDF value and world-space gradient
    /// `None` - If the point is in an unallocated region
    #[inline]
    pub fn query_with_gradient(&self, point: Point3) -> Option<(f32, [f32; 3])> {
        let (values, grads) = self.inner.query_with_gradient(point)?;
        Some((values[0], grads[0]))
    }

    /// Fast collision check - the most common use case.
    ///
    /// Returns `true` if the SDF value at the point is less than the threshold,
    /// indicating the point is inside an obstacle (or within `threshold` distance).
    ///
    /// Unknown regions (unallocated blocks) are treated as safe (returns `false`).
    ///
    /// # Arguments
    /// * `point` - World-space point to check
    /// * `threshold` - Collision threshold (robot radius + safety margin)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let robot_radius = 0.3;
    /// let safety_margin = 0.1;
    /// if grid.in_collision(robot_pos, robot_radius + safety_margin) {
    ///     // Take evasive action
    /// }
    /// ```
    #[inline]
    pub fn in_collision(&self, point: Point3, threshold: f32) -> bool {
        self.inner.in_collision(point, threshold)
    }

    /// Get the surface normal at a point (normalized gradient).
    ///
    /// The normal points outward from the surface.
    /// Returns `None` if the point is in an unknown region or if the
    /// gradient magnitude is too small (flat region).
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(normal) = grid.surface_normal(surface_point) {
    ///     // Use normal for rendering, physics, etc.
    /// }
    /// ```
    pub fn surface_normal(&self, point: Point3) -> Option<Point3> {
        let (_, grad) = self.query_with_gradient(point)?;
        let len_sq = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];

        if len_sq < 1e-16 {
            return None;
        }

        let len = crate::libm::sqrtf(len_sq);
        Some(Point3::new(grad[0] / len, grad[1] / len, grad[2] / len))
    }

    /// Query SDF value with bounds checking.
    ///
    /// Like `query()` but also returns whether the point is within allocated space.
    /// This is useful for diagnostics and debugging.
    #[inline]
    pub fn query_checked(&self, point: Point3) -> (Option<f32>, bool) {
        let (block, _cell, _local) =
            ash_core::decompose_point(point, self.inner.cell_size(), self.inner.grid_dim());

        let has_block = self.inner.has_block(block);
        let value = self.query(point);

        (value, has_block)
    }
}

// Safety: SparseDenseGrid is safe to share across threads for reading
unsafe impl Sync for SparseDenseGrid {}
unsafe impl Send for SparseDenseGrid {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridBuilder;
    use ash_core::SdfProvider;

    fn make_test_grid() -> SparseDenseGrid {
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        // Create a simple grid with a sphere SDF
        let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

        // Add blocks around the origin
        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    let coord = BlockCoord::new(bx, by, bz);
                    builder = builder.add_block_fn(coord, |pos| {
                        (pos - center).length() - radius
                    });
                }
            }
        }

        builder.build().unwrap()
    }

    #[test]
    fn test_grid_creation() {
        let grid = make_test_grid();
        assert_eq!(grid.num_blocks(), 8);
        assert_eq!(grid.config().grid_dim, 8);
    }

    #[test]
    fn test_has_block() {
        let grid = make_test_grid();
        assert!(grid.has_block(BlockCoord::new(0, 0, 0)));
        assert!(grid.has_block(BlockCoord::new(1, 1, 1)));
        assert!(!grid.has_block(BlockCoord::new(5, 5, 5)));
    }

    #[test]
    fn test_query_inside_sphere() {
        let grid = make_test_grid();
        // Query at center of sphere (should be negative)
        let center = Point3::new(0.4, 0.4, 0.4);
        let sdf = grid.query(center);
        assert!(sdf.is_some());
        assert!(sdf.unwrap() < 0.0, "Center should be inside sphere");
    }

    #[test]
    fn test_query_outside_sphere() {
        let grid = make_test_grid();
        // Query outside sphere but within allocated blocks
        // Block size is 0.8, so query in the interior of the allocated region
        let outside = Point3::new(1.2, 1.2, 1.2);
        let sdf = grid.query(outside);
        assert!(sdf.is_some());
        assert!(sdf.unwrap() > 0.0, "Point should be outside sphere: {}", sdf.unwrap());
    }

    #[test]
    fn test_query_unknown_region() {
        let grid = make_test_grid();
        // Query in unallocated block
        let unknown = Point3::new(-1.0, -1.0, -1.0);
        assert!(grid.query(unknown).is_none());
    }

    #[test]
    fn test_in_collision() {
        let grid = make_test_grid();
        let center = Point3::new(0.4, 0.4, 0.4);

        // Inside sphere with 0 threshold
        assert!(grid.in_collision(center, 0.0));

        // Outside sphere but within allocated blocks
        let outside = Point3::new(1.2, 1.2, 1.2);
        assert!(!grid.in_collision(outside, 0.1));

        // Unknown region
        let unknown = Point3::new(-5.0, -5.0, -5.0);
        assert!(!grid.in_collision(unknown, 0.1), "Unknown should be safe");
    }

    #[test]
    fn test_gradient_direction() {
        let grid = make_test_grid();

        // Query at a point on +X side of sphere
        let point = Point3::new(0.7, 0.4, 0.4);
        let result = grid.query_with_gradient(point);

        assert!(result.is_some());
        let (_, grad) = result.unwrap();

        // Gradient should point mostly in +X direction
        assert!(grad[0] > 0.5, "Gradient X should be positive: {:?}", grad);
    }

    #[test]
    fn test_surface_normal() {
        let grid = make_test_grid();

        // Point on surface in +X direction
        let point = Point3::new(0.7, 0.4, 0.4);
        let normal = grid.surface_normal(point);

        assert!(normal.is_some());
        let n = normal.unwrap();

        // Normal should be approximately unit length
        let len = n.length();
        assert!(
            (len - 1.0).abs() < 0.01,
            "Normal should be unit length: {}",
            len
        );

        // Normal should point mostly in +X
        assert!(n.x > 0.5, "Normal should point +X: {:?}", n);
    }

    #[test]
    fn test_cell_value_provider_impl() {
        let grid = make_test_grid();

        // Verify CellValueProvider methods
        assert_eq!(grid.grid_dim(), 8);
        assert!((grid.cell_size() - 0.1).abs() < 1e-6);
        assert!((grid.block_size() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_get_corner_values() {
        let grid = make_test_grid();

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(4, 4, 4);

        // Should be able to get corner values (N=1)
        let values = grid.get_corner_values(block, cell, (0, 0, 0));
        assert!(values.is_some());

        // Also test the SdfProvider convenience method
        let value = grid.get_corner_value(block, cell, (0, 0, 0));
        assert!(value.is_some());
        assert!((value.unwrap() - values.unwrap()[0]).abs() < 1e-6);

        // Unknown block should return None
        let values = grid.get_corner_values(BlockCoord::new(-10, -10, -10), cell, (0, 0, 0));
        assert!(values.is_none());
    }

    #[test]
    fn test_block_coords_iteration() {
        let grid = make_test_grid();

        let coords: Vec<_> = grid.block_coords().collect();
        assert_eq!(coords.len(), 8);

        // All should be in the expected range
        for coord in coords {
            assert!(coord.x >= 0 && coord.x <= 1);
            assert!(coord.y >= 0 && coord.y <= 1);
            assert!(coord.z >= 0 && coord.z <= 1);
        }
    }
}
