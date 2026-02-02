//! Query interface for SparseDenseGrid.
//!
//! Extends the grid with batch query methods and parallel operations.

use ash_core::Point3;

use crate::grid::SparseDenseGrid;
use crate::simd::{query_batch, query_batch_with_gradients, BatchResult, BatchResultWithGradients};

impl SparseDenseGrid {
    /// Query multiple points in batch using SIMD optimization.
    ///
    /// This is significantly faster than calling `query()` in a loop,
    /// especially for large numbers of points.
    ///
    /// # Performance
    /// - Uses AVX2 on x86-64 (8 points per iteration)
    /// - Uses NEON on ARM64 (4 points per iteration)
    /// - Falls back to optimized scalar code otherwise
    /// - Target: < 50μs for 1000 points
    ///
    /// # Example
    ///
    /// ```ignore
    /// let points: Vec<Point3> = robot_path_points();
    /// let results = grid.query_batch(&points);
    ///
    /// for (i, sdf) in results.valid_entries() {
    ///     if sdf < collision_threshold {
    ///         println!("Collision at point {}", i);
    ///     }
    /// }
    /// ```
    pub fn query_batch(&self, points: &[Point3]) -> BatchResult {
        query_batch(self, points)
    }

    /// Query multiple points with gradients in batch.
    ///
    /// Like `query_batch` but also computes analytical gradients.
    pub fn query_batch_with_gradients(&self, points: &[Point3]) -> BatchResultWithGradients {
        query_batch_with_gradients(self, points)
    }

    /// Check multiple points for collision in batch.
    ///
    /// Returns a vector of booleans indicating collision status.
    /// Unknown regions are treated as safe (no collision).
    ///
    /// # Arguments
    /// * `points` - Points to check
    /// * `threshold` - Collision threshold (typically robot radius + margin)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let robot_radius = 0.3;
    /// let collisions = grid.check_collisions_batch(&path_points, robot_radius);
    /// if collisions.iter().any(|&c| c) {
    ///     println!("Path has collisions!");
    /// }
    /// ```
    pub fn check_collisions_batch(&self, points: &[Point3], threshold: f32) -> Vec<bool> {
        let results = self.query_batch(points);
        results
            .values
            .iter()
            .zip(results.valid_mask.iter())
            .map(|(&val, &valid)| valid && val < threshold)
            .collect()
    }

    /// Find the closest point to the surface along a ray.
    ///
    /// Performs sphere tracing (ray marching) to find the intersection
    /// with the zero-level set.
    ///
    /// # Arguments
    /// * `origin` - Ray origin
    /// * `direction` - Ray direction (should be normalized)
    /// * `max_distance` - Maximum distance to march
    /// * `tolerance` - Distance tolerance for considering "on surface"
    ///
    /// # Returns
    /// `Some((t, point))` - Parameter `t` and intersection point
    /// `None` - If no intersection found within max_distance
    ///
    /// # Example
    ///
    /// ```ignore
    /// let hit = grid.raycast(
    ///     camera_pos,
    ///     ray_direction.normalize(),
    ///     100.0,  // max distance
    ///     0.001,  // tolerance
    /// );
    /// if let Some((t, point)) = hit {
    ///     println!("Hit at distance {}", t);
    /// }
    /// ```
    pub fn raycast(
        &self,
        origin: Point3,
        direction: Point3,
        max_distance: f32,
        tolerance: f32,
    ) -> Option<(f32, Point3)> {
        let mut t = 0.0;
        let max_iterations = 256;
        let cell_size = self.config().cell_size;

        for _ in 0..max_iterations {
            if t > max_distance {
                return None;
            }

            let point = Point3::new(
                origin.x + direction.x * t,
                origin.y + direction.y * t,
                origin.z + direction.z * t,
            );

            match self.query(point) {
                Some(sdf) => {
                    if sdf.abs() < tolerance {
                        return Some((t, point));
                    }
                    // Step by the SDF value (sphere tracing)
                    // Use a safety factor to avoid overshooting
                    t += sdf.abs().max(tolerance);
                }
                None => {
                    // Unknown region - step by a small amount
                    t += cell_size;
                }
            }
        }

        None
    }

    /// Get the bounding box of all allocated blocks.
    ///
    /// Returns the minimum and maximum world-space coordinates
    /// that are covered by allocated blocks.
    ///
    /// # Returns
    /// `Some((min, max))` - Bounding box corners
    /// `None` - If no blocks are allocated
    pub fn bounding_box(&self) -> Option<(Point3, Point3)> {
        if self.num_blocks() == 0 {
            return None;
        }

        let block_size = self.config().block_size();
        let mut min = Point3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Point3::new(f32::MIN, f32::MIN, f32::MIN);

        for coord in self.block_coords() {
            let block_min = Point3::new(
                coord.x as f32 * block_size,
                coord.y as f32 * block_size,
                coord.z as f32 * block_size,
            );
            let block_max = Point3::new(
                (coord.x + 1) as f32 * block_size,
                (coord.y + 1) as f32 * block_size,
                (coord.z + 1) as f32 * block_size,
            );

            min = min.min(block_min);
            max = max.max(block_max);
        }

        Some((min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridBuilder;
    use ash_core::BlockCoord;

    fn make_sphere_grid() -> SparseDenseGrid {
        let dim = 8;
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        let mut builder = GridBuilder::new(dim as u32, 0.1).with_capacity(8);

        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    let coord = BlockCoord::new(bx, by, bz);
                    builder = builder.add_block_fn(coord, |pos| (pos - center).length() - radius);
                }
            }
        }

        builder.build().unwrap()
    }

    #[test]
    fn test_batch_query() {
        let grid = make_sphere_grid();

        let points = vec![
            Point3::new(0.4, 0.4, 0.4), // Inside
            Point3::new(1.2, 1.2, 1.2), // Outside but within allocated blocks
            Point3::new(0.7, 0.4, 0.4), // Near surface
        ];

        let results = grid.query_batch(&points);

        assert_eq!(results.len(), 3);
        assert!(results.valid_mask[0], "Center should be valid");
        assert!(results.valid_mask[1], "Interior point should be valid");
        assert!(results.valid_mask[2], "Near surface should be valid");

        // Center should be negative
        assert!(results.values[0] < 0.0);
    }

    #[test]
    fn test_check_collisions_batch() {
        let grid = make_sphere_grid();

        let points = vec![
            Point3::new(0.4, 0.4, 0.4), // Inside sphere
            Point3::new(1.2, 1.2, 1.2), // Outside sphere but within allocated blocks
        ];

        let collisions = grid.check_collisions_batch(&points, 0.0);

        assert!(collisions[0], "Center should be in collision");
        assert!(!collisions[1], "Far point should not be in collision");
    }

    #[test]
    fn test_raycast() {
        let grid = make_sphere_grid();

        // Cast ray from outside toward center
        let origin = Point3::new(2.0, 0.4, 0.4);
        let direction = Point3::new(-1.0, 0.0, 0.0);

        let hit = grid.raycast(origin, direction, 10.0, 0.01);

        assert!(hit.is_some());
        let (t, point) = hit.unwrap();

        // Should hit near x = 0.7 (center at 0.4, radius 0.3)
        assert!(t > 0.0);
        assert!((point.x - 0.7).abs() < 0.05, "Hit point x: {}", point.x);
    }

    #[test]
    fn test_raycast_miss() {
        let grid = make_sphere_grid();

        // Cast ray that misses sphere
        let origin = Point3::new(2.0, 2.0, 0.4);
        let direction = Point3::new(0.0, 1.0, 0.0);

        let hit = grid.raycast(origin, direction, 5.0, 0.01);

        assert!(hit.is_none());
    }

    #[test]
    fn test_bounding_box() {
        let grid = make_sphere_grid();

        let bbox = grid.bounding_box();
        assert!(bbox.is_some());

        let (min, max) = bbox.unwrap();

        // Grid covers blocks 0,0,0 to 1,1,1 with block_size 0.8
        assert!((min.x - 0.0).abs() < 1e-6);
        assert!((min.y - 0.0).abs() < 1e-6);
        assert!((min.z - 0.0).abs() < 1e-6);
        assert!((max.x - 1.6).abs() < 1e-6);
        assert!((max.y - 1.6).abs() < 1e-6);
        assert!((max.z - 1.6).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_empty() {
        let grid = GridBuilder::new(8, 0.1).with_capacity(10).build().unwrap();

        assert!(grid.bounding_box().is_none());
    }
}
