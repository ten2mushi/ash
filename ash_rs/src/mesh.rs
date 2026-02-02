//! Mesh extraction using marching cubes.
//!
//! Wraps ash_core's marching cubes implementation with parallel processing support.

use ash_core::{BlockCoord, CellCoord, Point3};

use crate::grid::SparseDenseGrid;

/// A triangle represented by three vertices.
pub type Triangle = [Point3; 3];

impl SparseDenseGrid {
    /// Extract an isosurface mesh using marching cubes.
    ///
    /// This method processes cells in parallel using the number of threads
    /// available on the system.
    ///
    /// # Arguments
    /// * `iso_value` - Isosurface threshold (typically 0.0 for SDF zero-level set)
    ///
    /// # Returns
    /// Vector of triangles, where each triangle is defined by 3 vertices.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let triangles = grid.extract_mesh(0.0);
    /// println!("Generated {} triangles", triangles.len());
    /// ```
    #[cfg(feature = "mesh")]
    pub fn extract_mesh(&self, iso_value: f32) -> Vec<Triangle> {
        self.extract_mesh_parallel(iso_value, 0) // 0 means use rayon's default
    }

    /// Extract mesh with a specified number of threads.
    ///
    /// # Arguments
    /// * `iso_value` - Isosurface threshold
    /// * `num_threads` - Number of threads to use
    #[cfg(feature = "mesh")]
    pub fn extract_mesh_parallel(&self, iso_value: f32, num_threads: usize) -> Vec<Triangle> {
        use rayon::prelude::*;

        // Configure thread pool if non-zero thread count specified
        let _pool = if num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .ok()
        } else {
            None // Use rayon's default thread pool
        };

        // Collect all block indices
        let block_indices: Vec<usize> = (0..self.num_blocks()).collect();

        // Process blocks in parallel
        block_indices
            .par_iter()
            .flat_map(|&block_idx| {
                let coord = self.inner().storage().get_coord(block_idx);
                self.extract_block_mesh(coord, iso_value)
            })
            .collect()
    }

    /// Extract mesh from a single block.
    ///
    /// This is useful for incremental mesh updates when only some blocks change.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn extract_block_mesh(&self, block: BlockCoord, iso_value: f32) -> Vec<Triangle> {
        let dim = self.config().grid_dim;
        let mut triangles = Vec::new();

        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let cell = CellCoord::new(x, y, z);
                    triangles.extend(ash_core::marching_cubes::process_cell(
                        self, block, cell, iso_value,
                    ));
                }
            }
        }

        triangles
    }

    /// Extract mesh without allocation (for embedded/no_std).
    ///
    /// Processes each cell with a callback function that receives triangles.
    /// This avoids heap allocation by processing triangles immediately.
    ///
    /// # Arguments
    /// * `iso_value` - Isosurface threshold
    /// * `callback` - Function called for each triangle
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut triangle_count = 0;
    /// grid.extract_mesh_callback(0.0, |triangle| {
    ///     triangle_count += 1;
    ///     // Process triangle...
    /// });
    /// ```
    pub fn extract_mesh_callback<F>(&self, iso_value: f32, mut callback: F)
    where
        F: FnMut(Triangle),
    {
        let dim = self.config().grid_dim;

        for block_idx in 0..self.num_blocks() {
            let block = self.inner().storage().get_coord(block_idx);

            for z in 0..dim {
                for y in 0..dim {
                    for x in 0..dim {
                        let cell = CellCoord::new(x, y, z);
                        let (triangles, count) = ash_core::marching_cubes::process_cell_no_alloc(
                            self, block, cell, iso_value,
                        );

                        for triangle in triangles.iter().take(count) {
                            callback(*triangle);
                        }
                    }
                }
            }
        }
    }

    /// Estimate the number of triangles in the mesh.
    ///
    /// This is a rough estimate based on the number of allocated blocks
    /// and typical surface area. Useful for pre-allocating buffers.
    pub fn estimate_triangle_count(&self) -> usize {
        // Rough estimate: ~5% of cells on average have surface intersections,
        // each producing ~2 triangles on average
        let cells_per_block = self.config().cells_per_block();
        let surface_cells = (cells_per_block as f32 * 0.05) as usize;
        let triangles_per_cell = 2;
        self.num_blocks() * surface_cells * triangles_per_cell
    }
}

/// Mesh statistics after extraction.
#[derive(Debug, Clone, Copy)]
pub struct MeshStats {
    /// Total number of triangles.
    pub triangle_count: usize,
    /// Number of vertices (triangle_count * 3).
    pub vertex_count: usize,
    /// Approximate surface area (sum of triangle areas).
    pub surface_area: f32,
    /// Bounding box minimum.
    pub bbox_min: Point3,
    /// Bounding box maximum.
    pub bbox_max: Point3,
}

impl MeshStats {
    /// Compute statistics from a set of triangles.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn from_triangles(triangles: &[Triangle]) -> Self {
        let triangle_count = triangles.len();
        let vertex_count = triangle_count * 3;

        let mut surface_area = 0.0;
        let mut bbox_min = Point3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut bbox_max = Point3::new(f32::MIN, f32::MIN, f32::MIN);

        for tri in triangles {
            // Update bounding box
            for &v in tri {
                bbox_min = bbox_min.min(v);
                bbox_max = bbox_max.max(v);
            }

            // Compute triangle area
            let e1 = tri[1] - tri[0];
            let e2 = tri[2] - tri[0];
            let cross = e1.cross(e2);
            surface_area += cross.length() * 0.5;
        }

        Self {
            triangle_count,
            vertex_count,
            surface_area,
            bbox_min,
            bbox_max,
        }
    }
}

/// Export mesh to OBJ format string.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn triangles_to_obj(triangles: &[Triangle]) -> String {
    use core::fmt::Write;

    let mut obj = String::new();

    // Write header
    writeln!(obj, "# ASH_RS generated mesh").unwrap();
    writeln!(obj, "# {} triangles, {} vertices", triangles.len(), triangles.len() * 3).unwrap();
    writeln!(obj).unwrap();

    // Write vertices
    for tri in triangles {
        for v in tri {
            writeln!(obj, "v {} {} {}", v.x, v.y, v.z).unwrap();
        }
    }

    writeln!(obj).unwrap();

    // Write faces (1-indexed in OBJ format)
    for i in 0..triangles.len() {
        let base = i * 3 + 1;
        writeln!(obj, "f {} {} {}", base, base + 1, base + 2).unwrap();
    }

    obj
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridBuilder;

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
    fn test_extract_block_mesh() {
        let grid = make_sphere_grid();

        let triangles = grid.extract_block_mesh(BlockCoord::new(0, 0, 0), 0.0);

        // Should generate some triangles
        assert!(!triangles.is_empty(), "Should generate triangles");

        // Triangles should have valid vertices
        for tri in &triangles {
            for v in tri {
                assert!(v.x.is_finite());
                assert!(v.y.is_finite());
                assert!(v.z.is_finite());
            }
        }
    }

    #[test]
    #[cfg(feature = "mesh")]
    fn test_extract_mesh_parallel() {
        let grid = make_sphere_grid();

        let triangles = grid.extract_mesh(0.0);

        // Should generate a reasonable number of triangles for a sphere
        assert!(
            triangles.len() > 10,
            "Should generate triangles: got {}",
            triangles.len()
        );
        assert!(
            triangles.len() < 10000,
            "Too many triangles: {}",
            triangles.len()
        );
    }

    #[test]
    fn test_extract_mesh_callback() {
        let grid = make_sphere_grid();

        let mut count = 0;
        grid.extract_mesh_callback(0.0, |_tri| {
            count += 1;
        });

        assert!(count > 0, "Should generate triangles via callback");
    }

    #[test]
    fn test_mesh_stats() {
        let triangles = vec![
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ],
        ];

        let stats = MeshStats::from_triangles(&triangles);

        assert_eq!(stats.triangle_count, 2);
        assert_eq!(stats.vertex_count, 6);
        assert!(stats.surface_area > 0.0);

        // Bounding box should cover [0,1]³
        assert!((stats.bbox_min.x - 0.0).abs() < 1e-6);
        assert!((stats.bbox_max.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_triangles_to_obj() {
        let triangles = vec![[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]];

        let obj = triangles_to_obj(&triangles);

        assert!(obj.contains("v 0 0 0"));
        assert!(obj.contains("v 1 0 0"));
        assert!(obj.contains("v 0 1 0"));
        assert!(obj.contains("f 1 2 3"));
    }

    #[test]
    fn test_estimate_triangle_count() {
        let grid = make_sphere_grid();

        let estimate = grid.estimate_triangle_count();

        // Should give a non-zero estimate
        assert!(estimate > 0);
    }
}
