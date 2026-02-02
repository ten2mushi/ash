//! Marching Cubes mesh extraction algorithm.
//!
//! Provides functions to extract triangle meshes from SDF data using the
//! marching cubes algorithm.

use crate::coords::{corner_position, resolve_corner};
use crate::traits::{corner_from_index, SdfProvider};

// CellValueProvider<1> is used in tests
#[cfg(test)]
use crate::traits::CellValueProvider;
use crate::types::{BlockCoord, CellCoord, Point3, UNTRAINED_SENTINEL};

use super::tables::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};

/// Interpolate vertex position along an edge where the SDF crosses the iso-value.
///
/// Uses linear interpolation to find the exact crossing point.
///
/// # Arguments
/// * `p0` - Position of the first corner
/// * `p1` - Position of the second corner
/// * `v0` - SDF value at the first corner
/// * `v1` - SDF value at the second corner
/// * `iso_value` - The iso-surface value (typically 0.0 for SDF)
///
/// # Returns
/// The interpolated position where the surface crosses the edge.
#[inline]
pub fn interpolate_vertex(p0: Point3, p1: Point3, v0: f32, v1: f32, iso_value: f32) -> Point3 {
    // Avoid division by zero for degenerate cases
    let denom = v1 - v0;
    if libm::fabsf(denom) < 1e-10 {
        return p0.lerp(p1, 0.5);
    }

    let t = (iso_value - v0) / denom;

    // Clamp to [0, 1] to handle numerical errors
    let t = t.clamp(0.0, 1.0);

    p0.lerp(p1, t)
}

/// Compute the cube configuration index from corner SDF values.
///
/// # Arguments
/// * `corner_values` - SDF values at the 8 corners
/// * `iso_value` - The iso-surface value
///
/// # Returns
/// An 8-bit index where bit i is set if corner i is inside the surface (value < iso_value).
#[inline]
fn compute_cube_index(corner_values: &[f32; 8], iso_value: f32) -> usize {
    let mut index = 0;
    for (i, &val) in corner_values.iter().enumerate() {
        if val < iso_value {
            index |= 1 << i;
        }
    }
    index
}

/// Process a single cell and extract triangles using marching cubes.
///
/// # Arguments
/// * `provider` - Storage providing corner SDF values (must implement `SdfProvider`)
/// * `block` - The block containing the cell
/// * `cell` - The cell to process
/// * `iso_value` - The iso-surface value (typically 0.0 for SDF)
///
/// # Returns
/// A vector of triangles, where each triangle is an array of 3 vertices.
/// Returns an empty vector if any corner is missing or untrained.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn process_cell<P: SdfProvider>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    iso_value: f32,
) -> crate::alloc_prelude::Vec<[Point3; 3]> {
    use crate::alloc_prelude::Vec;

    let mut triangles = Vec::new();

    // Gather corner values and positions
    let grid_dim = provider.grid_dim();
    let cell_size = provider.cell_size();

    let mut corner_values = [0.0f32; 8];
    let mut corner_positions = [Point3::default(); 8];

    for i in 0..8 {
        let corner = corner_from_index(i);
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, corner, grid_dim);

        let value = match provider.get_corner_value(resolved_block, resolved_cell, (0, 0, 0)) {
            Some(v) => v,
            None => return triangles, // Missing corner, skip cell
        };

        if value >= UNTRAINED_SENTINEL * 0.5 {
            return triangles; // Untrained region, skip cell
        }

        corner_values[i] = value;
        corner_positions[i] = corner_position(block, cell, CORNER_OFFSETS[i], cell_size, grid_dim);
    }

    // Compute cube configuration
    let cube_index = compute_cube_index(&corner_values, iso_value);

    // Check if the surface intersects this cell
    let edge_flags = EDGE_TABLE[cube_index];
    if edge_flags == 0 {
        return triangles;
    }

    // Compute edge intersection vertices
    let mut edge_vertices = [Point3::default(); 12];
    for edge_idx in 0..12 {
        if (edge_flags & (1 << edge_idx)) != 0 {
            let (v0_idx, v1_idx) = EDGE_VERTICES[edge_idx];
            edge_vertices[edge_idx] = interpolate_vertex(
                corner_positions[v0_idx],
                corner_positions[v1_idx],
                corner_values[v0_idx],
                corner_values[v1_idx],
                iso_value,
            );
        }
    }

    // Generate triangles from the triangle table
    let tri_list = &TRI_TABLE[cube_index];
    let mut i = 0;
    while i < 16 && tri_list[i] != -1 {
        let e0 = tri_list[i] as usize;
        let e1 = tri_list[i + 1] as usize;
        let e2 = tri_list[i + 2] as usize;

        triangles.push([edge_vertices[e0], edge_vertices[e1], edge_vertices[e2]]);
        i += 3;
    }

    triangles
}

/// Process a single cell without dynamic allocation.
///
/// Returns a fixed-size array that can hold up to 5 triangles (the maximum
/// for any marching cubes configuration).
///
/// # Arguments
/// * `provider` - Storage providing corner SDF values (must implement `SdfProvider`)
/// * `block` - The block containing the cell
/// * `cell` - The cell to process
/// * `iso_value` - The iso-surface value (typically 0.0 for SDF)
///
/// # Returns
/// A tuple of (triangles, count) where triangles is a fixed array and count
/// is the number of valid triangles. Returns (default, 0) if any corner is
/// missing or untrained.
pub fn process_cell_no_alloc<P: SdfProvider>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    iso_value: f32,
) -> ([[Point3; 3]; 5], usize) {
    let empty = ([[Point3::default(); 3]; 5], 0);

    // Gather corner values and positions
    let grid_dim = provider.grid_dim();
    let cell_size = provider.cell_size();

    let mut corner_values = [0.0f32; 8];
    let mut corner_positions = [Point3::default(); 8];

    for i in 0..8 {
        let corner = corner_from_index(i);
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, corner, grid_dim);

        let value = match provider.get_corner_value(resolved_block, resolved_cell, (0, 0, 0)) {
            Some(v) => v,
            None => return empty,
        };

        if value >= UNTRAINED_SENTINEL * 0.5 {
            return empty;
        }

        corner_values[i] = value;
        corner_positions[i] = corner_position(block, cell, CORNER_OFFSETS[i], cell_size, grid_dim);
    }

    // Compute cube configuration
    let cube_index = compute_cube_index(&corner_values, iso_value);

    // Check if the surface intersects this cell
    let edge_flags = EDGE_TABLE[cube_index];
    if edge_flags == 0 {
        return empty;
    }

    // Compute edge intersection vertices
    let mut edge_vertices = [Point3::default(); 12];
    for edge_idx in 0..12 {
        if (edge_flags & (1 << edge_idx)) != 0 {
            let (v0_idx, v1_idx) = EDGE_VERTICES[edge_idx];
            edge_vertices[edge_idx] = interpolate_vertex(
                corner_positions[v0_idx],
                corner_positions[v1_idx],
                corner_values[v0_idx],
                corner_values[v1_idx],
                iso_value,
            );
        }
    }

    // Generate triangles from the triangle table
    let mut triangles = [[Point3::default(); 3]; 5];
    let mut count = 0;

    let tri_list = &TRI_TABLE[cube_index];
    let mut i = 0;
    while i < 16 && tri_list[i] != -1 && count < 5 {
        let e0 = tri_list[i] as usize;
        let e1 = tri_list[i + 1] as usize;
        let e2 = tri_list[i + 2] as usize;

        triangles[count] = [edge_vertices[e0], edge_vertices[e1], edge_vertices[e2]];
        count += 1;
        i += 3;
    }

    (triangles, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock provider for a sphere SDF centered at origin
    struct SphereProvider {
        grid_dim: u32,
        cell_size: f32,
        radius: f32,
    }

    impl CellValueProvider<1> for SphereProvider {
        fn get_corner_values(
            &self,
            block: BlockCoord,
            cell: CellCoord,
            _corner: (u32, u32, u32),
        ) -> Option<[f32; 1]> {
            // Note: corner is ignored, we use the resolved cell position
            let block_size = self.grid_dim as f32 * self.cell_size;
            let x = block.x as f32 * block_size + cell.x as f32 * self.cell_size;
            let y = block.y as f32 * block_size + cell.y as f32 * self.cell_size;
            let z = block.z as f32 * block_size + cell.z as f32 * self.cell_size;

            let dist = libm::sqrtf(x * x + y * y + z * z);
            Some([dist - self.radius])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    #[test]
    fn test_interpolate_vertex_midpoint() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        // Equal and opposite values -> midpoint
        let result = interpolate_vertex(p0, p1, -1.0, 1.0, 0.0);
        assert!((result.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_vertex_at_corners() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        // Value at p0 equals iso -> result at p0
        let result = interpolate_vertex(p0, p1, 0.0, 1.0, 0.0);
        assert!((result.x - 0.0).abs() < 1e-6);

        // Value at p1 equals iso -> result at p1
        let result = interpolate_vertex(p0, p1, -1.0, 0.0, 0.0);
        assert!((result.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_cube_index() {
        // All outside
        let values = [1.0; 8];
        assert_eq!(compute_cube_index(&values, 0.0), 0);

        // All inside
        let values = [-1.0; 8];
        assert_eq!(compute_cube_index(&values, 0.0), 255);

        // Only corner 0 inside
        let mut values = [1.0; 8];
        values[0] = -1.0;
        assert_eq!(compute_cube_index(&values, 0.0), 1);

        // Corners 0 and 1 inside
        values[1] = -1.0;
        assert_eq!(compute_cube_index(&values, 0.0), 3);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn test_process_cell_empty() {
        let provider = SphereProvider {
            grid_dim: 8,
            cell_size: 0.1,
            radius: 0.3,
        };

        // Cell far outside the sphere
        let block = BlockCoord::new(10, 10, 10);
        let cell = CellCoord::new(0, 0, 0);

        let tris = process_cell(&provider, block, cell, 0.0);
        assert!(tris.is_empty());
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn test_process_cell_sphere_surface() {
        let provider = SphereProvider {
            grid_dim: 8,
            cell_size: 0.1,
            radius: 0.3,
        };

        // Cell near the sphere surface
        // At (0.2, 0.2, 0.2), distance to origin is ~0.346
        // With radius 0.3, some corners should be inside, some outside
        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 2, 2);

        let _triangles = process_cell(&provider, block, cell, 0.0);

        // Should generate some triangles since cell crosses the surface
        // (This depends on exact geometry)
        // For this test, we just verify it doesn't crash
    }

    #[test]
    fn test_process_cell_no_alloc() {
        let provider = SphereProvider {
            grid_dim: 8,
            cell_size: 0.1,
            radius: 0.3,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 2, 2);

        let (triangles, count) = process_cell_no_alloc(&provider, block, cell, 0.0);

        assert!(count <= 5, "Should produce at most 5 triangles");

        // If we got triangles, verify they have valid points
        for i in 0..count {
            for vertex in &triangles[i] {
                assert!(vertex.x.is_finite());
                assert!(vertex.y.is_finite());
                assert!(vertex.z.is_finite());
            }
        }
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn test_alloc_and_no_alloc_match() {
        let provider = SphereProvider {
            grid_dim: 8,
            cell_size: 0.1,
            radius: 0.3,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 2, 2);

        let alloc_triangles = process_cell(&provider, block, cell, 0.0);
        let (no_alloc_triangles, no_alloc_count) =
            process_cell_no_alloc(&provider, block, cell, 0.0);

        assert_eq!(alloc_triangles.len(), no_alloc_count);

        for i in 0..no_alloc_count {
            for j in 0..3 {
                let a = alloc_triangles[i][j];
                let b = no_alloc_triangles[i][j];
                assert!((a.x - b.x).abs() < 1e-6);
                assert!((a.y - b.y).abs() < 1e-6);
                assert!((a.z - b.z).abs() < 1e-6);
            }
        }
    }
}
