//! Narrow band block allocation for mesh import.
//!
//! Instead of allocating blocks for the entire bounding box, narrow band
//! allocation only creates blocks that are within a specified distance
//! of the mesh surface. This dramatically reduces memory usage and
//! computation time for sparse meshes.

use ash_core::{BlockCoord, Point3};

/// Configuration for narrow band allocation.
#[derive(Debug, Clone)]
pub struct NarrowBandConfig {
    /// Cell size in world units.
    pub cell_size: f32,
    /// Grid dimension (cells per block side).
    pub grid_dim: u32,
    /// Number of extra block layers around the surface (dilation).
    pub dilation: u32,
}

impl NarrowBandConfig {
    /// Create a new narrow band configuration.
    pub fn new(cell_size: f32, grid_dim: u32, dilation: u32) -> Self {
        Self {
            cell_size,
            grid_dim,
            dilation,
        }
    }

    /// Get the block size in world units.
    #[inline]
    pub fn block_size(&self) -> f32 {
        self.cell_size * self.grid_dim as f32
    }
}

impl Default for NarrowBandConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.1,
            grid_dim: 8,
            dilation: 1,
        }
    }
}

/// Compute block coordinates from a world-space point.
#[inline]
fn point_to_block(point: Point3, block_size: f32) -> BlockCoord {
    BlockCoord::new(
        libm::floorf(point.x / block_size) as i32,
        libm::floorf(point.y / block_size) as i32,
        libm::floorf(point.z / block_size) as i32,
    )
}

/// Compute blocks touching a set of surface points.
///
/// This uses a hash-based approach:
/// 1. Convert each point to its containing block
/// 2. Optionally dilate by adding neighboring blocks
/// 3. Deduplicate using sorting
///
/// # Arguments
/// * `points` - Surface sample points
/// * `config` - Narrow band configuration
///
/// # Returns
/// Sorted, deduplicated list of block coordinates.
///
/// # Performance
/// O(n log n) via parallel sort + dedup.
pub fn compute_narrow_band(points: &[Point3], config: &NarrowBandConfig) -> Vec<BlockCoord> {
    if points.is_empty() {
        return Vec::new();
    }

    let block_size = config.block_size();
    let dilation = config.dilation as i32;

    // Convert points to blocks with dilation
    let mut blocks: Vec<BlockCoord> = Vec::with_capacity(points.len() * (2 * dilation as usize + 1).pow(3));

    for point in points {
        let base = point_to_block(*point, block_size);

        // Add dilated neighborhood
        for dz in -dilation..=dilation {
            for dy in -dilation..=dilation {
                for dx in -dilation..=dilation {
                    blocks.push(BlockCoord::new(base.x + dx, base.y + dy, base.z + dz));
                }
            }
        }
    }

    // Sort by Morton code for spatial locality and deduplication
    blocks.sort_by(|a, b| {
        use ash_core::morton_encode_signed;
        morton_encode_signed(*a).cmp(&morton_encode_signed(*b))
    });

    // Deduplicate
    blocks.dedup();

    blocks
}

/// Compute narrow band blocks from a triangle mesh.
///
/// Samples the mesh at vertices, edge midpoints, and face centers,
/// then computes the narrow band from these samples.
///
/// # Arguments
/// * `vertices` - Mesh vertex positions
/// * `triangles` - Triangle indices (3 per triangle)
/// * `config` - Narrow band configuration
///
/// # Returns
/// Sorted, deduplicated list of block coordinates.
pub fn narrow_band_from_triangles(
    vertices: &[Point3],
    triangles: &[[usize; 3]],
    config: &NarrowBandConfig,
) -> Vec<BlockCoord> {
    if vertices.is_empty() || triangles.is_empty() {
        return Vec::new();
    }

    // Estimate sample count: vertices + 3 edge midpoints + 1 centroid per triangle
    let estimated_samples = vertices.len() + triangles.len() * 4;
    let mut samples = Vec::with_capacity(estimated_samples);

    // Add all vertices
    samples.extend_from_slice(vertices);

    // Add edge midpoints and centroids for each triangle
    for &[i0, i1, i2] in triangles {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Edge midpoints
        samples.push(Point3::new(
            (v0.x + v1.x) * 0.5,
            (v0.y + v1.y) * 0.5,
            (v0.z + v1.z) * 0.5,
        ));
        samples.push(Point3::new(
            (v1.x + v2.x) * 0.5,
            (v1.y + v2.y) * 0.5,
            (v1.z + v2.z) * 0.5,
        ));
        samples.push(Point3::new(
            (v2.x + v0.x) * 0.5,
            (v2.y + v0.y) * 0.5,
            (v2.z + v0.z) * 0.5,
        ));

        // Centroid
        samples.push(Point3::new(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        ));
    }

    compute_narrow_band(&samples, config)
}

/// Compute narrow band blocks with adaptive sampling based on triangle size.
///
/// For large triangles that span multiple blocks, this adds additional
/// interior samples to ensure complete coverage.
///
/// # Arguments
/// * `vertices` - Mesh vertex positions
/// * `triangles` - Triangle indices (3 per triangle)
/// * `config` - Narrow band configuration
///
/// # Returns
/// Sorted, deduplicated list of block coordinates.
pub fn narrow_band_from_triangles_adaptive(
    vertices: &[Point3],
    triangles: &[[usize; 3]],
    config: &NarrowBandConfig,
) -> Vec<BlockCoord> {
    if vertices.is_empty() || triangles.is_empty() {
        return Vec::new();
    }

    let block_size = config.block_size();
    let mut samples = Vec::with_capacity(vertices.len() * 2);

    // Add all vertices
    samples.extend_from_slice(vertices);

    // Adaptively sample each triangle based on its size
    for &[i0, i1, i2] in triangles {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Compute triangle bounding box
        let tri_min = v0.min(v1).min(v2);
        let tri_max = v0.max(v1).max(v2);

        // Estimate how many blocks this triangle spans
        let extent = tri_max - tri_min;
        let spans_x = (extent.x / block_size).ceil() as i32;
        let spans_y = (extent.y / block_size).ceil() as i32;
        let spans_z = (extent.z / block_size).ceil() as i32;
        let max_spans = spans_x.max(spans_y).max(spans_z);

        if max_spans <= 1 {
            // Small triangle: just add centroid
            samples.push(Point3::new(
                (v0.x + v1.x + v2.x) / 3.0,
                (v0.y + v1.y + v2.y) / 3.0,
                (v0.z + v1.z + v2.z) / 3.0,
            ));
        } else {
            // Large triangle: sample interior
            let subdiv = (max_spans * 2) as usize;
            let step = 1.0 / subdiv as f32;

            for i in 0..=subdiv {
                for j in 0..=(subdiv - i) {
                    let u = i as f32 * step;
                    let v = j as f32 * step;
                    let w = 1.0 - u - v;

                    samples.push(Point3::new(
                        v0.x * u + v1.x * v + v2.x * w,
                        v0.y * u + v1.y * v + v2.y * w,
                        v0.z * u + v1.z * v + v2.z * w,
                    ));
                }
            }
        }
    }

    compute_narrow_band(&samples, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_band_unit_cube() {
        // Unit cube vertices
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];

        let triangles = vec![
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 5, 1],
            [0, 4, 5],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ];

        let config = NarrowBandConfig::new(0.5, 4, 0);

        let blocks = narrow_band_from_triangles(&vertices, &triangles, &config);

        // Should have some blocks (not computing exact count)
        assert!(!blocks.is_empty());

        // All blocks should be near the cube
        for block in &blocks {
            let block_size = config.block_size();
            let bx = block.x as f32 * block_size;
            let by = block.y as f32 * block_size;
            let bz = block.z as f32 * block_size;

            // Block should overlap or be near the cube [0,1]³
            assert!(bx + block_size >= -0.1 && bx <= 1.1);
            assert!(by + block_size >= -0.1 && by <= 1.1);
            assert!(bz + block_size >= -0.1 && bz <= 1.1);
        }
    }

    #[test]
    fn narrow_band_dilation() {
        let points = vec![Point3::new(0.5, 0.5, 0.5)];

        let config_no_dilation = NarrowBandConfig::new(1.0, 8, 0);
        let config_dilation_1 = NarrowBandConfig::new(1.0, 8, 1);
        let config_dilation_2 = NarrowBandConfig::new(1.0, 8, 2);

        let blocks_0 = compute_narrow_band(&points, &config_no_dilation);
        let blocks_1 = compute_narrow_band(&points, &config_dilation_1);
        let blocks_2 = compute_narrow_band(&points, &config_dilation_2);

        // Single point with dilation 0 → 1 block
        assert_eq!(blocks_0.len(), 1);

        // Dilation 1 → 3³ = 27 blocks
        assert_eq!(blocks_1.len(), 27);

        // Dilation 2 → 5³ = 125 blocks
        assert_eq!(blocks_2.len(), 125);
    }

    #[test]
    fn narrow_band_empty() {
        let points: Vec<Point3> = vec![];
        let config = NarrowBandConfig::default();

        let blocks = compute_narrow_band(&points, &config);
        assert!(blocks.is_empty());
    }

    #[test]
    fn narrow_band_deduplication() {
        // Multiple points in the same block
        let points = vec![
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.2, 0.2, 0.2),
            Point3::new(0.3, 0.3, 0.3),
        ];

        let config = NarrowBandConfig::new(1.0, 8, 0);

        let blocks = compute_narrow_band(&points, &config);

        // All points are in block (0,0,0), so should have 1 block
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], BlockCoord::new(0, 0, 0));
    }

    #[test]
    fn narrow_band_negative_coords() {
        // With block_size = cell_size * grid_dim = 1.0 * 8 = 8.0
        // Point at (-0.5, -0.5, -0.5) is in block (-1, -1, -1)
        // Point at (-9.0, -9.0, -9.0) is in block (-2, -2, -2)
        let points = vec![
            Point3::new(-0.5, -0.5, -0.5),
            Point3::new(-9.0, -9.0, -9.0),
        ];

        let config = NarrowBandConfig::new(1.0, 8, 0);

        let blocks = compute_narrow_band(&points, &config);

        // Should have 2 blocks in negative space
        assert_eq!(blocks.len(), 2);
        assert!(blocks.iter().any(|b| b.x == -1 && b.y == -1 && b.z == -1));
        assert!(blocks.iter().any(|b| b.x == -2 && b.y == -2 && b.z == -2));
    }

    #[test]
    fn narrow_band_adaptive_large_triangle() {
        // A large triangle that spans multiple blocks
        // With cell_size=0.5 and grid_dim=4, block_size = 2.0
        // Triangle from (0,0,0) to (10,0,0) to (5,10,0) spans 5-6 blocks per axis
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(5.0, 10.0, 0.0),
        ];
        let triangles = vec![[0, 1, 2]];

        let config = NarrowBandConfig::new(0.5, 4, 0);

        let blocks = narrow_band_from_triangles_adaptive(&vertices, &triangles, &config);

        // Should have many blocks covering the triangle (triangle area = 50 sq units)
        // With block_size = 2.0, we expect roughly 50/4 = 12+ blocks
        assert!(blocks.len() >= 10, "Large triangle should produce many blocks, got {}", blocks.len());

        // All blocks should be in the triangle's plane (z=0)
        for block in &blocks {
            assert_eq!(block.z, 0, "All blocks should have z=0 for flat triangle");
        }
    }
}
