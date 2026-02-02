//! OBJ import - convert triangle meshes to SDF grids.
//!
//! Parses Wavefront OBJ files and computes signed distance fields
//! using brute-force point-to-triangle distance calculations.
//!
//! # Algorithm
//!
//! 1. Parse OBJ file to extract vertices and triangle faces
//! 2. Compute axis-aligned bounding box with padding
//! 3. Determine grid resolution based on cell size
//! 4. For each cell center, compute signed distance to mesh:
//!    - Unsigned distance: minimum distance to any triangle
//!    - Sign: determined by pseudo-normal test (average of face normals at closest point)
//!
//! # Performance Notes
//!
//! The brute-force approach is O(blocks * cells * triangles). For large meshes,
//! consider using spatial acceleration structures. This implementation prioritizes
//! correctness and simplicity over raw speed.

#[cfg(feature = "std")]
use std::io::{BufRead, BufReader, Read};

use ash_core::{BlockCoord, Point3};

use crate::error::{AshIoError, Result};
use crate::memory::{GridBuilder, InMemoryGrid};

/// Configuration for OBJ import.
#[derive(Debug, Clone)]
pub struct ObjImportConfig {
    /// Cell size in world units.
    pub cell_size: f32,
    /// Grid dimension (cells per block side, must be power of 2).
    pub grid_dim: u32,
    /// Padding around the mesh bounding box (in world units).
    pub padding: f32,
    /// Maximum number of blocks to allocate.
    pub max_blocks: usize,
}

impl Default for ObjImportConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.1,
            grid_dim: 8,
            padding: 0.5,
            max_blocks: 100_000,
        }
    }
}

/// Parsed triangle mesh.
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    /// Vertex positions.
    pub vertices: Vec<Point3>,
    /// Triangle indices (3 per triangle).
    pub triangles: Vec<[usize; 3]>,
    /// Computed vertex normals (for sign determination).
    pub vertex_normals: Vec<Point3>,
}

impl TriangleMesh {
    /// Compute the axis-aligned bounding box.
    pub fn bounding_box(&self) -> (Point3, Point3) {
        if self.vertices.is_empty() {
            return (Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        }

        let mut min = self.vertices[0];
        let mut max = self.vertices[0];

        for v in &self.vertices {
            min = min.min(*v);
            max = max.max(*v);
        }

        (min, max)
    }

    /// Compute face normal for a triangle.
    fn face_normal(&self, tri_idx: usize) -> Point3 {
        let [i0, i1, i2] = self.triangles[tri_idx];
        let v0 = self.vertices[i0];
        let v1 = self.vertices[i1];
        let v2 = self.vertices[i2];

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        e1.cross(e2).normalize()
    }

    /// Compute vertex normals as average of adjacent face normals.
    pub fn compute_vertex_normals(&mut self) {
        self.vertex_normals = vec![Point3::new(0.0, 0.0, 0.0); self.vertices.len()];

        // Accumulate face normals at each vertex
        for (tri_idx, &[i0, i1, i2]) in self.triangles.iter().enumerate() {
            let normal = self.face_normal(tri_idx);
            self.vertex_normals[i0] = self.vertex_normals[i0] + normal;
            self.vertex_normals[i1] = self.vertex_normals[i1] + normal;
            self.vertex_normals[i2] = self.vertex_normals[i2] + normal;
        }

        // Normalize
        for n in &mut self.vertex_normals {
            let len = n.length();
            if len > 1e-10 {
                *n = Point3::new(n.x / len, n.y / len, n.z / len);
            }
        }
    }

    /// Compute signed distance from a point to this mesh.
    ///
    /// Uses brute-force search over all triangles.
    pub fn signed_distance(&self, point: Point3) -> f32 {
        if self.triangles.is_empty() {
            return f32::MAX;
        }

        let mut min_dist_sq = f32::MAX;
        let mut closest_normal = Point3::new(0.0, 1.0, 0.0);
        let mut closest_point = point;

        for &[i0, i1, i2] in &self.triangles {
            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];

            let (cp, bary) = closest_point_on_triangle(point, v0, v1, v2);
            let dist_sq = (point - cp).length_squared();

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                closest_point = cp;

                // Interpolate normal at closest point
                let n0 = self.vertex_normals[i0];
                let n1 = self.vertex_normals[i1];
                let n2 = self.vertex_normals[i2];
                closest_normal = Point3::new(
                    n0.x * bary[0] + n1.x * bary[1] + n2.x * bary[2],
                    n0.y * bary[0] + n1.y * bary[1] + n2.y * bary[2],
                    n0.z * bary[0] + n1.z * bary[1] + n2.z * bary[2],
                );
            }
        }

        let dist = libm::sqrtf(min_dist_sq);

        // Determine sign using pseudo-normal
        let to_point = point - closest_point;
        let sign = if to_point.dot(closest_normal) >= 0.0 {
            1.0
        } else {
            -1.0
        };

        sign * dist
    }
}

/// Compute the closest point on a triangle to a given point.
/// Returns (closest_point, barycentric_coordinates).
fn closest_point_on_triangle(p: Point3, a: Point3, b: Point3, c: Point3) -> (Point3, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let point = a + ab * v;
        return (point, [1.0 - v, v, 0.0]);
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let point = a + ac * w;
        return (point, [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let point = b + (c - b) * w;
        return (point, [0.0, 1.0 - w, w]);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let point = a + ab * v + ac * w;
    (point, [1.0 - v - w, v, w])
}

/// Parse an OBJ file from a reader.
#[cfg(feature = "std")]
pub fn parse_obj<R: Read>(reader: R) -> Result<TriangleMesh> {
    let buf_reader = BufReader::new(reader);
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    for line in buf_reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                // Vertex: v x y z
                let x: f32 = parts
                    .next()
                    .ok_or_else(|| AshIoError::Io("Missing vertex x".to_string()))?
                    .parse()
                    .map_err(|_| AshIoError::Io("Invalid vertex x".to_string()))?;
                let y: f32 = parts
                    .next()
                    .ok_or_else(|| AshIoError::Io("Missing vertex y".to_string()))?
                    .parse()
                    .map_err(|_| AshIoError::Io("Invalid vertex y".to_string()))?;
                let z: f32 = parts
                    .next()
                    .ok_or_else(|| AshIoError::Io("Missing vertex z".to_string()))?
                    .parse()
                    .map_err(|_| AshIoError::Io("Invalid vertex z".to_string()))?;
                vertices.push(Point3::new(x, y, z));
            }
            Some("f") => {
                // Face: f v1 v2 v3 [v4 ...]
                // Supports: "f 1 2 3", "f 1/1 2/2 3/3", "f 1/1/1 2/2/2 3/3/3", "f 1//1 2//2 3//3"
                let mut face_indices: Vec<usize> = Vec::new();
                for part in parts {
                    // Parse vertex index (before first /)
                    let idx_str = part.split('/').next().unwrap_or(part);
                    let idx: i32 = idx_str
                        .parse()
                        .map_err(|_| AshIoError::Io(format!("Invalid face index: {}", idx_str)))?;

                    // OBJ indices are 1-based, can be negative (relative)
                    let idx = if idx > 0 {
                        (idx - 1) as usize
                    } else {
                        (vertices.len() as i32 + idx) as usize
                    };

                    face_indices.push(idx);
                }

                // Triangulate if more than 3 vertices (fan triangulation)
                if face_indices.len() >= 3 {
                    for i in 1..face_indices.len() - 1 {
                        triangles.push([face_indices[0], face_indices[i], face_indices[i + 1]]);
                    }
                }
            }
            _ => {
                // Ignore other lines (vn, vt, etc.)
            }
        }
    }

    let mut mesh = TriangleMesh {
        vertices,
        triangles,
        vertex_normals: Vec::new(),
    };
    mesh.compute_vertex_normals();

    Ok(mesh)
}

/// Parse an OBJ file from a path.
#[cfg(feature = "std")]
pub fn parse_obj_file<P: AsRef<std::path::Path>>(path: P) -> Result<TriangleMesh> {
    let file = std::fs::File::open(path)?;
    parse_obj(file)
}

/// Import statistics.
#[derive(Debug, Clone)]
pub struct ImportStats {
    /// Number of vertices in the input mesh.
    pub input_vertices: usize,
    /// Number of triangles in the input mesh.
    pub input_triangles: usize,
    /// Mesh bounding box minimum.
    pub bbox_min: Point3,
    /// Mesh bounding box maximum.
    pub bbox_max: Point3,
    /// Number of blocks allocated.
    pub blocks_allocated: usize,
    /// Total cells processed.
    pub cells_processed: usize,
}

/// Import an OBJ mesh and convert to an SDF grid.
///
/// # Progress Callback
///
/// The optional progress callback is called with (blocks_done, total_blocks).
#[cfg(feature = "std")]
pub fn import_obj_to_grid<F>(
    mesh: &TriangleMesh,
    config: &ObjImportConfig,
    mut progress: Option<F>,
) -> Result<(InMemoryGrid<1>, ImportStats)>
where
    F: FnMut(usize, usize),
{
    let (bbox_min, bbox_max) = mesh.bounding_box();

    // Add padding
    let padded_min = bbox_min - Point3::new(config.padding, config.padding, config.padding);
    let padded_max = bbox_max + Point3::new(config.padding, config.padding, config.padding);

    // Compute block size
    let block_size = config.cell_size * config.grid_dim as f32;

    // Compute block range
    let block_min = BlockCoord::new(
        libm::floorf(padded_min.x / block_size) as i32,
        libm::floorf(padded_min.y / block_size) as i32,
        libm::floorf(padded_min.z / block_size) as i32,
    );
    let block_max = BlockCoord::new(
        libm::ceilf(padded_max.x / block_size) as i32,
        libm::ceilf(padded_max.y / block_size) as i32,
        libm::ceilf(padded_max.z / block_size) as i32,
    );

    let num_blocks_x = (block_max.x - block_min.x) as usize;
    let num_blocks_y = (block_max.y - block_min.y) as usize;
    let num_blocks_z = (block_max.z - block_min.z) as usize;
    let total_blocks = num_blocks_x * num_blocks_y * num_blocks_z;

    let capacity = total_blocks.min(config.max_blocks);
    let cells_per_block = (config.grid_dim * config.grid_dim * config.grid_dim) as usize;

    let mut builder: GridBuilder<1> = GridBuilder::new(config.grid_dim, config.cell_size)
        .with_capacity(capacity);

    let mut blocks_done = 0;
    let mut cells_processed = 0;

    for bz in block_min.z..block_max.z {
        for by in block_min.y..block_max.y {
            for bx in block_min.x..block_max.x {
                if blocks_done >= capacity {
                    break;
                }

                let block = BlockCoord::new(bx, by, bz);
                let block_origin = Point3::new(
                    bx as f32 * block_size,
                    by as f32 * block_size,
                    bz as f32 * block_size,
                );

                // Compute SDF values for each cell
                let mut values = Vec::with_capacity(cells_per_block);
                for z in 0..config.grid_dim {
                    for y in 0..config.grid_dim {
                        for x in 0..config.grid_dim {
                            let cell_center = block_origin
                                + Point3::new(
                                    (x as f32 + 0.5) * config.cell_size,
                                    (y as f32 + 0.5) * config.cell_size,
                                    (z as f32 + 0.5) * config.cell_size,
                                );
                            let sdf = mesh.signed_distance(cell_center);
                            values.push([sdf]);
                            cells_processed += 1;
                        }
                    }
                }

                builder = builder
                    .add_block(block, values)
                    .map_err(|e| AshIoError::Io(e.to_string()))?;

                blocks_done += 1;

                if let Some(ref mut cb) = progress {
                    cb(blocks_done, total_blocks.min(capacity));
                }
            }
        }
    }

    let grid = builder
        .build()
        .map_err(|e| AshIoError::Io(e.to_string()))?;

    let stats = ImportStats {
        input_vertices: mesh.vertices.len(),
        input_triangles: mesh.triangles.len(),
        bbox_min,
        bbox_max,
        blocks_allocated: grid.num_blocks(),
        cells_processed,
    };

    Ok((grid, stats))
}

/// Import an OBJ file directly to a grid.
#[cfg(feature = "std")]
pub fn import_obj_file_to_grid<P: AsRef<std::path::Path>>(
    path: P,
    config: &ObjImportConfig,
) -> Result<(InMemoryGrid<1>, ImportStats)> {
    let mesh = parse_obj_file(path)?;
    import_obj_to_grid(&mesh, config, None::<fn(usize, usize)>)
}

/// Import OBJ with narrow band allocation and optional BVH acceleration.
///
/// This is much more efficient than `import_obj_to_grid` for meshes that don't
/// fill their bounding box, as it only allocates blocks near the surface.
///
/// # Arguments
/// * `mesh` - Parsed triangle mesh
/// * `config` - Import configuration
/// * `use_bvh` - Whether to use BVH for distance queries (recommended for >1000 triangles)
/// * `progress` - Optional progress callback `(blocks_done, total_blocks)`
///
/// # Performance
/// - With BVH: O(blocks * cells * log(triangles)) instead of O(blocks * cells * triangles)
/// - Narrow band: Only processes blocks near the surface, not entire bounding box
#[cfg(feature = "std")]
pub fn import_obj_narrow_band<F>(
    mesh: &TriangleMesh,
    config: &ObjImportConfig,
    use_bvh: bool,
    mut progress: Option<F>,
) -> Result<(InMemoryGrid<1>, ImportStats)>
where
    F: FnMut(usize, usize),
{
    use crate::spatial::{narrow_band_from_triangles, NarrowBandConfig, TriangleBvh};

    let (bbox_min, bbox_max) = mesh.bounding_box();

    // Compute narrow band blocks
    let nb_config = NarrowBandConfig::new(config.cell_size, config.grid_dim, 1);
    let narrow_blocks = narrow_band_from_triangles(&mesh.vertices, &mesh.triangles, &nb_config);

    let total_blocks = narrow_blocks.len().min(config.max_blocks);
    let cells_per_block = (config.grid_dim * config.grid_dim * config.grid_dim) as usize;

    let mut builder: GridBuilder<1> = GridBuilder::new(config.grid_dim, config.cell_size)
        .with_capacity(total_blocks);

    // Build BVH if requested
    let bvh = if use_bvh && mesh.triangles.len() > 100 {
        Some(TriangleBvh::build(&mesh.vertices, &mesh.triangles, 8))
    } else {
        None
    };

    let block_size = config.cell_size * config.grid_dim as f32;
    let mut cells_processed = 0;

    for (block_idx, &block) in narrow_blocks.iter().take(total_blocks).enumerate() {
        let block_origin = Point3::new(
            block.x as f32 * block_size,
            block.y as f32 * block_size,
            block.z as f32 * block_size,
        );

        // Compute SDF values for each cell
        let mut values = Vec::with_capacity(cells_per_block);
        for z in 0..config.grid_dim {
            for y in 0..config.grid_dim {
                for x in 0..config.grid_dim {
                    let cell_center = block_origin
                        + Point3::new(
                            (x as f32 + 0.5) * config.cell_size,
                            (y as f32 + 0.5) * config.cell_size,
                            (z as f32 + 0.5) * config.cell_size,
                        );

                    let sdf = if let Some(ref bvh) = bvh {
                        bvh.signed_distance(
                            &mesh.vertices,
                            &mesh.triangles,
                            &mesh.vertex_normals,
                            cell_center,
                        )
                    } else {
                        mesh.signed_distance(cell_center)
                    };

                    values.push([sdf]);
                    cells_processed += 1;
                }
            }
        }

        builder = builder
            .add_block(block, values)
            .map_err(|e| AshIoError::Io(e.to_string()))?;

        if let Some(ref mut cb) = progress {
            cb(block_idx + 1, total_blocks);
        }
    }

    let grid = builder
        .build()
        .map_err(|e| AshIoError::Io(e.to_string()))?;

    let stats = ImportStats {
        input_vertices: mesh.vertices.len(),
        input_triangles: mesh.triangles.len(),
        bbox_min,
        bbox_max,
        blocks_allocated: grid.num_blocks(),
        cells_processed,
    };

    Ok((grid, stats))
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn make_cube_obj() -> &'static str {
        // Cube with outward-facing normals (CCW winding when viewed from outside)
        r#"
# Simple cube with correct winding for outward normals
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1

# Front face (z=0, normal points -z)
f 1 3 2
f 1 4 3
# Back face (z=1, normal points +z)
f 5 6 7
f 5 7 8
# Bottom face (y=0, normal points -y)
f 1 2 6
f 1 6 5
# Top face (y=1, normal points +y)
f 4 8 7
f 4 7 3
# Left face (x=0, normal points -x)
f 1 5 8
f 1 8 4
# Right face (x=1, normal points +x)
f 2 3 7
f 2 7 6
"#
    }

    #[test]
    fn test_parse_obj() {
        let mesh = parse_obj(make_cube_obj().as_bytes()).unwrap();

        assert_eq!(mesh.vertices.len(), 8);
        assert_eq!(mesh.triangles.len(), 12); // 6 faces * 2 triangles
    }

    #[test]
    fn test_bounding_box() {
        let mesh = parse_obj(make_cube_obj().as_bytes()).unwrap();
        let (min, max) = mesh.bounding_box();

        assert!((min.x - 0.0).abs() < 1e-6);
        assert!((min.y - 0.0).abs() < 1e-6);
        assert!((min.z - 0.0).abs() < 1e-6);
        assert!((max.x - 1.0).abs() < 1e-6);
        assert!((max.y - 1.0).abs() < 1e-6);
        assert!((max.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_signed_distance() {
        let mesh = parse_obj(make_cube_obj().as_bytes()).unwrap();

        // Center of cube should be inside (negative)
        let center_sdf = mesh.signed_distance(Point3::new(0.5, 0.5, 0.5));
        assert!(center_sdf < 0.0, "Center should be inside: {}", center_sdf);

        // Point outside should be positive
        let outside_sdf = mesh.signed_distance(Point3::new(2.0, 0.5, 0.5));
        assert!(outside_sdf > 0.0, "Outside should be positive: {}", outside_sdf);
    }

    #[test]
    fn test_import_to_grid() {
        let mesh = parse_obj(make_cube_obj().as_bytes()).unwrap();

        let config = ObjImportConfig {
            cell_size: 0.2,
            grid_dim: 4,
            padding: 0.2,
            max_blocks: 100,
        };

        let (grid, stats) = import_obj_to_grid(&mesh, &config, None::<fn(usize, usize)>).unwrap();

        assert!(grid.num_blocks() > 0);
        assert_eq!(stats.input_vertices, 8);
        assert_eq!(stats.input_triangles, 12);
    }

    #[test]
    fn test_closest_point_on_triangle() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);

        // Point above triangle center
        let p = Point3::new(0.25, 0.25, 1.0);
        let (closest, bary) = closest_point_on_triangle(p, a, b, c);

        assert!((closest.x - 0.25).abs() < 1e-5);
        assert!((closest.y - 0.25).abs() < 1e-5);
        assert!(closest.z.abs() < 1e-5);
        assert!((bary[0] + bary[1] + bary[2] - 1.0).abs() < 1e-5);
    }
}
