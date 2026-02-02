//! OBJ export for grid visualization.
//!
//! Exports the SDF isosurface as a standard Wavefront OBJ mesh file
//! using marching cubes algorithm from ash_core.

#[cfg(feature = "std")]
use std::io::Write;

use ash_core::{marching_cubes::process_cell, CellCoord};

use crate::error::Result;
use crate::memory::InMemoryGrid;

/// Configuration for OBJ export.
#[derive(Debug, Clone)]
pub struct ObjExportConfig {
    /// Iso-value for surface extraction (default: 0.0).
    pub iso_value: f32,
}

impl Default for ObjExportConfig {
    fn default() -> Self {
        Self { iso_value: 0.0 }
    }
}

/// Export a grid's SDF isosurface as OBJ mesh.
///
/// Only works with SDF grids (N=1) since marching cubes requires
/// SDF values for isosurface extraction.
///
/// # Arguments
/// * `grid` - The SDF grid to export
/// * `writer` - Writer to output OBJ data
/// * `config` - Export configuration
///
/// # Example
///
/// ```ignore
/// use ash_io::{InMemoryGrid, export_obj, ObjExportConfig};
/// use std::fs::File;
///
/// let grid: InMemoryGrid<1> = /* create grid */;
/// let mut file = File::create("scene.obj")?;
/// export_obj(&grid, &mut file, ObjExportConfig::default())?;
/// ```
#[cfg(feature = "std")]
pub fn export_obj<W: Write>(
    grid: &InMemoryGrid<1>,
    writer: &mut W,
    config: ObjExportConfig,
) -> Result<()> {
    writeln!(writer, "# ASH OBJ Export")?;
    writeln!(writer, "# Grid dim: {}, cell size: {}", grid.config().grid_dim, grid.config().cell_size)?;
    writeln!(writer, "# Num blocks: {}", grid.num_blocks())?;
    writeln!(writer, "# Iso value: {}", config.iso_value)?;
    writeln!(writer)?;

    let grid_dim = grid.config().grid_dim;
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    // Process each block
    for block in grid.block_coords() {
        // Process each cell in the block
        for z in 0..grid_dim {
            for y in 0..grid_dim {
                for x in 0..grid_dim {
                    let cell = CellCoord::new(x, y, z);

                    // Get triangles from marching cubes
                    let triangles = process_cell(grid, block, cell, config.iso_value);

                    // Add triangles to the mesh
                    for tri in triangles {
                        let base_idx = vertices.len();
                        vertices.push([tri[0].x, tri[0].y, tri[0].z]);
                        vertices.push([tri[1].x, tri[1].y, tri[1].z]);
                        vertices.push([tri[2].x, tri[2].y, tri[2].z]);
                        faces.push([base_idx + 1, base_idx + 2, base_idx + 3]); // OBJ uses 1-based indexing
                    }
                }
            }
        }
    }

    // Write vertices
    writeln!(writer, "# {} vertices", vertices.len())?;
    for v in &vertices {
        writeln!(writer, "v {} {} {}", v[0], v[1], v[2])?;
    }

    writeln!(writer)?;

    // Write faces
    writeln!(writer, "# {} faces", faces.len())?;
    for f in &faces {
        writeln!(writer, "f {} {} {}", f[0], f[1], f[2])?;
    }

    Ok(())
}

/// Export a grid's SDF isosurface to a file.
#[cfg(feature = "std")]
pub fn export_obj_to_file<P: AsRef<std::path::Path>>(
    grid: &InMemoryGrid<1>,
    path: P,
    config: ObjExportConfig,
) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    export_obj(grid, &mut file, config)
}

/// Mesh statistics returned by export functions.
#[derive(Debug, Clone)]
pub struct MeshStats {
    /// Number of vertices in the mesh.
    pub vertex_count: usize,
    /// Number of triangles in the mesh.
    pub triangle_count: usize,
}

/// Export with statistics - returns mesh info along with writing.
#[cfg(feature = "std")]
pub fn export_obj_with_stats<W: Write>(
    grid: &InMemoryGrid<1>,
    writer: &mut W,
    config: ObjExportConfig,
) -> Result<MeshStats> {
    writeln!(writer, "# ASH OBJ Export")?;
    writeln!(writer, "# Grid dim: {}, cell size: {}", grid.config().grid_dim, grid.config().cell_size)?;
    writeln!(writer, "# Num blocks: {}", grid.num_blocks())?;
    writeln!(writer, "# Iso value: {}", config.iso_value)?;
    writeln!(writer)?;

    let grid_dim = grid.config().grid_dim;
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    // Process each block
    for block in grid.block_coords() {
        for z in 0..grid_dim {
            for y in 0..grid_dim {
                for x in 0..grid_dim {
                    let cell = CellCoord::new(x, y, z);
                    let triangles = process_cell(grid, block, cell, config.iso_value);

                    for tri in triangles {
                        let base_idx = vertices.len();
                        vertices.push([tri[0].x, tri[0].y, tri[0].z]);
                        vertices.push([tri[1].x, tri[1].y, tri[1].z]);
                        vertices.push([tri[2].x, tri[2].y, tri[2].z]);
                        faces.push([base_idx + 1, base_idx + 2, base_idx + 3]);
                    }
                }
            }
        }
    }

    let stats = MeshStats {
        vertex_count: vertices.len(),
        triangle_count: faces.len(),
    };

    // Write vertices
    writeln!(writer, "# {} vertices", vertices.len())?;
    for v in &vertices {
        writeln!(writer, "v {} {} {}", v[0], v[1], v[2])?;
    }

    writeln!(writer)?;

    // Write faces
    writeln!(writer, "# {} faces", faces.len())?;
    for f in &faces {
        writeln!(writer, "f {} {} {}", f[0], f[1], f[2])?;
    }

    Ok(stats)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::memory::GridBuilder;
    use ash_core::{BlockCoord, Point3};

    fn make_sphere_grid() -> InMemoryGrid<1> {
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        let mut builder: GridBuilder<1> = GridBuilder::new(8, 0.1).with_capacity(8);

        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    let coord = BlockCoord::new(bx, by, bz);
                    builder = builder.add_block_fn(coord, |pos| {
                        [(pos - center).length() - radius]
                    });
                }
            }
        }

        builder.build().unwrap()
    }

    #[test]
    fn test_export_obj_basic() {
        let grid = make_sphere_grid();
        let mut output = Vec::new();

        export_obj(&grid, &mut output, ObjExportConfig::default()).unwrap();

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("# ASH OBJ Export"));
        assert!(output_str.contains("v ")); // Has vertices
        assert!(output_str.contains("f ")); // Has faces
    }

    #[test]
    fn test_export_obj_with_stats() {
        let grid = make_sphere_grid();
        let mut output = Vec::new();

        let stats = export_obj_with_stats(&grid, &mut output, ObjExportConfig::default()).unwrap();

        // Sphere should generate some triangles
        assert!(stats.vertex_count > 0);
        assert!(stats.triangle_count > 0);
        assert_eq!(stats.vertex_count, stats.triangle_count * 3);
    }

    #[test]
    fn test_export_empty_grid() {
        let grid: InMemoryGrid<1> = GridBuilder::new(8, 0.1).with_capacity(1).build().unwrap();

        let mut output = Vec::new();
        let stats = export_obj_with_stats(&grid, &mut output, ObjExportConfig::default()).unwrap();

        assert_eq!(stats.vertex_count, 0);
        assert_eq!(stats.triangle_count, 0);
    }

    #[test]
    fn test_export_different_iso_value() {
        // Test with a different iso value
        let grid = make_sphere_grid();
        let mut output = Vec::new();

        let config = ObjExportConfig { iso_value: 0.1 }; // Slightly outside surface
        let stats = export_obj_with_stats(&grid, &mut output, config).unwrap();

        // Should still generate triangles but fewer than at iso=0
        assert!(stats.triangle_count > 0);
    }
}
