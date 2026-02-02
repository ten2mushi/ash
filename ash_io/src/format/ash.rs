//! ASH binary format read/write implementation.
//!
//! The .ash format is optimized for fast loading and cross-platform compatibility.
//!
//! # Format Specification
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ HEADER (32 bytes)                                                  │
//! ├────────────────────────────────────────────────────────────────────┤
//! │  0-3:   Magic "ASHG" (4 bytes)                                     │
//! │  4-7:   cell_size (f32 LE)                                         │
//! │  8-11:  grid_dim (u32 LE)                                          │
//! │ 12-15:  num_blocks (u32 LE)                                        │
//! │ 16-17:  feature_dim (u16 LE) = N                                   │
//! │ 18-19:  flags (u16 LE)                                             │
//! │ 20-31:  reserved (12 bytes)                                        │
//! ├────────────────────────────────────────────────────────────────────┤
//! │ BLOCK COORDINATES (12 bytes per block)                             │
//! │  For each block: (i32 x, i32 y, i32 z) LE                          │
//! ├────────────────────────────────────────────────────────────────────┤
//! │ FEATURE DATA (SoA layout)                                          │
//! │  For each feature dimension (0..N):                                │
//! │    For each block:                                                 │
//! │      For each cell (grid_dim³):                                    │
//! │        f32 value (LE)                                              │
//! └────────────────────────────────────────────────────────────────────┘
//! ```

#[cfg(feature = "std")]
use std::io::{Read, Write};

use ash_core::BlockCoord;

use super::header::{AshHeader, HEADER_SIZE};
use crate::error::{AshIoError, Result};
use crate::memory::{GridBuilder, InMemoryGrid};

/// Save a grid to a writer in .ash binary format.
///
/// # Type Parameters
/// * `N` - Number of features per cell
///
/// # Performance
/// Target: < 100ms for 10MB grid to SSD
#[cfg(feature = "std")]
pub fn save_grid<const N: usize, W: Write>(grid: &InMemoryGrid<N>, writer: &mut W) -> Result<()> {
    let num_blocks = grid.num_blocks();
    let grid_dim = grid.config().grid_dim;
    let cell_size = grid.config().cell_size;

    // Write header
    let header = AshHeader::new(cell_size, grid_dim, num_blocks as u32, N as u16);
    writer.write_all(&header.to_bytes())?;

    // Write block coordinates
    for block_idx in 0..num_blocks {
        let coord = grid.storage.get_coord(block_idx);
        writer.write_all(&coord.x.to_le_bytes())?;
        writer.write_all(&coord.y.to_le_bytes())?;
        writer.write_all(&coord.z.to_le_bytes())?;
    }

    // Write feature data in SoA order
    // For each feature dimension, write all blocks' data for that feature
    for feature_idx in 0..N {
        for block_idx in 0..num_blocks {
            let values = grid.storage.block_feature_values(block_idx, feature_idx);
            // Write as bytes (4 bytes per f32)
            for &value in values {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
    }

    Ok(())
}

/// Load a grid from a reader in .ash binary format.
///
/// # Type Parameters
/// * `N` - Number of features per cell (must match the file)
///
/// # Performance
/// Target: < 50ms for 10MB grid from SSD
///
/// # Errors
/// Returns `FeatureDimensionMismatch` if the file's feature dimension doesn't match N.
#[cfg(feature = "std")]
pub fn load_grid<const N: usize, R: Read>(reader: &mut R) -> Result<InMemoryGrid<N>> {
    // Read header
    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = AshHeader::from_bytes(&header_bytes);

    // Validate header
    if !header.is_valid() {
        return Err(AshIoError::InvalidFormat {
            message: "invalid magic bytes (expected ASHG)",
        });
    }

    // Check feature dimension matches
    if header.feature_dim as usize != N {
        return Err(AshIoError::FeatureDimensionMismatch {
            expected: N,
            got: header.feature_dim as usize,
        });
    }

    let num_blocks = header.num_blocks as usize;
    let grid_dim = header.grid_dim;
    let cell_size = header.cell_size;
    let cells_per_block = (grid_dim as usize).pow(3);

    // Read block coordinates
    let mut coords = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let mut x_bytes = [0u8; 4];
        let mut y_bytes = [0u8; 4];
        let mut z_bytes = [0u8; 4];
        reader.read_exact(&mut x_bytes)?;
        reader.read_exact(&mut y_bytes)?;
        reader.read_exact(&mut z_bytes)?;
        coords.push(BlockCoord::new(
            i32::from_le_bytes(x_bytes),
            i32::from_le_bytes(y_bytes),
            i32::from_le_bytes(z_bytes),
        ));
    }

    // Read feature data in SoA order
    let total_cells = num_blocks * cells_per_block;
    let mut all_feature_data: Vec<Vec<f32>> = (0..N)
        .map(|_| vec![0.0f32; total_cells])
        .collect();

    for feature_idx in 0..N {
        let mut bytes = vec![0u8; total_cells * 4];
        reader.read_exact(&mut bytes)?;

        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            all_feature_data[feature_idx][i] = f32::from_le_bytes(arr);
        }
    }

    // Build the grid
    let mut builder: GridBuilder<N> =
        GridBuilder::new(grid_dim, cell_size).with_capacity(num_blocks.max(1));

    for (block_idx, coord) in coords.into_iter().enumerate() {
        let mut block_values: Vec<[f32; N]> = Vec::with_capacity(cells_per_block);

        for cell_idx in 0..cells_per_block {
            let global_idx = block_idx * cells_per_block + cell_idx;
            let mut cell_values = [0.0f32; N];
            for feature_idx in 0..N {
                cell_values[feature_idx] = all_feature_data[feature_idx][global_idx];
            }
            block_values.push(cell_values);
        }

        builder = builder.add_block(coord, block_values)?;
    }

    builder.build()
}

/// Save a grid to a file path.
#[cfg(feature = "std")]
pub fn save_to_file<const N: usize, P: AsRef<std::path::Path>>(
    grid: &InMemoryGrid<N>,
    path: P,
) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    save_grid(grid, &mut file)
}

/// Load a grid from a file path.
#[cfg(feature = "std")]
pub fn load_from_file<const N: usize, P: AsRef<std::path::Path>>(path: P) -> Result<InMemoryGrid<N>> {
    let mut file = std::fs::File::open(path)?;
    load_grid(&mut file)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::memory::GridBuilder;
    use ash_core::Point3;
    use std::io::Cursor;

    fn make_test_grid_n1() -> InMemoryGrid<1> {
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

    fn make_test_grid_n4() -> InMemoryGrid<4> {
        let mut builder: GridBuilder<4> = GridBuilder::new(4, 0.1).with_capacity(2);

        builder = builder.add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
            [pos.length(), pos.x, pos.y, pos.z]
        });

        builder = builder.add_block_fn(BlockCoord::new(1, 0, 0), |pos| {
            [pos.length() * 2.0, pos.x * 2.0, pos.y * 2.0, pos.z * 2.0]
        });

        builder.build().unwrap()
    }

    #[test]
    fn test_save_load_roundtrip_n1() {
        let original = make_test_grid_n1();

        // Save to buffer
        let mut buffer = Vec::new();
        save_grid(&original, &mut buffer).unwrap();

        // Load from buffer
        let mut cursor = Cursor::new(buffer);
        let loaded: InMemoryGrid<1> = load_grid(&mut cursor).unwrap();

        // Verify properties match
        assert_eq!(loaded.num_blocks(), original.num_blocks());
        assert_eq!(loaded.config().grid_dim, original.config().grid_dim);
        assert!((loaded.config().cell_size - original.config().cell_size).abs() < 1e-6);

        // Verify queries produce same results
        let test_points = vec![
            Point3::new(0.4, 0.4, 0.4),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.1, 0.1, 0.1),
        ];

        for point in test_points {
            let orig_val = original.query(point);
            let loaded_val = loaded.query(point);
            match (orig_val, loaded_val) {
                (Some(v1), Some(v2)) => {
                    assert!((v1[0] - v2[0]).abs() < 1e-6, "Mismatch at {:?}", point);
                }
                (None, None) => {}
                _ => panic!("Validity mismatch at {:?}", point),
            }
        }
    }

    #[test]
    fn test_save_load_roundtrip_n4() {
        let original = make_test_grid_n4();

        // Save to buffer
        let mut buffer = Vec::new();
        save_grid(&original, &mut buffer).unwrap();

        // Load from buffer
        let mut cursor = Cursor::new(buffer);
        let loaded: InMemoryGrid<4> = load_grid(&mut cursor).unwrap();

        // Verify properties match
        assert_eq!(loaded.num_blocks(), original.num_blocks());
        assert_eq!(loaded.config().grid_dim, original.config().grid_dim);

        // Verify multi-feature queries
        let test_points = vec![
            Point3::new(0.15, 0.15, 0.15),
            Point3::new(0.25, 0.25, 0.25),
        ];

        for point in test_points {
            let orig_val = original.query(point);
            let loaded_val = loaded.query(point);
            match (orig_val, loaded_val) {
                (Some(v1), Some(v2)) => {
                    for i in 0..4 {
                        assert!(
                            (v1[i] - v2[i]).abs() < 1e-6,
                            "Mismatch at {:?} feature {}: {} vs {}",
                            point, i, v1[i], v2[i]
                        );
                    }
                }
                (None, None) => {}
                _ => panic!("Validity mismatch at {:?}", point),
            }
        }
    }

    #[test]
    fn test_invalid_magic() {
        // Create a 32-byte header with wrong magic
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"BADM");
        // Rest can be zeros

        let mut cursor = Cursor::new(data);
        let result: Result<InMemoryGrid<1>> = load_grid(&mut cursor);
        assert!(matches!(result, Err(AshIoError::InvalidFormat { .. })));
    }

    #[test]
    fn test_feature_dimension_mismatch() {
        let grid = make_test_grid_n1();

        // Save as N=1
        let mut buffer = Vec::new();
        save_grid(&grid, &mut buffer).unwrap();

        // Try to load as N=4
        let mut cursor = Cursor::new(buffer);
        let result: Result<InMemoryGrid<4>> = load_grid(&mut cursor);
        assert!(matches!(
            result,
            Err(AshIoError::FeatureDimensionMismatch { expected: 4, got: 1 })
        ));
    }

    #[test]
    fn test_empty_grid_roundtrip() {
        let original: InMemoryGrid<1> =
            GridBuilder::new(8, 0.1).with_capacity(10).build().unwrap();

        let mut buffer = Vec::new();
        save_grid(&original, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let loaded: InMemoryGrid<1> = load_grid(&mut cursor).unwrap();

        assert_eq!(loaded.num_blocks(), 0);
    }

    #[test]
    fn test_negative_coords_roundtrip() {
        let mut builder: GridBuilder<1> = GridBuilder::new(8, 0.1).with_capacity(10);

        builder = builder
            .add_block(BlockCoord::new(-5, -10, 15), vec![[1.0]; 512])
            .unwrap()
            .add_block(BlockCoord::new(100, -200, 300), vec![[2.0]; 512])
            .unwrap();

        let original = builder.build().unwrap();

        let mut buffer = Vec::new();
        save_grid(&original, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let loaded: InMemoryGrid<1> = load_grid(&mut cursor).unwrap();

        assert!(loaded.has_block(BlockCoord::new(-5, -10, 15)));
        assert!(loaded.has_block(BlockCoord::new(100, -200, 300)));
    }
}
