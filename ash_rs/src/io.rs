//! Binary serialization for the .ash file format.
//!
//! This module delegates to `ash_io::format` for actual serialization.
//! Provides backward-compatible API for ash_rs users.

#[cfg(feature = "serde")]
use std::io::{Read, Write};

use crate::error::{AshError, Result};
use crate::grid::SparseDenseGrid;

// Re-export format constants from ash_io
pub use ash_io::{ASH_MAGIC, HEADER_SIZE};

/// Current format version.
pub const ASH_VERSION: u32 = 1;

#[cfg(feature = "serde")]
impl SparseDenseGrid {
    /// Save the grid to a writer in .ash binary format.
    ///
    /// # Performance
    /// Target: < 100ms for 10MB grid to SSD
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::fs::File;
    ///
    /// let mut file = File::create("environment.ash")?;
    /// grid.save(&mut file)?;
    /// ```
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        ash_io::save_grid(self.inner(), writer).map_err(|e| {
            match e {
                ash_io::AshIoError::Io(msg) => AshError::Io(msg),
                ash_io::AshIoError::InvalidFormat { message } => AshError::InvalidFormat { message },
                _ => AshError::InvalidFormat { message: "unexpected error" },
            }
        })
    }

    /// Load a grid from a reader in .ash binary format.
    ///
    /// # Performance
    /// Target: < 50ms for 10MB grid from SSD
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::fs::File;
    ///
    /// let mut file = File::open("environment.ash")?;
    /// let grid = SparseDenseGrid::load(&mut file)?;
    /// ```
    pub fn load<R: Read>(reader: &mut R) -> Result<Self> {
        let inner = ash_io::load_grid::<1, R>(reader).map_err(|e| {
            match e {
                ash_io::AshIoError::Io(msg) => AshError::Io(msg),
                ash_io::AshIoError::InvalidFormat { message } => AshError::InvalidFormat { message },
                ash_io::AshIoError::FeatureDimensionMismatch { expected, got } => {
                    AshError::DimensionMismatch {
                        expected: expected as u32,
                        got: got as u32,
                    }
                }
                _ => AshError::InvalidFormat { message: "unexpected error" },
            }
        })?;
        Ok(Self::from_inner(inner))
    }

    /// Save the grid to a file path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// grid.save_to_file("environment.ash")?;
    /// ```
    #[cfg(feature = "std")]
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.save(&mut file)
    }

    /// Load a grid from a file path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grid = SparseDenseGrid::load_from_file("environment.ash")?;
    /// ```
    #[cfg(feature = "std")]
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        Self::load(&mut file)
    }
}

/// Compute the expected file size for a grid.
///
/// Useful for progress reporting or pre-allocation.
pub fn compute_file_size(num_blocks: usize, grid_dim: u32) -> usize {
    ash_io::compute_file_size::<1>(num_blocks, grid_dim)
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::GridBuilder;
    use ash_core::{BlockCoord, Point3};
    use std::io::Cursor;

    fn make_test_grid() -> SparseDenseGrid {
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

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
    fn test_save_load_roundtrip() {
        let original = make_test_grid();

        // Save to buffer
        let mut buffer = Vec::new();
        original.save(&mut buffer).unwrap();

        // Load from buffer
        let mut cursor = Cursor::new(buffer);
        let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

        // Verify properties match
        assert_eq!(loaded.num_blocks(), original.num_blocks());
        assert_eq!(loaded.config().grid_dim, original.config().grid_dim);
        assert!(
            (loaded.config().cell_size - original.config().cell_size).abs() < 1e-6
        );

        // Verify block coordinates match
        let orig_coords: Vec<_> = original.block_coords().collect();
        let loaded_coords: Vec<_> = loaded.block_coords().collect();
        assert_eq!(orig_coords.len(), loaded_coords.len());

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
                    assert!((v1 - v2).abs() < 1e-6, "Mismatch at {:?}", point);
                }
                (None, None) => {}
                _ => panic!("Validity mismatch at {:?}", point),
            }
        }
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"BADM\x01\x00\x00\x00"; // Wrong magic
        let mut cursor = Cursor::new(data.to_vec());
        let result = SparseDenseGrid::load(&mut cursor);
        assert!(matches!(result, Err(AshError::InvalidFormat { .. })));
    }

    #[test]
    fn test_empty_grid_roundtrip() {
        let original = GridBuilder::new(8, 0.1).with_capacity(10).build().unwrap();

        let mut buffer = Vec::new();
        original.save(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

        assert_eq!(loaded.num_blocks(), 0);
    }

    #[test]
    fn test_compute_file_size() {
        let size = compute_file_size(8, 8);
        // Header: 32 bytes
        // Coords: 8 * 12 = 96 bytes
        // Values: 8 * 512 * 4 = 16384 bytes
        // Total: 16512 bytes
        assert!(size > 16000);
    }

    #[test]
    fn test_negative_coords_roundtrip() {
        let mut builder = GridBuilder::new(8, 0.1).with_capacity(10);

        builder = builder
            .add_block(BlockCoord::new(-5, -10, 15), vec![1.0; 512])
            .unwrap()
            .add_block(BlockCoord::new(100, -200, 300), vec![2.0; 512])
            .unwrap();

        let original = builder.build().unwrap();

        let mut buffer = Vec::new();
        original.save(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

        assert!(loaded.has_block(BlockCoord::new(-5, -10, 15)));
        assert!(loaded.has_block(BlockCoord::new(100, -200, 300)));
    }
}
