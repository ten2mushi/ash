//! Export to .ash file format.

#![allow(dead_code)]

use std::path::Path;

use ash_io::{save_to_file, InMemoryGrid};

use crate::error::{NeuralAshError, Result};

/// Export an InMemoryGrid to a .ash file.
pub fn export_to_file<const N: usize, P: AsRef<Path>>(
    grid: &InMemoryGrid<N>,
    path: P,
) -> Result<()> {
    save_to_file(grid, path).map_err(|e| NeuralAshError::ExportError {
        message: format!("failed to save .ash file: {}", e),
    })
}

/// Export statistics.
#[derive(Debug, Clone)]
pub struct ExportStats {
    /// Number of blocks exported.
    pub num_blocks: usize,
    /// Total number of cells.
    pub num_cells: usize,
    /// File size in bytes.
    pub file_size: usize,
    /// Grid dimensions.
    pub grid_dim: u32,
    /// Cell size in world units.
    pub cell_size: f32,
}

impl ExportStats {
    /// Create export stats from a grid.
    pub fn from_grid<const N: usize>(grid: &InMemoryGrid<N>) -> Self {
        let config = grid.config();
        let num_blocks = grid.num_blocks();
        let cells_per_block = (config.grid_dim * config.grid_dim * config.grid_dim) as usize;

        Self {
            num_blocks,
            num_cells: num_blocks * cells_per_block,
            file_size: ash_io::compute_file_size(num_blocks, config.grid_dim, N),
            grid_dim: config.grid_dim,
            cell_size: config.cell_size,
        }
    }

    /// Log the export statistics.
    pub fn log(&self, prefix: &str) {
        log::info!(
            "{} blocks={} cells={} file_size={:.2}MB dim={} cell_size={}",
            prefix,
            self.num_blocks,
            self.num_cells,
            self.file_size as f64 / (1024.0 * 1024.0),
            self.grid_dim,
            self.cell_size,
        );
    }
}

/// Export with statistics.
pub fn export_with_stats<const N: usize, P: AsRef<Path>>(
    grid: &InMemoryGrid<N>,
    path: P,
) -> Result<ExportStats> {
    let stats = ExportStats::from_grid(grid);
    export_to_file(grid, path)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash_core::BlockCoord;
    use ash_io::GridBuilder;
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("neural_ash_test_{}.ash", name))
    }

    #[test]
    fn test_export_to_file() {
        let path = temp_path("export");

        // Create a simple grid
        let mut builder = GridBuilder::<1>::new(4, 0.1).with_capacity(1);
        let values = vec![[0.5f32]; 64]; // 4³ cells
        builder = builder.add_block(BlockCoord::new(0, 0, 0), values).unwrap();
        let grid = builder.build().unwrap();

        // Export
        let result = export_to_file(&grid, &path);
        assert!(result.is_ok());

        // Verify file exists
        assert!(path.exists());

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_export_stats() {
        let mut builder = GridBuilder::<1>::new(8, 0.05).with_capacity(10);
        for i in 0..3 {
            let values = vec![[0.0f32]; 512]; // 8³ cells
            builder = builder
                .add_block(BlockCoord::new(i, 0, 0), values)
                .unwrap();
        }
        let grid = builder.build().unwrap();

        let stats = ExportStats::from_grid(&grid);
        assert_eq!(stats.num_blocks, 3);
        assert_eq!(stats.num_cells, 3 * 512);
        assert_eq!(stats.grid_dim, 8);
    }
}
