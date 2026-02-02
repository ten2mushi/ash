//! Grid configuration types.

/// Grid configuration parameters (immutable after construction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridConfig {
    /// Number of cells per axis per block (typically 8).
    pub grid_dim: u32,
    /// World units per cell.
    pub cell_size: f32,
    /// Maximum number of blocks the grid can hold.
    pub capacity: usize,
}

impl GridConfig {
    /// Create a new grid configuration.
    ///
    /// # Arguments
    /// * `grid_dim` - Cells per axis per block (typically 8)
    /// * `cell_size` - World units per cell
    /// * `capacity` - Maximum number of blocks
    #[inline]
    pub const fn new(grid_dim: u32, cell_size: f32, capacity: usize) -> Self {
        Self {
            grid_dim,
            cell_size,
            capacity,
        }
    }

    /// Total number of cells per block (grid_dim³).
    #[inline]
    pub const fn cells_per_block(&self) -> usize {
        (self.grid_dim as usize)
            * (self.grid_dim as usize)
            * (self.grid_dim as usize)
    }

    /// Size of each block in world units.
    #[inline]
    pub fn block_size(&self) -> f32 {
        self.grid_dim as f32 * self.cell_size
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            grid_dim: 8,
            cell_size: 0.1,
            capacity: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_config() {
        let config = GridConfig::new(8, 0.1, 1024);
        assert_eq!(config.cells_per_block(), 512);
        assert!((config.block_size() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_grid_config_default() {
        let config = GridConfig::default();
        assert_eq!(config.grid_dim, 8);
        assert_eq!(config.capacity, 1024);
    }
}
