//! Differentiable grid configuration.

use burn::config::Config;

/// Configuration for the differentiable SDF grid.
#[derive(Config, Debug)]
pub struct DiffGridConfig {
    /// Number of cells per axis in each block.
    pub grid_dim: u32,

    /// Size of each cell in world units.
    pub cell_size: f32,

    /// Maximum number of blocks to allocate.
    #[config(default = 10000)]
    pub capacity: usize,

    /// Margin around points for block allocation (world units).
    #[config(default = 0.2)]
    pub allocation_margin: f32,

    /// Initial value for uninitialized embeddings.
    /// Using a large value indicates "untrained" regions.
    #[config(default = 1e9)]
    pub init_value: f32,
}

impl DiffGridConfig {
    /// Compute the size of a block in world units.
    #[inline]
    pub fn block_size(&self) -> f32 {
        self.grid_dim as f32 * self.cell_size
    }

    /// Compute the number of cells per block.
    #[inline]
    pub fn cells_per_block(&self) -> usize {
        (self.grid_dim * self.grid_dim * self.grid_dim) as usize
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.grid_dim == 0 {
            return Err("grid_dim must be positive".to_string());
        }
        if self.cell_size <= 0.0 {
            return Err("cell_size must be positive".to_string());
        }
        if self.capacity == 0 {
            return Err("capacity must be positive".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DiffGridConfig::new(8, 0.05);
        assert_eq!(config.grid_dim, 8);
        assert_eq!(config.cell_size, 0.05);
        assert_eq!(config.capacity, 10000);
    }

    #[test]
    fn test_block_size() {
        let config = DiffGridConfig::new(8, 0.1);
        assert!((config.block_size() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cells_per_block() {
        let config = DiffGridConfig::new(8, 0.1);
        assert_eq!(config.cells_per_block(), 512);
    }

    #[test]
    fn test_validation() {
        let valid = DiffGridConfig::new(8, 0.05);
        assert!(valid.validate().is_ok());

        let invalid_dim = DiffGridConfig {
            grid_dim: 0,
            ..valid.clone()
        };
        assert!(invalid_dim.validate().is_err());

        let invalid_size = DiffGridConfig {
            cell_size: -0.1,
            ..valid.clone()
        };
        assert!(invalid_size.validate().is_err());
    }
}
