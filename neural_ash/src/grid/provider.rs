//! CellValueProvider implementation for tensor-backed grids.

use burn::prelude::*;

use ash_core::{resolve_corner, BlockCoord, CellCoord, CellValueProvider, UNTRAINED_SENTINEL};
use ash_io::BlockMap;

use crate::config::DiffGridConfig;

/// A CPU-side view of the tensor grid for implementing CellValueProvider.
///
/// This extracts values from the GPU tensor to CPU for use with ash_core's
/// interpolation algorithms.
pub struct TensorCellValueProvider<const N: usize> {
    /// Flat array of all cell values.
    values: Vec<f32>,
    /// Block map for lookups.
    block_map: std::sync::Arc<BlockMap>,
    /// Grid configuration.
    config: DiffGridConfig,
    /// Number of allocated blocks.
    num_blocks: usize,
}

impl<const N: usize> TensorCellValueProvider<N> {
    /// Create a new provider from tensor data.
    pub fn new<B: Backend>(
        embeddings: &Tensor<B, 2>,
        block_map: std::sync::Arc<BlockMap>,
        config: DiffGridConfig,
        num_blocks: usize,
    ) -> Self {
        let data = embeddings.to_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        Self {
            values,
            block_map,
            config,
            num_blocks,
        }
    }

    /// Get the number of cells per block.
    fn cells_per_block(&self) -> usize {
        self.config.cells_per_block()
    }
}

impl<const N: usize> CellValueProvider<N> for TensorCellValueProvider<N> {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
    ) -> Option<[f32; N]> {
        // Resolve corner that may cross block boundary
        let (resolved_block, resolved_cell) =
            resolve_corner(block, cell, corner, self.config.grid_dim);

        // Look up block index
        let block_idx = self.block_map.get(resolved_block)?;

        if block_idx >= self.num_blocks {
            return None;
        }

        // Compute flat cell index
        let cell_idx = resolved_cell.flat_index(self.config.grid_dim);
        let flat_idx = block_idx * self.cells_per_block() + cell_idx;
        let value_start = flat_idx * N;

        if value_start + N > self.values.len() {
            return None;
        }

        // Extract N values
        let mut result = [0.0f32; N];
        result.copy_from_slice(&self.values[value_start..value_start + N]);

        // Check for untrained sentinel
        if result[0] >= UNTRAINED_SENTINEL * 0.5 {
            return None;
        }

        Some(result)
    }

    fn grid_dim(&self) -> u32 {
        self.config.grid_dim
    }

    fn cell_size(&self) -> f32 {
        self.config.cell_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash_io::BlockMap;
    use burn::backend::NdArray;
    use std::sync::Arc;

    type TestBackend = NdArray;

    #[test]
    fn test_provider_creation() {
        let device = Default::default();
        let config = DiffGridConfig::new(4, 0.1).with_capacity(2);
        let cells_per_block = config.cells_per_block();

        // Create a tensor with known values
        let embeddings = Tensor::<TestBackend, 2>::zeros([2 * cells_per_block, 1], &device);

        let block_map = Arc::new(BlockMap::with_capacity(4));
        block_map.insert(BlockCoord::new(0, 0, 0), 0).unwrap();

        let provider = TensorCellValueProvider::<1>::new(&embeddings, block_map, config, 1);

        assert_eq!(provider.grid_dim(), 4);
        assert_eq!(provider.cell_size(), 0.1);
    }

    #[test]
    fn test_provider_get_values() {
        let device = Default::default();
        let config = DiffGridConfig::new(4, 0.1).with_capacity(2);
        let cells_per_block = config.cells_per_block();

        // Create embeddings initialized to 0.5
        let embeddings =
            Tensor::<TestBackend, 2>::full([2 * cells_per_block, 1], 0.5, &device);

        let block_map = Arc::new(BlockMap::with_capacity(4));
        block_map.insert(BlockCoord::new(0, 0, 0), 0).unwrap();

        let provider = TensorCellValueProvider::<1>::new(&embeddings, block_map, config, 1);

        // Should be able to get values from allocated block
        let values = provider.get_corner_values(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
        );
        assert!(values.is_some());
        assert!((values.unwrap()[0] - 0.5).abs() < 1e-6);

        // Should return None for unallocated block
        let values = provider.get_corner_values(
            BlockCoord::new(1, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
        );
        assert!(values.is_none());
    }
}
