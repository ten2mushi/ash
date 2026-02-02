//! Block storage with SoA (Structure of Arrays) layout for N-feature grids.
//!
//! Memory layout: values[feature_idx][block_idx * cells_per_block + cell_idx]
//! This provides optimal cache utilization for batch queries on individual features.

use ash_core::BlockCoord;

use crate::config::GridConfig;

/// Structure-of-Arrays storage for block data with N features per cell.
///
/// This layout provides optimal cache utilization for batch queries,
/// as all values for a single feature dimension are stored contiguously.
///
/// # Memory Layout
///
/// For N features, we store N separate arrays:
/// - `values[0]` contains all SDF (or first feature) values
/// - `values[1]` contains all second feature values
/// - etc.
///
/// Within each array, values are ordered by `block_idx * cells_per_block + cell_idx`.
pub struct BlockStorage<const N: usize> {
    /// Feature arrays (one contiguous array per feature dimension).
    /// Layout: values[feature_idx][block_idx * cells_per_block + cell_idx]
    pub(crate) values: [Box<[f32]>; N],

    /// Block coordinates for reverse lookup.
    pub(crate) coords: Box<[BlockCoord]>,

    /// Number of cells per block (grid_dim³).
    pub(crate) cells_per_block: usize,

    /// Current number of allocated blocks.
    pub(crate) num_blocks: usize,
}

impl<const N: usize> BlockStorage<N> {
    /// Create new block storage with the given configuration.
    pub fn new(config: &GridConfig) -> Self {
        let cells_per_block = config.cells_per_block();
        let total_cells = config.capacity * cells_per_block;

        // Initialize each feature array with untrained sentinel
        let values: [Box<[f32]>; N] = core::array::from_fn(|_| {
            vec![ash_core::UNTRAINED_SENTINEL; total_cells].into_boxed_slice()
        });

        let coords = vec![BlockCoord::default(); config.capacity].into_boxed_slice();

        Self {
            values,
            coords,
            cells_per_block,
            num_blocks: 0,
        }
    }

    /// Get all N feature values at a specific cell.
    #[inline]
    pub fn get_values(&self, block_idx: usize, cell_idx: usize) -> [f32; N] {
        let idx = block_idx * self.cells_per_block + cell_idx;
        let mut result = [0.0f32; N];
        for i in 0..N {
            result[i] = self.values[i][idx];
        }
        result
    }

    /// Set all N feature values at a specific cell.
    #[inline]
    pub fn set_values(&mut self, block_idx: usize, cell_idx: usize, values: [f32; N]) {
        let idx = block_idx * self.cells_per_block + cell_idx;
        for i in 0..N {
            self.values[i][idx] = values[i];
        }
    }

    /// Get a single feature value at a specific cell.
    #[inline]
    pub fn get_value(&self, block_idx: usize, cell_idx: usize, feature_idx: usize) -> f32 {
        let idx = block_idx * self.cells_per_block + cell_idx;
        self.values[feature_idx][idx]
    }

    /// Set a single feature value at a specific cell.
    #[inline]
    pub fn set_value(&mut self, block_idx: usize, cell_idx: usize, feature_idx: usize, value: f32) {
        let idx = block_idx * self.cells_per_block + cell_idx;
        self.values[feature_idx][idx] = value;
    }

    /// Get a slice of the first feature (SDF) values for a specific block.
    #[inline]
    pub fn block_sdf_values(&self, block_idx: usize) -> &[f32] {
        let start = block_idx * self.cells_per_block;
        let end = start + self.cells_per_block;
        &self.values[0][start..end]
    }

    /// Get a mutable slice of the first feature (SDF) values for a specific block.
    #[inline]
    pub fn block_sdf_values_mut(&mut self, block_idx: usize) -> &mut [f32] {
        let start = block_idx * self.cells_per_block;
        let end = start + self.cells_per_block;
        &mut self.values[0][start..end]
    }

    /// Get a specific feature array slice for a block.
    #[inline]
    pub fn block_feature_values(&self, block_idx: usize, feature_idx: usize) -> &[f32] {
        let start = block_idx * self.cells_per_block;
        let end = start + self.cells_per_block;
        &self.values[feature_idx][start..end]
    }

    /// Get a mutable feature array slice for a block.
    #[inline]
    pub fn block_feature_values_mut(&mut self, block_idx: usize, feature_idx: usize) -> &mut [f32] {
        let start = block_idx * self.cells_per_block;
        let end = start + self.cells_per_block;
        &mut self.values[feature_idx][start..end]
    }

    /// Get the entire feature array (for SIMD batch operations).
    #[inline]
    pub fn feature_array(&self, feature_idx: usize) -> &[f32] {
        &self.values[feature_idx]
    }

    /// Get the coordinate of a block by its index.
    #[inline]
    pub fn get_coord(&self, block_idx: usize) -> BlockCoord {
        self.coords[block_idx]
    }

    /// Number of allocated blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Number of cells per block.
    #[inline]
    pub fn cells_per_block(&self) -> usize {
        self.cells_per_block
    }

    /// Allocate a new block and return its index.
    /// Returns None if capacity is exceeded.
    pub fn allocate_block(&mut self, coord: BlockCoord) -> Option<usize> {
        if self.num_blocks >= self.coords.len() {
            return None;
        }
        let idx = self.num_blocks;
        self.coords[idx] = coord;
        self.num_blocks += 1;
        Some(idx)
    }

    /// Set all feature values for an entire block at once.
    ///
    /// # Arguments
    /// * `block_idx` - Block index
    /// * `values` - Array of N feature arrays, each with cells_per_block values
    pub fn set_block_values(&mut self, block_idx: usize, values: &[[f32; N]]) {
        debug_assert_eq!(values.len(), self.cells_per_block);
        for (cell_idx, cell_values) in values.iter().enumerate() {
            self.set_values(block_idx, cell_idx, *cell_values);
        }
    }

    /// Set all values for a single feature of an entire block.
    pub fn set_block_feature(&mut self, block_idx: usize, feature_idx: usize, values: &[f32]) {
        debug_assert_eq!(values.len(), self.cells_per_block);
        let dest = self.block_feature_values_mut(block_idx, feature_idx);
        dest.copy_from_slice(values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_n1() {
        let config = GridConfig::new(8, 0.1, 10);
        let mut storage: BlockStorage<1> = BlockStorage::new(&config);

        let coord = BlockCoord::new(1, 2, 3);
        let idx = storage.allocate_block(coord).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(storage.num_blocks(), 1);
        assert_eq!(storage.get_coord(idx), coord);

        // Test single value operations
        storage.set_values(idx, 0, [0.5]);
        assert_eq!(storage.get_values(idx, 0), [0.5]);
        assert_eq!(storage.get_value(idx, 0, 0), 0.5);
    }

    #[test]
    fn test_storage_n4() {
        let config = GridConfig::new(8, 0.1, 10);
        let mut storage: BlockStorage<4> = BlockStorage::new(&config);

        let coord = BlockCoord::new(0, 0, 0);
        let idx = storage.allocate_block(coord).unwrap();

        // Set 4 features at a cell
        let values = [0.1, 0.2, 0.3, 0.4];
        storage.set_values(idx, 0, values);

        assert_eq!(storage.get_values(idx, 0), values);
        assert_eq!(storage.get_value(idx, 0, 0), 0.1);
        assert_eq!(storage.get_value(idx, 0, 3), 0.4);
    }

    #[test]
    fn test_feature_array_access() {
        let config = GridConfig::new(4, 0.1, 2); // 4³ = 64 cells per block
        let mut storage: BlockStorage<2> = BlockStorage::new(&config);

        storage.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();
        storage.allocate_block(BlockCoord::new(1, 0, 0)).unwrap();

        // Set values in block 0
        storage.set_value(0, 10, 0, 1.0);
        storage.set_value(0, 10, 1, 2.0);

        // Set values in block 1
        storage.set_value(1, 10, 0, 3.0);
        storage.set_value(1, 10, 1, 4.0);

        // Access feature arrays
        let sdf_array = storage.feature_array(0);
        assert_eq!(sdf_array[10], 1.0);
        assert_eq!(sdf_array[64 + 10], 3.0); // Block 1, cell 10
    }

    #[test]
    fn test_block_feature_slices() {
        let config = GridConfig::new(4, 0.1, 2);
        let mut storage: BlockStorage<3> = BlockStorage::new(&config);

        let idx = storage.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();

        // Set feature 1 for entire block
        let feature_data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        storage.set_block_feature(idx, 1, &feature_data);

        let slice = storage.block_feature_values(idx, 1);
        assert_eq!(slice.len(), 64);
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[63], 63.0);
    }
}
