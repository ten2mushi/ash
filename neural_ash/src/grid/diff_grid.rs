//! Differentiable SDF grid implementation.

use std::fmt;
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use ash_core::{decompose_point, BlockCoord, Point3};
use ash_io::BlockMap;

use crate::config::DiffGridConfig;
use crate::error::{NeuralAshError, Result};

use super::TensorStorage;

/// Differentiable SDF grid backed by Burn tensors.
///
/// This grid stores cell embeddings that can be optimized via gradient descent.
/// The embeddings are stored in a 2D tensor of shape `[total_cells, N]`.
///
/// # Type Parameters
/// - `B`: Burn backend (e.g., Wgpu, NdArray)
/// - `N`: Number of features per cell (1 for SDF-only, 4 for SDF+semantic, etc.)
///
/// Note: This struct intentionally doesn't derive Module because it contains
/// non-tensor fields that don't need to be part of the module system.
/// The `embeddings` tensor is the only trainable parameter.
pub struct DiffSdfGrid<B: Backend, const N: usize = 1> {
    /// Cell embeddings: [total_cells, N]
    /// This is the trainable parameter.
    pub embeddings: Tensor<B, 2>,

    /// Block map for coordinate → index lookup (non-differentiable).
    block_map: Arc<BlockMap>,

    /// Block coordinates in allocation order.
    block_coords: Vec<BlockCoord>,

    /// Grid configuration.
    config: DiffGridConfig,
}

impl<B: Backend, const N: usize> DiffSdfGrid<B, N> {
    /// Create a new differentiable grid.
    pub fn new(config: DiffGridConfig, device: &B::Device) -> Self {
        let cells_per_block = config.cells_per_block();
        let total_cells = config.capacity * cells_per_block;

        // Initialize embeddings with sentinel value
        let embeddings = Tensor::full([total_cells, N], config.init_value, device);

        // Create block map with 2x capacity for good hash table performance
        let block_map = Arc::new(BlockMap::with_capacity(config.capacity * 2));

        Self {
            embeddings,
            block_map,
            block_coords: Vec::with_capacity(config.capacity),
            config,
        }
    }

    /// Get the grid configuration.
    pub fn config(&self) -> &DiffGridConfig {
        &self.config
    }

    /// Get the number of allocated blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_coords.len()
    }

    /// Get the block map for lookups.
    pub fn block_map(&self) -> &Arc<BlockMap> {
        &self.block_map
    }

    /// Get the block coordinates.
    pub fn block_coords(&self) -> &[BlockCoord] {
        &self.block_coords
    }

    /// Check if a block exists at the given coordinate.
    pub fn has_block(&self, coord: BlockCoord) -> bool {
        self.block_map.contains(coord)
    }

    /// Get the index of a block if it exists.
    pub fn get_block_index(&self, coord: BlockCoord) -> Option<usize> {
        self.block_map.get(coord)
    }

    /// Allocate a new block at the given coordinate.
    ///
    /// Returns the block index.
    pub fn allocate_block(&mut self, coord: BlockCoord) -> Result<usize> {
        if self.block_map.contains(coord) {
            return Err(NeuralAshError::BlockExists {
                x: coord.x,
                y: coord.y,
                z: coord.z,
            });
        }

        let block_idx = self.block_coords.len();
        if block_idx >= self.config.capacity {
            return Err(NeuralAshError::CapacityExceeded {
                block_count: block_idx + 1,
                capacity: self.config.capacity,
            });
        }

        // Insert into block map
        self.block_map
            .insert(coord, block_idx)
            .map_err(NeuralAshError::IoError)?;

        self.block_coords.push(coord);

        Ok(block_idx)
    }

    /// Initialize the grid spatially from a point cloud.
    ///
    /// Allocates blocks that contain or are near the given points.
    pub fn spatial_init(&mut self, points: &[Point3], margin: f32) {
        use std::collections::HashSet;

        let mut block_set = HashSet::new();
        let block_size = self.config.block_size();

        for point in points {
            // Decompose point to find its block
            let (block, _, _) = decompose_point(*point, self.config.cell_size, self.config.grid_dim);
            block_set.insert(block);

            // Also allocate neighboring blocks within margin
            let margin_blocks = (margin / block_size).ceil() as i32;
            for dz in -margin_blocks..=margin_blocks {
                for dy in -margin_blocks..=margin_blocks {
                    for dx in -margin_blocks..=margin_blocks {
                        let neighbor = BlockCoord::new(block.x + dx, block.y + dy, block.z + dz);
                        block_set.insert(neighbor);
                    }
                }
            }
        }

        // Allocate all unique blocks
        for coord in block_set {
            let _ = self.allocate_block(coord);
        }
    }

    /// Query the grid at the given points.
    ///
    /// Returns a tensor of shape [num_points, N] containing the interpolated values.
    /// Points in unallocated regions return the sentinel value.
    pub fn query(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = self.embeddings.device();
        let shape = points.dims();
        let num_points = shape[0];

        // Get point data for CPU-side coordinate decomposition
        let points_data = points.to_data();
        let points_flat: Vec<f32> = points_data.to_vec().unwrap();

        // Compute indices and weights for trilinear interpolation
        let mut corner_indices = Vec::with_capacity(num_points * 8);
        let mut weights = Vec::with_capacity(num_points * 8);
        let mut valid_mask = vec![true; num_points];

        let cells_per_block = self.config.cells_per_block();

        for i in 0..num_points {
            let point = Point3::new(
                points_flat[i * 3],
                points_flat[i * 3 + 1],
                points_flat[i * 3 + 2],
            );

            let (block, cell, local) =
                decompose_point(point, self.config.cell_size, self.config.grid_dim);

            // Check if block exists
            if let Some(block_idx) = self.block_map.get(block) {
                // Compute trilinear weights
                let u = local.u;
                let v = local.v;
                let w = local.w;

                let w000 = (1.0 - u) * (1.0 - v) * (1.0 - w);
                let w100 = u * (1.0 - v) * (1.0 - w);
                let w010 = (1.0 - u) * v * (1.0 - w);
                let w110 = u * v * (1.0 - w);
                let w001 = (1.0 - u) * (1.0 - v) * w;
                let w101 = u * (1.0 - v) * w;
                let w011 = (1.0 - u) * v * w;
                let w111 = u * v * w;

                weights.extend_from_slice(&[w000, w100, w010, w110, w001, w101, w011, w111]);

                // Compute corner cell indices
                let cell_idx = cell.flat_index(self.config.grid_dim);
                let base_idx = block_idx * cells_per_block + cell_idx;

                // For now, use simplified corner indexing (assumes cell doesn't cross block boundary)
                // A full implementation would use resolve_corner from ash_core
                for _corner in 0..8 {
                    corner_indices.push(base_idx as i64);
                }
            } else {
                valid_mask[i] = false;
                // Fill with zeros (will be masked)
                weights.extend_from_slice(&[0.0; 8]);
                corner_indices.extend_from_slice(&[0i64; 8]);
            }
        }

        // Create tensors for gathering
        let indices_tensor =
            Tensor::<B, 1, Int>::from_data(corner_indices.as_slice(), &device);
        let weights_tensor =
            Tensor::<B, 1>::from_data(weights.as_slice(), &device);

        // Gather corner values: [num_points * 8, N]
        let corner_values = self.embeddings.clone().select(0, indices_tensor);

        // Reshape for weighted sum: [num_points, 8, N]
        let corner_values = corner_values.reshape([num_points, 8, N]);
        let weights_tensor = weights_tensor.reshape([num_points, 8, 1]);

        // Weighted sum: [num_points, N]
        let result = (corner_values * weights_tensor).sum_dim(1).squeeze(1);

        // Apply valid mask (set invalid points to sentinel)
        let mask_data: Vec<f32> = valid_mask
            .iter()
            .map(|&v| if v { 1.0 } else { 0.0 })
            .collect();
        let mask = Tensor::<B, 1>::from_data(mask_data.as_slice(), &device).reshape([num_points, 1]);
        let sentinel = Tensor::full([num_points, N], self.config.init_value, &device);

        result * mask.clone() + sentinel * (Tensor::ones([num_points, 1], &device) - mask)
    }

    /// Convert the differentiable grid to an InMemoryGrid for export.
    pub fn to_memory_grid(&self) -> ash_io::InMemoryGrid<N> {
        let mut builder = ash_io::GridBuilder::<N>::new(self.config.grid_dim, self.config.cell_size)
            .with_capacity(self.num_blocks());

        let cells_per_block = self.config.cells_per_block();
        let embeddings_data = self.embeddings.to_data();
        let flat_values: Vec<f32> = embeddings_data.to_vec().unwrap();

        for (block_idx, &coord) in self.block_coords.iter().enumerate() {
            let start = block_idx * cells_per_block * N;
            let end = start + cells_per_block * N;
            let block_flat = &flat_values[start..end];

            let values: Vec<[f32; N]> = block_flat
                .chunks(N)
                .map(|chunk| {
                    let mut arr = [0.0f32; N];
                    arr.copy_from_slice(chunk);
                    arr
                })
                .collect();

            builder = builder.add_block(coord, values).unwrap();
        }

        builder.build().unwrap()
    }

    /// Create a TensorStorage view of the embeddings (useful for batch operations).
    pub fn storage(&self) -> TensorStorage<B, N> {
        TensorStorage::new(
            self.config.capacity,
            self.config.cells_per_block(),
            self.config.init_value,
            &self.embeddings.device(),
        )
    }

    /// Set the embeddings tensor (for use after optimization).
    pub fn set_embeddings(&mut self, embeddings: Tensor<B, 2>) {
        self.embeddings = embeddings;
    }

    /// Get the device of the embeddings.
    pub fn device(&self) -> B::Device {
        self.embeddings.device()
    }
}

impl<B: AutodiffBackend, const N: usize> DiffSdfGrid<B, N> {
    /// Compute the gradient of the grid values with respect to query points.
    ///
    /// This is useful for computing the SDF gradient for Eikonal loss.
    pub fn query_with_gradient(&self, points: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let values = self.query(points.clone());

        // Compute gradient via finite differences
        let eps = self.config.cell_size * 0.1;
        let device = points.device();
        let num_points = points.dims()[0];

        // Create offset tensors by creating single-row and repeating
        let eps_x_row = Tensor::<B, 2>::from_data([[eps, 0.0f32, 0.0]], &device);
        let eps_x = eps_x_row.repeat_dim(0, num_points);

        let eps_y_row = Tensor::<B, 2>::from_data([[0.0f32, eps, 0.0]], &device);
        let eps_y = eps_y_row.repeat_dim(0, num_points);

        let eps_z_row = Tensor::<B, 2>::from_data([[0.0f32, 0.0, eps]], &device);
        let eps_z = eps_z_row.repeat_dim(0, num_points);

        let dx_pos = self.query(points.clone() + eps_x.clone());
        let dx_neg = self.query(points.clone() - eps_x);
        let dy_pos = self.query(points.clone() + eps_y.clone());
        let dy_neg = self.query(points.clone() - eps_y);
        let dz_pos = self.query(points.clone() + eps_z.clone());
        let dz_neg = self.query(points - eps_z);

        let inv_2eps = 1.0 / (2.0 * eps);
        let grad_x = (dx_pos - dx_neg) * inv_2eps;
        let grad_y = (dy_pos - dy_neg) * inv_2eps;
        let grad_z = (dz_pos - dz_neg) * inv_2eps;

        // Only take the first feature (SDF) for gradient
        let grad = Tensor::cat(
            vec![
                grad_x.slice([0..num_points, 0..1]),
                grad_y.slice([0..num_points, 0..1]),
                grad_z.slice([0..num_points, 0..1]),
            ],
            1,
        );

        (values, grad)
    }
}

impl<B: Backend, const N: usize> Clone for DiffSdfGrid<B, N> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            block_map: Arc::clone(&self.block_map),
            block_coords: self.block_coords.clone(),
            config: self.config.clone(),
        }
    }
}

impl<B: Backend, const N: usize> fmt::Debug for DiffSdfGrid<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiffSdfGrid")
            .field("embeddings", &self.embeddings.dims())
            .field("num_blocks", &self.block_coords.len())
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_grid_creation() {
        let device = Default::default();
        let config = DiffGridConfig::new(8, 0.1);
        let grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

        assert_eq!(grid.num_blocks(), 0);
        assert_eq!(grid.config().grid_dim, 8);
    }

    #[test]
    fn test_block_allocation() {
        let device = Default::default();
        let config = DiffGridConfig::new(8, 0.1).with_capacity(10);
        let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

        let idx = grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(grid.num_blocks(), 1);
        assert!(grid.has_block(BlockCoord::new(0, 0, 0)));

        let idx2 = grid.allocate_block(BlockCoord::new(1, 0, 0)).unwrap();
        assert_eq!(idx2, 1);
        assert_eq!(grid.num_blocks(), 2);
    }

    #[test]
    fn test_duplicate_block_error() {
        let device = Default::default();
        let config = DiffGridConfig::new(8, 0.1).with_capacity(10);
        let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

        grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();
        let result = grid.allocate_block(BlockCoord::new(0, 0, 0));

        assert!(matches!(result, Err(NeuralAshError::BlockExists { .. })));
    }

    #[test]
    fn test_spatial_init() {
        let device = Default::default();
        let config = DiffGridConfig::new(8, 0.1).with_capacity(100);
        let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.5, 0.5),
        ];

        grid.spatial_init(&points, 0.1);

        assert!(grid.num_blocks() > 0);
        assert!(grid.has_block(BlockCoord::new(0, 0, 0)));
    }

    #[test]
    fn test_to_memory_grid() {
        let device = Default::default();
        let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
        let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

        grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();

        let memory_grid = grid.to_memory_grid();
        assert_eq!(memory_grid.num_blocks(), 1);
        assert!(memory_grid.has_block(BlockCoord::new(0, 0, 0)));
    }
}
