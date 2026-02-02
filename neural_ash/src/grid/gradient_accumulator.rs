//! Tensor-based gradient accumulator for neural SDF training.
//!
//! Implements `ash_core::GradientAccumulator<N>` using Burn tensors for efficient
//! gradient accumulation during backpropagation through trilinear interpolation.

use std::sync::Arc;

use burn::prelude::*;

use ash_core::{resolve_corner, BlockCoord, CellCoord, GradientAccumulator};
use ash_io::BlockMap;

use crate::config::DiffGridConfig;

/// Tensor-based gradient accumulator that implements ash_core's GradientAccumulator trait.
///
/// This accumulator collects gradients from the backward pass of trilinear interpolation
/// and stores them in a format compatible with Burn's optimizer infrastructure.
///
/// # Type Parameters
/// - `B`: Burn backend (e.g., NdArray, Wgpu)
/// - `N`: Number of features per cell (1 for SDF-only, 4 for SDF+semantic, etc.)
pub struct TensorGradientAccumulator<B: Backend, const N: usize> {
    /// Accumulated gradients: same shape as embeddings [total_cells, N]
    gradients: Vec<f32>,
    /// Block map for coordinate → index lookup
    block_map: Arc<BlockMap>,
    /// Grid configuration
    config: DiffGridConfig,
    /// Number of allocated blocks
    num_blocks: usize,
    /// Device for tensor conversion
    device: B::Device,
}

impl<B: Backend, const N: usize> TensorGradientAccumulator<B, N> {
    /// Create a new gradient accumulator.
    ///
    /// # Arguments
    /// * `block_map` - Block map for coordinate lookups
    /// * `config` - Grid configuration
    /// * `num_blocks` - Number of allocated blocks
    /// * `device` - Burn device for tensor operations
    pub fn new(
        block_map: Arc<BlockMap>,
        config: DiffGridConfig,
        num_blocks: usize,
        device: &B::Device,
    ) -> Self {
        let cells_per_block = config.cells_per_block();
        let total_cells = num_blocks * cells_per_block;

        Self {
            gradients: vec![0.0; total_cells * N],
            block_map,
            config,
            num_blocks,
            device: device.clone(),
        }
    }

    /// Get the number of cells per block.
    fn cells_per_block(&self) -> usize {
        self.config.cells_per_block()
    }

    /// Add a gradient contribution at a specific flat index.
    fn add_gradient_at(&mut self, flat_idx: usize, grad_contribution: &[f32; N]) {
        let start = flat_idx * N;
        if start + N <= self.gradients.len() {
            for i in 0..N {
                self.gradients[start + i] += grad_contribution[i];
            }
        }
    }

    /// Convert accumulated gradients to a Burn tensor.
    ///
    /// The returned tensor has shape `[total_cells, N]` and can be used
    /// with Burn's optimizer infrastructure.
    pub fn to_tensor(&self) -> Tensor<B, 2> {
        let total_cells = self.num_blocks * self.cells_per_block();
        Tensor::from_data(
            TensorData::new(self.gradients.clone(), [total_cells, N]),
            &self.device,
        )
    }

    /// Clear all accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.gradients.fill(0.0);
    }

    /// Get the total number of cells.
    pub fn total_cells(&self) -> usize {
        self.num_blocks * self.cells_per_block()
    }

    /// Get the L2 norm of the gradient (useful for monitoring).
    pub fn gradient_norm(&self) -> f32 {
        let sum_sq: f32 = self.gradients.iter().map(|g| g * g).sum();
        sum_sq.sqrt()
    }
}

impl<B: Backend, const N: usize> GradientAccumulator<N> for TensorGradientAccumulator<B, N> {
    fn accumulate_gradient(
        &mut self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
        weight: f32,
        upstream_grad: [f32; N],
    ) {
        // Resolve corner to actual cell (may cross block boundary)
        let (resolved_block, resolved_cell) =
            resolve_corner(block, cell, corner, self.config.grid_dim);

        // Look up block index
        if let Some(block_idx) = self.block_map.get(resolved_block) {
            if block_idx >= self.num_blocks {
                return;
            }

            let cell_idx = resolved_cell.flat_index(self.config.grid_dim);
            let flat_idx = block_idx * self.cells_per_block() + cell_idx;

            // Compute weighted gradient contribution
            let mut grad_contribution = [0.0f32; N];
            for i in 0..N {
                grad_contribution[i] = weight * upstream_grad[i];
            }

            self.add_gradient_at(flat_idx, &grad_contribution);
        }
    }
}

/// Batch gradient accumulator that processes multiple points efficiently.
///
/// This version accumulates gradients in batches before converting to tensor,
/// which is more efficient for large training batches.
pub struct BatchGradientAccumulator<B: Backend, const N: usize> {
    /// Base accumulator
    inner: TensorGradientAccumulator<B, N>,
    /// Number of samples accumulated (for averaging)
    sample_count: usize,
}

impl<B: Backend, const N: usize> BatchGradientAccumulator<B, N> {
    /// Create a new batch gradient accumulator.
    pub fn new(
        block_map: Arc<BlockMap>,
        config: DiffGridConfig,
        num_blocks: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            inner: TensorGradientAccumulator::new(block_map, config, num_blocks, device),
            sample_count: 0,
        }
    }

    /// Increment the sample count.
    pub fn increment_samples(&mut self) {
        self.sample_count += 1;
    }

    /// Get the number of samples accumulated.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Convert to averaged gradient tensor.
    ///
    /// Divides accumulated gradients by sample count to get mean gradient.
    pub fn to_averaged_tensor(&self) -> Tensor<B, 2> {
        if self.sample_count == 0 {
            return self.inner.to_tensor();
        }

        let scale = 1.0 / self.sample_count as f32;
        self.inner.to_tensor() * scale
    }

    /// Reset for next batch.
    pub fn reset(&mut self) {
        self.inner.zero_grad();
        self.sample_count = 0;
    }

    /// Get gradient norm (before averaging).
    pub fn gradient_norm(&self) -> f32 {
        self.inner.gradient_norm()
    }
}

impl<B: Backend, const N: usize> GradientAccumulator<N> for BatchGradientAccumulator<B, N> {
    fn accumulate_gradient(
        &mut self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
        weight: f32,
        upstream_grad: [f32; N],
    ) {
        self.inner
            .accumulate_gradient(block, cell, corner, weight, upstream_grad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    fn create_test_accumulator() -> TensorGradientAccumulator<TestBackend, 1> {
        let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
        let block_map = Arc::new(BlockMap::with_capacity(20));
        block_map.insert(BlockCoord::new(0, 0, 0), 0).unwrap();
        block_map.insert(BlockCoord::new(1, 0, 0), 1).unwrap();

        let device = Default::default();
        TensorGradientAccumulator::new(block_map, config, 2, &device)
    }

    #[test]
    fn test_accumulator_creation() {
        let accum = create_test_accumulator();
        assert_eq!(accum.total_cells(), 2 * 64); // 2 blocks * 4^3 cells
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut accum = create_test_accumulator();

        // Accumulate a gradient at cell (0,0,0) in block (0,0,0)
        accum.accumulate_gradient(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
            0.5,
            [2.0],
        );

        // The gradient should be 0.5 * 2.0 = 1.0
        let tensor = accum.to_tensor();
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulation_additive() {
        let mut accum = create_test_accumulator();

        // Accumulate multiple gradients at the same cell
        accum.accumulate_gradient(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
            0.5,
            [2.0],
        );
        accum.accumulate_gradient(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
            0.3,
            [1.0],
        );

        // Total should be 0.5*2.0 + 0.3*1.0 = 1.3
        let tensor = accum.to_tensor();
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
        assert!((data[0] - 1.3).abs() < 1e-6);
    }

    #[test]
    fn test_zero_grad() {
        let mut accum = create_test_accumulator();

        accum.accumulate_gradient(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
            1.0,
            [5.0],
        );

        accum.zero_grad();

        let tensor = accum.to_tensor();
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
        assert!(data.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_cross_block_boundary() {
        let mut accum = create_test_accumulator();

        // Accumulate at cell (3,3,3) with corner (1,1,1) which would go to next block
        // Since we have block (1,0,0), this should work for the x direction
        accum.accumulate_gradient(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(3, 0, 0),
            (1, 0, 0), // +x corner crosses to block (1,0,0)
            0.5,
            [4.0],
        );

        let tensor = accum.to_tensor();
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

        // The gradient should be accumulated in block 1, cell (0,0,0)
        let block1_start = 64; // block 1 starts at index 64
        assert!((data[block1_start] - 2.0).abs() < 1e-6); // 0.5 * 4.0 = 2.0
    }

    #[test]
    fn test_batch_accumulator() {
        let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
        let block_map = Arc::new(BlockMap::with_capacity(20));
        block_map.insert(BlockCoord::new(0, 0, 0), 0).unwrap();

        let device = Default::default();
        let mut batch_accum =
            BatchGradientAccumulator::<TestBackend, 1>::new(block_map, config, 1, &device);

        // Accumulate gradients for 2 samples
        batch_accum.accumulate_gradient(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
            1.0,
            [4.0],
        );
        batch_accum.increment_samples();

        batch_accum.accumulate_gradient(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            (0, 0, 0),
            1.0,
            [2.0],
        );
        batch_accum.increment_samples();

        // Averaged gradient should be (4.0 + 2.0) / 2 = 3.0
        let tensor = batch_accum.to_averaged_tensor();
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
        assert!((data[0] - 3.0).abs() < 1e-6);
    }
}
