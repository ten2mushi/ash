//! Tensor-based storage for differentiable grid data.

use burn::prelude::*;

/// Tensor-based storage for grid embeddings.
///
/// Stores cell embeddings in a 2D tensor of shape `[total_cells, N]`
/// where total_cells = capacity * cells_per_block.
#[derive(Debug)]
pub struct TensorStorage<B: Backend, const N: usize> {
    /// Cell embeddings tensor: [total_cells, N]
    embeddings: Tensor<B, 2>,
    /// Number of cells per block
    cells_per_block: usize,
    /// Capacity (max blocks)
    capacity: usize,
}

impl<B: Backend, const N: usize> TensorStorage<B, N> {
    /// Create new tensor storage with the given capacity.
    pub fn new(capacity: usize, cells_per_block: usize, init_value: f32, device: &B::Device) -> Self {
        let total_cells = capacity * cells_per_block;
        let embeddings = Tensor::full([total_cells, N], init_value, device);

        Self {
            embeddings,
            cells_per_block,
            capacity,
        }
    }

    /// Get the embeddings tensor.
    pub fn embeddings(&self) -> &Tensor<B, 2> {
        &self.embeddings
    }

    /// Get a mutable reference to the embeddings tensor.
    pub fn embeddings_mut(&mut self) -> &mut Tensor<B, 2> {
        &mut self.embeddings
    }

    /// Set the embeddings tensor (for use after optimization step).
    pub fn set_embeddings(&mut self, embeddings: Tensor<B, 2>) {
        self.embeddings = embeddings;
    }

    /// Get the total number of cells.
    pub fn total_cells(&self) -> usize {
        self.capacity * self.cells_per_block
    }

    /// Get the number of cells per block.
    pub fn cells_per_block(&self) -> usize {
        self.cells_per_block
    }

    /// Get the capacity (max blocks).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Compute the flat index for a given block and cell.
    #[inline]
    pub fn flat_index(&self, block_idx: usize, cell_idx: usize) -> usize {
        block_idx * self.cells_per_block + cell_idx
    }

    /// Get values at specific indices as a tensor slice.
    ///
    /// Returns a tensor of shape [num_indices, N].
    pub fn get_values_batch(&self, indices: &[usize]) -> Tensor<B, 2> {
        let device = self.embeddings.device();
        let index_tensor = Tensor::<B, 1, Int>::from_data(
            indices.iter().map(|&i| i as i64).collect::<Vec<_>>().as_slice(),
            &device,
        );
        self.embeddings.clone().select(0, index_tensor)
    }

    /// Set values at specific indices.
    ///
    /// `values` should have shape [num_indices, N].
    pub fn set_values_batch(&mut self, indices: &[usize], values: Tensor<B, 2>) {
        let device = self.embeddings.device();
        let index_tensor = Tensor::<B, 1, Int>::from_data(
            indices.iter().map(|&i| i as i64).collect::<Vec<_>>().as_slice(),
            &device,
        );

        // Scatter the values into the embeddings tensor
        self.embeddings = self.embeddings.clone().select_assign(0, index_tensor, values);
    }

    /// Initialize a block with values from a function.
    ///
    /// The function takes (cell_x, cell_y, cell_z) and returns [f32; N].
    pub fn init_block<F>(&mut self, block_idx: usize, grid_dim: u32, mut f: F)
    where
        F: FnMut(u32, u32, u32) -> [f32; N],
    {
        let mut values = Vec::with_capacity(self.cells_per_block * N);

        for z in 0..grid_dim {
            for y in 0..grid_dim {
                for x in 0..grid_dim {
                    let v = f(x, y, z);
                    values.extend_from_slice(&v);
                }
            }
        }

        let device = self.embeddings.device();
        let block_values = Tensor::<B, 2>::from_data(
            TensorData::new(values, [self.cells_per_block, N]),
            &device,
        );

        let start = block_idx * self.cells_per_block;
        let ranges = [start..start + self.cells_per_block, 0..N];
        self.embeddings = self.embeddings.clone().slice_assign(ranges, block_values);
    }

    /// Extract values for a single block as a Vec.
    pub fn extract_block_values(&self, block_idx: usize) -> Vec<[f32; N]> {
        let start = block_idx * self.cells_per_block;
        let end = start + self.cells_per_block;

        let block_tensor = self.embeddings.clone().slice([start..end, 0..N]);
        let data = block_tensor.to_data();
        let flat: Vec<f32> = data.to_vec().unwrap();

        flat.chunks(N)
            .map(|chunk| {
                let mut arr = [0.0f32; N];
                arr.copy_from_slice(chunk);
                arr
            })
            .collect()
    }
}

impl<B: Backend, const N: usize> Clone for TensorStorage<B, N> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            cells_per_block: self.cells_per_block,
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_tensor_storage_creation() {
        let device = Default::default();
        let storage = TensorStorage::<TestBackend, 1>::new(10, 512, 1e9, &device);

        assert_eq!(storage.capacity(), 10);
        assert_eq!(storage.cells_per_block(), 512);
        assert_eq!(storage.total_cells(), 5120);
    }

    #[test]
    fn test_flat_index() {
        let device = Default::default();
        let storage = TensorStorage::<TestBackend, 1>::new(10, 512, 1e9, &device);

        assert_eq!(storage.flat_index(0, 0), 0);
        assert_eq!(storage.flat_index(0, 511), 511);
        assert_eq!(storage.flat_index(1, 0), 512);
        assert_eq!(storage.flat_index(1, 100), 612);
    }
}
