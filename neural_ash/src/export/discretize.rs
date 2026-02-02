//! Discretization of differentiable grids to InMemoryGrid.

#![allow(dead_code)]

use burn::prelude::*;

use ash_core::{BlockCoord, Point3};
use ash_io::{GridBuilder, InMemoryGrid};

use crate::config::DiffGridConfig;
use crate::grid::DiffSdfGrid;

/// Discretize a differentiable grid to an InMemoryGrid.
///
/// This extracts the tensor values and converts them to the ash_io format.
pub fn discretize_grid<B: Backend, const N: usize>(
    grid: &DiffSdfGrid<B, N>,
) -> InMemoryGrid<N> {
    grid.to_memory_grid()
}

/// Configuration for grid discretization.
#[derive(Debug, Clone)]
pub struct DiscretizeConfig {
    /// Whether to apply any post-processing.
    pub apply_smoothing: bool,
    /// Smoothing kernel size (must be odd).
    pub smoothing_kernel_size: usize,
    /// Whether to clamp values to a range.
    pub clamp_values: bool,
    /// Clamp range.
    pub clamp_range: (f32, f32),
}

impl Default for DiscretizeConfig {
    fn default() -> Self {
        Self {
            apply_smoothing: false,
            smoothing_kernel_size: 3,
            clamp_values: false,
            clamp_range: (-10.0, 10.0),
        }
    }
}

/// Discretize with configuration.
pub fn discretize_with_config<B: Backend, const N: usize>(
    grid: &DiffSdfGrid<B, N>,
    _config: &DiscretizeConfig,
) -> InMemoryGrid<N> {
    let memory_grid = grid.to_memory_grid();

    // Apply post-processing if configured
    // (Currently just returns as-is; smoothing would require mutable access to storage)

    memory_grid
}

/// Sample a neural network decoder to create a grid.
///
/// This is useful when you want to evaluate a trained decoder
/// at specific grid points rather than using stored embeddings.
pub fn sample_decoder_to_grid<B: Backend, F>(
    config: &DiffGridConfig,
    blocks: &[BlockCoord],
    query_fn: F,
    device: &B::Device,
) -> InMemoryGrid<1>
where
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    let mut builder = GridBuilder::<1>::new(config.grid_dim, config.cell_size)
        .with_capacity(blocks.len());

    let cells_per_block = config.cells_per_block();
    let block_size = config.block_size();

    for &coord in blocks {
        // Generate query points for this block
        let mut query_points = Vec::with_capacity(cells_per_block * 3);
        let block_origin = Point3::new(
            coord.x as f32 * block_size,
            coord.y as f32 * block_size,
            coord.z as f32 * block_size,
        );

        for z in 0..config.grid_dim {
            for y in 0..config.grid_dim {
                for x in 0..config.grid_dim {
                    let p = block_origin
                        + Point3::new(
                            x as f32 * config.cell_size,
                            y as f32 * config.cell_size,
                            z as f32 * config.cell_size,
                        );
                    query_points.extend_from_slice(&[p.x, p.y, p.z]);
                }
            }
        }

        // Query the decoder
        let points_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(query_points, [cells_per_block, 3]),
            device,
        );
        let sdf_values = query_fn(points_tensor);

        // Extract values
        let sdf_data = sdf_values.to_data();
        let values: Vec<f32> = sdf_data.to_vec().unwrap();

        let block_values: Vec<[f32; 1]> = values.iter().map(|&v| [v]).collect();

        builder = builder.add_block(coord, block_values).unwrap();
    }

    builder.build().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_discretize_grid() {
        let device = Default::default();
        let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
        let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

        grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();

        let memory_grid = discretize_grid(&grid);
        assert_eq!(memory_grid.num_blocks(), 1);
    }

    #[test]
    fn test_sample_decoder() {
        let device = Default::default();
        let config = DiffGridConfig::new(4, 0.1);
        let blocks = vec![BlockCoord::new(0, 0, 0)];

        // Simple sphere SDF query function
        let query_fn = |points: Tensor<TestBackend, 2>| -> Tensor<TestBackend, 2> {
            let center = 0.2f32;
            let radius = 0.15f32;

            // Compute distance from center
            let centered = points - center;
            let dist_sq = (centered.clone() * centered).sum_dim(1);
            let dist = dist_sq.sqrt();

            dist - radius
        };

        let grid = sample_decoder_to_grid(&config, &blocks, query_fn, &device);
        assert_eq!(grid.num_blocks(), 1);
    }
}
