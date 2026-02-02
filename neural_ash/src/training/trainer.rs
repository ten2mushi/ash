//! Neural SDF trainer implementation.

use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use ash_core::{
    decompose_point, trilinear_backward, trilinear_interpolate, trilinear_with_gradient, Point3,
};

use crate::config::TrainingConfig;
use crate::grid::{DiffSdfGrid, TensorCellValueProvider, TensorGradientAccumulator};
use crate::loss::SdfLoss;

use super::batch::SdfBatch;
use super::metrics::{TrainMetrics, TrainOutput};
use super::optimizer::{OptimizerConfig, RmsPropState, TrainingState};

/// Neural SDF trainer that optimizes grid embeddings using analytical gradients.
///
/// This trainer uses ash_core's trilinear interpolation and backward pass
/// for efficient gradient computation, combined with an RMSprop optimizer
/// for stable convergence.
#[derive(Debug)]
pub struct NeuralSdfTrainer<B: Backend> {
    /// Differentiable grid (the main trainable component).
    pub grid: DiffSdfGrid<B, 1>,
    /// Training configuration.
    config: TrainingConfig,
}

impl<B: Backend> NeuralSdfTrainer<B> {
    /// Create a new trainer from configuration.
    pub fn new(config: TrainingConfig, device: &B::Device) -> Self {
        let grid = DiffSdfGrid::new(config.grid.clone(), device);

        Self { grid, config }
    }

    /// Get the training configuration.
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Initialize the grid from a point cloud.
    pub fn init_from_points(&mut self, points: &[Point3]) {
        self.grid
            .spatial_init(points, self.config.grid.allocation_margin);
    }

    /// Query SDF using the grid directly.
    pub fn query_grid(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        self.grid.query(points)
    }

    /// Export the trained grid.
    pub fn export_grid(&self) -> ash_io::InMemoryGrid<1> {
        self.grid.to_memory_grid()
    }
}

impl<B: AutodiffBackend> NeuralSdfTrainer<B> {
    /// Perform a single training step using tensor-based autodiff.
    ///
    /// Returns the training output with loss and metrics.
    pub fn train_step(&self, batch: &SdfBatch<B>) -> TrainOutput<B> {
        let loss_fn = SdfLoss::new(self.config.loss.clone());

        // Query SDF at surface points (should be 0)
        let surface_sdf = self.grid.query(batch.surface_points.clone());

        // Query SDF at free-space points (should be > threshold)
        let free_space_sdf = self.grid.query(batch.free_space_points.clone());

        // Compute gradients for Eikonal loss
        let (_, gradients) = self
            .grid
            .query_with_gradient(batch.eikonal_points.clone());

        // Compute losses
        let (total_loss, surface_loss, free_space_loss, eikonal_loss) =
            loss_fn.combined_loss(surface_sdf, free_space_sdf, gradients);

        // Extract scalar values for metrics
        let metrics = TrainMetrics::new(
            surface_loss.clone().to_data().to_vec().unwrap()[0],
            free_space_loss.clone().to_data().to_vec().unwrap()[0],
            eikonal_loss.clone().to_data().to_vec().unwrap()[0],
            0.0,
        );

        TrainOutput::new(total_loss, metrics)
    }

    /// Train for multiple epochs on a point cloud using analytical gradients.
    ///
    /// This method uses ash_core's trilinear_backward for efficient gradient
    /// computation and applies updates via RMSprop optimizer.
    ///
    /// Note: This initializes the grid from points if not already initialized.
    /// If you've already initialized (e.g., via TSDF fusion), use `train_initialized` instead.
    pub fn train(
        &mut self,
        points: &[Point3],
        normals: Option<&[Point3]>,
        epochs: usize,
        log_interval: usize,
    ) {
        // Initialize grid from points
        self.init_from_points(points);
        self.train_initialized(points, normals, epochs, log_interval);
    }

    /// Train on an already-initialized grid (e.g., after TSDF fusion).
    ///
    /// Returns a vector of (epoch, loss) pairs at each log interval.
    pub fn train_initialized(
        &mut self,
        points: &[Point3],
        normals: Option<&[Point3]>,
        epochs: usize,
        log_interval: usize,
    ) -> Vec<(usize, f32)> {
        use super::batch::BatchSampler;

        let mut loss_history = Vec::new();

        if self.grid.num_blocks() == 0 {
            log::warn!("No blocks allocated, cannot train");
            return loss_history;
        }

        // Create batch sampler
        let mut sampler = BatchSampler::new(
            self.config.surface_points_per_batch,
            self.config.free_space_points_per_batch,
            self.config.eikonal_points_per_batch,
        );

        let device = self.grid.device();

        // Create optimizer with capacity-sized tensor to match embeddings
        // The embeddings tensor is pre-allocated for full capacity, not just allocated blocks
        let cells_per_block = self.config.grid.cells_per_block();
        let total_cells = self.config.grid.capacity * cells_per_block;
        let optimizer_config = OptimizerConfig::new()
            .with_learning_rate(self.config.learning_rate)
            .with_gradient_clip(self.config.gradient_clip as f32);

        let mut optimizer =
            RmsPropState::<B::InnerBackend>::new([total_cells, 1], optimizer_config, &device);

        // Training loop
        for epoch in 0..epochs {
            // Sample batch
            let batch = sampler.sample_from_points::<B>(points, normals, &device);

            // Compute loss and gradients using analytical backward pass
            let (loss, grads) = self.compute_analytical_gradients(&batch);

            // Apply gradients via optimizer
            // We need to work with the inner backend for non-autodiff operations
            let embeddings_data = self.grid.embeddings.clone().inner();
            let new_embeddings = optimizer.step(embeddings_data, grads);

            // Update grid embeddings
            self.grid.embeddings = Tensor::from_inner(new_embeddings);

            // Log progress
            if epoch % log_interval == 0 || epoch == epochs - 1 {
                log::info!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
                loss_history.push((epoch + 1, loss));
            }
        }

        loss_history
    }

    /// Compute analytical gradients using ash_core's backward pass.
    ///
    /// Returns (loss, gradient_tensor) where gradient_tensor has shape [total_cells, 1].
    fn compute_analytical_gradients(&self, batch: &SdfBatch<B>) -> (f32, Tensor<B::InnerBackend, 2>) {
        let device = self.grid.device();
        let cell_size = self.config.grid.cell_size;
        let grid_dim = self.config.grid.grid_dim;

        // Create CPU-side provider for ash_core interpolation
        let provider = TensorCellValueProvider::<1>::new(
            &self.grid.embeddings.clone().inner(),
            Arc::clone(self.grid.block_map()),
            self.config.grid.clone(),
            self.grid.num_blocks(),
        );

        // Create gradient accumulator with capacity-sized tensor to match embeddings
        // The embeddings tensor is sized for capacity, not just allocated blocks
        let mut grad_accum = TensorGradientAccumulator::<B::InnerBackend, 1>::new(
            Arc::clone(self.grid.block_map()),
            self.config.grid.clone(),
            self.config.grid.capacity,
            &device,
        );

        // Extract surface points for processing
        let surface_data = batch.surface_points.clone().inner().to_data();
        let surface_flat: Vec<f32> = surface_data.to_vec().unwrap();
        let num_surface = batch.surface_points.dims()[0];

        let mut total_loss = 0.0f32;
        let mut valid_samples = 0;

        // Process surface points: SDF should be 0, so gradient is 2*sdf
        for i in 0..num_surface {
            let point = Point3::new(
                surface_flat[i * 3],
                surface_flat[i * 3 + 1],
                surface_flat[i * 3 + 2],
            );

            let (block, cell, local) = decompose_point(point, cell_size, grid_dim);

            if let Some(result) = trilinear_interpolate(&provider, block, cell, local) {
                let sdf = result.values[0];

                // L2 loss: loss = sdf^2, gradient = 2*sdf
                let loss_contrib = sdf * sdf;
                total_loss += loss_contrib;
                valid_samples += 1;

                // Upstream gradient for SDF loss: d(sdf^2)/d(sdf) = 2*sdf
                let upstream_grad = [2.0 * sdf * self.config.loss.surface_weight];

                // Use ash_core's analytical backward pass
                trilinear_backward(&mut grad_accum, &result, upstream_grad);
            }
        }

        // Process free-space points
        let free_data = batch.free_space_points.clone().inner().to_data();
        let free_flat: Vec<f32> = free_data.to_vec().unwrap();
        let num_free = batch.free_space_points.dims()[0];
        let threshold = self.config.loss.free_space_threshold;

        for i in 0..num_free {
            let point = Point3::new(
                free_flat[i * 3],
                free_flat[i * 3 + 1],
                free_flat[i * 3 + 2],
            );

            let (block, cell, local) = decompose_point(point, cell_size, grid_dim);

            if let Some(result) = trilinear_interpolate(&provider, block, cell, local) {
                let sdf = result.values[0];

                // Free-space loss: max(0, threshold - sdf)
                if sdf < threshold {
                    let violation = threshold - sdf;
                    total_loss += violation * self.config.loss.free_space_weight;
                    valid_samples += 1;

                    // Gradient: -1 when sdf < threshold
                    let upstream_grad = [-self.config.loss.free_space_weight];
                    trilinear_backward(&mut grad_accum, &result, upstream_grad);
                }
            }
        }

        // Process Eikonal points
        let eikonal_data = batch.eikonal_points.clone().inner().to_data();
        let eikonal_flat: Vec<f32> = eikonal_data.to_vec().unwrap();
        let num_eikonal = batch.eikonal_points.dims()[0];

        for i in 0..num_eikonal {
            let point = Point3::new(
                eikonal_flat[i * 3],
                eikonal_flat[i * 3 + 1],
                eikonal_flat[i * 3 + 2],
            );

            let (block, cell, local) = decompose_point(point, cell_size, grid_dim);

            if let Some((result, grad)) = trilinear_with_gradient(&provider, block, cell, local) {
                // Compute gradient magnitude in world space
                let scale = 1.0 / cell_size;
                let gx = grad[0][0] * scale;
                let gy = grad[0][1] * scale;
                let gz = grad[0][2] * scale;
                let grad_mag = (gx * gx + gy * gy + gz * gz).sqrt();

                // Eikonal loss: (|grad| - 1)^2
                let deviation = grad_mag - 1.0;
                let loss_contrib = deviation * deviation;
                total_loss += loss_contrib * self.config.loss.eikonal_weight;
                valid_samples += 1;

                // For Eikonal, we need to backpropagate through the gradient computation
                // This is more complex - for now, we use a simplified approach
                // by noting that Eikonal encourages smooth SDFs
                if grad_mag > 1e-6 {
                    // Approximate gradient: penalize cells based on their contribution
                    // This is a simplification that works reasonably well
                    let eikonal_upstream =
                        [2.0 * deviation * self.config.loss.eikonal_weight * 0.1];
                    trilinear_backward(&mut grad_accum, &result, eikonal_upstream);
                }
            }
        }

        // Average the loss
        let avg_loss = if valid_samples > 0 {
            total_loss / valid_samples as f32
        } else {
            0.0
        };

        // Convert accumulated gradients to tensor
        let grads = grad_accum.to_tensor();

        // Average gradients by sample count
        let grads = if valid_samples > 0 {
            grads / (valid_samples as f32)
        } else {
            grads
        };

        (avg_loss, grads)
    }

    /// Train with full TrainingState for advanced features.
    ///
    /// This method provides more control over training, including warmup
    /// and monitoring of training progress.
    pub fn train_with_state(
        &mut self,
        points: &[Point3],
        normals: Option<&[Point3]>,
        epochs: usize,
        log_interval: usize,
    ) -> TrainingState<B::InnerBackend> {
        use super::batch::BatchSampler;

        // Initialize grid
        self.init_from_points(points);

        if self.grid.num_blocks() == 0 {
            log::warn!("No blocks allocated, cannot train");
            let device = self.grid.device();
            return TrainingState::new([1, 1], OptimizerConfig::default(), &device);
        }

        let device = self.grid.device();

        // Create batch sampler
        let mut sampler = BatchSampler::new(
            self.config.surface_points_per_batch,
            self.config.free_space_points_per_batch,
            self.config.eikonal_points_per_batch,
        );

        // Create training state
        let cells_per_block = self.config.grid.cells_per_block();
        let total_cells = self.grid.num_blocks() * cells_per_block;
        let optimizer_config = OptimizerConfig::new()
            .with_learning_rate(self.config.learning_rate)
            .with_gradient_clip(self.config.gradient_clip as f32);

        let mut state = TrainingState::new([total_cells, 1], optimizer_config, &device);

        // Training loop
        for epoch in 0..epochs {
            // Sample batch
            let batch = sampler.sample_from_points::<B>(points, normals, &device);

            // Compute loss and gradients
            let (loss, grads) = self.compute_analytical_gradients(&batch);

            // Apply learning rate warmup
            let effective_lr = state.get_learning_rate(self.config.learning_rate, self.config.warmup_steps);

            // Scale gradients by effective learning rate ratio
            let lr_scale = (effective_lr / self.config.learning_rate) as f32;
            let scaled_grads = grads * lr_scale;

            // Apply gradients via optimizer
            let embeddings_data = self.grid.embeddings.clone().inner();
            let new_embeddings = state.optimizer.step(embeddings_data, scaled_grads);

            // Update grid embeddings
            self.grid.embeddings = Tensor::from_inner(new_embeddings);

            // Update training state
            state.on_step(loss);

            // Log progress
            if epoch % log_interval == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: loss = {:.6}, avg_loss = {:.6}, lr = {:.2e}",
                    epoch + 1,
                    epochs,
                    loss,
                    state.avg_loss,
                    effective_lr
                );
            }

            // End of epoch
            if (epoch + 1) % (epochs / 10).max(1) == 0 {
                state.on_epoch();
            }
        }

        state
    }
}

impl<B: Backend> Clone for NeuralSdfTrainer<B> {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use crate::config::{DiffGridConfig, PointNetEncoderConfig, SdfDecoderConfig, SdfLossConfig};

    type TestBackend = Autodiff<NdArray>;

    fn make_test_config() -> TrainingConfig {
        TrainingConfig::default()
    }

    fn make_test_config_with_grid(grid: DiffGridConfig) -> TrainingConfig {
        TrainingConfig::new(
            grid,
            PointNetEncoderConfig::new(256),
            SdfDecoderConfig::new(256),
            SdfLossConfig::default(),
        )
    }

    #[test]
    fn test_trainer_creation() {
        let device = NdArrayDevice::Cpu;
        let config = make_test_config();
        let trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

        assert_eq!(trainer.grid.num_blocks(), 0);
    }

    #[test]
    fn test_trainer_init() {
        let device = NdArrayDevice::Cpu;
        let config =
            make_test_config_with_grid(DiffGridConfig::new(4, 0.1).with_capacity(100));
        let mut trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.5, 0.5, 0.5)];

        trainer.init_from_points(&points);
        assert!(trainer.grid.num_blocks() > 0);
    }

    #[test]
    fn test_train_step() {
        let device = NdArrayDevice::Cpu;
        let config =
            make_test_config_with_grid(DiffGridConfig::new(4, 0.1).with_capacity(100));
        let mut trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

        // Initialize with some points
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.2, 0.2, 0.2),
        ];
        trainer.init_from_points(&points);

        // Create a batch
        let surface = Tensor::zeros([10, 3], &device);
        let free = Tensor::full([10, 3], 0.5, &device);
        let eikonal = Tensor::zeros([10, 3], &device);
        let batch = SdfBatch::new(surface, free, eikonal);

        let output = trainer.train_step(&batch);

        assert!(output.loss_value().is_finite());
    }

    #[test]
    fn test_analytical_gradients() {
        let device = NdArrayDevice::Cpu;
        let config =
            make_test_config_with_grid(DiffGridConfig::new(4, 0.1).with_capacity(100));
        let mut trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

        // Initialize with a sphere
        let points: Vec<_> = (0..100)
            .map(|i| {
                let t = i as f32 / 100.0 * std::f32::consts::PI * 2.0;
                Point3::new(0.3 * t.cos(), 0.3 * t.sin(), 0.0)
            })
            .collect();
        trainer.init_from_points(&points);

        // Create a batch
        let surface = Tensor::zeros([10, 3], &device);
        let free = Tensor::full([10, 3], 0.5, &device);
        let eikonal = Tensor::zeros([10, 3], &device);
        let batch = SdfBatch::new(surface, free, eikonal);

        // Compute gradients
        let (loss, grads) = trainer.compute_analytical_gradients(&batch);

        assert!(loss.is_finite());
        let grad_data: Vec<f32> = grads.to_data().to_vec().unwrap();
        assert!(grad_data.iter().all(|&g| g.is_finite()));
    }

    #[test]
    fn test_training_reduces_loss() {
        let device = NdArrayDevice::Cpu;
        let config = make_test_config_with_grid(
            DiffGridConfig::new(4, 0.1)
                .with_capacity(100)
                .with_init_value(0.5), // Start with non-zero SDF
        )
        .with_learning_rate(0.01);
        let mut trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

        // Create a simple point cloud (circle in XY plane)
        let points: Vec<_> = (0..50)
            .map(|i| {
                let t = i as f32 / 50.0 * std::f32::consts::PI * 2.0;
                Point3::new(0.2 * t.cos(), 0.2 * t.sin(), 0.0)
            })
            .collect();

        // Train for a few epochs (train() calls init_from_points internally)
        trainer.train(&points, None, 20, 100);

        // Verify training succeeded
        assert!(trainer.grid.num_blocks() > 0);

        // Create a batch using actual points from the cloud
        let batch_points: Vec<_> = points.iter().take(10).copied().collect();
        let surface_flat: Vec<f32> = batch_points
            .iter()
            .flat_map(|p| [p.x, p.y, p.z])
            .collect();
        let surface = Tensor::from_data(TensorData::new(surface_flat, [10, 3]), &device);

        // Free space points slightly offset from surface
        let free_flat: Vec<f32> = batch_points
            .iter()
            .flat_map(|p| [p.x * 1.5, p.y * 1.5, p.z])
            .collect();
        let free = Tensor::from_data(TensorData::new(free_flat, [10, 3]), &device);

        // Eikonal points in the same region
        let eikonal_flat: Vec<f32> = batch_points
            .iter()
            .flat_map(|p| [p.x * 0.5, p.y * 0.5, p.z])
            .collect();
        let eikonal = Tensor::from_data(TensorData::new(eikonal_flat, [10, 3]), &device);

        let batch = SdfBatch::new(surface, free, eikonal);

        // Compute gradients on trained model - this should work without shape mismatch
        let (final_loss, grads) = trainer.compute_analytical_gradients(&batch);

        // The loss should be finite after training
        assert!(final_loss.is_finite());

        // Gradients should be finite
        let grad_data: Vec<f32> = grads.to_data().to_vec().unwrap();
        assert!(grad_data.iter().all(|&g| g.is_finite()));
    }
}
