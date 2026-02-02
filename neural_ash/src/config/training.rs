//! Training configuration types.

use burn::config::Config;

use super::{DiffGridConfig, PointNetEncoderConfig, SdfDecoderConfig};

/// Configuration for SDF loss functions.
#[derive(Config, Debug)]
pub struct SdfLossConfig {
    /// Weight for surface loss (SDF should be 0 at surface points).
    #[config(default = 1.0)]
    pub surface_weight: f32,

    /// Weight for free-space loss (SDF should be > threshold in free space).
    #[config(default = 0.1)]
    pub free_space_weight: f32,

    /// Threshold for free-space loss.
    #[config(default = 0.05)]
    pub free_space_threshold: f32,

    /// Weight for Eikonal regularization (|∇SDF| = 1).
    #[config(default = 0.1)]
    pub eikonal_weight: f32,

    /// Weight for smoothness regularization.
    #[config(default = 0.01)]
    pub smoothness_weight: f32,

    /// Epsilon for finite difference gradient computation.
    #[config(default = 0.005)]
    pub gradient_epsilon: f32,
}

impl Default for SdfLossConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SdfLossConfig {
    /// Create a configuration optimized for fast convergence.
    pub fn fast() -> Self {
        Self::new()
            .with_free_space_weight(0.05)
            .with_free_space_threshold(0.1)
            .with_eikonal_weight(0.05)
            .with_smoothness_weight(0.0)
            .with_gradient_epsilon(0.01)
    }

    /// Create a configuration optimized for accuracy.
    pub fn accurate() -> Self {
        Self::new()
            .with_free_space_weight(0.2)
            .with_free_space_threshold(0.02)
            .with_eikonal_weight(0.2)
            .with_smoothness_weight(0.02)
            .with_gradient_epsilon(0.002)
    }
}

/// Configuration for the neural SDF trainer.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Grid configuration.
    pub grid: DiffGridConfig,

    /// Encoder configuration.
    pub encoder: PointNetEncoderConfig,

    /// Decoder configuration.
    pub decoder: SdfDecoderConfig,

    /// Loss configuration.
    pub loss: SdfLossConfig,

    /// Learning rate.
    #[config(default = 1e-4)]
    pub learning_rate: f64,

    /// Weight decay for regularization.
    #[config(default = 0.0)]
    pub weight_decay: f64,

    /// Batch size for training.
    #[config(default = 4096)]
    pub batch_size: usize,

    /// Number of surface points per batch.
    #[config(default = 2048)]
    pub surface_points_per_batch: usize,

    /// Number of free-space points per batch.
    #[config(default = 2048)]
    pub free_space_points_per_batch: usize,

    /// Number of points for Eikonal sampling.
    #[config(default = 1024)]
    pub eikonal_points_per_batch: usize,

    /// Gradient clipping threshold (0 = no clipping).
    #[config(default = 1.0)]
    pub gradient_clip: f64,

    /// Number of warmup steps for learning rate.
    #[config(default = 1000)]
    pub warmup_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new(
            DiffGridConfig::new(8, 0.05),
            PointNetEncoderConfig::new(256),
            SdfDecoderConfig::new(256),
            SdfLossConfig::default(),
        )
    }
}

impl TrainingConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.grid.validate()?;

        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".to_string());
        }
        if self.surface_points_per_batch == 0 {
            return Err("surface_points_per_batch must be positive".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_training_config() {
        let config = TrainingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        // Test that optional fields have with_* methods
        let config = TrainingConfig::default()
            .with_learning_rate(1e-3);

        assert_eq!(config.learning_rate, 1e-3);
    }

    #[test]
    fn test_custom_grid_config() {
        // Required fields must be passed to new()
        let config = TrainingConfig::new(
            DiffGridConfig::new(8, 0.1),
            PointNetEncoderConfig::new(256),
            SdfDecoderConfig::new(256),
            SdfLossConfig::default(),
        );

        assert_eq!(config.grid.grid_dim, 8);
    }

    #[test]
    fn test_loss_presets() {
        let fast = SdfLossConfig::fast();
        let accurate = SdfLossConfig::accurate();

        assert!(fast.gradient_epsilon > accurate.gradient_epsilon);
        assert!(fast.smoothness_weight < accurate.smoothness_weight);
    }
}
