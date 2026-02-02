//! # neural_ash
//!
//! Differentiable SDF learning with Burn for the ASH ecosystem.
//!
//! This crate provides neural network-based training and inference for
//! signed distance fields (SDFs), integrating with the ash_core and ash_io
//! crates for robotics applications.
//!
//! ## Features
//!
//! - **Differentiable grid**: `DiffSdfGrid<B, N>` stores learnable embeddings
//! - **Neural encoders**: PointNet for point clouds, Fourier for positional encoding
//! - **SDF decoder**: MLP-based decoder with skip connections
//! - **Loss functions**: Surface, free-space, Eikonal, and regularization losses
//! - **Online mode**: Real-time updates to shared memory for ash_rs
//!
//! ## Quick Start
//!
//! ```ignore
//! use neural_ash::{
//!     config::{DiffGridConfig, TrainingConfig},
//!     training::NeuralSdfTrainer,
//! };
//! use burn::backend::{Autodiff, NdArray};
//!
//! type MyBackend = Autodiff<NdArray>;
//!
//! // Create trainer
//! let config = TrainingConfig::new();
//! let device = Default::default();
//! let mut trainer = NeuralSdfTrainer::<MyBackend>::new(config, &device);
//!
//! // Initialize from point cloud
//! let points = load_point_cloud("scan.pcd");
//! trainer.init_from_points(&points);
//!
//! // Train
//! trainer.train(&points, None, 1000, 100);
//!
//! // Export
//! let grid = trainer.export_grid();
//! ash_io::save_to_file(&grid, "trained.ash").unwrap();
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ash_core (pure math)
//!     │
//!     ├──────────────────┬──────────────────┐
//!     ▼                  ▼                  ▼
//! ash_io             ash_rs            neural_ash
//! (storage)        (runtime)          (training)
//!     │                  ▲                  │
//!     └──────────────────┴──────────────────┘
//!               .ash file / shared memory
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default): Standard library support
//! - `wgpu` (default): GPU acceleration via WebGPU
//! - `ndarray`: CPU-only backend using ndarray
//! - `online`: Real-time inference with tokio

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod data;
pub mod error;
pub mod export;
pub mod grid;
pub mod loss;
pub mod nn;
pub mod online;
pub mod training;

// Re-export key types for convenience
pub use config::{DiffGridConfig, SdfLossConfig, TrainingConfig};
pub use error::{NeuralAshError, Result};
pub use grid::DiffSdfGrid;
pub use loss::SdfLoss;
pub use training::{NeuralSdfTrainer, TrainOutput};

// Re-export from ash_core and ash_io for convenience
pub use ash_core::{BlockCoord, CellCoord, Point3};
pub use ash_io::InMemoryGrid;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::config::{
        DiffGridConfig, FourierEncoderConfig, PointNetEncoderConfig, SdfDecoderConfig,
        SdfLossConfig, TrainingConfig,
    };
    pub use crate::data::{
        generate_orbit_poses, generate_sphere_poses, DepthCameraSimulator, DepthImage,
        PointCloud, PointCloudDataset, Pose, SensorDataBatch,
    };
    pub use crate::error::{NeuralAshError, Result};
    pub use crate::export::{discretize_grid, export_to_file};
    pub use crate::grid::{
        BatchGradientAccumulator, DiffSdfGrid, InterpolationMode, TensorCellValueProvider,
        TensorGradientAccumulator,
    };
    pub use crate::loss::{NormalLoss, NormalLossConfig, SdfLoss, SmoothnessLoss};
    pub use crate::nn::{FourierEncoder, PointNetEncoder, SdfDecoder, SemanticDecoder};
    pub use crate::training::{
        checkpoint_exists, find_latest_checkpoint, load_checkpoint, save_checkpoint,
        BatchSampler, CheckpointMetadata, NeuralSdfTrainer, OptimizerConfig, RmsPropState,
        SdfBatch, SgdState, TrainMetrics, TrainOutput, TrainingState,
    };

    pub use ash_core::{BlockCoord, CellCoord, Point3};
    pub use ash_io::InMemoryGrid;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use config::{PointNetEncoderConfig, SdfDecoderConfig};

    type TestBackend = Autodiff<NdArray>;

    fn make_training_config(grid: DiffGridConfig) -> TrainingConfig {
        TrainingConfig::new(
            grid,
            PointNetEncoderConfig::new(256),
            SdfDecoderConfig::new(256),
            SdfLossConfig::default(),
        )
    }

    #[test]
    fn test_public_api() {
        // Verify that the public API is accessible
        let _config = TrainingConfig::default();
        let _grid_config = DiffGridConfig::new(8, 0.05);
        let _loss_config = SdfLossConfig::default();
    }

    #[test]
    fn test_grid_creation() {
        use burn::backend::ndarray::NdArrayDevice;

        let device = NdArrayDevice::Cpu;
        let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
        let grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

        assert_eq!(grid.num_blocks(), 0);
    }

    #[test]
    fn test_trainer_creation() {
        use burn::backend::ndarray::NdArrayDevice;

        let device = NdArrayDevice::Cpu;
        let config = make_training_config(DiffGridConfig::new(4, 0.1).with_capacity(10));
        let trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

        assert_eq!(trainer.grid.num_blocks(), 0);
    }

    #[test]
    fn test_point_cloud_roundtrip() {
        use burn::backend::ndarray::NdArrayDevice;

        let device = NdArrayDevice::Cpu;
        let config = make_training_config(DiffGridConfig::new(4, 0.1).with_capacity(100));
        let mut trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

        // Create a simple point cloud (unit sphere surface)
        let mut points = Vec::new();
        for i in 0..100 {
            let theta = (i as f32 / 100.0) * std::f32::consts::PI * 2.0;
            for j in 0..50 {
                let phi = (j as f32 / 50.0) * std::f32::consts::PI;
                points.push(Point3::new(
                    phi.sin() * theta.cos() * 0.3,
                    phi.sin() * theta.sin() * 0.3,
                    phi.cos() * 0.3,
                ));
            }
        }

        trainer.init_from_points(&points);
        assert!(trainer.grid.num_blocks() > 0);

        // Export and verify
        let grid = trainer.export_grid();
        assert!(grid.num_blocks() > 0);
    }
}
