//! Training infrastructure for neural SDF learning.
//!
//! This module provides:
//! - `NeuralSdfTrainer`: Main training orchestrator
//! - Batch samplers for training data
//! - Training metrics and logging
//! - Optimizer configuration and state
//! - Checkpoint save/load for training resumption

mod batch;
mod checkpoint;
mod metrics;
mod optimizer;
mod trainer;

pub use batch::{BatchSampler, PointCloudBatch, SdfBatch};
pub use checkpoint::{
    checkpoint_exists, find_latest_checkpoint, load_checkpoint, save_checkpoint,
    CheckpointMetadata,
};
pub use metrics::{TrainMetrics, TrainOutput};
pub use optimizer::{OptimizerConfig, RmsPropState, SgdState, TrainingState};
pub use trainer::NeuralSdfTrainer;
