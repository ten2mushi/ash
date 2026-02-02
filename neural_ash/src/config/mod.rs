//! Configuration types for neural_ash.
//!
//! This module provides Burn-style configuration structs for the differentiable grid,
//! neural networks, training, and loss functions.

mod grid;
mod network;
mod training;

pub use grid::DiffGridConfig;
pub use network::{FourierEncoderConfig, PointNetEncoderConfig, SdfDecoderConfig, SemanticDecoderConfig};
pub use training::{SdfLossConfig, TrainingConfig};
