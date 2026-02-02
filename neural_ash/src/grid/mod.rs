//! Differentiable grid module.
//!
//! Provides `DiffSdfGrid<B, N>`, a differentiable grid structure backed by
//! Burn tensors for gradient-based optimization.

mod diff_grid;
mod gradient_accumulator;
mod interpolation;
mod provider;
mod tensor_storage;

pub use diff_grid::DiffSdfGrid;
pub use gradient_accumulator::{BatchGradientAccumulator, TensorGradientAccumulator};
pub use interpolation::{
    compute_weights_with_mode, interpolate, interpolate_gradient, smootherstep, smoothstep,
    InterpolationMode,
};
pub use provider::TensorCellValueProvider;
pub use tensor_storage::TensorStorage;
