//! Loss functions for SDF learning.
//!
//! This module provides various loss functions commonly used in neural SDF training:
//! - Surface loss: SDF should be zero at surface points
//! - Free-space loss: SDF should be positive (> threshold) in free space
//! - Eikonal loss: |∇SDF| should equal 1 everywhere
//! - Normal loss: Predicted normals should match target normals
//! - Regularization: Smoothness and other regularization terms

mod normal;
mod regularization;
mod sdf;

pub use normal::{
    angular_error, batch_predicted_normals, cosine_normal_loss, get_predicted_normal,
    l1_normal_loss, NormalLoss, NormalLossConfig,
};
pub use regularization::{RegularizationLoss, SmoothnessLoss};
pub use sdf::SdfLoss;
