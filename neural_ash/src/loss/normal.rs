//! Normal-based loss functions for SDF training.
//!
//! These losses compare predicted surface normals (computed as the gradient of the SDF)
//! with target normals from the training data.

use std::sync::Arc;

use burn::prelude::*;

use ash_core::{decompose_point, trilinear_gradient, CellValueProvider, Point3};

use crate::config::DiffGridConfig;
use crate::grid::TensorCellValueProvider;

/// Configuration for normal loss functions.
#[derive(Debug, Clone, Copy)]
pub struct NormalLossConfig {
    /// Weight for L1 normal loss.
    pub l1_weight: f32,
    /// Weight for cosine similarity loss.
    pub cosine_weight: f32,
    /// Minimum gradient magnitude to consider valid.
    pub min_gradient_magnitude: f32,
}

impl Default for NormalLossConfig {
    fn default() -> Self {
        Self {
            l1_weight: 0.0,
            cosine_weight: 0.1,
            min_gradient_magnitude: 1e-6,
        }
    }
}

impl NormalLossConfig {
    /// Create a new normal loss configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set L1 weight.
    pub fn with_l1_weight(mut self, weight: f32) -> Self {
        self.l1_weight = weight;
        self
    }

    /// Set cosine similarity weight.
    pub fn with_cosine_weight(mut self, weight: f32) -> Self {
        self.cosine_weight = weight;
        self
    }
}

/// Compute predicted normal at a point using ash_core's analytical gradient.
///
/// The normal is the normalized gradient of the SDF.
///
/// # Arguments
/// * `provider` - Cell value provider (e.g., TensorCellValueProvider)
/// * `point` - World-space point to query
/// * `cell_size` - Size of each cell
/// * `grid_dim` - Cells per axis in each block
///
/// # Returns
/// * `Some(normal)` - Normalized gradient direction
/// * `None` - If the point is in an unallocated region
pub fn get_predicted_normal<const N: usize, P: CellValueProvider<N>>(
    provider: &P,
    point: Point3,
    cell_size: f32,
    grid_dim: u32,
) -> Option<Point3> {
    let (block, cell, local) = decompose_point(point, cell_size, grid_dim);
    let grad = trilinear_gradient(provider, block, cell, local)?;

    // Convert from cell-space to world-space
    let scale = 1.0 / cell_size;
    let g = grad[0];
    let gx = g[0] * scale;
    let gy = g[1] * scale;
    let gz = g[2] * scale;

    let len = (gx * gx + gy * gy + gz * gz).sqrt();
    if len < 1e-6 {
        return None;
    }

    Some(Point3::new(gx / len, gy / len, gz / len))
}

/// Compute L1 normal loss between predicted and target normals.
///
/// L1 loss: mean(|pred - target|)
///
/// # Arguments
/// * `pred` - Predicted normals, shape [batch, 3]
/// * `target` - Target normals, shape [batch, 3]
///
/// # Returns
/// Scalar loss tensor
pub fn l1_normal_loss<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = pred - target;
    let abs_diff = diff.abs();
    abs_diff.mean()
}

/// Compute cosine similarity loss between predicted and target normals.
///
/// Cosine loss: 1 - mean(pred · target / (|pred| |target|))
///
/// This is more robust than L1/L2 for direction-only comparisons.
///
/// # Arguments
/// * `pred` - Predicted normals, shape [batch, 3]
/// * `target` - Target normals, shape [batch, 3]
///
/// # Returns
/// Scalar loss tensor
pub fn cosine_normal_loss<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    // Compute dot products: sum(pred * target, dim=1)
    let dot = (pred.clone() * target.clone()).sum_dim(1);

    // Compute magnitudes
    let pred_mag_sq = (pred.clone() * pred).sum_dim(1);
    let target_mag_sq = (target.clone() * target).sum_dim(1);
    let pred_mag = pred_mag_sq.sqrt();
    let target_mag = target_mag_sq.sqrt();

    // Cosine similarity = dot / (|pred| * |target|)
    let eps = 1e-8;
    let cos_sim = dot / (pred_mag * target_mag + eps);

    // Loss = 1 - mean(cos_sim)
    let one = Tensor::ones(cos_sim.dims(), &cos_sim.device());
    (one - cos_sim).mean()
}

/// Compute angular error between predicted and target normals in radians.
///
/// Angular error = arccos(pred · target)
///
/// # Arguments
/// * `pred` - Predicted normals, shape [batch, 3]
/// * `target` - Target normals, shape [batch, 3]
///
/// # Returns
/// Mean angular error in radians
pub fn angular_error<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    // Compute dot products
    let dot = (pred.clone() * target.clone()).sum_dim(1);

    // Clamp to [-1, 1] for numerical stability
    let clamped = dot.clamp(-1.0, 1.0);

    // acos to get angle
    // Note: Burn may not have acos directly, so we use a polynomial approximation
    // For small angles: acos(x) ≈ sqrt(2(1-x))
    let one = Tensor::ones(clamped.dims(), &clamped.device());
    let approx_acos = ((one - clamped) * 2.0).sqrt();

    approx_acos.mean()
}

/// Combined normal loss using both L1 and cosine similarity.
pub struct NormalLoss {
    config: NormalLossConfig,
}

impl NormalLoss {
    /// Create a new normal loss calculator.
    pub fn new(config: NormalLossConfig) -> Self {
        Self { config }
    }

    /// Compute combined normal loss.
    ///
    /// # Returns
    /// (total_loss, l1_loss, cosine_loss)
    pub fn compute<B: Backend>(
        &self,
        pred: Tensor<B, 2>,
        target: Tensor<B, 2>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let l1 = l1_normal_loss(pred.clone(), target.clone());
        let cosine = cosine_normal_loss(pred, target);

        let total = l1.clone() * self.config.l1_weight + cosine.clone() * self.config.cosine_weight;

        (total, l1, cosine)
    }
}

/// Batch compute predicted normals from a grid.
///
/// # Arguments
/// * `embeddings` - Grid embeddings tensor
/// * `block_map` - Block coordinate map
/// * `config` - Grid configuration
/// * `num_blocks` - Number of allocated blocks
/// * `points` - Query points, shape [batch, 3]
///
/// # Returns
/// Predicted normals tensor, shape [batch, 3]
/// Points in unallocated regions get zero normals.
pub fn batch_predicted_normals<B: Backend>(
    embeddings: &Tensor<B, 2>,
    block_map: Arc<ash_io::BlockMap>,
    config: &DiffGridConfig,
    num_blocks: usize,
    points: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let device = embeddings.device();
    let [num_points, _] = points.dims();

    // Create CPU-side provider
    let provider = TensorCellValueProvider::<1>::new(embeddings, block_map, config.clone(), num_blocks);

    // Extract points data
    let points_data = points.to_data();
    let points_flat: Vec<f32> = points_data.to_vec().unwrap();

    // Compute normals for each point
    let mut normals_data = Vec::with_capacity(num_points * 3);

    for i in 0..num_points {
        let point = Point3::new(
            points_flat[i * 3],
            points_flat[i * 3 + 1],
            points_flat[i * 3 + 2],
        );

        if let Some(normal) = get_predicted_normal(&provider, point, config.cell_size, config.grid_dim)
        {
            normals_data.extend_from_slice(&[normal.x, normal.y, normal.z]);
        } else {
            normals_data.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    }

    Tensor::from_data(TensorData::new(normals_data, [num_points, 3]), &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_l1_normal_loss_zero() {
        let device = Default::default();

        // Same normals should give zero loss
        let pred = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device,
        );
        let target = pred.clone();

        let loss = l1_normal_loss(pred, target);
        let val: f32 = loss.to_data().to_vec().unwrap()[0];

        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_l1_normal_loss_nonzero() {
        let device = Default::default();

        let pred = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0]],
            &device,
        );
        let target = Tensor::<TestBackend, 2>::from_data(
            [[0.0f32, 1.0, 0.0]],
            &device,
        );

        let loss = l1_normal_loss(pred, target);
        let val: f32 = loss.to_data().to_vec().unwrap()[0];

        // L1 difference: |1-0| + |0-1| + |0-0| = 2, mean = 2/3
        assert!((val - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_loss_parallel() {
        let device = Default::default();

        // Parallel vectors should give zero cosine loss
        let pred = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &device,
        );
        let target = pred.clone();

        let loss = cosine_normal_loss(pred, target);
        let val: f32 = loss.to_data().to_vec().unwrap()[0];

        assert!(val.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_loss_perpendicular() {
        let device = Default::default();

        // Perpendicular vectors should give loss of 1
        let pred = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0]],
            &device,
        );
        let target = Tensor::<TestBackend, 2>::from_data(
            [[0.0f32, 1.0, 0.0]],
            &device,
        );

        let loss = cosine_normal_loss(pred, target);
        let val: f32 = loss.to_data().to_vec().unwrap()[0];

        // cos(90°) = 0, so loss = 1 - 0 = 1
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_loss_opposite() {
        let device = Default::default();

        // Opposite vectors should give loss of 2
        let pred = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0]],
            &device,
        );
        let target = Tensor::<TestBackend, 2>::from_data(
            [[-1.0f32, 0.0, 0.0]],
            &device,
        );

        let loss = cosine_normal_loss(pred, target);
        let val: f32 = loss.to_data().to_vec().unwrap()[0];

        // cos(180°) = -1, so loss = 1 - (-1) = 2
        assert!((val - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_combined_loss() {
        let device = Default::default();
        let config = NormalLossConfig::new()
            .with_l1_weight(0.5)
            .with_cosine_weight(0.5);
        let loss_fn = NormalLoss::new(config);

        let pred = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0]],
            &device,
        );
        let target = pred.clone();

        let (total, l1, cosine) = loss_fn.compute(pred, target);

        // All losses should be zero for identical inputs
        assert!(total.to_data().to_vec::<f32>().unwrap()[0].abs() < 1e-5);
        assert!(l1.to_data().to_vec::<f32>().unwrap()[0].abs() < 1e-5);
        assert!(cosine.to_data().to_vec::<f32>().unwrap()[0].abs() < 1e-5);
    }

    #[test]
    fn test_angular_error() {
        let device = Default::default();

        // Parallel vectors should have zero angular error
        let pred = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0]],
            &device,
        );
        let target = pred.clone();

        let error = angular_error(pred, target);
        let val: f32 = error.to_data().to_vec().unwrap()[0];

        assert!(val.abs() < 1e-5);
    }
}
