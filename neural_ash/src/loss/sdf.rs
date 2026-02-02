//! SDF-specific loss functions.

#![allow(dead_code)]

use burn::prelude::*;

use crate::config::SdfLossConfig;

/// SDF loss function calculator.
///
/// Provides methods for computing various SDF-related losses:
/// - Surface loss: Points on the surface should have SDF = 0
/// - Free-space loss: Points in free space should have SDF > threshold
/// - Eikonal loss: Gradient magnitude should equal 1
pub struct SdfLoss {
    config: SdfLossConfig,
}

impl SdfLoss {
    /// Create a new SDF loss calculator.
    pub fn new(config: SdfLossConfig) -> Self {
        Self { config }
    }

    /// Compute surface loss.
    ///
    /// L_surface = mean(sdf²)
    ///
    /// This encourages the SDF to be zero at surface points.
    ///
    /// Input: sdf_pred of shape [batch, 1] - predicted SDF values at surface points
    /// Output: scalar loss
    pub fn surface_loss<B: Backend>(&self, sdf_pred: Tensor<B, 2>) -> Tensor<B, 1> {
        let squared = sdf_pred.clone() * sdf_pred;
        squared.mean()
    }

    /// Compute free-space loss.
    ///
    /// L_free = mean(max(0, threshold - sdf))
    ///
    /// This encourages the SDF to be greater than the threshold in free space.
    ///
    /// Input: sdf_pred of shape [batch, 1] - predicted SDF values at free-space points
    /// Output: scalar loss
    pub fn free_space_loss<B: Backend>(&self, sdf_pred: Tensor<B, 2>) -> Tensor<B, 1> {
        let threshold = self.config.free_space_threshold;
        let device = sdf_pred.device();

        // max(0, threshold - sdf)
        let threshold_tensor = Tensor::full(sdf_pred.dims(), threshold, &device);
        let violation = (threshold_tensor - sdf_pred).clamp_min(0.0);

        violation.mean()
    }

    /// Compute Eikonal loss.
    ///
    /// L_eikonal = mean((|∇sdf| - 1)²)
    ///
    /// This regularizes the SDF to have unit gradient magnitude, which is a
    /// necessary property of a valid signed distance function.
    ///
    /// Inputs:
    /// - gradients: [batch, 3] - gradient vectors at sample points
    ///
    /// Output: scalar loss
    pub fn eikonal_loss<B: Backend>(&self, gradients: Tensor<B, 2>) -> Tensor<B, 1> {
        // Compute gradient magnitude: sqrt(gx² + gy² + gz²)
        let grad_sq = gradients.clone() * gradients;
        let grad_mag_sq = grad_sq.sum_dim(1);
        let grad_mag = grad_mag_sq.sqrt();

        // (|∇sdf| - 1)²
        let device = grad_mag.device();
        let one = Tensor::ones(grad_mag.dims(), &device);
        let deviation = grad_mag - one;
        let deviation_sq = deviation.clone() * deviation;

        deviation_sq.mean()
    }

    /// Compute combined SDF loss.
    ///
    /// Combines surface, free-space, and Eikonal losses with configured weights.
    ///
    /// Returns: (total_loss, surface_loss, free_space_loss, eikonal_loss)
    pub fn combined_loss<B: Backend>(
        &self,
        surface_sdf: Tensor<B, 2>,
        free_space_sdf: Tensor<B, 2>,
        gradients: Tensor<B, 2>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let surface = self.surface_loss(surface_sdf);
        let free_space = self.free_space_loss(free_space_sdf);
        let eikonal = self.eikonal_loss(gradients);

        let total = surface.clone() * self.config.surface_weight
            + free_space.clone() * self.config.free_space_weight
            + eikonal.clone() * self.config.eikonal_weight;

        (total, surface, free_space, eikonal)
    }

    /// Compute gradients via finite differences.
    ///
    /// This is useful when autodiff is not available or too expensive.
    ///
    /// Inputs:
    /// - query_fn: Function that takes [batch, 3] points and returns [batch, 1] SDF values
    /// - points: [batch, 3] - points to compute gradients at
    /// - epsilon: step size for finite differences
    ///
    /// Output: [batch, 3] - gradient vectors
    pub fn finite_difference_gradient<B: Backend, F>(
        &self,
        query_fn: F,
        points: Tensor<B, 2>,
    ) -> Tensor<B, 2>
    where
        F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let eps = self.config.gradient_epsilon;
        let device = points.device();
        let [batch, _] = points.dims();

        // Create offset tensors by creating single-row and repeating
        let eps_x_row = Tensor::<B, 2>::from_data([[eps, 0.0f32, 0.0]], &device);
        let eps_x = eps_x_row.repeat_dim(0, batch);

        let eps_y_row = Tensor::<B, 2>::from_data([[0.0f32, eps, 0.0]], &device);
        let eps_y = eps_y_row.repeat_dim(0, batch);

        let eps_z_row = Tensor::<B, 2>::from_data([[0.0f32, 0.0, eps]], &device);
        let eps_z = eps_z_row.repeat_dim(0, batch);

        // Central differences
        let dx_pos = query_fn(points.clone() + eps_x.clone());
        let dx_neg = query_fn(points.clone() - eps_x);
        let dy_pos = query_fn(points.clone() + eps_y.clone());
        let dy_neg = query_fn(points.clone() - eps_y);
        let dz_pos = query_fn(points.clone() + eps_z.clone());
        let dz_neg = query_fn(points - eps_z);

        let inv_2eps = 1.0 / (2.0 * eps);
        let grad_x = (dx_pos - dx_neg) * inv_2eps;
        let grad_y = (dy_pos - dy_neg) * inv_2eps;
        let grad_z = (dz_pos - dz_neg) * inv_2eps;

        Tensor::cat(vec![grad_x, grad_y, grad_z], 1)
    }

    /// Get the configuration.
    pub fn config(&self) -> &SdfLossConfig {
        &self.config
    }
}

/// Compute L1 SDF loss (alternative to L2 surface loss).
///
/// L1 loss is more robust to outliers.
pub fn l1_sdf_loss<B: Backend>(sdf_pred: Tensor<B, 2>) -> Tensor<B, 1> {
    sdf_pred.abs().mean()
}

/// Compute truncated SDF loss.
///
/// Clamps the SDF values to [-truncation, truncation] before computing loss.
/// This is useful for focusing on the near-surface region.
pub fn truncated_sdf_loss<B: Backend>(
    sdf_pred: Tensor<B, 2>,
    sdf_target: Tensor<B, 2>,
    truncation: f32,
) -> Tensor<B, 1> {
    let pred_clamped = sdf_pred.clamp(-truncation, truncation);
    let target_clamped = sdf_target.clamp(-truncation, truncation);

    let diff = pred_clamped - target_clamped;
    (diff.clone() * diff).mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_surface_loss() {
        let device = Default::default();
        let loss = SdfLoss::new(SdfLossConfig::default());

        // Zero SDF should give zero loss
        let zero_sdf = Tensor::<TestBackend, 2>::zeros([10, 1], &device);
        let l = loss.surface_loss(zero_sdf);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val.abs() < 1e-6);

        // Non-zero SDF should give positive loss
        let nonzero_sdf = Tensor::<TestBackend, 2>::full([10, 1], 0.5, &device);
        let l = loss.surface_loss(nonzero_sdf);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val > 0.0);
    }

    #[test]
    fn test_free_space_loss() {
        let device = Default::default();
        let config = SdfLossConfig::new()
            .with_free_space_threshold(0.1);
        let loss = SdfLoss::new(config);

        // SDF > threshold should give zero loss
        let large_sdf = Tensor::<TestBackend, 2>::full([10, 1], 0.5, &device);
        let l = loss.free_space_loss(large_sdf);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val.abs() < 1e-6);

        // SDF < threshold should give positive loss
        let small_sdf = Tensor::<TestBackend, 2>::full([10, 1], 0.01, &device);
        let l = loss.free_space_loss(small_sdf);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val > 0.0);
    }

    #[test]
    fn test_eikonal_loss() {
        let device = Default::default();
        let loss = SdfLoss::new(SdfLossConfig::default());

        // Unit gradient should give zero loss
        let unit_grad = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device,
        );
        let l = loss.eikonal_loss(unit_grad);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val.abs() < 1e-5);

        // Non-unit gradient should give positive loss
        let non_unit = Tensor::<TestBackend, 2>::from_data(
            [[2.0f32, 0.0, 0.0], [0.0, 0.5, 0.0]],
            &device,
        );
        let l = loss.eikonal_loss(non_unit);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val > 0.0);
    }

    #[test]
    fn test_combined_loss() {
        let device = Default::default();
        let loss = SdfLoss::new(SdfLossConfig::default());

        let surface = Tensor::<TestBackend, 2>::full([10, 1], 0.1, &device);
        let free = Tensor::<TestBackend, 2>::full([10, 1], 0.5, &device);
        // Create unit gradient vectors (pointing along x-axis)
        let grads_data: [[f32; 3]; 10] = [[1.0f32, 0.0, 0.0]; 10];
        let grads = Tensor::<TestBackend, 2>::from_data(grads_data, &device);

        let (total, surf, free_l, eik) = loss.combined_loss(surface, free, grads);

        // All losses should be finite
        assert!(total.to_data().to_vec::<f32>().unwrap()[0].is_finite());
        assert!(surf.to_data().to_vec::<f32>().unwrap()[0].is_finite());
        assert!(free_l.to_data().to_vec::<f32>().unwrap()[0].is_finite());
        assert!(eik.to_data().to_vec::<f32>().unwrap()[0].is_finite());
    }

    #[test]
    fn test_l1_loss() {
        let device = Default::default();

        let sdf = Tensor::<TestBackend, 2>::from_data([[0.5f32], [-0.3], [0.1]], &device);
        let l = l1_sdf_loss(sdf);
        let val: f32 = l.to_data().to_vec().unwrap()[0];

        // (0.5 + 0.3 + 0.1) / 3 = 0.3
        assert!((val - 0.3).abs() < 1e-5);
    }
}
