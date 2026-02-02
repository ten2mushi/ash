//! Regularization loss functions.

#![allow(dead_code)]

use burn::prelude::*;

/// Smoothness regularization loss.
///
/// Encourages smooth SDF values by penalizing large second derivatives.
pub struct SmoothnessLoss {
    /// Weight for the smoothness term.
    weight: f32,
    /// Epsilon for finite differences.
    epsilon: f32,
}

impl SmoothnessLoss {
    /// Create a new smoothness loss.
    pub fn new(weight: f32, epsilon: f32) -> Self {
        Self { weight, epsilon }
    }

    /// Compute smoothness loss via Laplacian regularization.
    ///
    /// Approximates ∇²f and penalizes large values.
    ///
    /// Inputs:
    /// - query_fn: Function that takes [batch, 3] and returns [batch, 1]
    /// - points: [batch, 3] - sample points
    ///
    /// Output: scalar loss
    pub fn laplacian_loss<B: Backend, F>(&self, query_fn: F, points: Tensor<B, 2>) -> Tensor<B, 1>
    where
        F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let eps = self.epsilon;
        let device = points.device();
        let [batch, _] = points.dims();

        // Create offset tensors by creating single-row and repeating
        let eps_x_row = Tensor::<B, 2>::from_data([[eps, 0.0f32, 0.0]], &device);
        let eps_x = eps_x_row.repeat_dim(0, batch);

        let eps_y_row = Tensor::<B, 2>::from_data([[0.0f32, eps, 0.0]], &device);
        let eps_y = eps_y_row.repeat_dim(0, batch);

        let eps_z_row = Tensor::<B, 2>::from_data([[0.0f32, 0.0, eps]], &device);
        let eps_z = eps_z_row.repeat_dim(0, batch);

        // Query at center and neighbors
        let f_center = query_fn(points.clone());

        let f_px = query_fn(points.clone() + eps_x.clone());
        let f_nx = query_fn(points.clone() - eps_x);
        let f_py = query_fn(points.clone() + eps_y.clone());
        let f_ny = query_fn(points.clone() - eps_y);
        let f_pz = query_fn(points.clone() + eps_z.clone());
        let f_nz = query_fn(points - eps_z);

        // Laplacian approximation: (f(x+e) + f(x-e) - 2f(x)) / e² for each dimension
        let inv_eps2 = 1.0 / (eps * eps);
        let laplacian = (f_px + f_nx + f_py + f_ny + f_pz + f_nz - f_center * 6.0) * inv_eps2;

        // L2 loss on Laplacian
        (laplacian.clone() * laplacian).mean() * self.weight
    }

    /// Compute total variation loss.
    ///
    /// Penalizes gradient magnitude, encouraging piecewise constant solutions.
    ///
    /// Inputs:
    /// - gradients: [batch, 3] - gradient vectors
    ///
    /// Output: scalar loss
    pub fn total_variation_loss<B: Backend>(&self, gradients: Tensor<B, 2>) -> Tensor<B, 1> {
        let grad_sq = gradients.clone() * gradients;
        let grad_mag = grad_sq.sum_dim(1).sqrt();

        grad_mag.mean() * self.weight
    }
}

/// General regularization loss combining multiple terms.
pub struct RegularizationLoss {
    /// L2 weight decay coefficient.
    pub weight_decay: f32,
    /// Smoothness loss.
    pub smoothness: Option<SmoothnessLoss>,
}

impl RegularizationLoss {
    /// Create a new regularization loss.
    pub fn new(weight_decay: f32) -> Self {
        Self {
            weight_decay,
            smoothness: None,
        }
    }

    /// Add smoothness regularization.
    pub fn with_smoothness(mut self, weight: f32, epsilon: f32) -> Self {
        self.smoothness = Some(SmoothnessLoss::new(weight, epsilon));
        self
    }

    /// Compute L2 weight decay loss on a tensor.
    ///
    /// This is typically applied to network weights.
    pub fn weight_decay_loss<B: Backend>(&self, weights: Tensor<B, 2>) -> Tensor<B, 1> {
        let sq = weights.clone() * weights;
        sq.mean() * self.weight_decay
    }

    /// Compute embedding regularization loss.
    ///
    /// Penalizes large embedding values to prevent explosion.
    pub fn embedding_loss<B: Backend>(&self, embeddings: Tensor<B, 2>) -> Tensor<B, 1> {
        let sq = embeddings.clone() * embeddings;
        sq.mean() * self.weight_decay
    }
}

/// Lipschitz regularization for neural networks.
///
/// Encourages the network to have bounded Lipschitz constant,
/// which helps ensure the output is a valid SDF.
pub fn spectral_norm_loss<B: Backend>(weight: Tensor<B, 2>, target_norm: f32) -> Tensor<B, 1> {
    // Simple approximation: use Frobenius norm as proxy for spectral norm
    let frob_sq = (weight.clone() * weight.clone()).sum();
    let frob = frob_sq.sqrt();

    // Target: spectral norm ≈ sqrt(max(rows, cols)) * entry_scale
    let target = Tensor::full([1], target_norm, &frob.device());
    let diff = frob - target;
    diff.clone() * diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_smoothness_loss() {
        let device = Default::default();
        let loss = SmoothnessLoss::new(1.0, 0.01);

        // Constant function should have zero Laplacian
        let points = Tensor::<TestBackend, 2>::zeros([10, 3], &device);
        let l = loss.laplacian_loss(
            |p| Tensor::full([p.dims()[0], 1], 1.0, &p.device()),
            points,
        );

        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val.abs() < 1e-3, "Expected near-zero for constant function, got {}", val);
    }

    #[test]
    fn test_total_variation() {
        let device = Default::default();
        let loss = SmoothnessLoss::new(1.0, 0.01);

        // Zero gradient should give zero TV loss
        let zero_grad = Tensor::<TestBackend, 2>::zeros([10, 3], &device);
        let l = loss.total_variation_loss(zero_grad);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val.abs() < 1e-6);

        // Non-zero gradient should give positive TV loss
        let nonzero_grad = Tensor::<TestBackend, 2>::ones([10, 3], &device);
        let l = loss.total_variation_loss(nonzero_grad);
        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val > 0.0);
    }

    #[test]
    fn test_weight_decay() {
        let device = Default::default();
        let reg = RegularizationLoss::new(0.01);

        let weights = Tensor::<TestBackend, 2>::ones([100, 100], &device);
        let l = reg.weight_decay_loss(weights);

        let val: f32 = l.to_data().to_vec().unwrap()[0];
        assert!(val > 0.0);
        // 0.01 * mean(1.0) = 0.01
        assert!((val - 0.01).abs() < 1e-5);
    }
}
