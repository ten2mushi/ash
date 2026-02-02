//! PointNet encoder for point cloud processing.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

use crate::config::PointNetEncoderConfig;

/// PointNet-style encoder for point cloud data.
///
/// Takes a batch of point clouds and produces a global latent code for each.
/// Uses max-pooling for permutation invariance.
#[derive(Module, Debug)]
pub struct PointNetEncoder<B: Backend> {
    /// MLP layers applied to each point.
    mlp_layers: Vec<Linear<B>>,
    /// Final projection to latent dimension.
    proj: Linear<B>,
    /// Activation function.
    activation: Relu,
}

impl<B: Backend> PointNetEncoder<B> {
    /// Create a new PointNet encoder from configuration.
    pub fn new(config: &PointNetEncoderConfig, device: &B::Device) -> Self {
        let mut mlp_layers = Vec::new();
        let mut in_dim = 3; // 3D points

        for &out_dim in &config.hidden_dims {
            mlp_layers.push(LinearConfig::new(in_dim, out_dim).init(device));
            in_dim = out_dim;
        }

        let proj = LinearConfig::new(in_dim, config.latent_dim).init(device);

        Self {
            mlp_layers,
            proj,
            activation: Relu::new(),
        }
    }

    /// Forward pass.
    ///
    /// Input: points tensor of shape [batch, num_points, 3]
    /// Output: latent tensor of shape [batch, latent_dim]
    pub fn forward(&self, points: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, num_points, _] = points.dims();

        // Reshape for MLP: [batch * num_points, 3]
        let mut x = points.reshape([batch * num_points, 3]);

        // Apply MLP layers
        for layer in &self.mlp_layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Project to latent dimension
        x = self.proj.forward(x);

        // Reshape back: [batch, num_points, latent_dim]
        let latent_dim = x.dims()[1];
        let x_3d: Tensor<B, 3> = x.reshape([batch, num_points, latent_dim]);

        // Max pooling over points: [batch, latent_dim]
        x_3d.max_dim(1).squeeze(1)
    }

    /// Forward pass with per-point features (for decoder conditioning).
    ///
    /// Input: points tensor of shape [batch, num_points, 3]
    /// Output: per-point features of shape [batch, num_points, latent_dim]
    pub fn forward_per_point(&self, points: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, num_points, _] = points.dims();

        // Reshape for MLP: [batch * num_points, 3]
        let mut x = points.reshape([batch * num_points, 3]);

        // Apply MLP layers
        for layer in &self.mlp_layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Project to latent dimension
        x = self.proj.forward(x);

        // Reshape back: [batch, num_points, latent_dim]
        let latent_dim = x.dims()[1];
        x.reshape([batch, num_points, latent_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_pointnet_forward() {
        let device = Default::default();
        let config = PointNetEncoderConfig::new(256);
        let encoder = PointNetEncoder::<TestBackend>::new(&config, &device);

        // Batch of 2 point clouds, each with 100 points
        let points = Tensor::zeros([2, 100, 3], &device);
        let output = encoder.forward(points);

        assert_eq!(output.dims(), [2, 256]);
    }

    #[test]
    fn test_pointnet_permutation_invariance() {
        let device = Default::default();
        let config = PointNetEncoderConfig::new(64).with_hidden_dims(vec![32, 64]);
        let encoder = PointNetEncoder::<TestBackend>::new(&config, &device);

        // Create a small point cloud
        let points_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let points = Tensor::from_data(
            TensorData::new(points_data.clone(), [1, 3, 3]),
            &device,
        );

        // Permute the points
        let permuted_data: Vec<f32> = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let permuted = Tensor::from_data(
            TensorData::new(permuted_data, [1, 3, 3]),
            &device,
        );

        let output1 = encoder.forward(points);
        let output2 = encoder.forward(permuted);

        // Due to max pooling, outputs should be identical (or very close)
        let diff = (output1 - output2).abs().max().to_data();
        let max_diff: f32 = diff.to_vec().unwrap()[0];

        assert!(max_diff < 1e-5, "Max diff: {}", max_diff);
    }

    #[test]
    fn test_pointnet_per_point() {
        let device = Default::default();
        let config = PointNetEncoderConfig::new(128);
        let encoder = PointNetEncoder::<TestBackend>::new(&config, &device);

        let points = Tensor::zeros([2, 50, 3], &device);
        let output = encoder.forward_per_point(points);

        assert_eq!(output.dims(), [2, 50, 128]);
    }
}
