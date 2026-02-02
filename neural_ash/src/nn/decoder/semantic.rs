//! Semantic decoder network.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

use crate::config::{FourierEncoderConfig, SemanticDecoderConfig};
use crate::nn::encoder::FourierEncoder;

/// Semantic decoder that maps latent codes and coordinates to semantic features.
///
/// Outputs multiple semantic channels (e.g., class probabilities or continuous features).
#[derive(Module, Debug)]
pub struct SemanticDecoder<B: Backend> {
    /// Positional encoder for coordinates.
    pos_encoder: FourierEncoder<B>,
    /// Hidden layers.
    layers: Vec<Linear<B>>,
    /// Output layer to semantic features.
    output: Linear<B>,
    /// Activation function.
    activation: Relu,
    /// Latent dimension.
    #[module(skip)]
    latent_dim: usize,
}

impl<B: Backend> SemanticDecoder<B> {
    /// Create a new semantic decoder from configuration.
    pub fn new(config: &SemanticDecoderConfig, device: &B::Device) -> Self {
        // Create positional encoder
        let fourier_config = FourierEncoderConfig::new(config.positional_encoding_bands);
        let pos_encoder = FourierEncoder::new(&fourier_config, device);
        let pos_dim = pos_encoder.output_dim();

        // Input dimension: latent + positional encoding
        let mut in_dim = config.latent_dim + pos_dim;

        let mut layers = Vec::new();
        for &out_dim in &config.hidden_dims {
            layers.push(LinearConfig::new(in_dim, out_dim).init(device));
            in_dim = out_dim;
        }

        // Output layer
        let output = LinearConfig::new(in_dim, config.num_classes).init(device);

        Self {
            pos_encoder,
            layers,
            output,
            activation: Relu::new(),
            latent_dim: config.latent_dim,
        }
    }

    /// Forward pass.
    ///
    /// Inputs:
    /// - latent: [batch, latent_dim] - latent codes
    /// - xyz: [batch, 3] - 3D coordinates
    ///
    /// Output: [batch, num_classes] - semantic features
    pub fn forward(&self, latent: Tensor<B, 2>, xyz: Tensor<B, 2>) -> Tensor<B, 2> {
        // Encode coordinates
        let pos_encoded = self.pos_encoder.forward(xyz);

        // Concatenate latent and positional encoding
        let mut x = Tensor::cat(vec![latent, pos_encoded], 1);

        // Hidden layers
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Output (no final activation - could apply softmax externally for classification)
        self.output.forward(x)
    }

    /// Forward pass for batched coordinates with shared latent.
    ///
    /// Inputs:
    /// - latent: [batch, latent_dim] - latent codes
    /// - xyz: [batch, num_points, 3] - 3D coordinates
    ///
    /// Output: [batch, num_points, num_classes] - semantic features
    pub fn forward_batched(&self, latent: Tensor<B, 2>, xyz: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, num_points, _] = xyz.dims();
        // In Burn, Linear weight is [in_features, out_features]
        let num_classes = self.output.weight.dims()[1];

        // Expand latent for each point: [batch, num_points, latent_dim]
        let latent_3d: Tensor<B, 3> = latent.unsqueeze_dim(1);
        let latent_expanded = latent_3d.repeat_dim(1, num_points);

        // Reshape for 2D forward pass
        let latent_flat = latent_expanded.reshape([batch * num_points, self.latent_dim]);
        let xyz_flat = xyz.reshape([batch * num_points, 3]);

        // Forward pass
        let output_flat = self.forward(latent_flat, xyz_flat);

        // Reshape back
        output_flat.reshape([batch, num_points, num_classes])
    }

    /// Get the number of output classes.
    pub fn num_classes(&self) -> usize {
        // In Burn, Linear weight is [in_features, out_features]
        self.output.weight.dims()[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_semantic_decoder_forward() {
        let device = Default::default();
        let config = SemanticDecoderConfig::new(256, 10);
        let decoder = SemanticDecoder::<TestBackend>::new(&config, &device);

        let latent = Tensor::zeros([4, 256], &device);
        let xyz = Tensor::zeros([4, 3], &device);
        let output = decoder.forward(latent, xyz);

        assert_eq!(output.dims(), [4, 10]);
    }

    #[test]
    fn test_semantic_decoder_batched() {
        let device = Default::default();
        let config = SemanticDecoderConfig::new(128, 5);
        let decoder = SemanticDecoder::<TestBackend>::new(&config, &device);

        let latent = Tensor::zeros([2, 128], &device);
        let xyz = Tensor::zeros([2, 100, 3], &device);
        let output = decoder.forward_batched(latent, xyz);

        assert_eq!(output.dims(), [2, 100, 5]);
    }

    #[test]
    fn test_semantic_decoder_num_classes() {
        let device = Default::default();
        let config = SemanticDecoderConfig::new(64, 20);
        let decoder = SemanticDecoder::<TestBackend>::new(&config, &device);

        assert_eq!(decoder.num_classes(), 20);
    }
}
