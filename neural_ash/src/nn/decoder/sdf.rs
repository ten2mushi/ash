//! SDF decoder network.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

use crate::config::{FourierEncoderConfig, SdfDecoderConfig};
use crate::nn::encoder::FourierEncoder;

/// SDF decoder that maps latent codes and coordinates to signed distance values.
///
/// Architecture:
/// 1. Concatenate latent code with positionally-encoded coordinates
/// 2. Pass through MLP with optional skip connection
/// 3. Output scalar SDF value
#[derive(Module, Debug)]
pub struct SdfDecoder<B: Backend> {
    /// Positional encoder for coordinates.
    pos_encoder: FourierEncoder<B>,
    /// First half of hidden layers (before skip connection).
    layers_first: Vec<Linear<B>>,
    /// Second half of hidden layers (after skip connection).
    layers_second: Vec<Linear<B>>,
    /// Output layer to scalar SDF.
    output: Linear<B>,
    /// Activation function.
    activation: Relu,
    /// Whether to use skip connection.
    #[module(skip)]
    skip_connection: bool,
    /// Latent dimension for skip connection.
    #[module(skip)]
    latent_dim: usize,
    /// Positional encoding dimension.
    #[module(skip)]
    pos_dim: usize,
}

impl<B: Backend> SdfDecoder<B> {
    /// Create a new SDF decoder from configuration.
    pub fn new(config: &SdfDecoderConfig, device: &B::Device) -> Self {
        // Create positional encoder
        let fourier_config = FourierEncoderConfig::new(config.positional_encoding_bands);
        let pos_encoder = FourierEncoder::new(&fourier_config, device);
        let pos_dim = pos_encoder.output_dim();

        // Input dimension: latent + positional encoding
        let input_dim = config.latent_dim + pos_dim;

        // Split hidden layers for skip connection
        let num_layers = config.hidden_dims.len();
        let skip_layer = num_layers / 2;

        let mut layers_first = Vec::new();
        let mut layers_second = Vec::new();

        let mut in_dim = input_dim;

        for (i, &out_dim) in config.hidden_dims.iter().enumerate() {
            if i < skip_layer {
                layers_first.push(LinearConfig::new(in_dim, out_dim).init(device));
                in_dim = out_dim;
            } else {
                // Add skip connection input at the skip layer
                let actual_in = if i == skip_layer && config.skip_connection {
                    in_dim + input_dim
                } else {
                    in_dim
                };
                layers_second.push(LinearConfig::new(actual_in, out_dim).init(device));
                in_dim = out_dim;
            }
        }

        // Output layer
        let output = LinearConfig::new(in_dim, 1).init(device);

        Self {
            pos_encoder,
            layers_first,
            layers_second,
            output,
            activation: Relu::new(),
            skip_connection: config.skip_connection,
            latent_dim: config.latent_dim,
            pos_dim,
        }
    }

    /// Forward pass.
    ///
    /// Inputs:
    /// - latent: [batch, latent_dim] - latent codes
    /// - xyz: [batch, 3] - 3D coordinates
    ///
    /// Output: [batch, 1] - SDF values
    pub fn forward(&self, latent: Tensor<B, 2>, xyz: Tensor<B, 2>) -> Tensor<B, 2> {
        // Encode coordinates
        let pos_encoded = self.pos_encoder.forward(xyz);

        // Concatenate latent and positional encoding
        let input = Tensor::cat(vec![latent, pos_encoded], 1);

        // First half of layers
        let mut x = input.clone();
        for layer in &self.layers_first {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Skip connection and second half
        if self.skip_connection && !self.layers_second.is_empty() {
            x = Tensor::cat(vec![x, input], 1);
        }

        for layer in &self.layers_second {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Output
        self.output.forward(x)
    }

    /// Forward pass for batched coordinates with shared latent.
    ///
    /// This is more efficient when querying many points with the same latent code.
    ///
    /// Inputs:
    /// - latent: [batch, latent_dim] - latent codes (one per batch)
    /// - xyz: [batch, num_points, 3] - 3D coordinates
    ///
    /// Output: [batch, num_points, 1] - SDF values
    pub fn forward_batched(&self, latent: Tensor<B, 2>, xyz: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, num_points, _] = xyz.dims();

        // Expand latent for each point: [batch, num_points, latent_dim]
        let latent_3d: Tensor<B, 3> = latent.unsqueeze_dim(1);
        let latent_expanded = latent_3d.repeat_dim(1, num_points);

        // Reshape for 2D forward pass
        let latent_flat = latent_expanded.reshape([batch * num_points, self.latent_dim]);
        let xyz_flat = xyz.reshape([batch * num_points, 3]);

        // Forward pass
        let output_flat = self.forward(latent_flat, xyz_flat);

        // Reshape back
        output_flat.reshape([batch, num_points, 1])
    }

    /// Get the expected input dimension (latent + pos encoding).
    pub fn input_dim(&self) -> usize {
        self.latent_dim + self.pos_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_sdf_decoder_forward() {
        let device = Default::default();
        let config = SdfDecoderConfig::new(256);
        let decoder = SdfDecoder::<TestBackend>::new(&config, &device);

        let latent = Tensor::zeros([4, 256], &device);
        let xyz = Tensor::zeros([4, 3], &device);
        let output = decoder.forward(latent, xyz);

        assert_eq!(output.dims(), [4, 1]);
    }

    #[test]
    fn test_sdf_decoder_batched() {
        let device = Default::default();
        let config = SdfDecoderConfig::new(128).with_hidden_dims(vec![128, 128]);
        let decoder = SdfDecoder::<TestBackend>::new(&config, &device);

        let latent = Tensor::zeros([2, 128], &device);
        let xyz = Tensor::zeros([2, 100, 3], &device);
        let output = decoder.forward_batched(latent, xyz);

        assert_eq!(output.dims(), [2, 100, 1]);
    }

    #[test]
    fn test_sdf_decoder_with_skip() {
        let device = Default::default();
        let config = SdfDecoderConfig::new(64)
            .with_hidden_dims(vec![128, 128, 64, 64])
            .with_skip_connection(true)
            .with_positional_encoding_bands(4)
            .with_geometric_init(false)
            .with_dropout(0.0);
        let decoder = SdfDecoder::<TestBackend>::new(&config, &device);

        let latent = Tensor::zeros([8, 64], &device);
        let xyz = Tensor::zeros([8, 3], &device);
        let output = decoder.forward(latent, xyz);

        assert_eq!(output.dims(), [8, 1]);
    }

    #[test]
    fn test_sdf_decoder_without_skip() {
        let device = Default::default();
        let config = SdfDecoderConfig::new(64)
            .with_hidden_dims(vec![128, 64])
            .with_skip_connection(false)
            .with_positional_encoding_bands(4)
            .with_geometric_init(false)
            .with_dropout(0.0);
        let decoder = SdfDecoder::<TestBackend>::new(&config, &device);

        let latent = Tensor::zeros([8, 64], &device);
        let xyz = Tensor::zeros([8, 3], &device);
        let output = decoder.forward(latent, xyz);

        assert_eq!(output.dims(), [8, 1]);
    }
}
