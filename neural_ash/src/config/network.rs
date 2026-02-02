//! Neural network configuration types.

use burn::config::Config;

/// Configuration for the PointNet encoder.
#[derive(Config, Debug)]
pub struct PointNetEncoderConfig {
    /// Output latent dimension.
    pub latent_dim: usize,

    /// Hidden layer dimensions.
    #[config(default = "vec![64, 128, 256]")]
    pub hidden_dims: Vec<usize>,

    /// Whether to use batch normalization.
    #[config(default = true)]
    pub use_batch_norm: bool,

    /// Dropout probability (0.0 = no dropout).
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Configuration for Fourier positional encoding.
#[derive(Config, Debug)]
pub struct FourierEncoderConfig {
    /// Number of frequency bands.
    pub num_bands: usize,

    /// Maximum frequency (log2 scale).
    #[config(default = 4.0)]
    pub max_freq_log2: f32,

    /// Whether to include the original coordinates.
    #[config(default = true)]
    pub include_input: bool,
}

impl FourierEncoderConfig {
    /// Compute the output dimension of the encoder.
    ///
    /// For 3D input with `num_bands` frequencies:
    /// - If include_input: 3 + 3 * 2 * num_bands = 3 + 6 * num_bands
    /// - Otherwise: 3 * 2 * num_bands = 6 * num_bands
    pub fn output_dim(&self) -> usize {
        let fourier_dim = 3 * 2 * self.num_bands; // sin + cos for each axis
        if self.include_input {
            3 + fourier_dim
        } else {
            fourier_dim
        }
    }
}

/// Configuration for the SDF decoder.
#[derive(Config, Debug)]
pub struct SdfDecoderConfig {
    /// Input latent dimension from encoder.
    pub latent_dim: usize,

    /// Hidden layer dimensions.
    #[config(default = "vec![256, 256, 256]")]
    pub hidden_dims: Vec<usize>,

    /// Whether to use a skip connection at the middle layer.
    #[config(default = true)]
    pub skip_connection: bool,

    /// Number of positional encoding bands for coordinates.
    #[config(default = 6)]
    pub positional_encoding_bands: usize,

    /// Whether to use geometric initialization for the final layer.
    #[config(default = true)]
    pub geometric_init: bool,

    /// Dropout probability.
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl SdfDecoderConfig {
    /// Compute the input dimension for the decoder.
    ///
    /// Input is: latent + positional_encoded_xyz
    pub fn input_dim(&self) -> usize {
        let fourier = FourierEncoderConfig::new(self.positional_encoding_bands);
        self.latent_dim + fourier.output_dim()
    }
}

/// Configuration for the semantic decoder.
#[derive(Config, Debug)]
pub struct SemanticDecoderConfig {
    /// Input latent dimension from encoder.
    pub latent_dim: usize,

    /// Number of semantic classes or feature dimensions.
    pub num_classes: usize,

    /// Hidden layer dimensions.
    #[config(default = "vec![128, 64]")]
    pub hidden_dims: Vec<usize>,

    /// Number of positional encoding bands.
    #[config(default = 4)]
    pub positional_encoding_bands: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointnet_config() {
        let config = PointNetEncoderConfig::new(256);
        assert_eq!(config.latent_dim, 256);
        assert_eq!(config.hidden_dims, vec![64, 128, 256]);
    }

    #[test]
    fn test_fourier_output_dim() {
        let config = FourierEncoderConfig::new(6);
        // 3 (input) + 6 bands * 3 axes * 2 (sin/cos) = 3 + 36 = 39
        assert_eq!(config.output_dim(), 39);

        let no_input = FourierEncoderConfig {
            include_input: false,
            ..config
        };
        assert_eq!(no_input.output_dim(), 36);
    }

    #[test]
    fn test_sdf_decoder_config() {
        let config = SdfDecoderConfig::new(256);
        // 256 (latent) + 39 (positional encoding with 6 bands) = 295
        assert_eq!(config.input_dim(), 256 + 39);
    }
}
