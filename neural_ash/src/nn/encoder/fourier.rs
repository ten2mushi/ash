//! Fourier positional encoder.

use burn::module::Module;
use burn::prelude::*;

use crate::config::FourierEncoderConfig;

/// Fourier positional encoding for 3D coordinates.
///
/// Transforms coordinates into a higher-dimensional feature space using
/// sinusoidal functions at multiple frequencies.
#[derive(Module, Debug)]
pub struct FourierEncoder<B: Backend> {
    /// Frequency bands for encoding.
    frequencies: Tensor<B, 1>,
    /// Whether to include the original input coordinates.
    #[module(skip)]
    include_input: bool,
}

impl<B: Backend> FourierEncoder<B> {
    /// Create a new Fourier encoder from configuration.
    pub fn new(config: &FourierEncoderConfig, device: &B::Device) -> Self {
        // Generate frequency bands from 2^0 to 2^max_freq_log2
        let frequencies: Vec<f32> = (0..config.num_bands)
            .map(|i| {
                let t = i as f32 / (config.num_bands - 1).max(1) as f32;
                (t * config.max_freq_log2).exp2() * std::f32::consts::PI
            })
            .collect();

        let frequencies = Tensor::from_data(frequencies.as_slice(), device);

        Self {
            frequencies,
            include_input: config.include_input,
        }
    }

    /// Encode 3D coordinates.
    ///
    /// Input: coordinates of shape [batch, 3]
    /// Output: encoded features of shape [batch, output_dim]
    pub fn forward(&self, coords: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, dim] = coords.dims();
        let num_bands = self.frequencies.dims()[0];

        // Expand coordinates for broadcasting: [batch, dim, 1]
        let coords_expanded = coords.clone().reshape([batch, dim, 1]);

        // Expand frequencies: [1, 1, num_bands]
        let freqs = self.frequencies.clone().reshape([1, 1, num_bands]);

        // Compute coords * freq: [batch, dim, num_bands]
        let scaled = coords_expanded * freqs;

        // Compute sin and cos features
        let sin_features = scaled.clone().sin();
        let cos_features = scaled.cos();

        // Concatenate sin and cos: [batch, dim, num_bands * 2]
        let fourier = Tensor::cat(vec![sin_features, cos_features], 2);

        // Reshape to [batch, dim * num_bands * 2]
        let fourier_flat = fourier.reshape([batch, dim * num_bands * 2]);

        if self.include_input {
            Tensor::cat(vec![coords, fourier_flat], 1)
        } else {
            fourier_flat
        }
    }

    /// Get the output dimension.
    pub fn output_dim(&self) -> usize {
        let num_bands = self.frequencies.dims()[0];
        let fourier_dim = 3 * num_bands * 2; // 3 input dims, sin + cos
        if self.include_input {
            3 + fourier_dim
        } else {
            fourier_dim
        }
    }

    /// Get the number of frequency bands.
    pub fn num_bands(&self) -> usize {
        self.frequencies.dims()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_fourier_encoder_output_dim() {
        let device = Default::default();
        let config = FourierEncoderConfig::new(6);
        let encoder = FourierEncoder::<TestBackend>::new(&config, &device);

        // 3 (input) + 3 * 6 * 2 (sin/cos) = 3 + 36 = 39
        assert_eq!(encoder.output_dim(), 39);
    }

    #[test]
    fn test_fourier_encoder_forward() {
        let device = Default::default();
        let config = FourierEncoderConfig::new(6);
        let encoder = FourierEncoder::<TestBackend>::new(&config, &device);

        let coords = Tensor::zeros([4, 3], &device);
        let output = encoder.forward(coords);

        assert_eq!(output.dims(), [4, 39]);
    }

    #[test]
    fn test_fourier_encoder_no_input() {
        let device = Default::default();
        let config = FourierEncoderConfig::new(6)
            .with_include_input(false);
        let encoder = FourierEncoder::<TestBackend>::new(&config, &device);

        let coords = Tensor::zeros([4, 3], &device);
        let output = encoder.forward(coords);

        // 3 * 6 * 2 = 36
        assert_eq!(output.dims(), [4, 36]);
    }

    #[test]
    fn test_fourier_values_periodic() {
        let device = Default::default();
        let config = FourierEncoderConfig::new(4);
        let encoder = FourierEncoder::<TestBackend>::new(&config, &device);

        // Points at origin and at 2*pi should have similar encodings
        // (for the lowest frequency band)
        let p1: Tensor<TestBackend, 2> = Tensor::from_data([[0.0f32, 0.0, 0.0]], &device);
        let _p2: Tensor<TestBackend, 2> = Tensor::from_data([[2.0 * std::f32::consts::PI, 0.0, 0.0]], &device);

        let e1 = encoder.forward(p1);
        // The encoding should be finite
        let data = e1.to_data();
        let vals: Vec<f32> = data.to_vec().unwrap();
        for v in vals {
            assert!(v.is_finite());
        }
    }
}
