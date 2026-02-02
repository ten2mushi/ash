//! MLP (Multi-Layer Perceptron) building blocks.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;

/// Configuration for an MLP layer.
#[derive(Config, Debug)]
pub struct MlpConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Hidden layer dimensions.
    #[config(default = "vec![]")]
    pub hidden_dims: Vec<usize>,
    /// Dropout probability.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Whether to apply activation to the final layer.
    #[config(default = false)]
    pub final_activation: bool,
}

impl MlpConfig {
    /// Initialize the MLP.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let mut layers = Vec::new();
        let mut in_dim = self.input_dim;

        // Hidden layers
        for &out_dim in &self.hidden_dims {
            layers.push(LinearConfig::new(in_dim, out_dim).init(device));
            in_dim = out_dim;
        }

        // Output layer
        let output = LinearConfig::new(in_dim, self.output_dim).init(device);

        let dropout = if self.dropout > 0.0 {
            Some(DropoutConfig::new(self.dropout).init())
        } else {
            None
        };

        Mlp {
            layers,
            output,
            activation: Relu::new(),
            dropout,
            final_activation: self.final_activation,
        }
    }
}

/// Multi-Layer Perceptron module.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// Hidden layers.
    layers: Vec<Linear<B>>,
    /// Output layer.
    output: Linear<B>,
    /// Activation function.
    activation: Relu,
    /// Optional dropout.
    dropout: Option<Dropout>,
    /// Whether to apply activation to final layer.
    #[module(skip)]
    final_activation: bool,
}

impl<B: Backend> Mlp<B> {
    /// Forward pass.
    ///
    /// Input shape: [batch, input_dim]
    /// Output shape: [batch, output_dim]
    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Hidden layers with activation and optional dropout
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
            if let Some(ref dropout) = self.dropout {
                x = dropout.forward(x);
            }
        }

        // Output layer
        x = self.output.forward(x);

        if self.final_activation {
            x = self.activation.forward(x);
        }

        x
    }

    /// Forward pass for 3D input (batched sequences).
    ///
    /// Input shape: [batch, seq_len, input_dim]
    /// Output shape: [batch, seq_len, output_dim]
    pub fn forward_3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, input_dim] = x.dims();

        // Reshape to 2D for linear layers
        let x_flat = x.reshape([batch * seq_len, input_dim]);
        let y_flat = self.forward(x_flat);
        let output_dim = self.output.weight.dims()[0];

        y_flat.reshape([batch, seq_len, output_dim])
    }
}

/// Positional encoding using Fourier features.
#[derive(Module, Debug)]
pub struct FourierFeatures<B: Backend> {
    /// Frequency bands.
    frequencies: Tensor<B, 1>,
    /// Whether to include original input.
    #[module(skip)]
    include_input: bool,
}

impl<B: Backend> FourierFeatures<B> {
    /// Create new Fourier features.
    pub fn new(num_bands: usize, max_freq_log2: f32, include_input: bool, device: &B::Device) -> Self {
        // Frequencies from 2^0 to 2^max_freq_log2
        let frequencies: Vec<f32> = (0..num_bands)
            .map(|i| {
                let t = i as f32 / (num_bands - 1).max(1) as f32;
                (t * max_freq_log2).exp2() * std::f32::consts::PI
            })
            .collect();

        let frequencies = Tensor::from_data(frequencies.as_slice(), device);

        Self {
            frequencies,
            include_input,
        }
    }

    /// Encode input coordinates.
    ///
    /// Input shape: [batch, 3]
    /// Output shape: [batch, output_dim] where output_dim = 3 + 6 * num_bands (if include_input)
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, dim] = x.dims();
        let num_bands = self.frequencies.dims()[0];

        // Expand x for broadcasting: [batch, dim, 1]
        let x_expanded = x.clone().reshape([batch, dim, 1]);

        // Expand frequencies: [1, 1, num_bands]
        let freqs = self.frequencies.clone().reshape([1, 1, num_bands]);

        // Compute x * freq: [batch, dim, num_bands]
        let scaled = x_expanded * freqs;

        // Compute sin and cos
        let sin_features = scaled.clone().sin();
        let cos_features = scaled.cos();

        // Interleave sin/cos: [batch, dim, num_bands * 2]
        let fourier = Tensor::cat(vec![sin_features, cos_features], 2);

        // Reshape to [batch, dim * num_bands * 2]
        let fourier_flat = fourier.reshape([batch, dim * num_bands * 2]);

        if self.include_input {
            Tensor::cat(vec![x, fourier_flat], 1)
        } else {
            fourier_flat
        }
    }

    /// Get the output dimension.
    pub fn output_dim(&self, input_dim: usize) -> usize {
        let fourier_dim = input_dim * self.frequencies.dims()[0] * 2;
        if self.include_input {
            input_dim + fourier_dim
        } else {
            fourier_dim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_mlp_forward() {
        let device = Default::default();
        let config = MlpConfig::new(3, 1).with_hidden_dims(vec![64, 32]);
        let mlp = config.init::<TestBackend>(&device);

        let input = Tensor::zeros([4, 3], &device);
        let output = mlp.forward(input);

        assert_eq!(output.dims(), [4, 1]);
    }

    #[test]
    fn test_fourier_features() {
        let device = Default::default();
        let fourier = FourierFeatures::<TestBackend>::new(6, 4.0, true, &device);

        let input = Tensor::zeros([4, 3], &device);
        let output = fourier.forward(input);

        // 3 (input) + 3 * 6 * 2 (sin/cos) = 3 + 36 = 39
        assert_eq!(output.dims(), [4, 39]);
    }

    #[test]
    fn test_fourier_without_input() {
        let device = Default::default();
        let fourier = FourierFeatures::<TestBackend>::new(6, 4.0, false, &device);

        let input = Tensor::zeros([4, 3], &device);
        let output = fourier.forward(input);

        // 3 * 6 * 2 = 36
        assert_eq!(output.dims(), [4, 36]);
    }
}
