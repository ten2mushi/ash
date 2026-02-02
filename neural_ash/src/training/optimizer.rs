//! Optimizer configuration and training state for neural SDF training.
//!
//! Provides RMSprop optimizer with Burn integration for updating grid embeddings.

use burn::config::Config;
use burn::prelude::*;

/// Configuration for the optimizer.
#[derive(Config, Debug)]
pub struct OptimizerConfig {
    /// Learning rate.
    #[config(default = 1e-3)]
    pub learning_rate: f64,

    /// RMSprop alpha (smoothing constant).
    #[config(default = 0.99)]
    pub alpha: f32,

    /// Epsilon for numerical stability.
    #[config(default = 1e-8)]
    pub epsilon: f32,

    /// Weight decay (L2 regularization).
    #[config(default = 0.0)]
    pub weight_decay: f64,

    /// Momentum factor.
    #[config(default = 0.0)]
    pub momentum: f32,

    /// Whether to center the gradient (RMSprop variant).
    #[config(default = false)]
    pub centered: bool,

    /// Gradient clipping threshold (0 = no clipping).
    #[config(default = 1.0)]
    pub gradient_clip: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// RMSprop optimizer state for a single parameter tensor.
///
/// This implements a simple RMSprop optimizer that can be used with
/// the gradient accumulator for updating embeddings.
pub struct RmsPropState<B: Backend> {
    /// Running average of squared gradients.
    square_avg: Tensor<B, 2>,
    /// Momentum buffer (if momentum > 0).
    momentum_buffer: Option<Tensor<B, 2>>,
    /// Running average of gradients (if centered).
    grad_avg: Option<Tensor<B, 2>>,
    /// Configuration.
    config: OptimizerConfig,
    /// Current step count.
    step: usize,
}

impl<B: Backend> RmsPropState<B> {
    /// Create a new RMSprop state for a parameter tensor.
    pub fn new(shape: [usize; 2], config: OptimizerConfig, device: &B::Device) -> Self {
        let square_avg = Tensor::zeros(shape, device);
        let momentum_buffer = if config.momentum > 0.0 {
            Some(Tensor::zeros(shape, device))
        } else {
            None
        };
        let grad_avg = if config.centered {
            Some(Tensor::zeros(shape, device))
        } else {
            None
        };

        Self {
            square_avg,
            momentum_buffer,
            grad_avg,
            config,
            step: 0,
        }
    }

    /// Perform an optimization step.
    ///
    /// # Arguments
    /// * `param` - Current parameter values
    /// * `grad` - Gradient tensor
    ///
    /// # Returns
    /// Updated parameter tensor
    pub fn step(&mut self, param: Tensor<B, 2>, grad: Tensor<B, 2>) -> Tensor<B, 2> {
        self.step += 1;

        // Apply gradient clipping if configured
        let grad = if self.config.gradient_clip > 0.0 {
            clip_grad_norm(grad, self.config.gradient_clip)
        } else {
            grad
        };

        // Apply weight decay
        let grad = if self.config.weight_decay > 0.0 {
            grad + param.clone() * self.config.weight_decay as f32
        } else {
            grad
        };

        let alpha = self.config.alpha;
        let eps = self.config.epsilon;
        let lr = self.config.learning_rate as f32;

        // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
        self.square_avg = self.square_avg.clone() * alpha + grad.clone() * grad.clone() * (1.0 - alpha);

        let avg = if self.config.centered {
            // Update gradient average
            let grad_avg = self.grad_avg.take().unwrap();
            let new_grad_avg = grad_avg * alpha + grad.clone() * (1.0 - alpha);
            let avg = self.square_avg.clone() - new_grad_avg.clone() * new_grad_avg.clone();
            self.grad_avg = Some(new_grad_avg);
            avg
        } else {
            self.square_avg.clone()
        };

        // Compute update: grad / sqrt(avg + eps)
        let update = grad / (avg.sqrt() + eps);

        // Apply momentum if configured
        let update = if self.config.momentum > 0.0 {
            let momentum_buffer = self.momentum_buffer.take().unwrap();
            let new_buffer = momentum_buffer * self.config.momentum + update;
            let result = new_buffer.clone();
            self.momentum_buffer = Some(new_buffer);
            result
        } else {
            update
        };

        // Update parameters
        param - update * lr
    }

    /// Get the current step count.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Reset the optimizer state.
    pub fn reset(&mut self, device: &B::Device) {
        let shape = self.square_avg.dims();
        self.square_avg = Tensor::zeros(shape, device);
        if let Some(ref mut buf) = self.momentum_buffer {
            *buf = Tensor::zeros(shape, device);
        }
        if let Some(ref mut avg) = self.grad_avg {
            *avg = Tensor::zeros(shape, device);
        }
        self.step = 0;
    }
}

/// Clip gradient by L2 norm.
fn clip_grad_norm<B: Backend>(grad: Tensor<B, 2>, max_norm: f32) -> Tensor<B, 2> {
    let grad_sq = grad.clone() * grad.clone();
    let norm_sq = grad_sq.sum();
    let norm: f32 = norm_sq.to_data().to_vec().unwrap()[0];
    let norm = norm.sqrt();

    if norm > max_norm {
        grad * (max_norm / norm)
    } else {
        grad
    }
}

/// Training state that holds optimizer and training progress.
pub struct TrainingState<B: Backend> {
    /// Optimizer state for embeddings.
    pub optimizer: RmsPropState<B>,
    /// Current epoch.
    pub epoch: usize,
    /// Total training steps.
    pub total_steps: usize,
    /// Best loss achieved.
    pub best_loss: f32,
    /// Running average of recent losses.
    pub avg_loss: f32,
    /// Exponential moving average factor for loss.
    loss_ema_factor: f32,
}

impl<B: Backend> TrainingState<B> {
    /// Create a new training state.
    pub fn new(
        embedding_shape: [usize; 2],
        optimizer_config: OptimizerConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            optimizer: RmsPropState::new(embedding_shape, optimizer_config, device),
            epoch: 0,
            total_steps: 0,
            best_loss: f32::INFINITY,
            avg_loss: 0.0,
            loss_ema_factor: 0.99,
        }
    }

    /// Update training state after a step.
    pub fn on_step(&mut self, loss: f32) {
        self.total_steps += 1;

        // Update EMA loss
        if self.total_steps == 1 {
            self.avg_loss = loss;
        } else {
            self.avg_loss = self.loss_ema_factor * self.avg_loss + (1.0 - self.loss_ema_factor) * loss;
        }

        // Track best loss
        if loss < self.best_loss {
            self.best_loss = loss;
        }
    }

    /// Update training state after an epoch.
    pub fn on_epoch(&mut self) {
        self.epoch += 1;
    }

    /// Check if training has improved recently.
    pub fn is_improving(&self, _patience_steps: usize) -> bool {
        // Simple check: if avg_loss is close to best_loss
        self.avg_loss < self.best_loss * 1.1
    }

    /// Get learning rate with optional warmup.
    pub fn get_learning_rate(&self, base_lr: f64, warmup_steps: usize) -> f64 {
        if self.total_steps < warmup_steps {
            base_lr * (self.total_steps as f64 / warmup_steps as f64)
        } else {
            base_lr
        }
    }
}

/// Simple SGD optimizer for comparison/testing.
pub struct SgdState {
    /// Learning rate.
    pub learning_rate: f32,
    /// Momentum factor.
    pub momentum: f32,
    /// Step count.
    step: usize,
}

impl SgdState {
    /// Create a new SGD optimizer state.
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            step: 0,
        }
    }

    /// Perform an optimization step (stateless, no momentum buffer).
    pub fn step<B: Backend>(&mut self, param: Tensor<B, 2>, grad: Tensor<B, 2>) -> Tensor<B, 2> {
        self.step += 1;
        param - grad * self.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.alpha > 0.0 && config.alpha < 1.0);
    }

    #[test]
    fn test_rmsprop_step() {
        let device = Default::default();
        let config = OptimizerConfig::new().with_learning_rate(0.1);
        let mut state = RmsPropState::<TestBackend>::new([10, 1], config, &device);

        let param = Tensor::zeros([10, 1], &device);
        let grad = Tensor::full([10, 1], 1.0, &device);

        let new_param = state.step(param.clone(), grad);

        // Parameters should have moved in the negative gradient direction
        let data: Vec<f32> = new_param.to_data().to_vec().unwrap();
        assert!(data[0] < 0.0);
    }

    #[test]
    fn test_gradient_clipping() {
        let device: <TestBackend as Backend>::Device = Default::default();

        // Create a gradient with large norm
        let grad = Tensor::<TestBackend, 2>::full([10, 1], 10.0, &device);
        let clipped = clip_grad_norm(grad, 1.0);

        // Compute norm of clipped gradient
        let clipped_data: Vec<f32> = clipped.to_data().to_vec().unwrap();
        let norm: f32 = clipped_data.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_training_state() {
        let device = Default::default();
        let config = OptimizerConfig::default();
        let mut state = TrainingState::<TestBackend>::new([100, 1], config, &device);

        assert_eq!(state.epoch, 0);
        assert_eq!(state.total_steps, 0);

        state.on_step(1.0);
        assert_eq!(state.total_steps, 1);
        assert!((state.avg_loss - 1.0).abs() < 1e-6);

        state.on_epoch();
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_learning_rate_warmup() {
        let device = Default::default();
        let config = OptimizerConfig::default();
        let mut state = TrainingState::<TestBackend>::new([100, 1], config, &device);

        // At step 0, warmup should give 0
        let lr = state.get_learning_rate(0.01, 100);
        assert!((lr - 0.0).abs() < 1e-10);

        // After 50 steps, should be 50% of base lr
        for _ in 0..50 {
            state.on_step(1.0);
        }
        let lr = state.get_learning_rate(0.01, 100);
        assert!((lr - 0.005).abs() < 1e-6);

        // After 100 steps, should be full lr
        for _ in 0..50 {
            state.on_step(1.0);
        }
        let lr = state.get_learning_rate(0.01, 100);
        assert!((lr - 0.01).abs() < 1e-6);
    }
}
