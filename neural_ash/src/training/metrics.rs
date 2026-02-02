//! Training metrics and output types.

#![allow(dead_code)]

use burn::prelude::*;

/// Output from a single training step.
#[derive(Debug, Clone)]
pub struct TrainOutput<B: Backend> {
    /// Total loss (scalar).
    pub loss: Tensor<B, 1>,
    /// Individual loss components.
    pub metrics: TrainMetrics,
}

impl<B: Backend> TrainOutput<B> {
    /// Create a new training output.
    pub fn new(loss: Tensor<B, 1>, metrics: TrainMetrics) -> Self {
        Self { loss, metrics }
    }

    /// Get the loss as a scalar value.
    pub fn loss_value(&self) -> f32 {
        self.loss.clone().to_data().to_vec().unwrap()[0]
    }
}

/// Training metrics for a single step.
#[derive(Debug, Clone, Default)]
pub struct TrainMetrics {
    /// Surface loss component.
    pub surface_loss: f32,
    /// Free-space loss component.
    pub free_space_loss: f32,
    /// Eikonal loss component.
    pub eikonal_loss: f32,
    /// Regularization loss component.
    pub regularization_loss: f32,
    /// Gradient norm (for monitoring).
    pub gradient_norm: f32,
    /// Learning rate (if using scheduler).
    pub learning_rate: f32,
    /// Number of training steps.
    pub step: usize,
}

impl TrainMetrics {
    /// Create new training metrics.
    pub fn new(
        surface_loss: f32,
        free_space_loss: f32,
        eikonal_loss: f32,
        regularization_loss: f32,
    ) -> Self {
        Self {
            surface_loss,
            free_space_loss,
            eikonal_loss,
            regularization_loss,
            gradient_norm: 0.0,
            learning_rate: 0.0,
            step: 0,
        }
    }

    /// Get the total loss.
    pub fn total_loss(&self) -> f32 {
        self.surface_loss + self.free_space_loss + self.eikonal_loss + self.regularization_loss
    }

    /// Log metrics to standard output.
    pub fn log(&self, prefix: &str) {
        log::info!(
            "{} step={} total={:.6} surface={:.6} free={:.6} eik={:.6} reg={:.6} lr={:.2e}",
            prefix,
            self.step,
            self.total_loss(),
            self.surface_loss,
            self.free_space_loss,
            self.eikonal_loss,
            self.regularization_loss,
            self.learning_rate,
        );
    }
}

/// Running average tracker for metrics.
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    /// Window size for running average.
    window_size: usize,
    /// Recent surface losses.
    surface_losses: Vec<f32>,
    /// Recent free-space losses.
    free_space_losses: Vec<f32>,
    /// Recent Eikonal losses.
    eikonal_losses: Vec<f32>,
    /// Recent regularization losses.
    regularization_losses: Vec<f32>,
    /// Total steps.
    total_steps: usize,
}

impl MetricsTracker {
    /// Create a new metrics tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            surface_losses: Vec::with_capacity(window_size),
            free_space_losses: Vec::with_capacity(window_size),
            eikonal_losses: Vec::with_capacity(window_size),
            regularization_losses: Vec::with_capacity(window_size),
            total_steps: 0,
        }
    }

    /// Add metrics from a training step.
    pub fn add(&mut self, metrics: &TrainMetrics) {
        self.push_to_window(&mut self.surface_losses.clone(), metrics.surface_loss);
        self.push_to_window(&mut self.free_space_losses.clone(), metrics.free_space_loss);
        self.push_to_window(&mut self.eikonal_losses.clone(), metrics.eikonal_loss);
        self.push_to_window(&mut self.regularization_losses.clone(), metrics.regularization_loss);

        // Actually mutate
        Self::push_to_window_in_place(&mut self.surface_losses, metrics.surface_loss, self.window_size);
        Self::push_to_window_in_place(&mut self.free_space_losses, metrics.free_space_loss, self.window_size);
        Self::push_to_window_in_place(&mut self.eikonal_losses, metrics.eikonal_loss, self.window_size);
        Self::push_to_window_in_place(&mut self.regularization_losses, metrics.regularization_loss, self.window_size);

        self.total_steps += 1;
    }

    fn push_to_window_in_place(vec: &mut Vec<f32>, value: f32, window_size: usize) {
        if vec.len() >= window_size {
            vec.remove(0);
        }
        vec.push(value);
    }

    fn push_to_window(&self, vec: &mut Vec<f32>, value: f32) {
        if vec.len() >= self.window_size {
            vec.remove(0);
        }
        vec.push(value);
    }

    fn average(vec: &[f32]) -> f32 {
        if vec.is_empty() {
            0.0
        } else {
            vec.iter().sum::<f32>() / vec.len() as f32
        }
    }

    /// Get average metrics over the window.
    pub fn average_metrics(&self) -> TrainMetrics {
        TrainMetrics {
            surface_loss: Self::average(&self.surface_losses),
            free_space_loss: Self::average(&self.free_space_losses),
            eikonal_loss: Self::average(&self.eikonal_losses),
            regularization_loss: Self::average(&self.regularization_losses),
            gradient_norm: 0.0,
            learning_rate: 0.0,
            step: self.total_steps,
        }
    }

    /// Get total number of steps.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_total() {
        let metrics = TrainMetrics::new(0.1, 0.05, 0.02, 0.01);
        assert!((metrics.total_loss() - 0.18).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new(10);

        for i in 0..20 {
            let metrics = TrainMetrics::new(i as f32 * 0.1, 0.05, 0.02, 0.01);
            tracker.add(&metrics);
        }

        assert_eq!(tracker.total_steps(), 20);

        // Average should be over last 10 values
        let avg = tracker.average_metrics();
        // Last 10 surface losses: 1.0, 1.1, 1.2, ..., 1.9
        // Average: (1.0 + 1.1 + ... + 1.9) / 10 = 14.5 / 10 = 1.45
        assert!((avg.surface_loss - 1.45).abs() < 1e-5);
    }
}
