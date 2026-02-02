//! Online training and inference for real-time robotics applications.
//!
//! This module provides:
//! - `ConcurrentFeatureUpdater`: Updates shared memory grid concurrently
//! - `InferenceScheduler`: Manages update rates for different features
//! - Synchronization with ash_rs via shared memory

#[cfg(feature = "online")]
mod scheduler;
#[cfg(feature = "online")]
mod sync;
#[cfg(feature = "online")]
mod updater;

#[cfg(feature = "online")]
pub use scheduler::InferenceScheduler;
#[cfg(feature = "online")]
pub use sync::SharedGridSync;
#[cfg(feature = "online")]
pub use updater::{ConcurrentFeatureUpdater, FeatureUpdaterConfig};

/// Configuration for a feature updater.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Feature indices to update.
    pub feature_indices: Vec<usize>,
    /// Update rate in Hz.
    pub update_rate_hz: f64,
    /// Priority (higher = more important).
    pub priority: u32,
}

impl FeatureConfig {
    /// Create a new feature configuration.
    pub fn new(feature_indices: Vec<usize>, update_rate_hz: f64) -> Self {
        Self {
            feature_indices,
            update_rate_hz,
            priority: 0,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Default configurations for common use cases.
pub mod presets {
    use super::FeatureConfig;

    /// SDF-only configuration at 100Hz.
    pub fn sdf_only() -> FeatureConfig {
        FeatureConfig::new(vec![0], 100.0)
    }

    /// SDF + 3 semantic features, SDF at 100Hz, semantics at 10Hz.
    pub fn sdf_with_semantics() -> Vec<FeatureConfig> {
        vec![
            FeatureConfig::new(vec![0], 100.0).with_priority(10),
            FeatureConfig::new(vec![1, 2, 3], 10.0).with_priority(1),
        ]
    }

    /// High-frequency SDF for fast collision checking.
    pub fn high_freq_sdf() -> FeatureConfig {
        FeatureConfig::new(vec![0], 500.0).with_priority(100)
    }
}
