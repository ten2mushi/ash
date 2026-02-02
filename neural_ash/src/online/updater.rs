//! Concurrent feature updater for online inference.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use burn::prelude::*;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

use ash_io::SharedGridWriter;

use crate::error::Result;
use crate::grid::DiffSdfGrid;

/// Configuration for a feature updater.
#[derive(Debug, Clone)]
pub struct FeatureUpdaterConfig {
    /// Feature indices to update.
    pub feature_indices: Vec<usize>,
    /// Update rate in Hz.
    pub update_rate_hz: f64,
}

impl FeatureUpdaterConfig {
    /// Create a new configuration.
    pub fn new(feature_indices: Vec<usize>, update_rate_hz: f64) -> Self {
        Self {
            feature_indices,
            update_rate_hz,
        }
    }
}

/// Concurrent feature updater that writes to shared memory.
pub struct ConcurrentFeatureUpdater<B: Backend, const N: usize> {
    /// Shared memory writer.
    writer: Arc<RwLock<SharedGridWriter<N>>>,
    /// Differentiable grid (source of truth).
    grid: Arc<RwLock<DiffSdfGrid<B, N>>>,
    /// Updater configurations.
    configs: Vec<FeatureUpdaterConfig>,
    /// Running flag.
    running: Arc<AtomicBool>,
}

impl<B: Backend, const N: usize> ConcurrentFeatureUpdater<B, N> {
    /// Create a new concurrent updater.
    pub fn new(
        writer: Arc<RwLock<SharedGridWriter<N>>>,
        grid: Arc<RwLock<DiffSdfGrid<B, N>>>,
        configs: Vec<FeatureUpdaterConfig>,
    ) -> Self {
        Self {
            writer,
            grid,
            configs,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start all updater tasks.
    pub async fn start(&self) {
        self.running.store(true, Ordering::SeqCst);

        let mut handles = Vec::new();

        for config in &self.configs {
            let writer = Arc::clone(&self.writer);
            let grid = Arc::clone(&self.grid);
            let running = Arc::clone(&self.running);
            let feature_indices = config.feature_indices.clone();
            let update_rate = config.update_rate_hz;

            let handle = tokio::spawn(async move {
                let period = Duration::from_secs_f64(1.0 / update_rate);
                let mut tick = interval(period);

                while running.load(Ordering::SeqCst) {
                    tick.tick().await;

                    // Read current grid state
                    let grid_guard = grid.read().await;

                    // Extract embeddings for the specified features
                    let embeddings = grid_guard.embeddings.val();
                    let data = embeddings.to_data();
                    let values: Vec<f32> = data.to_vec().unwrap();

                    drop(grid_guard);

                    // Write to shared memory
                    let mut writer_guard = writer.write().await;

                    // Update each allocated block
                    let num_blocks = writer_guard.num_blocks();
                    for block_idx in 0..num_blocks {
                        // For each feature in this updater's responsibility
                        for &feat_idx in &feature_indices {
                            // Compute values for this block and feature
                            // (simplified: in real impl, would extract specific feature channel)
                            // writer_guard.write_feature(block_idx, feat_idx, &block_values);
                        }
                    }

                    // Increment version to signal update
                    // writer_guard.increment_version();
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Stop all updater tasks.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the updater is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// Single-feature updater for simpler use cases.
pub struct SingleFeatureUpdater<B: Backend> {
    /// Grid to read from.
    grid: Arc<RwLock<DiffSdfGrid<B, 1>>>,
    /// Shared memory writer.
    writer: Arc<RwLock<SharedGridWriter<1>>>,
    /// Update rate in Hz.
    update_rate_hz: f64,
    /// Running flag.
    running: Arc<AtomicBool>,
}

impl<B: Backend> SingleFeatureUpdater<B> {
    /// Create a new single-feature updater.
    pub fn new(
        grid: Arc<RwLock<DiffSdfGrid<B, 1>>>,
        writer: Arc<RwLock<SharedGridWriter<1>>>,
        update_rate_hz: f64,
    ) -> Self {
        Self {
            grid,
            writer,
            update_rate_hz,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run the updater (blocking).
    pub async fn run(&self) {
        self.running.store(true, Ordering::SeqCst);

        let period = Duration::from_secs_f64(1.0 / self.update_rate_hz);
        let mut tick = interval(period);

        while self.running.load(Ordering::SeqCst) {
            tick.tick().await;

            // Sync grid to shared memory
            let grid_guard = self.grid.read().await;
            let embeddings = grid_guard.embeddings.val();
            let data = embeddings.to_data();
            let _values: Vec<f32> = data.to_vec().unwrap();

            // In a full implementation, we would:
            // 1. Write values to SharedGridWriter
            // 2. Update the version number
            // 3. Signal completion

            drop(grid_guard);
        }
    }

    /// Stop the updater.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_config() {
        let config = FeatureUpdaterConfig::new(vec![0, 1], 50.0);
        assert_eq!(config.feature_indices, vec![0, 1]);
        assert_eq!(config.update_rate_hz, 50.0);
    }
}
