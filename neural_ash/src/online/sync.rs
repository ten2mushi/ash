//! Synchronization utilities for shared memory access.

use std::sync::atomic::{AtomicU64, Ordering};

use burn::prelude::*;

use ash_io::SharedGridWriter;

use crate::grid::DiffSdfGrid;

/// Synchronization helper for grid-to-shared-memory transfers.
pub struct SharedGridSync<const N: usize> {
    /// Version counter for change detection.
    version: AtomicU64,
    /// Last synchronized version.
    last_sync: AtomicU64,
}

impl<const N: usize> SharedGridSync<N> {
    /// Create a new sync helper.
    pub fn new() -> Self {
        Self {
            version: AtomicU64::new(0),
            last_sync: AtomicU64::new(0),
        }
    }

    /// Increment the version (call after grid update).
    pub fn mark_dirty(&self) {
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    /// Check if sync is needed.
    pub fn needs_sync(&self) -> bool {
        let current = self.version.load(Ordering::SeqCst);
        let last = self.last_sync.load(Ordering::SeqCst);
        current > last
    }

    /// Mark sync as complete.
    pub fn mark_synced(&self) {
        let current = self.version.load(Ordering::SeqCst);
        self.last_sync.store(current, Ordering::SeqCst);
    }

    /// Get the current version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    /// Synchronize a DiffSdfGrid to a SharedGridWriter.
    ///
    /// This extracts tensor values and writes them to shared memory.
    pub fn sync_grid<B: Backend>(
        &self,
        grid: &DiffSdfGrid<B, N>,
        writer: &mut SharedGridWriter<N>,
    ) {
        // Extract embeddings
        let embeddings = grid.embeddings.val();
        let data = embeddings.to_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        let cells_per_block = grid.config().cells_per_block();

        // Write each block
        for (block_idx, &_coord) in grid.block_coords().iter().enumerate() {
            let start = block_idx * cells_per_block * N;
            let end = start + cells_per_block * N;
            let block_values = &values[start..end];

            // Write to shared memory
            // Note: SharedGridWriter needs the appropriate method
            // For now, we write cell by cell
            for cell_idx in 0..cells_per_block {
                let value_start = cell_idx * N;
                let mut cell_values = [0.0f32; N];
                cell_values.copy_from_slice(&block_values[value_start..value_start + N]);
                writer.write_values(block_idx, cell_idx, cell_values);
            }
        }

        self.mark_synced();
    }
}

impl<const N: usize> Default for SharedGridSync<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limiter for sync operations.
pub struct SyncRateLimiter {
    /// Minimum interval between syncs in nanoseconds.
    min_interval_ns: u64,
    /// Last sync timestamp in nanoseconds.
    last_sync_ns: AtomicU64,
}

impl SyncRateLimiter {
    /// Create a new rate limiter with maximum rate in Hz.
    pub fn new(max_rate_hz: f64) -> Self {
        let min_interval_ns = (1e9 / max_rate_hz) as u64;
        Self {
            min_interval_ns,
            last_sync_ns: AtomicU64::new(0),
        }
    }

    /// Check if enough time has passed for another sync.
    pub fn should_sync(&self) -> bool {
        let now = std::time::Instant::now()
            .elapsed()
            .as_nanos() as u64;
        let last = self.last_sync_ns.load(Ordering::Relaxed);

        now.saturating_sub(last) >= self.min_interval_ns
    }

    /// Record a sync.
    pub fn record_sync(&self) {
        let now = std::time::Instant::now()
            .elapsed()
            .as_nanos() as u64;
        self.last_sync_ns.store(now, Ordering::Relaxed);
    }

    /// Attempt to sync if rate limit allows.
    pub fn try_sync(&self) -> bool {
        if self.should_sync() {
            self.record_sync();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_version() {
        let sync = SharedGridSync::<1>::new();

        assert!(!sync.needs_sync());

        sync.mark_dirty();
        assert!(sync.needs_sync());

        sync.mark_synced();
        assert!(!sync.needs_sync());
    }

    #[test]
    fn test_multiple_dirty() {
        let sync = SharedGridSync::<1>::new();

        sync.mark_dirty();
        sync.mark_dirty();
        sync.mark_dirty();

        assert_eq!(sync.version(), 3);
        assert!(sync.needs_sync());

        sync.mark_synced();
        assert!(!sync.needs_sync());
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = SyncRateLimiter::new(1000.0); // 1000 Hz = 1ms interval

        // First sync should always be allowed
        assert!(limiter.try_sync());

        // Immediate second sync should be blocked
        // (this test is timing-dependent, but for 1000Hz it should work)
        assert!(!limiter.should_sync());
    }
}
