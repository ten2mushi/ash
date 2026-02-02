//! Shared memory interface for zero-copy communication with neural_ash.
//!
//! This module re-exports types from `ash_io::shared` and provides
//! SDF-specific type aliases for convenience.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Shared Memory Region                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Header (64 bytes, cache-line aligned)                          │
//! │  ├─ magic: u32                                                  │
//! │  ├─ version: AtomicU64 (for epoch-based synchronization)       │
//! │  ├─ grid_dim: u32                                               │
//! │  ├─ cell_size: f32                                              │
//! │  ├─ capacity: u32                                               │
//! │  └─ num_blocks: AtomicU32                                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Map (lock-free hash table)                               │
//! │  └─ entries: [AtomicU64; capacity * 2]                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Coords (reverse lookup)                                  │
//! │  └─ coords: [BlockCoord; capacity]                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Values (SDF data, 32-byte aligned for SIMD)              │
//! │  └─ values: [f32; capacity * grid_dim³]                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Versions (per-block epoch for safe concurrent access)    │
//! │  └─ versions: [AtomicU64; capacity]                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Synchronization Strategy
//!
//! We use **seqlock** (sequence lock) semantics for lock-free reads:
//!
//! 1. **Writer (neural_ash)**:
//!    - Increment block version to ODD (signals "writing in progress")
//!    - Write block data
//!    - Memory fence
//!    - Increment block version to EVEN (signals "write complete")
//!
//! 2. **Reader (ash_rs)**:
//!    - Read block version (must be EVEN)
//!    - Read block data
//!    - Memory fence
//!    - Read block version again (must match)
//!    - If mismatch, retry read
//!
//! This provides:
//! - **Lock-free reads**: Readers never block
//! - **Wait-free writes**: Writers never wait for readers
//! - **Bounded retry**: At most one retry per concurrent write

// Re-export types from ash_io
pub use ash_io::shared::{
    compute_shared_size, SharedGridView, SharedGridWriter, SharedHeader, SharedLayout,
    SHARED_MAGIC,
};

/// Type alias for SDF-only shared view (N=1).
pub type SharedSdfView = SharedGridView<1>;

/// Type alias for SDF-only shared writer (N=1).
pub type SharedSdfWriter = SharedGridWriter<1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_header() {
        let header = SharedHeader::new(8, 0.1, 1000, 1);
        assert!(header.validate());
        assert!(header.validate_feature_dim::<1>());
        assert_eq!(header.cells_per_block(), 512);
        assert!((header.block_size() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_shared_layout() {
        let layout = SharedLayout::compute::<1>(8, 1000);

        // Verify offsets are properly ordered
        assert!(layout.block_map_offset < layout.coords_offset);
        assert!(layout.coords_offset < layout.feature_offsets[0]);
        assert!(layout.feature_offsets[0] < layout.versions_offset);
        assert!(layout.versions_offset < layout.total_size);

        // Verify values are 32-byte aligned
        assert_eq!(layout.feature_offsets[0] % 32, 0);
    }

    #[test]
    fn test_compute_shared_size() {
        let size = compute_shared_size::<1>(8, 1000);

        // At least 2MB for 1000 blocks
        assert!(size > 2_000_000);
        assert!(size < 3_000_000);
    }
}
