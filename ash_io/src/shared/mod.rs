//! Shared memory support for real-time grid access.
//!
//! This module provides types for sharing grids across processes
//! with lock-free read access using seqlock synchronization.
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
//! │  ├─ feature_dim: u16                                            │
//! │  └─ num_blocks: AtomicU32                                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Map (lock-free hash table)                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Coords (reverse lookup)                                  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Feature Data (SoA layout, 32-byte aligned)                     │
//! │  └─ For each feature dimension:                                 │
//! │       values: [f32; capacity * grid_dim³]                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Versions (per-block seqlock)                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Synchronization Strategy (Seqlock)
//!
//! **Writer (neural_ash)**:
//! 1. Increment block version to ODD (signals "writing in progress")
//! 2. Write block data
//! 3. Memory fence
//! 4. Increment block version to EVEN (signals "write complete")
//!
//! **Reader (ash_rs)**:
//! 1. Read block version (must be EVEN)
//! 2. Read block data
//! 3. Memory fence
//! 4. Read block version again (must match)
//! 5. If mismatch, retry read
//!
//! # Example
//!
//! ```ignore
//! use ash_io::{SharedGridWriter, SharedGridView, compute_shared_size, BlockCoord};
//!
//! // Writer side (neural_ash)
//! let size = compute_shared_size::<1>(8, 1000);
//! let mut buffer = vec![0u8; size];
//! let mut writer = unsafe {
//!     SharedGridWriter::<1>::initialize(buffer.as_mut_ptr(), size, 8, 0.1, 1000)?
//! };
//! let block_idx = writer.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();
//! writer.write_values(block_idx, 0, [0.5]);
//!
//! // Reader side (ash_rs)
//! let view = unsafe { SharedGridView::<1>::from_ptr(buffer.as_ptr())? };
//! let sdf = view.query(Point3::new(0.4, 0.4, 0.4));
//! ```

pub mod layout;
pub mod view;
pub mod writer;

pub use layout::{compute_shared_size, SharedHeader, SharedLayout, SHARED_MAGIC};
pub use view::SharedGridView;
pub use writer::SharedGridWriter;

/// Type alias for SDF-only shared view.
pub type SharedSdfView = SharedGridView<1>;

/// Type alias for SDF-only shared writer.
pub type SharedSdfWriter = SharedGridWriter<1>;
