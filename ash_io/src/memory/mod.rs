//! In-memory grid types with N-dimensional feature support.
//!
//! This module provides the core grid types:
//! - `InMemoryGrid<N>`: The main grid container with N features per cell
//! - `GridBuilder<N>`: Builder pattern for constructing grids
//! - `BlockStorage<N>`: SoA storage backend
//! - `BlockMap`: Lock-free hash map for block lookups
//! - `BatchLookupResult`: Results from batch coordinate lookups

pub mod batch_lookup;
pub mod block_map;
pub mod builder;
pub mod grid;
pub mod storage;

pub use batch_lookup::BatchLookupResult;
pub use block_map::BlockMap;
pub use builder::{GridBuilder, SdfGridBuilder};
pub use grid::InMemoryGrid;
pub use storage::BlockStorage;

/// Type alias for SDF-only grid (most common case).
pub type SdfGrid = InMemoryGrid<1>;
