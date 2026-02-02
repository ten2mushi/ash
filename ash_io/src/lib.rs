//! ash_io - I/O, storage, and shared memory for sparse-dense SDF grids.
//!
//! This crate provides the data storage and I/O layer for the ASH ecosystem,
//! with support for arbitrary feature dimensions per coordinate using
//! compile-time generics (`Grid<const N: usize>`).
//!
//! # Feature Dimensions
//!
//! The `N` type parameter specifies how many f32 features are stored per cell:
//! - `N=1`: SDF-only grid (most common for collision detection)
//! - `N=4`: SDF + 3 semantic features (e.g., material classification)
//! - `N=8+`: Higher-dimensional feature spaces
//!
//! # Core Types
//!
//! - [`InMemoryGrid<N>`]: The main in-memory grid container
//! - [`GridBuilder<N>`]: Builder pattern for constructing grids
//! - [`BlockStorage<N>`]: SoA storage backend for optimal cache performance
//! - [`BlockMap`]: Lock-free hash map for O(1) block lookups
//!
//! # Example
//!
//! ```ignore
//! use ash_io::{GridBuilder, InMemoryGrid, BlockCoord, Point3};
//!
//! // Create an SDF-only grid (N=1)
//! let grid: InMemoryGrid<1> = GridBuilder::new(8, 0.1)
//!     .with_capacity(100)
//!     .add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
//!         let center = Point3::new(0.4, 0.4, 0.4);
//!         [(pos - center).length() - 0.3]
//!     })
//!     .build()
//!     .unwrap();
//!
//! // Query the SDF value
//! if let Some([sdf]) = grid.query(Point3::new(0.4, 0.4, 0.4)) {
//!     println!("SDF at center: {}", sdf);
//! }
//!
//! // Create a multi-feature grid (N=4)
//! let grid: InMemoryGrid<4> = GridBuilder::new(8, 0.1)
//!     .with_capacity(100)
//!     .add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
//!         [
//!             pos.length() - 0.5,  // SDF
//!             pos.x,               // Feature 1
//!             pos.y,               // Feature 2
//!             pos.z,               // Feature 3
//!         ]
//!     })
//!     .build()
//!     .unwrap();
//! ```
//!
//! # Crate Features
//!
//! - `std` (default): Enables standard library support
//! - `alloc`: Enables heap allocation without full std

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod config;
pub mod convert;
pub mod error;
pub mod format;
pub mod memory;
pub mod shared;
pub mod spatial;

// Re-export core types from ash_core
pub use ash_core::{
    BlockCoord, CellCoord, CellValueProvider, InterpolationResult, LocalCoord, Point3,
    SdfInterpolationResult, SdfProvider, UNTRAINED_SENTINEL,
};

// Re-export main types
pub use config::GridConfig;
pub use error::{AshIoError, Result};
pub use memory::{BlockMap, BlockStorage, GridBuilder, InMemoryGrid, SdfGrid, SdfGridBuilder};

// Re-export format types
pub use format::{compute_file_size, load_grid, save_grid, AshHeader, ASH_MAGIC, HEADER_SIZE};
#[cfg(feature = "std")]
pub use format::{load_from_file, save_to_file};

// Re-export convert types (export)
pub use convert::{export_obj, MeshStats, ObjExportConfig};
#[cfg(feature = "std")]
pub use convert::{export_obj_to_file, export_obj_with_stats};

// Re-export convert types (import)
pub use convert::{ObjImportConfig, TriangleMesh, ImportStats};
#[cfg(feature = "std")]
pub use convert::{parse_obj, parse_obj_file, import_obj_to_grid, import_obj_file_to_grid, import_obj_narrow_band};

// Re-export shared memory types
pub use shared::{
    compute_shared_size, SharedGridView, SharedGridWriter, SharedHeader, SharedLayout,
    SharedSdfView, SharedSdfWriter, SHARED_MAGIC,
};

// Re-export spatial types
pub use spatial::{Aabb, TriangleBvh, NarrowBandConfig, compute_narrow_band, narrow_band_from_triangles};
