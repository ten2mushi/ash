//! # ash_rs
//!
//! Real-time SDF query runtime optimized for speed and memory efficiency in
//! sim2real robotics applications.
//!
//! This crate provides the production inference side of the ASH (Adaptive Sparse
//! Hierarchical) ecosystem. It is designed for:
//!
//! - **Speed**: < 100ns single-point queries, < 50μs batch queries of 1000 points
//! - **Memory efficiency**: 2KB per block (8³×4 bytes)
//! - **Deterministic timing**: Bounded worst-case latency for real-time control
//! - **Concurrent access**: Lock-free reads for multi-threaded applications
//!
//! ## Quick Start
//!
//! ```ignore
//! use ash_rs::{GridBuilder, SparseDenseGrid};
//! use ash_core::{BlockCoord, Point3};
//!
//! // Build a grid with SDF values
//! let grid = GridBuilder::new(8, 0.1)  // 8³ cells/block, 10cm cells
//!     .with_capacity(1000)
//!     .add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
//!         // Sphere SDF: distance to surface
//!         let center = Point3::new(0.4, 0.4, 0.4);
//!         (pos - center).length() - 0.3
//!     })
//!     .build()?;
//!
//! // Query the SDF at a point
//! if let Some(distance) = grid.query(Point3::new(0.5, 0.5, 0.5)) {
//!     println!("Distance to nearest surface: {}", distance);
//! }
//!
//! // Check for collision
//! let robot_radius = 0.2;
//! if grid.in_collision(robot_pos, robot_radius) {
//!     println!("Collision detected!");
//! }
//! ```
//!
//! ## Architecture
//!
//! `ash_rs` implements a sparse-dense grid structure:
//!
//! - **Sparse level**: Lock-free hash map from block coordinates to block indices
//! - **Dense level**: Contiguous arrays of SDF values for each block (SoA layout)
//!
//! This design provides O(1) block lookup and excellent cache utilization for
//! both single-point and batch queries.
//!
//! ## Feature Flags
//!
//! - `std` (default): Standard library support
//! - `alloc`: Heap allocation without full std
//! - `serde`: Binary serialization (.ash file format)
//! - `mesh`: Parallel mesh extraction via rayon
//! - `simd`: SIMD optimizations (auto-detected)
//!
//! ## Performance Guidelines
//!
//! For optimal performance:
//!
//! 1. Use `query_batch()` for multiple points instead of looping over `query()`
//! 2. Pre-allocate grids with appropriate capacity to avoid rebuilding
//! 3. Use `in_collision()` for simple collision checks (early exit optimization)
//! 4. For mesh extraction, use `extract_mesh()` which parallelizes automatically
//!
//! ## Serialization
//!
//! With the `serde` feature, grids can be saved/loaded in the `.ash` binary format:
//!
//! ```ignore
//! // Save
//! grid.save_to_file("environment.ash")?;
//!
//! // Load
//! let grid = SparseDenseGrid::load_from_file("environment.ash")?;
//! ```
//!
//! ## Design Philosophy
//!
//! `ash_rs` is for **querying** learned SDFs in production (fast, deterministic),
//! while `neural_ash` is for **learning** SDFs from sensor data (differentiable,
//! flexible). Training happens once on a workstation/cloud; deployment is
//! continuous on embedded systems.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

// Conditional std/alloc support
#[cfg(feature = "std")]
extern crate std;

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

// Re-export libm for internal use (ash_core uses it too)
pub(crate) use libm;

// Core modules
mod builder;
mod error;
mod grid;
mod query;

// Feature-gated modules
#[cfg(feature = "serde")]
mod io;

#[cfg(any(feature = "std", feature = "alloc"))]
mod mesh;

mod simd;

// Shared memory module for zero-copy neural_ash integration
#[cfg(any(feature = "std", feature = "alloc"))]
mod shared;

// Re-export core types
pub use builder::GridBuilder;
pub use error::{AshError, Result};
pub use grid::SparseDenseGrid;

// Re-export types from ash_io
pub use ash_io::{BlockMap, BlockStorage, GridConfig};

// Re-export SIMD types
pub use simd::{query_batch, BatchResult, BatchResultWithGradients};

// Re-export mesh types
#[cfg(any(feature = "std", feature = "alloc"))]
pub use mesh::{triangles_to_obj, MeshStats, Triangle};

// Re-export shared memory types for neural_ash integration
#[cfg(any(feature = "std", feature = "alloc"))]
pub use shared::{
    compute_shared_size, SharedGridView, SharedGridWriter, SharedHeader, SharedLayout,
    SharedSdfView, SharedSdfWriter, SHARED_MAGIC,
};

// Re-export serialization constants
#[cfg(feature = "serde")]
pub use io::{compute_file_size, ASH_MAGIC, ASH_VERSION, HEADER_SIZE};

// Re-export ash_core types for convenience
pub use ash_core::{
    BlockCoord, CellCoord, CellValueProvider, InterpolationResult, LocalCoord, Point3,
    SdfInterpolationResult, SdfProvider, UNTRAINED_SENTINEL,
};

/// Prelude module for convenient imports.
///
/// ```ignore
/// use ash_rs::prelude::*;
/// ```
pub mod prelude {
    pub use crate::builder::GridBuilder;
    pub use crate::error::{AshError, Result};
    pub use crate::grid::SparseDenseGrid;
    pub use crate::simd::{BatchResult, BatchResultWithGradients};

    #[cfg(any(feature = "std", feature = "alloc"))]
    pub use crate::mesh::{MeshStats, Triangle};

    #[cfg(any(feature = "std", feature = "alloc"))]
    pub use crate::shared::{SharedGridView, SharedSdfView};

    pub use ash_core::{BlockCoord, CellCoord, Point3};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // Create a simple grid
        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block_constant(BlockCoord::new(0, 0, 0), 0.5)
            .build()
            .unwrap();

        // Query
        let result = grid.query(Point3::new(0.4, 0.4, 0.4));
        assert!(result.is_some());
        assert!((result.unwrap() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_cell_value_provider() {
        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block_constant(BlockCoord::new(0, 0, 0), 1.0)
            .build()
            .unwrap();

        // Test CellValueProvider methods
        assert_eq!(grid.grid_dim(), 8);
        assert!((grid.cell_size() - 0.1).abs() < 1e-6);
        assert!((grid.block_size() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_query() {
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        // Need 2x2x2 blocks to have proper corner values for interpolation
        let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);
        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    builder = builder.add_block_fn(BlockCoord::new(bx, by, bz), |pos| {
                        (pos - center).length() - radius
                    });
                }
            }
        }
        let grid = builder.build().unwrap();

        // Query with gradient at a point on the +X side of the sphere
        let result = grid.query_with_gradient(Point3::new(0.65, 0.4, 0.4));
        assert!(result.is_some(), "Query should succeed within allocated blocks");

        let (_, grad) = result.unwrap();
        // Gradient should point mostly in +X direction (away from center)
        assert!(grad[0] > 0.0, "Gradient X should be positive: {:?}", grad);
    }

    #[test]
    fn test_batch_query() {
        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(8)
            .add_block_constant(BlockCoord::new(0, 0, 0), 0.5)
            .build()
            .unwrap();

        let points = vec![
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.2, 0.2, 0.2),
            Point3::new(0.3, 0.3, 0.3),
        ];

        let results = grid.query_batch(&points);
        assert_eq!(results.len(), 3);
        assert!(results.num_valid() > 0);
    }

    #[test]
    fn test_collision_check() {
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(8)
            .add_block_fn(BlockCoord::new(0, 0, 0), |pos| (pos - center).length() - radius)
            .build()
            .unwrap();

        // Inside sphere
        assert!(grid.in_collision(center, 0.0));

        // Outside sphere
        let outside = Point3::new(1.0, 1.0, 1.0);
        assert!(!grid.in_collision(outside, 0.0));
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn test_mesh_extraction() {
        let center = Point3::new(0.4, 0.4, 0.4);
        let radius = 0.3;

        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(8)
            .add_block_fn(BlockCoord::new(0, 0, 0), |pos| (pos - center).length() - radius)
            .build()
            .unwrap();

        let triangles = grid.extract_block_mesh(BlockCoord::new(0, 0, 0), 0.0);

        // Should generate some triangles
        assert!(!triangles.is_empty());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serialization() {
        use std::io::Cursor;

        let grid = GridBuilder::new(8, 0.1)
            .with_capacity(10)
            .add_block_constant(BlockCoord::new(0, 0, 0), 0.5)
            .build()
            .unwrap();

        let mut buffer = Vec::new();
        grid.save(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

        assert_eq!(loaded.num_blocks(), grid.num_blocks());
    }
}
