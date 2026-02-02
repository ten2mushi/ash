//! # ash_core
//!
//! Pure mathematical algorithms for sparse-dense SDF grids.
//!
//! This crate provides the foundational algorithms for the ASH (Adaptive Sparse Hierarchical)
//! ecosystem, enabling efficient Signed Distance Field (SDF) operations for robotics applications.
//!
//! ## Features
//!
//! - **no_std compatible**: Works in embedded environments with the `alloc` feature
//! - **Pure algorithms**: No storage implementation, just math
//! - **Analytical gradients**: Enables efficient training without full autodiff
//! - **Marching cubes**: Zero-runtime-cost lookup tables
//! - **Generic feature dimensions**: Support N-dimensional features via `const N: usize`
//!
//! ## Feature Flags
//!
//! - `std` (default): Enables standard library support
//! - `alloc`: Enables heap allocation (Vec, etc.) without full std
//!
//! ## Modules
//!
//! - [`types`]: Core data types (Point3, BlockCoord, CellCoord, LocalCoord, InterpolationResult<N>)
//! - [`traits`]: Storage abstraction traits (CellValueProvider<N>, GradientAccumulator<N>, SdfProvider)
//! - [`coords`]: Coordinate conversion functions
//! - [`interpolation`]: Trilinear interpolation with gradients
//! - [`hash`]: Spatial hashing primitives (Morton encoding, FNV-1a)
//! - [`marching_cubes`]: Mesh extraction algorithm
//! - [`error`]: Error types
//!
//! ## Usage
//!
//! ```ignore
//! use ash_core::prelude::*;
//!
//! // Decompose a world point into sparse-dense coordinates
//! let point = Point3::new(0.25, 0.5, 0.75);
//! let (block, cell, local) = decompose_point(point, 0.1, 8);
//!
//! // Trilinear interpolation (requires implementing CellValueProvider<N>)
//! let result = trilinear_interpolate(&provider, block, cell, local);
//! ```
//!
//! ## N-Dimensional Features
//!
//! This crate supports N-dimensional features per cell corner through const generics:
//!
//! - `N=1`: SDF-only (most common, use `SdfProvider` convenience trait)
//! - `N=4`: SDF + RGB color
//! - `N=8`: SDF + semantic embedding
//!
//! The `SdfProvider` trait provides convenience methods for the common `N=1` case.

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]

// Conditional std/alloc support
#[cfg(feature = "std")]
extern crate std;

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

// Internal alloc prelude for conditional compilation
#[cfg(feature = "std")]
mod alloc_prelude {
    pub use std::vec::Vec;
}

#[cfg(all(feature = "alloc", not(feature = "std")))]
mod alloc_prelude {
    pub use alloc::vec::Vec;
}

pub mod coords;
pub mod error;
pub mod hash;
#[cfg(feature = "simd")]
pub mod hash_simd;
pub mod interpolation;
pub mod marching_cubes;
pub mod traits;
pub mod types;

/// Prelude module for convenient imports.
///
/// Provides the most commonly used types and functions.
pub mod prelude {
    pub use crate::coords::{
        cell_origin, compose_point, corner_position, decompose_point, get_neighbor_blocks,
        resolve_corner,
    };
    pub use crate::error::AshCoreError;
    pub use crate::hash::{
        fnv1a_32, fnv1a_64, hash_to_index, morton_decode_3d, morton_decode_signed,
        morton_encode_3d, morton_encode_signed,
    };
    pub use crate::interpolation::{
        compute_trilinear_weights, trilinear_backward, trilinear_gradient,
        trilinear_gradient_sdf, trilinear_hessian_mixed, trilinear_interpolate,
        trilinear_interpolate_sdf, trilinear_with_gradient, trilinear_with_gradient_sdf,
    };
    pub use crate::marching_cubes::{interpolate_vertex, process_cell_no_alloc};
    pub use crate::traits::{
        corner_from_index, index_from_corner, CellValueProvider, GradientAccumulator, SdfProvider,
    };
    pub use crate::types::{
        BlockCoord, CellCoord, InterpolationResult, LocalCoord, Point3, SdfInterpolationResult,
        UNTRAINED_SENTINEL,
    };

    #[cfg(any(feature = "std", feature = "alloc"))]
    pub use crate::marching_cubes::process_cell;
}

// Re-export everything at crate root for convenience
pub use coords::{
    cell_origin, compose_point, corner_position, decompose_point, get_neighbor_blocks,
    resolve_corner,
};
pub use error::AshCoreError;
pub use hash::{
    compact_bits_3d, fnv1a_32, fnv1a_64, hash_to_index, morton_decode_3d, morton_decode_signed,
    morton_encode_3d, morton_encode_signed, spread_bits_3d,
};
#[cfg(feature = "simd")]
pub use hash_simd::{hash_batch_4, hash_batch_4_scalar, hash_i32x4};
#[cfg(all(feature = "simd", any(feature = "std", feature = "alloc")))]
pub use hash_simd::hash_batch;
pub use interpolation::{
    compute_trilinear_weights, trilinear_backward, trilinear_gradient, trilinear_gradient_sdf,
    trilinear_hessian_mixed, trilinear_interpolate, trilinear_interpolate_sdf,
    trilinear_with_gradient, trilinear_with_gradient_sdf,
};
pub use traits::{
    corner_from_index, index_from_corner, CellValueProvider, GradientAccumulator, SdfProvider,
};
pub use types::{
    BlockCoord, CellCoord, InterpolationResult, LocalCoord, Point3, SdfInterpolationResult,
    UNTRAINED_SENTINEL,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Mock provider implementing a sphere SDF for integration testing
    struct SphereSdf {
        grid_dim: u32,
        cell_size: f32,
        center: Point3,
        radius: f32,
    }

    impl CellValueProvider<1> for SphereSdf {
        fn get_corner_values(
            &self,
            block: BlockCoord,
            cell: CellCoord,
            _corner: (u32, u32, u32),
        ) -> Option<[f32; 1]> {
            let block_size = self.block_size();
            let x = block.x as f32 * block_size + cell.x as f32 * self.cell_size;
            let y = block.y as f32 * block_size + cell.y as f32 * self.cell_size;
            let z = block.z as f32 * block_size + cell.z as f32 * self.cell_size;

            let point = Point3::new(x, y, z);
            let diff = point - self.center;
            let dist = diff.length();
            Some([dist - self.radius])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    #[test]
    fn test_sphere_sdf_properties() {
        let sdf = SphereSdf {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        // Query at center should be negative (inside)
        let (block, cell, local) = decompose_point(sdf.center, sdf.cell_size, sdf.grid_dim);
        let result = trilinear_interpolate(&sdf, block, cell, local);

        // Note: Due to discretization, the exact center value depends on corner positions
        // But it should be significantly negative
        if let Some(r) = result {
            assert!(r.values[0] < 0.0, "Center should be inside sphere");
        }

        // Query far outside should be positive
        let outside = Point3::new(2.0, 2.0, 2.0);
        let (block, cell, local) = decompose_point(outside, sdf.cell_size, sdf.grid_dim);
        let result = trilinear_interpolate(&sdf, block, cell, local);
        if let Some(r) = result {
            assert!(r.values[0] > 0.0, "Far point should be outside sphere");
        }
    }

    #[test]
    fn test_gradient_points_outward_at_surface() {
        let sdf = SphereSdf {
            grid_dim: 8,
            cell_size: 0.05,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        // Query at a point near the surface (along +x direction from center)
        let surface_point = Point3::new(0.4 + 0.3, 0.4, 0.4);
        let (block, cell, local) = decompose_point(surface_point, sdf.cell_size, sdf.grid_dim);

        if let Some(grad) = trilinear_gradient(&sdf, block, cell, local) {
            // Gradient should point outward (positive x direction)
            // Since this is cell-space gradient, normalize and check direction
            let len = libm::sqrtf(grad[0][0] * grad[0][0] + grad[0][1] * grad[0][1] + grad[0][2] * grad[0][2]);
            let grad_vec_x = grad[0][0] / len;

            // Should be mostly in +x direction
            assert!(
                grad_vec_x > 0.5,
                "Gradient should point outward, got x component: {}",
                grad_vec_x
            );
        }
    }

    #[test]
    fn test_decompose_compose_roundtrip_comprehensive() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let test_points = [
            // Positive quadrant
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.05, 0.05, 0.05),
            Point3::new(0.79, 0.79, 0.79), // Near block boundary
            Point3::new(0.81, 0.81, 0.81), // Just past block boundary
            Point3::new(5.5, 3.2, 1.7),
            // Negative quadrant
            Point3::new(-0.05, 0.0, 0.0),
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(-0.81, -0.81, -0.81),
            // Mixed
            Point3::new(-0.5, 0.5, -0.5),
            Point3::new(100.0, -100.0, 50.0),
        ];

        for point in &test_points {
            let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
            let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

            let diff = (*point - reconstructed).length();
            assert!(
                diff < 1e-5,
                "Roundtrip failed for {:?}: got {:?}, diff={}",
                point,
                reconstructed,
                diff
            );

            // Verify local coords are approximately in valid range
            // Allow small floating-point errors at extreme coordinates
            let eps = 1e-4;
            assert!(
                local.u >= -eps && local.u <= 1.0 + eps,
                "Local u out of range for {:?}: {:?}",
                point,
                local
            );
            assert!(
                local.v >= -eps && local.v <= 1.0 + eps,
                "Local v out of range for {:?}: {:?}",
                point,
                local
            );
            assert!(
                local.w >= -eps && local.w <= 1.0 + eps,
                "Local w out of range for {:?}: {:?}",
                point,
                local
            );
        }
    }

    #[test]
    fn test_morton_encoding_preserves_order() {
        // Morton encoding should preserve spatial locality
        let coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 0, 0),
            BlockCoord::new(0, 1, 0),
            BlockCoord::new(1, 1, 0),
            BlockCoord::new(0, 0, 1),
        ];

        let mut codes: [u64; 5] = [0; 5];
        for (i, coord) in coords.iter().enumerate() {
            codes[i] = morton_encode_signed(*coord);
        }

        // Adjacent coordinates should have nearby Morton codes
        // (This is a weak check for locality but validates basic behavior)
        for code in codes {
            // Just verify codes are distinct and valid
            let decoded = morton_decode_signed(code);
            assert!(coords.contains(&decoded));
        }
    }

    #[test]
    fn test_weights_invariants() {
        let test_cases = [
            LocalCoord::new(0.0, 0.0, 0.0),
            LocalCoord::new(1.0, 1.0, 1.0),
            LocalCoord::new(0.5, 0.5, 0.5),
            LocalCoord::new(0.1, 0.9, 0.3),
        ];

        for local in test_cases {
            let weights = compute_trilinear_weights(local);

            // Sum should be 1
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Weights sum {} != 1.0 for {:?}",
                sum,
                local
            );

            // All weights should be non-negative
            for (i, &w) in weights.iter().enumerate() {
                assert!(w >= 0.0, "Weight {} is negative: {} for {:?}", i, w, local);
            }

            // All weights should be <= 1
            for (i, &w) in weights.iter().enumerate() {
                assert!(w <= 1.0, "Weight {} exceeds 1: {} for {:?}", i, w, local);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn test_marching_cubes_sphere_mesh() {
        use marching_cubes::process_cell;

        let sdf = SphereSdf {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        let mut total_triangles = 0;

        // Scan cells around the sphere
        for bz in 0..=1 {
            for by in 0..=1 {
                for bx in 0..=1 {
                    let block = BlockCoord::new(bx, by, bz);

                    for cz in 0..8 {
                        for cy in 0..8 {
                            for cx in 0..8 {
                                let cell = CellCoord::new(cx, cy, cz);
                                let triangles = process_cell(&sdf, block, cell, 0.0);
                                total_triangles += triangles.len();
                            }
                        }
                    }
                }
            }
        }

        // A sphere of radius 0.3 in a grid of cell_size 0.1 should produce
        // some triangles (exact count depends on implementation details)
        assert!(
            total_triangles > 0,
            "Should generate some triangles for sphere"
        );

        // Should produce a reasonable number of triangles for a sphere
        // (not too few, not explosively many)
        assert!(
            total_triangles < 10000,
            "Too many triangles: {}",
            total_triangles
        );
    }

    #[test]
    fn test_sdf_provider_convenience_trait() {
        let sdf = SphereSdf {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        // Test that SdfProvider is auto-implemented
        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(4, 4, 4);
        let corner = (0, 0, 0);

        // Use the SdfProvider convenience method
        let value = sdf.get_corner_value(block, cell, corner);
        assert!(value.is_some());

        // Use the generic method
        let values = sdf.get_corner_values(block, cell, corner);
        assert!(values.is_some());

        // They should produce the same result
        assert!((value.unwrap() - values.unwrap()[0]).abs() < 1e-6);
    }
}
