//! Spatial acceleration structures for geometry queries.
//!
//! This module provides:
//! - `bvh`: Bounding Volume Hierarchy for O(log n) triangle queries
//! - `narrow_band`: Surface-aware block allocation for mesh import

pub mod bvh;
pub mod narrow_band;

pub use bvh::{Aabb, TriangleBvh};
pub use narrow_band::{compute_narrow_band, narrow_band_from_triangles, NarrowBandConfig};
