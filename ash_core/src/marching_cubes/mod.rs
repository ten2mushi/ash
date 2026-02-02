//! Marching Cubes mesh extraction.
//!
//! This module provides the marching cubes algorithm for extracting triangle meshes
//! from SDF data. It includes:
//!
//! - Compile-time lookup tables for cube configurations
//! - Functions for processing individual cells
//! - Both allocating and no-alloc variants for flexibility
//!
//! # Example
//!
//! ```ignore
//! use ash_core::marching_cubes::{process_cell, process_cell_no_alloc};
//! use ash_core::types::{BlockCoord, CellCoord};
//!
//! // Process a cell with allocation
//! let triangles = process_cell(&provider, block, cell, 0.0);
//!
//! // Process a cell without allocation (for embedded/no_std)
//! let (triangles, count) = process_cell_no_alloc(&provider, block, cell, 0.0);
//! ```

mod algorithm;
mod tables;

pub use algorithm::{interpolate_vertex, process_cell_no_alloc};
pub use tables::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};

#[cfg(any(feature = "std", feature = "alloc"))]
pub use algorithm::process_cell;
