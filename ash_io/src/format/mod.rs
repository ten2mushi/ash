//! ASH file format support.
//!
//! This module provides reading and writing of the .ash binary format
//! for sparse-dense SDF grids with N features.
//!
//! # Format Overview
//!
//! The .ash format stores grids in a compact binary format optimized for
//! fast loading and cross-platform compatibility. Features are stored in
//! SoA (Structure of Arrays) layout for optimal cache performance.
//!
//! # Example
//!
//! ```ignore
//! use ash_io::{GridBuilder, InMemoryGrid, save_grid, load_grid};
//! use std::fs::File;
//!
//! // Save a grid
//! let grid: InMemoryGrid<1> = GridBuilder::new(8, 0.1)
//!     .add_block_fn(BlockCoord::new(0, 0, 0), |p| [p.length()])
//!     .build()?;
//!
//! let mut file = File::create("scene.ash")?;
//! save_grid(&grid, &mut file)?;
//!
//! // Load a grid
//! let mut file = File::open("scene.ash")?;
//! let loaded: InMemoryGrid<1> = load_grid(&mut file)?;
//! ```

pub mod ash;
pub mod header;

pub use ash::{load_grid, save_grid};
#[cfg(feature = "std")]
pub use ash::{load_from_file, save_to_file};
pub use header::{compute_file_size, AshHeader, ASH_MAGIC, HEADER_SIZE};
