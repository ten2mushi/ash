//! Export functionality for trained grids.
//!
//! Provides conversion from differentiable grids to ash_io formats
//! for use with ash_rs runtime.

mod ash_file;
mod discretize;

pub use ash_file::export_to_file;
pub use discretize::discretize_grid;
