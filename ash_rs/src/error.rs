//! Error types for ash_rs operations.
//!
//! Provides specific error variants for grid construction, serialization, and query failures.

use core::fmt;

/// Errors that can occur during ash_rs operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AshError {
    /// Attempted to create a grid with zero capacity.
    ZeroCapacity,

    /// Block count exceeds the configured capacity.
    CapacityExceeded {
        /// Number of blocks attempted to store.
        blocks: usize,
        /// Maximum capacity of the grid.
        capacity: usize,
    },

    /// Block data has incorrect size.
    InvalidBlockSize {
        /// Expected number of values.
        expected: usize,
        /// Actual number of values provided.
        got: usize,
    },

    /// Attempted to insert a duplicate block coordinate.
    DuplicateBlock {
        /// The duplicate block coordinate.
        x: i32,
        /// Y coordinate.
        y: i32,
        /// Z coordinate.
        z: i32,
    },

    /// Hash table is full (should not happen with proper capacity planning).
    HashTableFull,

    /// Invalid file format during deserialization.
    InvalidFormat {
        /// Description of the format error.
        message: &'static str,
    },

    /// I/O error during serialization/deserialization.
    #[cfg(feature = "std")]
    Io(String),

    /// Grid dimension mismatch (e.g., when loading a file).
    DimensionMismatch {
        /// Expected grid dimension.
        expected: u32,
        /// Actual grid dimension found.
        got: u32,
    },
}

impl fmt::Display for AshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AshError::ZeroCapacity => write!(f, "grid capacity cannot be zero"),
            AshError::CapacityExceeded { blocks, capacity } => {
                write!(
                    f,
                    "block count {} exceeds capacity {}",
                    blocks, capacity
                )
            }
            AshError::InvalidBlockSize { expected, got } => {
                write!(
                    f,
                    "invalid block size: expected {} values, got {}",
                    expected, got
                )
            }
            AshError::DuplicateBlock { x, y, z } => {
                write!(f, "duplicate block coordinate: ({}, {}, {})", x, y, z)
            }
            AshError::HashTableFull => write!(f, "hash table is full"),
            AshError::InvalidFormat { message } => {
                write!(f, "invalid file format: {}", message)
            }
            #[cfg(feature = "std")]
            AshError::Io(msg) => write!(f, "I/O error: {}", msg),
            AshError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "grid dimension mismatch: expected {}, got {}",
                    expected, got
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AshError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for AshError {
    fn from(err: std::io::Error) -> Self {
        AshError::Io(err.to_string())
    }
}

/// Result type alias for ash_rs operations.
pub type Result<T> = core::result::Result<T, AshError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AshError::ZeroCapacity;
        assert_eq!(format!("{}", err), "grid capacity cannot be zero");

        let err = AshError::CapacityExceeded {
            blocks: 100,
            capacity: 50,
        };
        assert!(format!("{}", err).contains("100"));
        assert!(format!("{}", err).contains("50"));
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(AshError::ZeroCapacity, AshError::ZeroCapacity);
        assert_ne!(
            AshError::ZeroCapacity,
            AshError::CapacityExceeded {
                blocks: 1,
                capacity: 1
            }
        );
    }
}
