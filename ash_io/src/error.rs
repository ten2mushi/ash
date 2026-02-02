//! Error types for ash_io operations.
//!
//! Provides specific error variants for grid construction, serialization, and query failures.

use core::fmt;

/// Errors that can occur during ash_io operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AshIoError {
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

    /// Feature dimension mismatch.
    FeatureDimensionMismatch {
        /// Expected feature dimension.
        expected: usize,
        /// Actual feature dimension found.
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

    /// Shared memory error.
    SharedMemory {
        /// Description of the error.
        message: &'static str,
    },
}

impl fmt::Display for AshIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AshIoError::ZeroCapacity => write!(f, "grid capacity cannot be zero"),
            AshIoError::CapacityExceeded { blocks, capacity } => {
                write!(
                    f,
                    "block count {} exceeds capacity {}",
                    blocks, capacity
                )
            }
            AshIoError::InvalidBlockSize { expected, got } => {
                write!(
                    f,
                    "invalid block size: expected {} values, got {}",
                    expected, got
                )
            }
            AshIoError::FeatureDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "feature dimension mismatch: expected {}, got {}",
                    expected, got
                )
            }
            AshIoError::DuplicateBlock { x, y, z } => {
                write!(f, "duplicate block coordinate: ({}, {}, {})", x, y, z)
            }
            AshIoError::HashTableFull => write!(f, "hash table is full"),
            AshIoError::InvalidFormat { message } => {
                write!(f, "invalid file format: {}", message)
            }
            #[cfg(feature = "std")]
            AshIoError::Io(msg) => write!(f, "I/O error: {}", msg),
            AshIoError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "grid dimension mismatch: expected {}, got {}",
                    expected, got
                )
            }
            AshIoError::SharedMemory { message } => {
                write!(f, "shared memory error: {}", message)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AshIoError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for AshIoError {
    fn from(err: std::io::Error) -> Self {
        AshIoError::Io(err.to_string())
    }
}

/// Result type alias for ash_io operations.
pub type Result<T> = core::result::Result<T, AshIoError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AshIoError::ZeroCapacity;
        assert_eq!(format!("{}", err), "grid capacity cannot be zero");

        let err = AshIoError::CapacityExceeded {
            blocks: 100,
            capacity: 50,
        };
        assert!(format!("{}", err).contains("100"));
        assert!(format!("{}", err).contains("50"));
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(AshIoError::ZeroCapacity, AshIoError::ZeroCapacity);
        assert_ne!(
            AshIoError::ZeroCapacity,
            AshIoError::CapacityExceeded {
                blocks: 1,
                capacity: 1
            }
        );
    }
}
