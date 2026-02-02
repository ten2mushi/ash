//! Error types for ash_core operations.
//!
//! Provides a simple error enum with no external dependencies for no_std compatibility.

use core::fmt;

/// Error types that can occur during ash_core operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AshCoreError {
    /// A corner value was not available from the provider.
    MissingCornerValue {
        /// The corner index (0-7) that was missing.
        corner_index: u8,
    },
    /// The cell coordinates are out of bounds for the grid dimension.
    CellOutOfBounds {
        /// The coordinate component that was out of bounds.
        coord: u32,
        /// The maximum valid value (grid_dim - 1).
        max: u32,
    },
    /// The local coordinates are outside the valid [0, 1] range.
    LocalCoordOutOfRange {
        /// The coordinate value that was out of range.
        value: f32,
    },
    /// A sentinel value (untrained region) was encountered during interpolation.
    UntrainedRegion,
    /// The capacity provided for a no-alloc operation was insufficient.
    InsufficientCapacity {
        /// The capacity that was required.
        required: usize,
        /// The capacity that was provided.
        provided: usize,
    },
}

impl fmt::Display for AshCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AshCoreError::MissingCornerValue { corner_index } => {
                write!(f, "missing corner value at index {}", corner_index)
            }
            AshCoreError::CellOutOfBounds { coord, max } => {
                write!(f, "cell coordinate {} exceeds maximum {}", coord, max)
            }
            AshCoreError::LocalCoordOutOfRange { value } => {
                write!(f, "local coordinate {} is outside [0, 1] range", value)
            }
            AshCoreError::UntrainedRegion => {
                write!(f, "encountered untrained region sentinel value")
            }
            AshCoreError::InsufficientCapacity { required, provided } => {
                write!(
                    f,
                    "insufficient capacity: required {}, provided {}",
                    required, provided
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AshCoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_error_display() {
        use std::format;

        let err = AshCoreError::MissingCornerValue { corner_index: 3 };
        assert_eq!(format!("{}", err), "missing corner value at index 3");

        let err = AshCoreError::CellOutOfBounds { coord: 10, max: 7 };
        assert_eq!(format!("{}", err), "cell coordinate 10 exceeds maximum 7");

        let err = AshCoreError::LocalCoordOutOfRange { value: 1.5 };
        assert_eq!(
            format!("{}", err),
            "local coordinate 1.5 is outside [0, 1] range"
        );

        let err = AshCoreError::UntrainedRegion;
        assert_eq!(
            format!("{}", err),
            "encountered untrained region sentinel value"
        );

        let err = AshCoreError::InsufficientCapacity {
            required: 10,
            provided: 5,
        };
        assert_eq!(
            format!("{}", err),
            "insufficient capacity: required 10, provided 5"
        );
    }

    #[test]
    fn test_error_equality() {
        let err1 = AshCoreError::MissingCornerValue { corner_index: 3 };
        let err2 = AshCoreError::MissingCornerValue { corner_index: 3 };
        let err3 = AshCoreError::MissingCornerValue { corner_index: 4 };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
