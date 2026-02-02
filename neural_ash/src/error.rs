//! Error types for neural_ash.

use thiserror::Error;

/// Errors that can occur during neural SDF operations.
#[derive(Error, Debug)]
pub enum NeuralAshError {
    /// Grid capacity exceeded.
    #[error("grid capacity exceeded: tried to allocate block {block_count} but capacity is {capacity}")]
    CapacityExceeded {
        /// Number of blocks currently allocated.
        block_count: usize,
        /// Maximum capacity.
        capacity: usize,
    },

    /// Block already exists at the given coordinate.
    #[error("block already exists at ({x}, {y}, {z})")]
    BlockExists {
        /// X coordinate.
        x: i32,
        /// Y coordinate.
        y: i32,
        /// Z coordinate.
        z: i32,
    },

    /// Block not found at the given coordinate.
    #[error("block not found at ({x}, {y}, {z})")]
    BlockNotFound {
        /// X coordinate.
        x: i32,
        /// Y coordinate.
        y: i32,
        /// Z coordinate.
        z: i32,
    },

    /// Invalid configuration.
    #[error("invalid configuration: {message}")]
    InvalidConfig {
        /// Description of the configuration error.
        message: String,
    },

    /// Tensor shape mismatch.
    #[error("tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Training error.
    #[error("training error: {message}")]
    TrainingError {
        /// Description of the error.
        message: String,
    },

    /// Export error.
    #[error("export error: {message}")]
    ExportError {
        /// Description of the error.
        message: String,
    },

    /// I/O error from ash_io.
    #[error("I/O error: {0}")]
    IoError(#[from] ash_io::AshIoError),

    /// Feature index out of bounds.
    #[error("feature index {index} out of bounds for {feature_dim} features")]
    FeatureIndexOutOfBounds {
        /// The requested feature index.
        index: usize,
        /// Total number of features.
        feature_dim: usize,
    },

    /// Device mismatch between tensors.
    #[error("device mismatch: tensors must be on the same device")]
    DeviceMismatch,

    /// Grid not initialized.
    #[error("grid not initialized: call spatial_init() first")]
    GridNotInitialized,

    /// Online mode error.
    #[error("online mode error: {message}")]
    OnlineError {
        /// Description of the error.
        message: String,
    },

    /// Invalid or corrupted data.
    #[error("invalid data: {0}")]
    InvalidData(String),
}

/// Result type for neural_ash operations.
pub type Result<T> = std::result::Result<T, NeuralAshError>;
