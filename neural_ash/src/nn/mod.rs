//! Neural network modules for SDF learning.
//!
//! This module provides:
//! - Encoders: Transform input data (point clouds, images) into latent representations
//! - Decoders: Map latent codes + coordinates to SDF and semantic values
//! - MLP building blocks for constructing custom architectures

pub mod decoder;
pub mod encoder;
pub mod mlp;

pub use decoder::{SdfDecoder, SemanticDecoder};
pub use encoder::{FourierEncoder, PointNetEncoder};
pub use mlp::{Mlp, MlpConfig};
