//! Decoder modules for mapping latent codes to SDF and semantic values.

mod sdf;
mod semantic;

pub use sdf::SdfDecoder;
pub use semantic::SemanticDecoder;
