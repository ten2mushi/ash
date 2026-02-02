//! Encoder modules for transforming input data to latent representations.

mod fourier;
mod pointnet;

pub use fourier::FourierEncoder;
pub use pointnet::PointNetEncoder;
