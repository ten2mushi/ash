//! Data loading and preprocessing for training.

mod depth_camera;
mod point_cloud;
mod sensor;

pub use depth_camera::{
    generate_orbit_poses, generate_sphere_poses, DepthCameraSimulator, DepthImage, Pose,
};
pub use point_cloud::{PointCloud, PointCloudDataset};
pub use sensor::SensorDataBatch;
