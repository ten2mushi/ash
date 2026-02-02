//! Simulated depth camera for realistic sensor initialization.
//!
//! Provides a depth camera simulator that generates depth images from a virtual scene,
//! which can then be fused into a TSDF-initialized grid. This mimics the workflow of
//! real robotics applications where sensor data drives reconstruction.

use std::f32::consts::PI;
use std::sync::Arc;

use ash_core::{decompose_point, Point3};
use ash_io::BlockMap;
use burn::prelude::*;

use crate::grid::DiffSdfGrid;

/// Camera pose in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    /// Camera position in world coordinates.
    pub position: Point3,
    /// Forward direction (normalized).
    pub forward: Point3,
    /// Up direction (normalized).
    pub up: Point3,
    /// Right direction (computed from forward x up).
    pub right: Point3,
}

impl Pose {
    /// Create a new pose from position and look-at target.
    pub fn look_at(position: Point3, target: Point3, up: Point3) -> Self {
        let forward = (target - position).normalize();
        let right = forward.cross(up).normalize();
        let up = right.cross(forward).normalize();

        Self {
            position,
            forward,
            up,
            right,
        }
    }

    /// Create a pose looking along the Z axis from the given position.
    pub fn looking_forward(position: Point3) -> Self {
        Self::look_at(
            position,
            position + Point3::new(0.0, 0.0, -1.0),
            Point3::new(0.0, 1.0, 0.0),
        )
    }

    /// Transform a direction from camera space to world space.
    pub fn transform_direction(&self, dir: Point3) -> Point3 {
        self.right * dir.x + self.up * dir.y + self.forward * dir.z
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self::looking_forward(Point3::new(0.0, 0.0, 1.0))
    }
}

/// A depth image with per-pixel depth values.
#[derive(Debug, Clone)]
pub struct DepthImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth values (row-major, [y * width + x]).
    /// Invalid pixels have depth = 0 or > max_depth.
    pub depths: Vec<f32>,
}

impl DepthImage {
    /// Create a new depth image.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depths: vec![0.0; (width * height) as usize],
        }
    }

    /// Get depth at pixel (x, y).
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            self.depths[(y * self.width + x) as usize]
        } else {
            0.0
        }
    }

    /// Set depth at pixel (x, y).
    pub fn set(&mut self, x: u32, y: u32, depth: f32) {
        if x < self.width && y < self.height {
            self.depths[(y * self.width + x) as usize] = depth;
        }
    }

    /// Get the number of valid depth pixels.
    pub fn valid_count(&self) -> usize {
        self.depths.iter().filter(|&&d| d > 0.0).count()
    }
}

/// Simulated depth camera for generating synthetic depth data.
pub struct DepthCameraSimulator {
    /// Image width in pixels.
    pub resolution_x: u32,
    /// Image height in pixels.
    pub resolution_y: u32,
    /// Horizontal field of view in radians.
    pub fov_h: f32,
    /// Vertical field of view (computed from aspect ratio if not set).
    pub fov_v: f32,
    /// Minimum depth (near plane).
    pub min_depth: f32,
    /// Maximum depth (far plane).
    pub max_depth: f32,
    /// Depth noise standard deviation.
    pub noise_sigma: f32,
    /// Random seed for noise generation.
    seed: u64,
}

impl DepthCameraSimulator {
    /// Create a new depth camera simulator.
    ///
    /// # Arguments
    /// * `resolution_x` - Image width in pixels
    /// * `resolution_y` - Image height in pixels
    /// * `fov_h` - Horizontal field of view in radians
    pub fn new(resolution_x: u32, resolution_y: u32, fov_h: f32) -> Self {
        let aspect = resolution_x as f32 / resolution_y as f32;
        let fov_v = 2.0 * ((fov_h / 2.0).tan() / aspect).atan();

        Self {
            resolution_x,
            resolution_y,
            fov_h,
            fov_v,
            min_depth: 0.1,
            max_depth: 10.0,
            noise_sigma: 0.002,
            seed: 42,
        }
    }

    /// Set the depth range.
    pub fn with_depth_range(mut self, min: f32, max: f32) -> Self {
        self.min_depth = min;
        self.max_depth = max;
        self
    }

    /// Set the noise level.
    pub fn with_noise(mut self, sigma: f32) -> Self {
        self.noise_sigma = sigma;
        self
    }

    /// Generate a random number using a simple LCG.
    fn rand(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.seed >> 33) as f32) / (1u64 << 31) as f32
    }

    /// Generate Gaussian noise using Box-Muller transform.
    fn gaussian_noise(&mut self) -> f32 {
        let u1 = self.rand().max(1e-10);
        let u2 = self.rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Compute ray direction for a pixel.
    fn pixel_to_ray(&self, x: u32, y: u32) -> Point3 {
        // Normalized device coordinates [-1, 1]
        let ndc_x = (2.0 * x as f32 + 1.0) / self.resolution_x as f32 - 1.0;
        let ndc_y = 1.0 - (2.0 * y as f32 + 1.0) / self.resolution_y as f32;

        // Camera space direction
        let tan_h = (self.fov_h / 2.0).tan();
        let tan_v = (self.fov_v / 2.0).tan();

        Point3::new(ndc_x * tan_h, ndc_y * tan_v, 1.0).normalize()
    }

    /// Render a depth image from a viewpoint using a signed distance function.
    ///
    /// # Arguments
    /// * `pose` - Camera pose
    /// * `sdf_query` - Function that returns signed distance at a point
    ///
    /// # Returns
    /// A depth image with per-pixel depth values.
    pub fn render_depth<F>(&mut self, pose: &Pose, sdf_query: F) -> DepthImage
    where
        F: Fn(Point3) -> Option<f32>,
    {
        let mut image = DepthImage::new(self.resolution_x, self.resolution_y);

        for y in 0..self.resolution_y {
            for x in 0..self.resolution_x {
                let ray_dir_camera = self.pixel_to_ray(x, y);
                let ray_dir_world = pose.transform_direction(ray_dir_camera);

                // Sphere trace to find surface
                if let Some(depth) = self.sphere_trace(pose.position, ray_dir_world, &sdf_query) {
                    // Add noise
                    let noisy_depth = depth + self.gaussian_noise() * self.noise_sigma * depth;
                    if noisy_depth >= self.min_depth && noisy_depth <= self.max_depth {
                        image.set(x, y, noisy_depth);
                    }
                }
            }
        }

        image
    }

    /// Sphere trace along a ray to find the surface.
    fn sphere_trace<F>(&self, origin: Point3, direction: Point3, sdf_query: &F) -> Option<f32>
    where
        F: Fn(Point3) -> Option<f32>,
    {
        let mut t = self.min_depth;
        let max_steps = 128;
        let hit_threshold = 0.0001;

        for _ in 0..max_steps {
            if t > self.max_depth {
                return None;
            }

            let p = origin + direction * t;
            let dist = sdf_query(p)?;

            if dist.abs() < hit_threshold {
                return Some(t);
            }

            // Step by the distance to nearest surface
            t += dist.max(hit_threshold);
        }

        None
    }

    /// Convert a depth image to a point cloud with normals.
    ///
    /// # Arguments
    /// * `depth` - The depth image
    /// * `pose` - Camera pose used to capture the image
    ///
    /// # Returns
    /// (points, normals) - Vectors of 3D points and their estimated normals.
    pub fn depth_to_points(&self, depth: &DepthImage, pose: &Pose) -> (Vec<Point3>, Vec<Point3>) {
        let mut points = Vec::new();
        let mut normals = Vec::new();

        for y in 0..depth.height {
            for x in 0..depth.width {
                let d = depth.get(x, y);
                if d <= 0.0 || d >= self.max_depth {
                    continue;
                }

                // Compute 3D point
                let ray_dir_camera = self.pixel_to_ray(x, y);
                let ray_dir_world = pose.transform_direction(ray_dir_camera);
                let point = pose.position + ray_dir_world * d;
                points.push(point);

                // Estimate normal from depth gradient
                let normal = self.estimate_normal(depth, pose, x, y);
                normals.push(normal);
            }
        }

        (points, normals)
    }

    /// Estimate surface normal from depth gradient.
    fn estimate_normal(&self, depth: &DepthImage, pose: &Pose, x: u32, y: u32) -> Point3 {
        let d = depth.get(x, y);
        if d <= 0.0 {
            return -pose.forward; // Default to facing camera
        }

        // Compute neighbors
        let dx = if x > 0 && x < depth.width - 1 {
            let d_left = depth.get(x - 1, y);
            let d_right = depth.get(x + 1, y);
            if d_left > 0.0 && d_right > 0.0 {
                (d_right - d_left) / 2.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        let dy = if y > 0 && y < depth.height - 1 {
            let d_up = depth.get(x, y - 1);
            let d_down = depth.get(x, y + 1);
            if d_up > 0.0 && d_down > 0.0 {
                (d_down - d_up) / 2.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Convert gradient to normal
        let _ray = self.pixel_to_ray(x, y);
        let tan_h = (self.fov_h / 2.0).tan();
        let tan_v = (self.fov_v / 2.0).tan();

        let dpdx = pose.right * (d * 2.0 * tan_h / self.resolution_x as f32) + pose.forward * dx;
        let dpdy = pose.up * (d * 2.0 * tan_v / self.resolution_y as f32) + pose.forward * dy;

        // Normal is cross product of tangent vectors
        let normal = dpdx.cross(dpdy).normalize();

        // Flip if pointing away from camera
        if normal.dot(-pose.forward) < 0.0 {
            -normal
        } else {
            normal
        }
    }

    /// Fuse multiple depth frames into a grid using TSDF.
    ///
    /// # Arguments
    /// * `frames` - Vector of (pose, depth_image) pairs
    /// * `grid` - The differentiable grid to initialize
    /// * `truncation` - TSDF truncation distance
    pub fn fuse_to_grid<B: Backend>(
        &self,
        frames: &[(Pose, DepthImage)],
        grid: &mut DiffSdfGrid<B, 1>,
        truncation: f32,
    ) {
        let device = grid.device();
        let cell_size = grid.config().cell_size;
        let _grid_dim = grid.config().grid_dim;
        let cells_per_block = grid.config().cells_per_block();

        // First pass: allocate blocks that contain observed points
        for (pose, depth) in frames {
            let (points, _) = self.depth_to_points(depth, pose);
            grid.spatial_init(&points, cell_size * 2.0);
        }

        if grid.num_blocks() == 0 {
            return;
        }

        // Use capacity for total cells to match the pre-allocated embeddings tensor
        // The grid's embeddings are always sized for capacity, not just allocated blocks
        let capacity = grid.config().capacity;
        let total_cells = capacity * cells_per_block;
        let allocated_cells = grid.num_blocks() * cells_per_block;

        // Create TSDF accumulator for allocated blocks only (for efficiency)
        let mut tsdf_sum = vec![0.0f32; allocated_cells];
        let mut weight_sum = vec![0.0f32; allocated_cells];

        // Second pass: fuse depth measurements
        for (pose, depth) in frames {
            self.fuse_frame(&mut tsdf_sum, &mut weight_sum, pose, depth, grid, truncation);
        }

        // Convert accumulated TSDF to full capacity-sized embeddings
        // Unallocated blocks remain at init_value
        let mut embeddings = vec![grid.config().init_value; total_cells];
        for i in 0..allocated_cells {
            if weight_sum[i] > 0.0 {
                embeddings[i] = tsdf_sum[i] / weight_sum[i];
            }
        }

        // Update grid embeddings (capacity-sized tensor)
        let new_embeddings =
            Tensor::from_data(TensorData::new(embeddings, [total_cells, 1]), &device);
        grid.set_embeddings(new_embeddings);
    }

    /// Fuse a single depth frame into the TSDF accumulator.
    fn fuse_frame<B: Backend>(
        &self,
        tsdf_sum: &mut [f32],
        weight_sum: &mut [f32],
        pose: &Pose,
        depth: &DepthImage,
        grid: &DiffSdfGrid<B, 1>,
        truncation: f32,
    ) {
        let cell_size = grid.config().cell_size;
        let grid_dim = grid.config().grid_dim;
        let cells_per_block = grid.config().cells_per_block();
        let block_map = grid.block_map();

        // For each pixel with valid depth
        for y in 0..depth.height {
            for x in 0..depth.width {
                let d = depth.get(x, y);
                if d <= 0.0 || d >= self.max_depth {
                    continue;
                }

                // Compute 3D point on surface
                let ray_dir_camera = self.pixel_to_ray(x, y);
                let ray_dir_world = pose.transform_direction(ray_dir_camera);
                let _surface_point = pose.position + ray_dir_world * d;

                // Update cells along the ray near the surface
                self.update_cells_along_ray(
                    tsdf_sum,
                    weight_sum,
                    pose.position,
                    ray_dir_world,
                    d,
                    truncation,
                    cell_size,
                    grid_dim,
                    cells_per_block,
                    block_map,
                );
            }
        }
    }

    /// Update cells along a ray for TSDF fusion.
    fn update_cells_along_ray(
        &self,
        tsdf_sum: &mut [f32],
        weight_sum: &mut [f32],
        origin: Point3,
        direction: Point3,
        depth: f32,
        truncation: f32,
        cell_size: f32,
        grid_dim: u32,
        cells_per_block: usize,
        block_map: &Arc<BlockMap>,
    ) {
        // Sample along ray from (depth - truncation) to (depth + truncation)
        let start_t = (depth - truncation).max(self.min_depth);
        let end_t = (depth + truncation).min(self.max_depth);
        let step = cell_size * 0.5;

        let mut t = start_t;
        while t <= end_t {
            let point = origin + direction * t;
            let (block, cell, _) = decompose_point(point, cell_size, grid_dim);

            if let Some(block_idx) = block_map.get(block) {
                let cell_idx = cell.flat_index(grid_dim);
                let flat_idx = block_idx * cells_per_block + cell_idx;

                if flat_idx < tsdf_sum.len() {
                    // Compute signed distance to surface
                    let sdf = depth - t;

                    // Truncate
                    let truncated_sdf = sdf.clamp(-truncation, truncation);

                    // Weight by viewing angle and distance
                    let cos_angle = direction.dot(-direction).abs().max(0.1); // Simplified
                    let weight = cos_angle / (t * t + 1.0);

                    // Accumulate
                    tsdf_sum[flat_idx] += truncated_sdf * weight;
                    weight_sum[flat_idx] += weight;
                }
            }

            t += step;
        }
    }
}

/// Generate orbit camera poses around a center point.
///
/// # Arguments
/// * `num_views` - Number of camera positions
/// * `radius` - Distance from center
/// * `center` - Point to orbit around
/// * `height` - Height above center
pub fn generate_orbit_poses(
    num_views: usize,
    radius: f32,
    center: Point3,
    height: f32,
) -> Vec<Pose> {
    let mut poses = Vec::with_capacity(num_views);

    for i in 0..num_views {
        let angle = 2.0 * PI * i as f32 / num_views as f32;
        let position = Point3::new(
            center.x + radius * angle.cos(),
            center.y + height,
            center.z + radius * angle.sin(),
        );

        poses.push(Pose::look_at(position, center, Point3::new(0.0, 1.0, 0.0)));
    }

    poses
}

/// Generate camera poses uniformly distributed on a sphere using Fibonacci spiral.
///
/// This provides much better coverage than a simple orbit, observing the target
/// from all directions including top and bottom.
///
/// # Arguments
/// * `num_views` - Number of camera positions (more = better coverage)
/// * `radius` - Distance from center to each camera
/// * `center` - Point to look at (target center)
pub fn generate_sphere_poses(num_views: usize, radius: f32, center: Point3) -> Vec<Pose> {
    let mut poses = Vec::with_capacity(num_views);

    // Golden ratio for Fibonacci sphere
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let golden_angle = 2.0 * PI / (golden_ratio * golden_ratio);

    for i in 0..num_views {
        // Fibonacci sphere sampling
        let y = 1.0 - (i as f32 / (num_views - 1) as f32) * 2.0; // -1 to 1
        let radius_at_y = (1.0 - y * y).sqrt();
        let theta = golden_angle * i as f32;

        let x = radius_at_y * theta.cos();
        let z = radius_at_y * theta.sin();

        let position = Point3::new(
            center.x + x * radius,
            center.y + y * radius,
            center.z + z * radius,
        );

        // Use a stable up vector - switch when looking straight up/down
        let up = if y.abs() > 0.99 {
            Point3::new(1.0, 0.0, 0.0) // Use X as up when looking along Y
        } else {
            Point3::new(0.0, 1.0, 0.0) // Normal up vector
        };

        poses.push(Pose::look_at(position, center, up));
    }

    poses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose_creation() {
        let pose = Pose::look_at(
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );

        // Forward should point toward origin
        assert!((pose.forward.z - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_depth_image() {
        let mut image = DepthImage::new(10, 10);
        image.set(5, 5, 1.5);

        assert!((image.get(5, 5) - 1.5).abs() < 1e-6);
        assert_eq!(image.valid_count(), 1);
    }

    #[test]
    fn test_camera_simulator_creation() {
        let camera = DepthCameraSimulator::new(640, 480, 60.0_f32.to_radians());

        assert_eq!(camera.resolution_x, 640);
        assert_eq!(camera.resolution_y, 480);
    }

    #[test]
    fn test_render_sphere() {
        let mut camera = DepthCameraSimulator::new(64, 64, 60.0_f32.to_radians())
            .with_depth_range(0.1, 5.0)
            .with_noise(0.0);

        // Sphere SDF centered at origin
        let sphere_sdf = |p: Point3| -> Option<f32> {
            let dist = p.length() - 0.3;
            Some(dist)
        };

        // Camera looking at sphere from +Z
        let pose = Pose::look_at(
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );

        let depth = camera.render_depth(&pose, sphere_sdf);

        // Should have some valid depths
        assert!(depth.valid_count() > 0);

        // Center pixel should have depth around 0.7 (1.0 - 0.3)
        let center_depth = depth.get(32, 32);
        assert!(center_depth > 0.5 && center_depth < 0.9);
    }

    #[test]
    fn test_depth_to_points() {
        let mut camera = DepthCameraSimulator::new(32, 32, 60.0_f32.to_radians())
            .with_depth_range(0.1, 5.0)
            .with_noise(0.0);

        // Simple sphere
        let sphere_sdf = |p: Point3| -> Option<f32> { Some(p.length() - 0.3) };

        let pose = Pose::look_at(
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );

        let depth = camera.render_depth(&pose, sphere_sdf);
        let (points, normals) = camera.depth_to_points(&depth, &pose);

        // Should have some points
        assert!(!points.is_empty());
        assert_eq!(points.len(), normals.len());

        // Points should be roughly on the sphere surface
        for p in &points {
            let dist_to_center = p.length();
            assert!(
                (dist_to_center - 0.3).abs() < 0.1,
                "Point at distance {} from center",
                dist_to_center
            );
        }
    }

    #[test]
    fn test_orbit_poses() {
        let poses = generate_orbit_poses(8, 1.0, Point3::new(0.0, 0.0, 0.0), 0.5);

        assert_eq!(poses.len(), 8);

        // All poses should look toward center
        for pose in &poses {
            let to_center = (Point3::new(0.0, 0.0, 0.0) - pose.position).normalize();
            let dot = pose.forward.dot(to_center);
            assert!(dot > 0.5, "Pose should look toward center, dot = {}", dot);
        }
    }
}
