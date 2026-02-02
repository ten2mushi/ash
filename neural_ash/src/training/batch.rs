//! Batch sampling for training.

use burn::prelude::*;

use ash_core::Point3;

/// A batch of SDF training data.
#[derive(Debug, Clone)]
pub struct SdfBatch<B: Backend> {
    /// Surface points where SDF should be 0.
    pub surface_points: Tensor<B, 2>,
    /// Free-space points where SDF should be > 0.
    pub free_space_points: Tensor<B, 2>,
    /// Points for Eikonal regularization.
    pub eikonal_points: Tensor<B, 2>,
}

impl<B: Backend> SdfBatch<B> {
    /// Create a new SDF batch from point tensors.
    pub fn new(
        surface_points: Tensor<B, 2>,
        free_space_points: Tensor<B, 2>,
        eikonal_points: Tensor<B, 2>,
    ) -> Self {
        Self {
            surface_points,
            free_space_points,
            eikonal_points,
        }
    }

    /// Get the device of this batch.
    pub fn device(&self) -> B::Device {
        self.surface_points.device()
    }
}

/// A batch of point cloud data.
#[derive(Debug, Clone)]
pub struct PointCloudBatch<B: Backend> {
    /// Point positions: [batch, num_points, 3]
    pub points: Tensor<B, 3>,
    /// Optional normals: [batch, num_points, 3]
    pub normals: Option<Tensor<B, 3>>,
}

impl<B: Backend> PointCloudBatch<B> {
    /// Create a new point cloud batch.
    pub fn new(points: Tensor<B, 3>) -> Self {
        Self {
            points,
            normals: None,
        }
    }

    /// Create a point cloud batch with normals.
    pub fn with_normals(points: Tensor<B, 3>, normals: Tensor<B, 3>) -> Self {
        Self {
            points,
            normals: Some(normals),
        }
    }
}

/// Batch sampler for generating training batches.
pub struct BatchSampler {
    /// Number of surface points per batch.
    surface_points: usize,
    /// Number of free-space points per batch.
    free_space_points: usize,
    /// Number of Eikonal points per batch.
    eikonal_points: usize,
    /// Free-space sampling range (distance from surface).
    free_space_range: (f32, f32),
    /// Current random seed.
    seed: u64,
}

impl BatchSampler {
    /// Create a new batch sampler.
    pub fn new(
        surface_points: usize,
        free_space_points: usize,
        eikonal_points: usize,
    ) -> Self {
        Self {
            surface_points,
            free_space_points,
            eikonal_points,
            free_space_range: (0.05, 0.5),
            seed: 42,
        }
    }

    /// Set the free-space sampling range.
    pub fn with_free_space_range(mut self, min: f32, max: f32) -> Self {
        self.free_space_range = (min, max);
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sample a batch from a point cloud.
    ///
    /// - Surface points: sampled from the input point cloud
    /// - Free-space points: offset from surface along normals (or random direction)
    /// - Eikonal points: random points in the bounding box
    pub fn sample_from_points<B: Backend>(
        &mut self,
        points: &[Point3],
        normals: Option<&[Point3]>,
        device: &B::Device,
    ) -> SdfBatch<B> {
        use std::f32::consts::PI;

        // Simple LCG for reproducible randomness
        let mut rng = || {
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.seed >> 33) as f32 / (1u64 << 31) as f32
        };

        let n = points.len();
        if n == 0 {
            // Return empty batch
            let empty = Tensor::zeros([0, 3], device);
            return SdfBatch::new(empty.clone(), empty.clone(), empty);
        }

        // Sample surface points
        let mut surface_data = Vec::with_capacity(self.surface_points * 3);
        for _ in 0..self.surface_points {
            let idx = (rng() * n as f32) as usize % n;
            let p = points[idx];
            surface_data.extend_from_slice(&[p.x, p.y, p.z]);
        }

        // Sample free-space points (offset from surface)
        let mut free_space_data = Vec::with_capacity(self.free_space_points * 3);
        for _ in 0..self.free_space_points {
            let idx = (rng() * n as f32) as usize % n;
            let p = points[idx];

            // Get offset direction (use normal if available, otherwise random)
            let dir = if let Some(norms) = normals {
                norms[idx]
            } else {
                // Random direction on unit sphere
                let theta = rng() * 2.0 * PI;
                let phi = (1.0 - 2.0 * rng()).acos();
                Point3::new(
                    phi.sin() * theta.cos(),
                    phi.sin() * theta.sin(),
                    phi.cos(),
                )
            };

            // Random offset distance
            let (min_d, max_d) = self.free_space_range;
            let dist = min_d + rng() * (max_d - min_d);

            let offset = p + dir * dist;
            free_space_data.extend_from_slice(&[offset.x, offset.y, offset.z]);
        }

        // Compute bounding box for Eikonal sampling
        let (mut min_p, mut max_p) = (points[0], points[0]);
        for p in points {
            min_p = min_p.min(*p);
            max_p = max_p.max(*p);
        }
        let margin = (max_p - min_p) * 0.1;
        min_p = min_p - margin;
        max_p = max_p + margin;

        // Sample Eikonal points uniformly in bounding box
        let mut eikonal_data = Vec::with_capacity(self.eikonal_points * 3);
        for _ in 0..self.eikonal_points {
            let x = min_p.x + rng() * (max_p.x - min_p.x);
            let y = min_p.y + rng() * (max_p.y - min_p.y);
            let z = min_p.z + rng() * (max_p.z - min_p.z);
            eikonal_data.extend_from_slice(&[x, y, z]);
        }

        // Create tensors
        let surface = Tensor::from_data(
            TensorData::new(surface_data, [self.surface_points, 3]),
            device,
        );
        let free_space = Tensor::from_data(
            TensorData::new(free_space_data, [self.free_space_points, 3]),
            device,
        );
        let eikonal = Tensor::from_data(
            TensorData::new(eikonal_data, [self.eikonal_points, 3]),
            device,
        );

        SdfBatch::new(surface, free_space, eikonal)
    }

    /// Sample a batch using tensor points directly.
    pub fn sample_from_tensor<B: Backend>(
        &mut self,
        points: Tensor<B, 2>,
        _normals: Option<Tensor<B, 2>>,
    ) -> SdfBatch<B> {
        let device = points.device();
        let [n, _] = points.dims();

        // Simple implementation: random indices
        let mut rng = || {
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.seed >> 33) as f32 / (1u64 << 31) as f32
        };

        // Sample surface indices
        let surface_indices: Vec<i64> = (0..self.surface_points)
            .map(|_| (rng() * n as f32) as i64 % n as i64)
            .collect();
        let surface_idx = Tensor::<B, 1, Int>::from_data(surface_indices.as_slice(), &device);
        let surface_points = points.clone().select(0, surface_idx);

        // Sample free-space (offset from surface)
        let free_indices: Vec<i64> = (0..self.free_space_points)
            .map(|_| (rng() * n as f32) as i64 % n as i64)
            .collect();
        let free_idx = Tensor::<B, 1, Int>::from_data(free_indices.as_slice(), &device);
        let base_points = points.clone().select(0, free_idx);

        // Random offsets
        let (min_d, max_d) = self.free_space_range;
        let offsets: Vec<f32> = (0..self.free_space_points * 3)
            .map(|_| {
                let dist = min_d + rng() * (max_d - min_d);
                (rng() - 0.5) * 2.0 * dist
            })
            .collect();
        let offset_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(offsets, [self.free_space_points, 3]),
            &device,
        );
        let free_space = base_points + offset_tensor;

        // Eikonal points: uniform in bounding box
        // Use CPU-side extraction for bounding box to avoid type inference issues
        let points_data = points.to_data();
        let points_flat: Vec<f32> = points_data.to_vec().unwrap();

        let mut min_xyz = [f32::INFINITY; 3];
        let mut max_xyz = [f32::NEG_INFINITY; 3];
        for i in 0..n {
            for j in 0..3 {
                let val = points_flat[i * 3 + j];
                min_xyz[j] = min_xyz[j].min(val);
                max_xyz[j] = max_xyz[j].max(val);
            }
        }

        let eikonal_data: Vec<f32> = (0..self.eikonal_points * 3)
            .map(|i| {
                let axis = i % 3;
                min_xyz[axis] + rng() * (max_xyz[axis] - min_xyz[axis])
            })
            .collect();
        let eikonal = Tensor::<B, 2>::from_data(
            TensorData::new(eikonal_data, [self.eikonal_points, 3]),
            &device,
        );

        SdfBatch::new(surface_points, free_space, eikonal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_batch_sampler() {
        let device = Default::default();
        let mut sampler = BatchSampler::new(100, 100, 50);

        let points: Vec<Point3> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0 * std::f32::consts::PI * 2.0;
                Point3::new(t.cos(), t.sin(), 0.0)
            })
            .collect();

        let batch = sampler.sample_from_points::<TestBackend>(&points, None, &device);

        assert_eq!(batch.surface_points.dims(), [100, 3]);
        assert_eq!(batch.free_space_points.dims(), [100, 3]);
        assert_eq!(batch.eikonal_points.dims(), [50, 3]);
    }

    #[test]
    fn test_batch_from_tensor() {
        let device = Default::default();
        let mut sampler = BatchSampler::new(50, 50, 25);

        let points = Tensor::<TestBackend, 2>::zeros([200, 3], &device);
        let batch = sampler.sample_from_tensor(points, None);

        assert_eq!(batch.surface_points.dims(), [50, 3]);
        assert_eq!(batch.free_space_points.dims(), [50, 3]);
        assert_eq!(batch.eikonal_points.dims(), [25, 3]);
    }
}
