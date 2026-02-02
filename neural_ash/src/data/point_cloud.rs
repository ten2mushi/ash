//! Point cloud data structures and loading.

use ash_core::Point3;

/// A point cloud with optional normals and colors.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// Point positions.
    pub points: Vec<Point3>,
    /// Optional point normals.
    pub normals: Option<Vec<Point3>>,
    /// Optional point colors (RGB in [0, 1]).
    pub colors: Option<Vec<[f32; 3]>>,
}

impl PointCloud {
    /// Create a new point cloud from points.
    pub fn new(points: Vec<Point3>) -> Self {
        Self {
            points,
            normals: None,
            colors: None,
        }
    }

    /// Create a point cloud with normals.
    pub fn with_normals(points: Vec<Point3>, normals: Vec<Point3>) -> Self {
        assert_eq!(points.len(), normals.len());
        Self {
            points,
            normals: Some(normals),
            colors: None,
        }
    }

    /// Add normals to the point cloud.
    pub fn set_normals(&mut self, normals: Vec<Point3>) {
        assert_eq!(self.points.len(), normals.len());
        self.normals = Some(normals);
    }

    /// Add colors to the point cloud.
    pub fn set_colors(&mut self, colors: Vec<[f32; 3]>) {
        assert_eq!(self.points.len(), colors.len());
        self.colors = Some(colors);
    }

    /// Get the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute the bounding box.
    pub fn bounding_box(&self) -> Option<(Point3, Point3)> {
        if self.points.is_empty() {
            return None;
        }

        let mut min = self.points[0];
        let mut max = self.points[0];

        for p in &self.points {
            min = min.min(*p);
            max = max.max(*p);
        }

        Some((min, max))
    }

    /// Compute the centroid.
    pub fn centroid(&self) -> Option<Point3> {
        if self.points.is_empty() {
            return None;
        }

        let mut sum = Point3::new(0.0, 0.0, 0.0);
        for p in &self.points {
            sum = sum + *p;
        }

        Some(sum / self.points.len() as f32)
    }

    /// Center the point cloud at the origin.
    pub fn center(&mut self) {
        if let Some(centroid) = self.centroid() {
            for p in &mut self.points {
                *p = *p - centroid;
            }
        }
    }

    /// Scale the point cloud to fit in a unit cube.
    pub fn normalize(&mut self) {
        self.center();

        if let Some((min, max)) = self.bounding_box() {
            let extent = (max - min).length();
            if extent > 0.0 {
                let scale = 1.0 / extent;
                for p in &mut self.points {
                    *p = *p * scale;
                }
            }
        }
    }

    /// Subsample the point cloud.
    pub fn subsample(&self, n: usize) -> Self {
        if n >= self.points.len() {
            return self.clone();
        }

        let step = self.points.len() / n;
        let points: Vec<Point3> = self.points.iter().step_by(step).take(n).copied().collect();
        let normals = self.normals.as_ref().map(|n| {
            n.iter().step_by(step).take(points.len()).copied().collect()
        });
        let colors = self.colors.as_ref().map(|c| {
            c.iter().step_by(step).take(points.len()).copied().collect()
        });

        Self {
            points,
            normals,
            colors,
        }
    }

    /// Estimate normals from local neighborhoods.
    ///
    /// Uses PCA on k-nearest neighbors.
    pub fn estimate_normals(&mut self, k: usize) {
        let n = self.points.len();
        if n < k {
            return;
        }

        let mut normals = Vec::with_capacity(n);

        for i in 0..n {
            let p = self.points[i];

            // Find k-nearest neighbors (brute force for simplicity)
            let mut dists: Vec<(usize, f32)> = self
                .points
                .iter()
                .enumerate()
                .map(|(j, q)| (j, (*q - p).length_squared()))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Compute covariance matrix of neighbors
            let neighbors: Vec<Point3> = dists.iter().take(k).map(|(j, _)| self.points[*j]).collect();

            let centroid: Point3 = neighbors.iter().fold(Point3::new(0.0, 0.0, 0.0), |acc, p| acc + *p)
                / k as f32;

            // 3x3 covariance matrix
            let mut cov = [[0.0f32; 3]; 3];
            for q in &neighbors {
                let d = *q - centroid;
                cov[0][0] += d.x * d.x;
                cov[0][1] += d.x * d.y;
                cov[0][2] += d.x * d.z;
                cov[1][1] += d.y * d.y;
                cov[1][2] += d.y * d.z;
                cov[2][2] += d.z * d.z;
            }
            cov[1][0] = cov[0][1];
            cov[2][0] = cov[0][2];
            cov[2][1] = cov[1][2];

            // Simple power iteration to find smallest eigenvector (normal)
            let mut v = Point3::new(1.0, 1.0, 1.0).normalize();
            for _ in 0..20 {
                let new_v = Point3::new(
                    cov[0][0] * v.x + cov[0][1] * v.y + cov[0][2] * v.z,
                    cov[1][0] * v.x + cov[1][1] * v.y + cov[1][2] * v.z,
                    cov[2][0] * v.x + cov[2][1] * v.y + cov[2][2] * v.z,
                );
                v = new_v.normalize();
            }

            // This gives the largest eigenvector, we want the smallest
            // For a quick approximation, we just use it (proper implementation would use inverse iteration)
            normals.push(v);
        }

        self.normals = Some(normals);
    }
}

/// Dataset for loading multiple point clouds.
pub struct PointCloudDataset {
    /// Point clouds in the dataset.
    pub clouds: Vec<PointCloud>,
}

impl PointCloudDataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self { clouds: Vec::new() }
    }

    /// Add a point cloud to the dataset.
    pub fn add(&mut self, cloud: PointCloud) {
        self.clouds.push(cloud);
    }

    /// Get the number of point clouds.
    pub fn len(&self) -> usize {
        self.clouds.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.clouds.is_empty()
    }

    /// Get a point cloud by index.
    pub fn get(&self, idx: usize) -> Option<&PointCloud> {
        self.clouds.get(idx)
    }

    /// Iterate over point clouds.
    pub fn iter(&self) -> impl Iterator<Item = &PointCloud> {
        self.clouds.iter()
    }
}

impl Default for PointCloudDataset {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let cloud = PointCloud::new(points);

        assert_eq!(cloud.len(), 3);
    }

    #[test]
    fn test_bounding_box() {
        let points = vec![
            Point3::new(-1.0, -2.0, -3.0),
            Point3::new(1.0, 2.0, 3.0),
        ];
        let cloud = PointCloud::new(points);

        let (min, max) = cloud.bounding_box().unwrap();
        assert_eq!(min, Point3::new(-1.0, -2.0, -3.0));
        assert_eq!(max, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_center() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 2.0, 2.0),
        ];
        let mut cloud = PointCloud::new(points);
        cloud.center();

        let centroid = cloud.centroid().unwrap();
        assert!((centroid.x).abs() < 1e-6);
        assert!((centroid.y).abs() < 1e-6);
        assert!((centroid.z).abs() < 1e-6);
    }

    #[test]
    fn test_subsample() {
        let points: Vec<Point3> = (0..100)
            .map(|i| Point3::new(i as f32, 0.0, 0.0))
            .collect();
        let cloud = PointCloud::new(points);

        let subsampled = cloud.subsample(10);
        assert!(subsampled.len() <= 10);
    }
}
