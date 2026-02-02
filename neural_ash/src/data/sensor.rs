//! Sensor data structures for robotics applications.

#![allow(dead_code)]

use ash_core::Point3;

/// A batch of sensor data for incremental SDF updates.
#[derive(Debug, Clone)]
pub struct SensorDataBatch {
    /// Point cloud from sensor.
    pub points: Vec<Point3>,
    /// Optional normals.
    pub normals: Option<Vec<Point3>>,
    /// Sensor origin/position.
    pub sensor_origin: Point3,
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Frame ID for tracking.
    pub frame_id: u64,
}

impl SensorDataBatch {
    /// Create a new sensor data batch.
    pub fn new(points: Vec<Point3>, sensor_origin: Point3) -> Self {
        Self {
            points,
            normals: None,
            sensor_origin,
            timestamp: 0.0,
            frame_id: 0,
        }
    }

    /// Set the timestamp.
    pub fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set the frame ID.
    pub fn with_frame_id(mut self, frame_id: u64) -> Self {
        self.frame_id = frame_id;
        self
    }

    /// Set the normals.
    pub fn with_normals(mut self, normals: Vec<Point3>) -> Self {
        assert_eq!(self.points.len(), normals.len());
        self.normals = Some(normals);
        self
    }

    /// Get the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute ray directions from sensor origin to each point.
    pub fn ray_directions(&self) -> Vec<Point3> {
        self.points
            .iter()
            .map(|p| (*p - self.sensor_origin).normalize())
            .collect()
    }

    /// Compute distances from sensor origin.
    pub fn distances(&self) -> Vec<f32> {
        self.points
            .iter()
            .map(|p| (*p - self.sensor_origin).length())
            .collect()
    }

    /// Filter points by distance range.
    pub fn filter_by_distance(&self, min_dist: f32, max_dist: f32) -> Self {
        let mut new_points = Vec::new();
        let mut new_normals = self.normals.as_ref().map(|_| Vec::new());

        for (i, p) in self.points.iter().enumerate() {
            let dist = (*p - self.sensor_origin).length();
            if dist >= min_dist && dist <= max_dist {
                new_points.push(*p);
                if let (Some(normals), Some(ref mut new_n)) = (&self.normals, &mut new_normals) {
                    new_n.push(normals[i]);
                }
            }
        }

        SensorDataBatch {
            points: new_points,
            normals: new_normals,
            sensor_origin: self.sensor_origin,
            timestamp: self.timestamp,
            frame_id: self.frame_id,
        }
    }

    /// Subsample to a maximum number of points.
    pub fn subsample(&self, max_points: usize) -> Self {
        if self.points.len() <= max_points {
            return self.clone();
        }

        let step = self.points.len() / max_points;
        let points: Vec<Point3> = self.points.iter().step_by(step).take(max_points).copied().collect();
        let normals = self.normals.as_ref().map(|n| {
            n.iter().step_by(step).take(max_points).copied().collect()
        });

        SensorDataBatch {
            points,
            normals,
            sensor_origin: self.sensor_origin,
            timestamp: self.timestamp,
            frame_id: self.frame_id,
        }
    }
}

/// Ring buffer for accumulating sensor data over time.
pub struct SensorDataBuffer {
    /// Buffer of sensor batches.
    batches: Vec<SensorDataBatch>,
    /// Maximum number of batches to keep.
    max_batches: usize,
    /// Maximum total points across all batches.
    max_points: usize,
}

impl SensorDataBuffer {
    /// Create a new sensor data buffer.
    pub fn new(max_batches: usize, max_points: usize) -> Self {
        Self {
            batches: Vec::new(),
            max_batches,
            max_points,
        }
    }

    /// Add a batch to the buffer.
    pub fn push(&mut self, batch: SensorDataBatch) {
        self.batches.push(batch);

        // Remove old batches if we exceed limits
        while self.batches.len() > self.max_batches {
            self.batches.remove(0);
        }

        // Also check total points
        while self.total_points() > self.max_points && self.batches.len() > 1 {
            self.batches.remove(0);
        }
    }

    /// Get total number of points across all batches.
    pub fn total_points(&self) -> usize {
        self.batches.iter().map(|b| b.len()).sum()
    }

    /// Get all points as a single vector.
    pub fn all_points(&self) -> Vec<Point3> {
        self.batches.iter().flat_map(|b| b.points.iter().copied()).collect()
    }

    /// Get all normals as a single vector (if available).
    pub fn all_normals(&self) -> Option<Vec<Point3>> {
        if self.batches.iter().any(|b| b.normals.is_none()) {
            return None;
        }

        Some(
            self.batches
                .iter()
                .flat_map(|b| b.normals.as_ref().unwrap().iter().copied())
                .collect(),
        )
    }

    /// Get the number of batches.
    pub fn len(&self) -> usize {
        self.batches.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// Clear all batches.
    pub fn clear(&mut self) {
        self.batches.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_batch_creation() {
        let points = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let origin = Point3::new(0.0, 0.0, 0.0);

        let batch = SensorDataBatch::new(points, origin);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_ray_directions() {
        let points = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let origin = Point3::new(0.0, 0.0, 0.0);

        let batch = SensorDataBatch::new(points, origin);
        let dirs = batch.ray_directions();

        assert!((dirs[0].x - 1.0).abs() < 1e-6);
        assert!((dirs[1].y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_filter_by_distance() {
        let points = vec![
            Point3::new(0.5, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let origin = Point3::new(0.0, 0.0, 0.0);

        let batch = SensorDataBatch::new(points, origin);
        let filtered = batch.filter_by_distance(0.6, 1.5);

        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_buffer() {
        let mut buffer = SensorDataBuffer::new(3, 1000);

        for i in 0..5 {
            let points = vec![Point3::new(i as f32, 0.0, 0.0)];
            let batch = SensorDataBatch::new(points, Point3::new(0.0, 0.0, 0.0))
                .with_frame_id(i as u64);
            buffer.push(batch);
        }

        // Should only keep 3 batches
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.total_points(), 3);
    }
}
