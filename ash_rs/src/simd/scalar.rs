//! Scalar fallback implementation for batch queries.
//!
//! Used when SIMD is not available or for remainder elements.

use ash_core::Point3;

use super::{BatchResult, BatchResultWithGradients};
use crate::SparseDenseGrid;

/// Batch query using scalar operations.
///
/// This is the fallback implementation used when SIMD is not available.
/// It's still reasonably fast due to good cache utilization.
pub fn query_batch_scalar(grid: &SparseDenseGrid, points: &[Point3]) -> BatchResult {
    let n = points.len();
    let mut values = Vec::with_capacity(n);
    let mut valid_mask = vec![false; n];

    for (i, &point) in points.iter().enumerate() {
        match grid.query(point) {
            Some(v) => {
                values.push(v);
                valid_mask[i] = true;
            }
            None => {
                values.push(f32::NAN);
            }
        }
    }

    BatchResult { values, valid_mask }
}

/// Batch query with gradients using scalar operations.
pub fn query_batch_with_gradients_scalar(
    grid: &SparseDenseGrid,
    points: &[Point3],
) -> BatchResultWithGradients {
    let n = points.len();
    let mut values = Vec::with_capacity(n);
    let mut gradients = Vec::with_capacity(n);
    let mut valid_mask = vec![false; n];

    for (i, &point) in points.iter().enumerate() {
        match grid.query_with_gradient(point) {
            Some((v, g)) => {
                values.push(v);
                gradients.push(g);
                valid_mask[i] = true;
            }
            None => {
                values.push(f32::NAN);
                gradients.push([f32::NAN; 3]);
            }
        }
    }

    BatchResultWithGradients {
        values,
        gradients,
        valid_mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridBuilder;
    use ash_core::BlockCoord;

    fn make_test_grid() -> SparseDenseGrid {
        let dim = 8;
        let cells_per_block = dim * dim * dim;

        let mut builder = GridBuilder::new(dim as u32, 0.1).with_capacity(8);

        // Add blocks with simple SDF values
        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    let coord = BlockCoord::new(bx, by, bz);
                    let values = vec![0.5; cells_per_block]; // Uniform positive SDF
                    builder = builder.add_block(coord, values).unwrap();
                }
            }
        }

        builder.build().unwrap()
    }

    #[test]
    fn test_batch_query_scalar() {
        let grid = make_test_grid();

        let points = vec![
            Point3::new(0.4, 0.4, 0.4),
            Point3::new(1.2, 0.4, 0.4),
            Point3::new(-1.0, -1.0, -1.0), // Out of bounds
        ];

        let result = query_batch_scalar(&grid, &points);

        assert_eq!(result.len(), 3);
        assert!(result.valid_mask[0]);
        assert!(result.valid_mask[1]);
        assert!(!result.valid_mask[2]);
    }

    #[test]
    fn test_batch_query_with_gradients_scalar() {
        let grid = make_test_grid();

        let points = vec![
            Point3::new(0.4, 0.4, 0.4),
            Point3::new(-1.0, -1.0, -1.0),
        ];

        let result = query_batch_with_gradients_scalar(&grid, &points);

        assert_eq!(result.values.len(), 2);
        assert_eq!(result.gradients.len(), 2);
        assert!(result.valid_mask[0]);
        assert!(!result.valid_mask[1]);
    }

    #[test]
    fn test_batch_query_empty() {
        let grid = make_test_grid();
        let points: Vec<Point3> = vec![];

        let result = query_batch_scalar(&grid, &points);
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_matches_sequential() {
        let grid = make_test_grid();

        let points = vec![
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.5, 0.5, 0.5),
            Point3::new(0.9, 0.9, 0.9),
        ];

        let batch_result = query_batch_scalar(&grid, &points);

        for (i, &point) in points.iter().enumerate() {
            let sequential = grid.query(point);
            match (batch_result.valid_mask[i], sequential) {
                (true, Some(v)) => {
                    assert!(
                        (batch_result.values[i] - v).abs() < 1e-6,
                        "Mismatch at index {}: batch={}, seq={}",
                        i,
                        batch_result.values[i],
                        v
                    );
                }
                (false, None) => {}
                _ => panic!(
                    "Validity mismatch at index {}: batch={}, seq={:?}",
                    i, batch_result.valid_mask[i], sequential
                ),
            }
        }
    }
}
