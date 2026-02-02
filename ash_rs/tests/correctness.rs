//! Property-based tests verifying correctness against ash_core reference.

use ash_core::{
    decompose_point, trilinear_interpolate_sdf, trilinear_with_gradient_sdf, BlockCoord,
    CellValueProvider, Point3,
};
use ash_rs::{GridBuilder, SparseDenseGrid};
use proptest::prelude::*;

/// Create a test grid with sphere SDF
fn make_sphere_grid(center: Point3, radius: f32) -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(27);

    // Create a 3x3x3 block neighborhood around origin
    for bz in -1..=1 {
        for by in -1..=1 {
            for bx in -1..=1 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| (pos - center).length() - radius);
            }
        }
    }

    builder.build().unwrap()
}

/// Create a test grid with linear SDF
fn make_linear_grid() -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                let coord = BlockCoord::new(bx, by, bz);
                // Linear SDF: f(x,y,z) = x + y + z
                builder = builder.add_block_fn(coord, |pos| pos.x + pos.y + pos.z);
            }
        }
    }

    builder.build().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test that query matches direct trilinear interpolation from ash_core
    #[test]
    fn query_matches_reference(
        x in 0.0f32..1.5,
        y in 0.0f32..1.5,
        z in 0.0f32..1.5,
    ) {
        let grid = make_linear_grid();
        let point = Point3::new(x, y, z);

        let result = grid.query(point);

        let (block, cell, local) = decompose_point(point, grid.cell_size(), grid.grid_dim());
        let expected = trilinear_interpolate_sdf(&grid, block, cell, local);

        match (result, expected) {
            (Some(v1), Some(r)) => {
                prop_assert!((v1 - r.value()).abs() < 1e-5,
                    "Mismatch at {:?}: grid={}, ref={}", point, v1, r.value());
            }
            (None, None) => {}
            _ => prop_assert!(false, "Validity mismatch at {:?}: grid={:?}, ref={:?}",
                point, result, expected),
        }
    }

    /// Test that gradient matches reference from ash_core
    #[test]
    fn gradient_matches_reference(
        x in 0.1f32..1.4,
        y in 0.1f32..1.4,
        z in 0.1f32..1.4,
    ) {
        let grid = make_linear_grid();
        let point = Point3::new(x, y, z);

        let result = grid.query_with_gradient(point);

        let (block, cell, local) = decompose_point(point, grid.cell_size(), grid.grid_dim());
        let expected = trilinear_with_gradient_sdf(&grid, block, cell, local);

        match (result, expected) {
            (Some((v1, g1)), Some((r, g2))) => {
                prop_assert!((v1 - r.value()).abs() < 1e-5,
                    "Value mismatch at {:?}: grid={}, ref={}", point, v1, r.value());

                // Convert reference gradient from cell-space to world-space
                let inv_cell = 1.0 / grid.cell_size();
                let ref_grad = [g2[0] * inv_cell, g2[1] * inv_cell, g2[2] * inv_cell];

                for i in 0..3 {
                    prop_assert!((g1[i] - ref_grad[i]).abs() < 1e-4,
                        "Gradient mismatch at {:?} axis {}: grid={}, ref={}",
                        point, i, g1[i], ref_grad[i]);
                }
            }
            (None, None) => {}
            _ => prop_assert!(false, "Validity mismatch at {:?}", point),
        }
    }

    /// Test that batch queries match sequential queries
    #[test]
    fn batch_matches_sequential(
        xs in prop::collection::vec(0.0f32..1.5, 1..50),
    ) {
        let grid = make_linear_grid();

        let points: Vec<Point3> = xs.iter()
            .map(|&x| Point3::new(x, x * 0.5, x * 0.3))
            .collect();

        let batch_result = grid.query_batch(&points);

        for (i, point) in points.iter().enumerate() {
            let sequential = grid.query(*point);
            match (batch_result.valid_mask[i], sequential) {
                (true, Some(v)) => {
                    prop_assert!((batch_result.values[i] - v).abs() < 1e-5,
                        "Batch/sequential mismatch at index {}: batch={}, seq={}",
                        i, batch_result.values[i], v);
                }
                (false, None) => {}
                _ => prop_assert!(false, "Validity mismatch at index {}", i),
            }
        }
    }

    /// Test interpolation weights sum to 1 (via CellValueProvider)
    #[test]
    fn weights_sum_to_one(
        u in 0.0f32..1.0,
        v in 0.0f32..1.0,
        w in 0.0f32..1.0,
    ) {
        let local = ash_core::LocalCoord::new(u, v, w);
        let weights = ash_core::compute_trilinear_weights(local);
        let sum: f32 = weights.iter().sum();

        prop_assert!((sum - 1.0).abs() < 1e-6,
            "Weights sum to {} instead of 1.0 for local {:?}", sum, local);
    }

    /// Test that collision detection is conservative
    #[test]
    fn collision_is_conservative(
        x in 0.0f32..0.8,
        y in 0.0f32..0.8,
        z in 0.0f32..0.8,
        threshold in 0.0f32..0.5,
    ) {
        let center = Point3::new(0.0, 0.0, 0.0);
        let radius = 0.3;
        let grid = make_sphere_grid(center, radius);

        let point = Point3::new(x, y, z);
        let collision = grid.in_collision(point, threshold);

        if let Some(sdf) = grid.query(point) {
            if collision {
                prop_assert!(sdf < threshold,
                    "Collision detected but SDF {} >= threshold {}",
                    sdf, threshold);
            } else {
                prop_assert!(sdf >= threshold,
                    "No collision but SDF {} < threshold {}",
                    sdf, threshold);
            }
        }
    }
}

#[test]
fn test_linear_gradient_is_constant() {
    let grid = make_linear_grid();

    // For f(x,y,z) = x + y + z (world coords), the SDF values at cell corners
    // are proportional to world position. The trilinear interpolation gradient
    // in cell-space is (cell_size, cell_size, cell_size), and when converted
    // to world-space by dividing by cell_size, we get (1, 1, 1).
    let test_points = [
        Point3::new(0.1, 0.1, 0.1),
        Point3::new(0.5, 0.5, 0.5),
        Point3::new(1.0, 1.0, 1.0),
    ];

    let expected_grad = 1.0; // World-space gradient of f(x,y,z) = x + y + z is (1,1,1)

    for point in &test_points {
        if let Some((_, grad)) = grid.query_with_gradient(*point) {
            for (i, &g) in grad.iter().enumerate() {
                assert!(
                    (g - expected_grad).abs() < 0.1,
                    "Gradient component {} at {:?}: expected {}, got {}",
                    i,
                    point,
                    expected_grad,
                    g
                );
            }
        }
    }
}

#[test]
fn test_sphere_gradient_points_outward() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = make_sphere_grid(center, radius);

    // Test points on +X, +Y, +Z axes, outside the sphere
    let test_cases = [
        (Point3::new(0.5, 0.0, 0.0), 0), // Should have +X gradient
        (Point3::new(0.0, 0.5, 0.0), 1), // Should have +Y gradient
        (Point3::new(0.0, 0.0, 0.5), 2), // Should have +Z gradient
    ];

    for (point, expected_axis) in &test_cases {
        if let Some((_, grad)) = grid.query_with_gradient(*point) {
            // The expected axis should have the largest positive component
            let max_axis = grad
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            assert_eq!(
                max_axis, *expected_axis,
                "Point {:?}: expected max gradient on axis {}, got {} ({:?})",
                point, expected_axis, max_axis, grad
            );
        }
    }
}

#[test]
fn test_decompose_compose_roundtrip() {
    let grid = make_linear_grid();

    let test_points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.05, 0.05, 0.05),
        Point3::new(0.79, 0.79, 0.79),
        Point3::new(0.81, 0.81, 0.81),
        Point3::new(1.5, 0.5, 0.25),
    ];

    for point in &test_points {
        let (block, cell, local) = decompose_point(*point, grid.cell_size(), grid.grid_dim());
        let reconstructed =
            ash_core::compose_point(block, cell, local, grid.cell_size(), grid.grid_dim());

        let diff = (*point - reconstructed).length();
        assert!(
            diff < 1e-5,
            "Roundtrip failed for {:?}: got {:?}, diff={}",
            point,
            reconstructed,
            diff
        );
    }
}
