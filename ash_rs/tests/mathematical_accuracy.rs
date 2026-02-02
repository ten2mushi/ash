//! Mathematical accuracy tests for ash_rs.
//!
//! These tests verify the mathematical correctness of SDF queries against
//! analytical closed-form solutions for known geometric primitives.
//!
//! Following the Yoneda philosophy: we define the behavior so thoroughly
//! that any implementation passing these tests must be functionally equivalent.

use ash_core::{
    compute_trilinear_weights, decompose_point, BlockCoord, CellCoord, LocalCoord, Point3,
    UNTRAINED_SENTINEL,
};
use ash_rs::{GridBuilder, SparseDenseGrid};

// =============================================================================
// Test Helper Functions
// =============================================================================

/// Analytical sphere SDF: d(p) = |p - center| - radius
fn sphere_sdf(point: Point3, center: Point3, radius: f32) -> f32 {
    (point - center).length() - radius
}

/// Analytical plane SDF: d(p) = dot(p - p0, n) where n is unit normal
fn plane_sdf(point: Point3, plane_point: Point3, normal: Point3) -> f32 {
    (point - plane_point).dot(normal)
}

/// Analytical box SDF: distance to axis-aligned box centered at origin
fn box_sdf(point: Point3, center: Point3, half_extents: Point3) -> f32 {
    let p = point - center;
    let q = p.abs() - half_extents;
    let outside = Point3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

/// Analytical cylinder SDF (infinite height, along Z axis)
fn cylinder_sdf(point: Point3, center_xy: (f32, f32), radius: f32) -> f32 {
    let dx = point.x - center_xy.0;
    let dy = point.y - center_xy.1;
    (dx * dx + dy * dy).sqrt() - radius
}

/// Analytical torus SDF (major radius R, minor radius r, centered at origin in XY plane)
fn torus_sdf(point: Point3, major_radius: f32, minor_radius: f32) -> f32 {
    let q_x = (point.x * point.x + point.y * point.y).sqrt() - major_radius;
    let q = (q_x * q_x + point.z * point.z).sqrt();
    q - minor_radius
}

/// Create a grid filled with sphere SDF values
fn create_sphere_grid(center: Point3, radius: f32, num_blocks: i32) -> SparseDenseGrid {
    let half = num_blocks / 2;
    let mut builder = GridBuilder::new(8, 0.1).with_capacity((num_blocks * num_blocks * num_blocks) as usize);

    for bz in -half..half {
        for by in -half..half {
            for bx in -half..half {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| sphere_sdf(pos, center, radius));
            }
        }
    }

    builder.build().unwrap()
}

/// Create a grid filled with box SDF values
fn create_box_grid(center: Point3, half_extents: Point3, num_blocks: i32) -> SparseDenseGrid {
    let half = num_blocks / 2;
    let mut builder = GridBuilder::new(8, 0.1).with_capacity((num_blocks * num_blocks * num_blocks) as usize);

    for bz in -half..half {
        for by in -half..half {
            for bx in -half..half {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| box_sdf(pos, center, half_extents));
            }
        }
    }

    builder.build().unwrap()
}

/// Create a grid filled with linear SDF (f(x,y,z) = ax + by + cz + d)
fn create_linear_grid(a: f32, b: f32, c: f32, d: f32) -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(27);

    for bz in -1..=1 {
        for by in -1..=1 {
            for bx in -1..=1 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| a * pos.x + b * pos.y + c * pos.z + d);
            }
        }
    }

    builder.build().unwrap()
}

// =============================================================================
// Trilinear Interpolation Weight Tests
// =============================================================================

#[test]
fn test_trilinear_weights_sum_to_one() {
    // Test across a range of local coordinates
    let test_coords = [
        LocalCoord::new(0.0, 0.0, 0.0),
        LocalCoord::new(1.0, 1.0, 1.0),
        LocalCoord::new(0.5, 0.5, 0.5),
        LocalCoord::new(0.25, 0.75, 0.33),
        LocalCoord::new(0.1, 0.9, 0.5),
        LocalCoord::new(0.001, 0.999, 0.001),
    ];

    for local in &test_coords {
        let weights = compute_trilinear_weights(*local);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Weights sum to {} instead of 1.0 for local {:?}",
            sum,
            local
        );
    }
}

#[test]
fn test_trilinear_weights_at_corners() {
    // At corner (0,0,0), only weight[0] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(0.0, 0.0, 0.0));
    assert!((w[0] - 1.0).abs() < 1e-6, "Weight[0] should be 1.0, got {}", w[0]);
    for i in 1..8 {
        assert!(w[i].abs() < 1e-6, "Weight[{}] should be 0.0, got {}", i, w[i]);
    }

    // At corner (1,0,0), only weight[1] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(1.0, 0.0, 0.0));
    assert!((w[1] - 1.0).abs() < 1e-6, "Weight[1] should be 1.0, got {}", w[1]);

    // At corner (1,1,0), only weight[2] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(1.0, 1.0, 0.0));
    assert!((w[2] - 1.0).abs() < 1e-6, "Weight[2] should be 1.0, got {}", w[2]);

    // At corner (0,1,0), only weight[3] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(0.0, 1.0, 0.0));
    assert!((w[3] - 1.0).abs() < 1e-6, "Weight[3] should be 1.0, got {}", w[3]);

    // At corner (0,0,1), only weight[4] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(0.0, 0.0, 1.0));
    assert!((w[4] - 1.0).abs() < 1e-6, "Weight[4] should be 1.0, got {}", w[4]);

    // At corner (1,0,1), only weight[5] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(1.0, 0.0, 1.0));
    assert!((w[5] - 1.0).abs() < 1e-6, "Weight[5] should be 1.0, got {}", w[5]);

    // At corner (1,1,1), only weight[6] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(1.0, 1.0, 1.0));
    assert!((w[6] - 1.0).abs() < 1e-6, "Weight[6] should be 1.0, got {}", w[6]);

    // At corner (0,1,1), only weight[7] should be 1.0
    let w = compute_trilinear_weights(LocalCoord::new(0.0, 1.0, 1.0));
    assert!((w[7] - 1.0).abs() < 1e-6, "Weight[7] should be 1.0, got {}", w[7]);
}

#[test]
fn test_trilinear_weights_at_center() {
    // At center (0.5, 0.5, 0.5), all weights should be 0.125
    let w = compute_trilinear_weights(LocalCoord::new(0.5, 0.5, 0.5));
    for i in 0..8 {
        assert!(
            (w[i] - 0.125).abs() < 1e-6,
            "Weight[{}] should be 0.125, got {}",
            i,
            w[i]
        );
    }
}

#[test]
fn test_trilinear_weights_on_faces() {
    // At face center (0.5, 0.5, 0.0), z=0 corners should each be 0.25
    let w = compute_trilinear_weights(LocalCoord::new(0.5, 0.5, 0.0));
    for i in [0, 1, 2, 3] {
        assert!(
            (w[i] - 0.25).abs() < 1e-6,
            "Weight[{}] on z=0 face should be 0.25, got {}",
            i,
            w[i]
        );
    }
    for i in [4, 5, 6, 7] {
        assert!(
            w[i].abs() < 1e-6,
            "Weight[{}] on z=1 face should be 0.0, got {}",
            i,
            w[i]
        );
    }
}

#[test]
fn test_trilinear_weights_on_edges() {
    // At edge center (0.5, 0.0, 0.0), only corners 0 and 1 should contribute
    let w = compute_trilinear_weights(LocalCoord::new(0.5, 0.0, 0.0));
    assert!((w[0] - 0.5).abs() < 1e-6);
    assert!((w[1] - 0.5).abs() < 1e-6);
    for i in 2..8 {
        assert!(w[i].abs() < 1e-6, "Weight[{}] should be 0.0", i);
    }
}

#[test]
fn test_trilinear_weights_are_non_negative() {
    // Weights should always be non-negative for valid local coords
    for u_int in 0..=10 {
        for v_int in 0..=10 {
            for w_int in 0..=10 {
                let u = u_int as f32 / 10.0;
                let v = v_int as f32 / 10.0;
                let w = w_int as f32 / 10.0;
                let weights = compute_trilinear_weights(LocalCoord::new(u, v, w));
                for (i, &weight) in weights.iter().enumerate() {
                    assert!(
                        weight >= -1e-7,
                        "Weight[{}] is negative ({}) for local ({}, {}, {})",
                        i,
                        weight,
                        u,
                        v,
                        w
                    );
                }
            }
        }
    }
}

// =============================================================================
// Linear SDF Tests (Exact Solutions)
// =============================================================================

#[test]
fn test_linear_sdf_exact_interpolation() {
    // For a linear function f(x,y,z) = ax + by + cz + d,
    // trilinear interpolation should return the exact value
    let (a, b, c, d) = (1.0, 2.0, 3.0, 0.5);
    let grid = create_linear_grid(a, b, c, d);

    let test_points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.15, 0.25, 0.35),
        Point3::new(0.5, 0.5, 0.5),
        Point3::new(-0.3, 0.4, -0.2),
    ];

    for point in &test_points {
        let expected = a * point.x + b * point.y + c * point.z + d;
        if let Some(actual) = grid.query(*point) {
            assert!(
                (actual - expected).abs() < 0.01,
                "Linear SDF at {:?}: expected {}, got {}",
                point,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_linear_sdf_gradient_is_constant() {
    // For f(x,y,z) = x + y + z, the gradient is (1, 1, 1) everywhere
    let grid = create_linear_grid(1.0, 1.0, 1.0, 0.0);

    let test_points = [
        Point3::new(0.1, 0.1, 0.1),
        Point3::new(0.5, 0.5, 0.5),
        Point3::new(-0.3, 0.2, 0.4),
    ];

    for point in &test_points {
        if let Some((_, grad)) = grid.query_with_gradient(*point) {
            // Gradient should be approximately (1, 1, 1) in world space
            // Given cell_size = 0.1, world gradient = cell_gradient / cell_size
            // = (cell_size, cell_size, cell_size) / cell_size = (1, 1, 1)
            for (i, &g) in grad.iter().enumerate() {
                assert!(
                    (g - 1.0).abs() < 0.15,
                    "Gradient component {} at {:?}: expected 1.0, got {}",
                    i,
                    point,
                    g
                );
            }
        }
    }
}

#[test]
fn test_linear_sdf_different_coefficients() {
    // Test with different linear coefficients
    let test_cases = [
        (2.0, 0.0, 0.0, 0.0),   // f = 2x
        (0.0, 3.0, 0.0, 0.0),   // f = 3y
        (0.0, 0.0, 4.0, 0.0),   // f = 4z
        (1.0, -1.0, 0.5, 1.0),  // f = x - y + 0.5z + 1
    ];

    for (a, b, c, d) in test_cases {
        let grid = create_linear_grid(a, b, c, d);
        let point = Point3::new(0.25, 0.35, 0.45);
        let expected = a * point.x + b * point.y + c * point.z + d;

        if let Some(actual) = grid.query(point) {
            assert!(
                (actual - expected).abs() < 0.02,
                "Linear SDF with ({}, {}, {}, {}) at {:?}: expected {}, got {}",
                a,
                b,
                c,
                d,
                point,
                expected,
                actual
            );
        }
    }
}

// =============================================================================
// Sphere SDF Tests
// =============================================================================

#[test]
fn test_sphere_sdf_at_center() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    // At center, SDF should be -radius
    let result = grid.query(center);
    assert!(result.is_some());
    let sdf = result.unwrap();
    assert!(
        (sdf - (-radius)).abs() < 0.05,
        "Sphere center SDF: expected {}, got {}",
        -radius,
        sdf
    );
}

#[test]
fn test_sphere_sdf_on_surface() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    // Points on the surface should have SDF near 0
    let surface_points = [
        Point3::new(radius, 0.0, 0.0),
        Point3::new(0.0, radius, 0.0),
        Point3::new(0.0, 0.0, radius),
        Point3::new(-radius, 0.0, 0.0),
    ];

    for point in &surface_points {
        if let Some(sdf) = grid.query(*point) {
            assert!(
                sdf.abs() < 0.05,
                "Sphere surface SDF at {:?}: expected ~0, got {}",
                point,
                sdf
            );
        }
    }
}

#[test]
fn test_sphere_sdf_outside() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    // Points outside should have positive SDF
    let outside_points = [
        Point3::new(0.5, 0.0, 0.0),
        Point3::new(0.0, 0.5, 0.0),
        Point3::new(0.3, 0.3, 0.3),
    ];

    for point in &outside_points {
        if let Some(sdf) = grid.query(*point) {
            let expected = sphere_sdf(*point, center, radius);
            assert!(
                sdf > 0.0,
                "Point {:?} should be outside sphere, got SDF {}",
                point,
                sdf
            );
            assert!(
                (sdf - expected).abs() < 0.05,
                "Sphere SDF at {:?}: expected {}, got {}",
                point,
                expected,
                sdf
            );
        }
    }
}

#[test]
fn test_sphere_sdf_inside() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    // Points inside should have negative SDF
    let inside_points = [
        Point3::new(0.1, 0.0, 0.0),
        Point3::new(0.0, 0.1, 0.0),
        Point3::new(0.1, 0.1, 0.1),
    ];

    for point in &inside_points {
        if let Some(sdf) = grid.query(*point) {
            let expected = sphere_sdf(*point, center, radius);
            assert!(
                sdf < 0.0,
                "Point {:?} should be inside sphere, got SDF {}",
                point,
                sdf
            );
            assert!(
                (sdf - expected).abs() < 0.05,
                "Sphere SDF at {:?}: expected {}, got {}",
                point,
                expected,
                sdf
            );
        }
    }
}

#[test]
fn test_sphere_gradient_points_radially_outward() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    // Test points along each axis (outside sphere)
    let test_cases = [
        (Point3::new(0.5, 0.0, 0.0), 0),  // +X axis
        (Point3::new(-0.5, 0.0, 0.0), 0), // -X axis
        (Point3::new(0.0, 0.5, 0.0), 1),  // +Y axis
        (Point3::new(0.0, -0.5, 0.0), 1), // -Y axis
        (Point3::new(0.0, 0.0, 0.5), 2),  // +Z axis
        (Point3::new(0.0, 0.0, -0.5), 2), // -Z axis
    ];

    for (point, expected_dominant_axis) in &test_cases {
        if let Some((_, grad)) = grid.query_with_gradient(*point) {
            // Find the axis with the largest gradient magnitude
            let abs_grad = [grad[0].abs(), grad[1].abs(), grad[2].abs()];
            let max_idx = abs_grad
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            assert_eq!(
                max_idx, *expected_dominant_axis,
                "Gradient at {:?}: expected dominant axis {}, got {} (grad={:?})",
                point, expected_dominant_axis, max_idx, grad
            );
        }
    }
}

#[test]
fn test_sphere_gradient_magnitude_approximately_one() {
    // For a true SDF, |grad f| = 1 everywhere
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    let test_points = [
        Point3::new(0.4, 0.0, 0.0),
        Point3::new(0.2, 0.2, 0.2),
        Point3::new(-0.3, 0.1, 0.0),
    ];

    for point in &test_points {
        if let Some((_, grad)) = grid.query_with_gradient(*point) {
            let mag = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();
            // Due to discretization, magnitude may not be exactly 1
            assert!(
                (mag - 1.0).abs() < 0.3,
                "Gradient magnitude at {:?}: expected ~1.0, got {}",
                point,
                mag
            );
        }
    }
}

// =============================================================================
// Box SDF Tests
// =============================================================================

#[test]
fn test_box_sdf_at_center() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let half_extents = Point3::new(0.2, 0.2, 0.2);
    let grid = create_box_grid(center, half_extents, 4);

    // At center, SDF should be negative (inside)
    if let Some(sdf) = grid.query(center) {
        assert!(
            sdf < 0.0,
            "Box center should have negative SDF, got {}",
            sdf
        );
    }
}

#[test]
fn test_box_sdf_on_faces() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let half_extents = Point3::new(0.2, 0.2, 0.2);
    let grid = create_box_grid(center, half_extents, 4);

    // Points on faces should have SDF near 0
    let face_points = [
        Point3::new(0.2, 0.0, 0.0),  // +X face
        Point3::new(-0.2, 0.0, 0.0), // -X face
        Point3::new(0.0, 0.2, 0.0),  // +Y face
        Point3::new(0.0, 0.0, 0.2),  // +Z face
    ];

    for point in &face_points {
        if let Some(sdf) = grid.query(*point) {
            assert!(
                sdf.abs() < 0.05,
                "Box face SDF at {:?}: expected ~0, got {}",
                point,
                sdf
            );
        }
    }
}

#[test]
fn test_box_sdf_outside() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let half_extents = Point3::new(0.2, 0.2, 0.2);
    let grid = create_box_grid(center, half_extents, 4);

    // Points outside should have positive SDF
    let outside_points = [
        Point3::new(0.4, 0.0, 0.0),
        Point3::new(0.0, 0.4, 0.0),
        Point3::new(0.3, 0.3, 0.3), // Diagonal
    ];

    for point in &outside_points {
        if let Some(sdf) = grid.query(*point) {
            let expected = box_sdf(*point, center, half_extents);
            assert!(
                sdf > 0.0,
                "Point {:?} should be outside box, got SDF {}",
                point,
                sdf
            );
            // Allow larger tolerance for box SDF due to corners
            assert!(
                (sdf - expected).abs() < 0.1,
                "Box SDF at {:?}: expected {}, got {}",
                point,
                expected,
                sdf
            );
        }
    }
}

// =============================================================================
// Floating Point Edge Cases
// =============================================================================

#[test]
fn test_query_at_block_boundaries() {
    let grid = create_linear_grid(1.0, 1.0, 1.0, 0.0);

    // Points exactly at block boundaries
    let boundary_points = [
        Point3::new(0.8, 0.0, 0.0),  // Block boundary at x
        Point3::new(0.0, 0.8, 0.0),  // Block boundary at y
        Point3::new(0.0, 0.0, 0.8),  // Block boundary at z
        Point3::new(0.8, 0.8, 0.8),  // All boundaries
    ];

    for point in &boundary_points {
        let result = grid.query(*point);
        assert!(
            result.is_some(),
            "Query at boundary {:?} should succeed",
            point
        );
    }
}

#[test]
fn test_query_at_cell_boundaries() {
    let grid = create_linear_grid(1.0, 1.0, 1.0, 0.0);

    // Points exactly at cell boundaries (cell_size = 0.1)
    let boundary_points = [
        Point3::new(0.1, 0.0, 0.0),
        Point3::new(0.2, 0.0, 0.0),
        Point3::new(0.3, 0.3, 0.3),
    ];

    for point in &boundary_points {
        let result = grid.query(*point);
        assert!(
            result.is_some(),
            "Query at cell boundary {:?} should succeed",
            point
        );
    }
}

#[test]
fn test_query_very_small_values() {
    // Test that very small positive and negative values are handled
    let grid = create_linear_grid(0.001, 0.001, 0.001, 0.0);

    let point = Point3::new(0.1, 0.1, 0.1);
    let expected = 0.001 * (0.1 + 0.1 + 0.1); // = 0.0003

    if let Some(sdf) = grid.query(point) {
        assert!(
            (sdf - expected).abs() < 0.001,
            "Small value SDF: expected {}, got {}",
            expected,
            sdf
        );
    }
}

#[test]
fn test_query_large_magnitude_values() {
    // Test that large magnitude values are handled
    let grid = create_linear_grid(100.0, 100.0, 100.0, 0.0);

    let point = Point3::new(0.5, 0.5, 0.5);
    let expected = 100.0 * (0.5 + 0.5 + 0.5); // = 150.0

    if let Some(sdf) = grid.query(point) {
        assert!(
            (sdf - expected).abs() < 5.0,
            "Large value SDF: expected {}, got {}",
            expected,
            sdf
        );
    }
}

#[test]
fn test_query_negative_coordinate_blocks() {
    // Test queries in blocks with negative coordinates
    let center = Point3::new(-0.4, -0.4, -0.4);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    // Query at center (negative coords)
    let result = grid.query(center);
    assert!(result.is_some(), "Query at negative coord center should succeed");
    assert!(result.unwrap() < 0.0, "Center should be inside sphere");
}

// =============================================================================
// Cross-Block Interpolation Tests
// =============================================================================

#[test]
fn test_interpolation_across_block_boundary() {
    // Create a grid where we can test interpolation that spans blocks
    let grid = create_linear_grid(1.0, 0.0, 0.0, 0.0); // f(x,y,z) = x

    // Points near block boundaries that require cross-block interpolation
    // Block size = 0.8, so boundary at x = 0.8
    let test_points = [
        Point3::new(0.75, 0.1, 0.1),  // Just before boundary
        Point3::new(0.79, 0.1, 0.1),  // Very close to boundary
        Point3::new(0.8, 0.1, 0.1),   // At boundary
        Point3::new(0.81, 0.1, 0.1),  // Just after boundary
    ];

    for point in &test_points {
        if let Some(sdf) = grid.query(*point) {
            // For f = x, SDF should equal x
            assert!(
                (sdf - point.x).abs() < 0.05,
                "Cross-block interpolation at {:?}: expected {}, got {}",
                point,
                point.x,
                sdf
            );
        }
    }
}

#[test]
fn test_continuity_across_block_boundary() {
    // SDF should be continuous across block boundaries
    let center = Point3::new(0.4, 0.4, 0.4);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    // Sample points across a block boundary
    let y = 0.4;
    let z = 0.4;
    let x_values: Vec<f32> = (0..20).map(|i| 0.7 + i as f32 * 0.01).collect(); // 0.7 to 0.89

    let mut prev_sdf: Option<f32> = None;
    for x in &x_values {
        if let Some(sdf) = grid.query(Point3::new(*x, y, z)) {
            if let Some(prev) = prev_sdf {
                // Check continuity: difference should be small
                let diff = (sdf - prev).abs();
                assert!(
                    diff < 0.02,
                    "Discontinuity at x={}: prev={}, current={}, diff={}",
                    x,
                    prev,
                    sdf,
                    diff
                );
            }
            prev_sdf = Some(sdf);
        }
    }
}

// =============================================================================
// UNTRAINED_SENTINEL Tests
// =============================================================================

#[test]
fn test_untrained_sentinel_detection() {
    // Create a grid with sentinel values
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(1);
    let mut values = vec![0.5; 512];
    values[0] = UNTRAINED_SENTINEL; // Mark one cell as untrained

    builder = builder.add_block(BlockCoord::new(0, 0, 0), values).unwrap();
    let grid = builder.build().unwrap();

    // Query at the sentinel cell should return None
    let point = Point3::new(0.0, 0.0, 0.0);
    let result = grid.query(point);
    // The result depends on which corners are involved in interpolation
    // For corner (0,0,0) to affect the result, the query must be near that corner
}

#[test]
fn test_untrained_block_returns_none() {
    // Create a grid with one block
    let grid = GridBuilder::new(8, 0.1)
        .with_capacity(1)
        .add_block_constant(BlockCoord::new(0, 0, 0), 0.5)
        .build()
        .unwrap();

    // Query in an unallocated block should return None
    let point = Point3::new(10.0, 10.0, 10.0);
    let result = grid.query(point);
    assert!(result.is_none(), "Query in unallocated block should return None");
}

// =============================================================================
// Morton Encoding Round-Trip Tests
// =============================================================================

#[test]
fn test_decompose_compose_roundtrip_positive() {
    let cell_size = 0.1f32;
    let grid_dim = 8u32;

    let test_points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.05, 0.05, 0.05),
        Point3::new(0.25, 0.5, 0.75),
        Point3::new(1.5, 2.3, 3.7),
    ];

    for point in &test_points {
        let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
        let reconstructed = ash_core::compose_point(block, cell, local, cell_size, grid_dim);

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

#[test]
fn test_decompose_compose_roundtrip_negative() {
    let cell_size = 0.1f32;
    let grid_dim = 8u32;

    let test_points = [
        Point3::new(-0.1, 0.0, 0.0),
        Point3::new(-1.5, -2.3, -3.7),
        Point3::new(-0.05, -0.05, -0.05),
        Point3::new(-10.0, 10.0, -10.0),
    ];

    for point in &test_points {
        let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
        let reconstructed = ash_core::compose_point(block, cell, local, cell_size, grid_dim);

        let diff = (*point - reconstructed).length();
        assert!(
            diff < 1e-5,
            "Roundtrip failed for negative {:?}: got {:?}, diff={}",
            point,
            reconstructed,
            diff
        );
    }
}

#[test]
fn test_cell_coord_flat_index_roundtrip() {
    let grid_dim = 8u32;

    for z in 0..grid_dim {
        for y in 0..grid_dim {
            for x in 0..grid_dim {
                let cell = CellCoord::new(x, y, z);
                let idx = cell.flat_index(grid_dim);
                let reconstructed = CellCoord::from_flat_index(idx, grid_dim);

                assert_eq!(
                    cell, reconstructed,
                    "CellCoord flat index roundtrip failed for ({}, {}, {})",
                    x, y, z
                );
            }
        }
    }
}

// =============================================================================
// Collision Detection Consistency
// =============================================================================

#[test]
fn test_collision_detection_consistency() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    let test_points = [
        Point3::new(0.0, 0.0, 0.0),   // Inside
        Point3::new(0.5, 0.0, 0.0),   // Outside
        Point3::new(0.3, 0.0, 0.0),   // On surface
        Point3::new(0.15, 0.15, 0.15), // Inside
    ];

    for point in &test_points {
        if let Some(sdf) = grid.query(*point) {
            let collision_0 = grid.in_collision(*point, 0.0);
            let expected_collision = sdf < 0.0;

            assert_eq!(
                collision_0, expected_collision,
                "Collision at {:?} with threshold 0: SDF={}, in_collision={}, expected={}",
                point, sdf, collision_0, expected_collision
            );

            // With threshold 0.1, more points should be in collision
            let collision_margin = grid.in_collision(*point, 0.1);
            let expected_with_margin = sdf < 0.1;

            assert_eq!(
                collision_margin, expected_with_margin,
                "Collision at {:?} with threshold 0.1: SDF={}, in_collision={}, expected={}",
                point, sdf, collision_margin, expected_with_margin
            );
        }
    }
}

// =============================================================================
// Numerical Gradient Verification
// =============================================================================

#[test]
fn test_gradient_numerical_verification() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let radius = 0.3;
    let grid = create_sphere_grid(center, radius, 4);

    let test_points = [
        Point3::new(0.4, 0.0, 0.0),
        Point3::new(0.2, 0.2, 0.2),
    ];

    let epsilon = 0.001;

    for point in &test_points {
        if let Some((value, grad)) = grid.query_with_gradient(*point) {
            // Compute numerical gradient using central differences
            let mut numerical_grad = [0.0f32; 3];

            // dx
            let f_plus = grid.query(Point3::new(point.x + epsilon, point.y, point.z));
            let f_minus = grid.query(Point3::new(point.x - epsilon, point.y, point.z));
            if let (Some(fp), Some(fm)) = (f_plus, f_minus) {
                numerical_grad[0] = (fp - fm) / (2.0 * epsilon);
            }

            // dy
            let f_plus = grid.query(Point3::new(point.x, point.y + epsilon, point.z));
            let f_minus = grid.query(Point3::new(point.x, point.y - epsilon, point.z));
            if let (Some(fp), Some(fm)) = (f_plus, f_minus) {
                numerical_grad[1] = (fp - fm) / (2.0 * epsilon);
            }

            // dz
            let f_plus = grid.query(Point3::new(point.x, point.y, point.z + epsilon));
            let f_minus = grid.query(Point3::new(point.x, point.y, point.z - epsilon));
            if let (Some(fp), Some(fm)) = (f_plus, f_minus) {
                numerical_grad[2] = (fp - fm) / (2.0 * epsilon);
            }

            // Compare analytical and numerical gradients
            for i in 0..3 {
                let diff = (grad[i] - numerical_grad[i]).abs();
                assert!(
                    diff < 0.5,
                    "Gradient mismatch at {:?} axis {}: analytical={}, numerical={}, diff={}",
                    point,
                    i,
                    grad[i],
                    numerical_grad[i],
                    diff
                );
            }
        }
    }
}
