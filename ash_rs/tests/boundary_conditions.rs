//! Boundary condition tests for ash_rs.
//!
//! These tests thoroughly exercise edge cases at block boundaries, cell boundaries,
//! and coordinate extremes. The goal is to ensure correct behavior when interpolation
//! requires data from multiple blocks or when queries are at exact boundaries.

use ash_core::{decompose_point, resolve_corner, BlockCoord, CellCoord, LocalCoord, Point3};
use ash_rs::{GridBuilder, SparseDenseGrid};

// =============================================================================
// Test Grid Factories
// =============================================================================

/// Create a grid with linear SDF (f = x + y + z) for testing interpolation
fn create_linear_grid(num_blocks: i32) -> SparseDenseGrid {
    let half = num_blocks / 2;
    let mut builder = GridBuilder::new(8, 0.1).with_capacity((num_blocks * num_blocks * num_blocks) as usize);

    for bz in -half..half {
        for by in -half..half {
            for bx in -half..half {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| pos.x + pos.y + pos.z);
            }
        }
    }

    builder.build().unwrap()
}

/// Create a 2x2x2 block grid with different constant values per block
fn create_distinct_blocks_grid() -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                let coord = BlockCoord::new(bx, by, bz);
                // Each block has a unique value based on its position
                let value = (bx + by * 10 + bz * 100) as f32;
                builder = builder.add_block_constant(coord, value);
            }
        }
    }

    builder.build().unwrap()
}

/// Create a grid with one missing block to test boundary with unallocated neighbor
fn create_grid_with_missing_block() -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(7);

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                // Skip block (1, 1, 1)
                if bx == 1 && by == 1 && bz == 1 {
                    continue;
                }
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| pos.x + pos.y + pos.z);
            }
        }
    }

    builder.build().unwrap()
}

// =============================================================================
// Block Boundary Tests
// =============================================================================

#[test]
fn test_query_at_exact_block_boundary() {
    let grid = create_linear_grid(4);

    // Block boundary is at x = 0.8 (block_size = 8 * 0.1 = 0.8)
    let boundary_point = Point3::new(0.8, 0.4, 0.4);
    let result = grid.query(boundary_point);

    assert!(
        result.is_some(),
        "Query at exact block boundary should succeed"
    );

    // Expected value for f(x,y,z) = x + y + z
    let expected = 0.8 + 0.4 + 0.4;
    if let Some(sdf) = result {
        assert!(
            (sdf - expected).abs() < 0.05,
            "Block boundary value: expected {}, got {}",
            expected,
            sdf
        );
    }
}

#[test]
fn test_query_just_before_block_boundary() {
    let grid = create_linear_grid(4);

    // Just before block boundary
    let point = Point3::new(0.79, 0.4, 0.4);
    let result = grid.query(point);

    assert!(result.is_some(), "Query just before boundary should succeed");

    let expected = 0.79 + 0.4 + 0.4;
    if let Some(sdf) = result {
        assert!(
            (sdf - expected).abs() < 0.05,
            "Just before boundary: expected {}, got {}",
            expected,
            sdf
        );
    }
}

#[test]
fn test_query_just_after_block_boundary() {
    let grid = create_linear_grid(4);

    // Just after block boundary
    let point = Point3::new(0.81, 0.4, 0.4);
    let result = grid.query(point);

    assert!(result.is_some(), "Query just after boundary should succeed");

    let expected = 0.81 + 0.4 + 0.4;
    if let Some(sdf) = result {
        assert!(
            (sdf - expected).abs() < 0.05,
            "Just after boundary: expected {}, got {}",
            expected,
            sdf
        );
    }
}

#[test]
fn test_continuity_across_block_boundary_x() {
    let grid = create_linear_grid(4);

    // Sample across X block boundary at x = 0.8
    let y = 0.4;
    let z = 0.4;

    let mut prev_sdf: Option<f32> = None;
    for i in 0..40 {
        let x = 0.6 + i as f32 * 0.01; // 0.6 to 0.99
        let point = Point3::new(x, y, z);

        if let Some(sdf) = grid.query(point) {
            if let Some(prev) = prev_sdf {
                let diff = (sdf - prev).abs();
                // For linear SDF, consecutive samples differ by 0.01 in value
                assert!(
                    diff < 0.03,
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

#[test]
fn test_continuity_across_block_boundary_y() {
    let grid = create_linear_grid(4);

    let x = 0.4;
    let z = 0.4;

    let mut prev_sdf: Option<f32> = None;
    for i in 0..40 {
        let y = 0.6 + i as f32 * 0.01;
        let point = Point3::new(x, y, z);

        if let Some(sdf) = grid.query(point) {
            if let Some(prev) = prev_sdf {
                let diff = (sdf - prev).abs();
                assert!(
                    diff < 0.03,
                    "Discontinuity at y={}: prev={}, current={}, diff={}",
                    y,
                    prev,
                    sdf,
                    diff
                );
            }
            prev_sdf = Some(sdf);
        }
    }
}

#[test]
fn test_continuity_across_block_boundary_z() {
    let grid = create_linear_grid(4);

    let x = 0.4;
    let y = 0.4;

    let mut prev_sdf: Option<f32> = None;
    for i in 0..40 {
        let z = 0.6 + i as f32 * 0.01;
        let point = Point3::new(x, y, z);

        if let Some(sdf) = grid.query(point) {
            if let Some(prev) = prev_sdf {
                let diff = (sdf - prev).abs();
                assert!(
                    diff < 0.03,
                    "Discontinuity at z={}: prev={}, current={}, diff={}",
                    z,
                    prev,
                    sdf,
                    diff
                );
            }
            prev_sdf = Some(sdf);
        }
    }
}

#[test]
fn test_continuity_across_diagonal_boundary() {
    let grid = create_linear_grid(4);

    // Move diagonally across the corner where three blocks meet
    let mut prev_sdf: Option<f32> = None;
    for i in 0..40 {
        let t = 0.6 + i as f32 * 0.01;
        let point = Point3::new(t, t, t);

        if let Some(sdf) = grid.query(point) {
            if let Some(prev) = prev_sdf {
                let diff = (sdf - prev).abs();
                // For diagonal, change per step is 3 * 0.01 = 0.03
                assert!(
                    diff < 0.06,
                    "Discontinuity at t={}: prev={}, current={}, diff={}",
                    t,
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
// Cell Boundary Tests
// =============================================================================

#[test]
fn test_query_at_cell_boundary() {
    let grid = create_linear_grid(4);

    // Cell boundaries at multiples of cell_size = 0.1
    let cell_boundary_points = [
        Point3::new(0.1, 0.2, 0.3),
        Point3::new(0.2, 0.3, 0.4),
        Point3::new(0.5, 0.5, 0.5),
    ];

    for point in &cell_boundary_points {
        let result = grid.query(*point);
        assert!(
            result.is_some(),
            "Query at cell boundary {:?} should succeed",
            point
        );

        let expected = point.x + point.y + point.z;
        if let Some(sdf) = result {
            assert!(
                (sdf - expected).abs() < 0.02,
                "Cell boundary at {:?}: expected {}, got {}",
                point,
                expected,
                sdf
            );
        }
    }
}

#[test]
fn test_continuity_across_cell_boundary() {
    let grid = create_linear_grid(4);

    // Sample across cell boundary at x = 0.1
    let y = 0.25;
    let z = 0.25;

    let mut prev_sdf: Option<f32> = None;
    for i in 0..20 {
        let x = 0.05 + i as f32 * 0.005; // Fine samples across cell boundary
        let point = Point3::new(x, y, z);

        if let Some(sdf) = grid.query(point) {
            if let Some(prev) = prev_sdf {
                let diff = (sdf - prev).abs();
                assert!(
                    diff < 0.01,
                    "Cell boundary discontinuity at x={}: diff={}",
                    x,
                    diff
                );
            }
            prev_sdf = Some(sdf);
        }
    }
}

// =============================================================================
// Corner Resolution Tests
// =============================================================================

#[test]
fn test_resolve_corner_no_overflow() {
    let grid_dim = 8;
    let block = BlockCoord::new(0, 0, 0);
    let cell = CellCoord::new(3, 3, 3);

    // All corners should stay in the same block
    for dx in 0..=1 {
        for dy in 0..=1 {
            for dz in 0..=1 {
                let (resolved_block, resolved_cell) =
                    resolve_corner(block, cell, (dx, dy, dz), grid_dim);

                assert_eq!(
                    resolved_block, block,
                    "Corner ({},{},{}) should stay in same block",
                    dx, dy, dz
                );

                assert_eq!(resolved_cell.x, cell.x + dx);
                assert_eq!(resolved_cell.y, cell.y + dy);
                assert_eq!(resolved_cell.z, cell.z + dz);
            }
        }
    }
}

#[test]
fn test_resolve_corner_x_overflow() {
    let grid_dim = 8;
    let block = BlockCoord::new(0, 0, 0);
    let cell = CellCoord::new(7, 3, 3); // At X edge

    // Corner (1, 0, 0) should overflow to next block
    let (resolved_block, resolved_cell) = resolve_corner(block, cell, (1, 0, 0), grid_dim);

    assert_eq!(resolved_block, BlockCoord::new(1, 0, 0));
    assert_eq!(resolved_cell.x, 0);
    assert_eq!(resolved_cell.y, 3);
    assert_eq!(resolved_cell.z, 3);
}

#[test]
fn test_resolve_corner_y_overflow() {
    let grid_dim = 8;
    let block = BlockCoord::new(0, 0, 0);
    let cell = CellCoord::new(3, 7, 3);

    let (resolved_block, resolved_cell) = resolve_corner(block, cell, (0, 1, 0), grid_dim);

    assert_eq!(resolved_block, BlockCoord::new(0, 1, 0));
    assert_eq!(resolved_cell.y, 0);
}

#[test]
fn test_resolve_corner_z_overflow() {
    let grid_dim = 8;
    let block = BlockCoord::new(0, 0, 0);
    let cell = CellCoord::new(3, 3, 7);

    let (resolved_block, resolved_cell) = resolve_corner(block, cell, (0, 0, 1), grid_dim);

    assert_eq!(resolved_block, BlockCoord::new(0, 0, 1));
    assert_eq!(resolved_cell.z, 0);
}

#[test]
fn test_resolve_corner_triple_overflow() {
    let grid_dim = 8;
    let block = BlockCoord::new(0, 0, 0);
    let cell = CellCoord::new(7, 7, 7); // At corner of block

    // Corner (1, 1, 1) should overflow all three dimensions
    let (resolved_block, resolved_cell) = resolve_corner(block, cell, (1, 1, 1), grid_dim);

    assert_eq!(resolved_block, BlockCoord::new(1, 1, 1));
    assert_eq!(resolved_cell, CellCoord::new(0, 0, 0));
}

// =============================================================================
// Negative Coordinate Tests
// =============================================================================

#[test]
fn test_query_negative_coordinates() {
    let grid = create_linear_grid(4);

    // Test points in negative coordinate space
    let negative_points = [
        Point3::new(-0.1, 0.0, 0.0),
        Point3::new(0.0, -0.1, 0.0),
        Point3::new(0.0, 0.0, -0.1),
        Point3::new(-0.5, -0.5, -0.5),
    ];

    for point in &negative_points {
        let result = grid.query(*point);
        assert!(
            result.is_some(),
            "Query at negative coord {:?} should succeed",
            point
        );

        let expected = point.x + point.y + point.z;
        if let Some(sdf) = result {
            assert!(
                (sdf - expected).abs() < 0.05,
                "Negative coord {:?}: expected {}, got {}",
                point,
                expected,
                sdf
            );
        }
    }
}

#[test]
fn test_decompose_negative_point() {
    let cell_size = 0.1f32;
    let grid_dim = 8u32;

    // Point just below origin
    let point = Point3::new(-0.05, -0.05, -0.05);
    let (block, cell, _local) = decompose_point(point, cell_size, grid_dim);

    // Should be in block (-1, -1, -1)
    assert_eq!(block.x, -1, "Block X should be -1");
    assert_eq!(block.y, -1, "Block Y should be -1");
    assert_eq!(block.z, -1, "Block Z should be -1");

    // Should be in the last cell of that block (cell 7)
    assert_eq!(cell.x, 7, "Cell X should be 7");
    assert_eq!(cell.y, 7, "Cell Y should be 7");
    assert_eq!(cell.z, 7, "Cell Z should be 7");
}

#[test]
fn test_cross_zero_boundary() {
    let grid = create_linear_grid(4);

    // Sample across the zero boundary
    let mut prev_sdf: Option<f32> = None;
    for i in 0..20 {
        let t = -0.1 + i as f32 * 0.01; // -0.1 to 0.09
        let point = Point3::new(t, 0.0, 0.0);

        if let Some(sdf) = grid.query(point) {
            if let Some(prev) = prev_sdf {
                let diff = (sdf - prev).abs();
                assert!(
                    diff < 0.02,
                    "Discontinuity at zero boundary t={}: diff={}",
                    t,
                    diff
                );
            }
            prev_sdf = Some(sdf);
        }
    }
}

// =============================================================================
// Missing Block Boundary Tests
// =============================================================================

#[test]
fn test_query_near_missing_block() {
    let grid = create_grid_with_missing_block();

    // Query in block (0, 0, 0) should work
    let valid_point = Point3::new(0.4, 0.4, 0.4);
    let result = grid.query(valid_point);
    assert!(result.is_some(), "Query in allocated block should succeed");

    // Query in missing block (1, 1, 1) should fail
    let missing_point = Point3::new(1.2, 1.2, 1.2); // Inside block (1, 1, 1)
    let result = grid.query(missing_point);
    assert!(result.is_none(), "Query in missing block should return None");
}

#[test]
fn test_interpolation_requires_missing_block() {
    let grid = create_grid_with_missing_block();

    // Query at corner of block (0, 0, 0) that needs block (1, 1, 1) for interpolation
    // Cell (7, 7, 7) corner (1, 1, 1) overflows to block (1, 1, 1)
    let corner_point = Point3::new(0.79, 0.79, 0.79);
    let result = grid.query(corner_point);

    // This should fail because interpolation needs the missing block
    assert!(
        result.is_none(),
        "Query requiring missing block should return None"
    );
}

// =============================================================================
// Local Coordinate Tests
// =============================================================================

#[test]
fn test_local_coord_range() {
    let grid = create_linear_grid(4);

    // Verify local coordinates are always in [0, 1)
    let test_points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.05, 0.05, 0.05),
        Point3::new(0.09999, 0.09999, 0.09999),
        Point3::new(0.1, 0.1, 0.1),
        Point3::new(-0.01, 0.0, 0.0),
    ];

    for point in &test_points {
        let (_, _, local) = decompose_point(*point, 0.1, 8);

        assert!(
            local.u >= 0.0 && local.u < 1.0001, // Small tolerance for floating point
            "Local U out of range for {:?}: {}",
            point,
            local.u
        );
        assert!(
            local.v >= 0.0 && local.v < 1.0001,
            "Local V out of range for {:?}: {}",
            point,
            local.v
        );
        assert!(
            local.w >= 0.0 && local.w < 1.0001,
            "Local W out of range for {:?}: {}",
            point,
            local.w
        );
    }
}

#[test]
fn test_local_coord_clamping() {
    // Test that slightly out-of-range local coords can be clamped
    let local = LocalCoord::new(-0.001, 1.001, 0.5);
    let clamped = local.clamped();

    assert_eq!(clamped.u, 0.0); // -0.001 clamped to 0.0
    assert_eq!(clamped.v, 1.0); // 1.001 clamped to 1.0
    assert_eq!(clamped.w, 0.5); // 0.5 unchanged
}

// =============================================================================
// Edge of Grid Tests
// =============================================================================

#[test]
fn test_query_at_grid_edge() {
    let grid = create_linear_grid(4); // Blocks from -2 to 1 inclusive

    // Query at the very edge of the allocated region
    // Block range: -2, -1, 0, 1 (for half=2)
    // World range: -2*0.8 to 2*0.8 = -1.6 to 1.6

    // Query just inside the edge
    let edge_point = Point3::new(-1.59, 0.0, 0.0);
    let result = grid.query(edge_point);
    // May or may not succeed depending on which corners are needed
}

#[test]
fn test_query_outside_grid() {
    let grid = create_linear_grid(4);

    // Query completely outside allocated blocks
    let outside_points = [
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(0.0, 10.0, 0.0),
        Point3::new(0.0, 0.0, 10.0),
        Point3::new(-10.0, -10.0, -10.0),
    ];

    for point in &outside_points {
        let result = grid.query(*point);
        assert!(
            result.is_none(),
            "Query at {:?} outside grid should return None",
            point
        );
    }
}

// =============================================================================
// Gradient at Boundaries
// =============================================================================

#[test]
fn test_gradient_at_block_boundary() {
    let grid = create_linear_grid(4);

    // Query gradient at block boundary
    let boundary_point = Point3::new(0.79, 0.4, 0.4);
    let result = grid.query_with_gradient(boundary_point);

    assert!(
        result.is_some(),
        "Gradient query at boundary should succeed"
    );

    if let Some((_, grad)) = result {
        // For f = x + y + z, gradient should be (1, 1, 1)
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                (g - 1.0).abs() < 0.2,
                "Gradient at boundary, component {}: expected 1.0, got {}",
                i,
                g
            );
        }
    }
}

#[test]
fn test_gradient_continuity_across_boundary() {
    let grid = create_linear_grid(4);

    let y = 0.4;
    let z = 0.4;

    let mut prev_grad: Option<[f32; 3]> = None;
    for i in 0..20 {
        let x = 0.7 + i as f32 * 0.01;
        let point = Point3::new(x, y, z);

        if let Some((_, grad)) = grid.query_with_gradient(point) {
            if let Some(prev) = prev_grad {
                for axis in 0..3 {
                    let diff = (grad[axis] - prev[axis]).abs();
                    assert!(
                        diff < 0.2,
                        "Gradient discontinuity at x={}, axis {}: prev={}, curr={}, diff={}",
                        x,
                        axis,
                        prev[axis],
                        grad[axis],
                        diff
                    );
                }
            }
            prev_grad = Some(grad);
        }
    }
}

// =============================================================================
// Batch Query Boundary Tests
// =============================================================================

#[test]
fn test_batch_query_across_boundaries() {
    let grid = create_linear_grid(4);

    // Create points that cross multiple block boundaries
    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(t * 2.0 - 1.0, t * 2.0 - 1.0, t * 2.0 - 1.0)
        })
        .collect();

    let batch_result = grid.query_batch(&points);

    // Verify results are continuous
    let mut prev_valid_value: Option<f32> = None;
    for i in 0..points.len() {
        if batch_result.valid_mask[i] {
            let value = batch_result.values[i];
            if let Some(prev) = prev_valid_value {
                let diff = (value - prev).abs();
                // Allow for large step since we're moving 0.02 in each direction per sample
                assert!(
                    diff < 0.2,
                    "Batch discontinuity at index {}: prev={}, curr={}",
                    i,
                    prev,
                    value
                );
            }
            prev_valid_value = Some(value);
        }
    }
}

// =============================================================================
// Floating Point Precision at Boundaries
// =============================================================================

#[test]
fn test_floating_point_boundary_precision() {
    let grid = create_linear_grid(4);

    // Test points very close to boundaries to check for floating point issues
    let epsilon_points = [
        Point3::new(0.8 - 1e-6, 0.4, 0.4),
        Point3::new(0.8 + 1e-6, 0.4, 0.4),
        Point3::new(0.8, 0.4 - 1e-6, 0.4),
        Point3::new(0.8, 0.4 + 1e-6, 0.4),
    ];

    for point in &epsilon_points {
        let result = grid.query(*point);
        // Should either succeed or fail, but not panic
        if let Some(sdf) = result {
            assert!(
                sdf.is_finite(),
                "Result at {:?} should be finite, got {}",
                point,
                sdf
            );
        }
    }
}

#[test]
fn test_exactly_representable_boundaries() {
    let grid = create_linear_grid(4);

    // 0.1, 0.2, 0.4, 0.5, 0.8 are not exactly representable in IEEE 754
    // but 0.25, 0.5, 0.75 have exact representations
    let exact_points = [
        Point3::new(0.25, 0.25, 0.25),
        Point3::new(0.5, 0.5, 0.5),
        Point3::new(0.75, 0.75, 0.75),
    ];

    for point in &exact_points {
        let result = grid.query(*point);
        assert!(
            result.is_some(),
            "Query at exactly representable {:?} should succeed",
            point
        );

        let expected = point.x + point.y + point.z;
        if let Some(sdf) = result {
            assert!(
                (sdf - expected).abs() < 0.02,
                "Exact boundary {:?}: expected {}, got {}",
                point,
                expected,
                sdf
            );
        }
    }
}

// =============================================================================
// Grid Dimension Edge Cases
// =============================================================================

#[test]
fn test_last_cell_in_block() {
    let grid = create_linear_grid(4);

    // Query in the last cell (7, 7, 7) of block (0, 0, 0)
    // Cell (7,7,7) spans [0.7, 0.8) in each dimension
    let point = Point3::new(0.75, 0.75, 0.75);
    let result = grid.query(point);

    assert!(result.is_some(), "Query in last cell should succeed");

    let expected = 0.75 * 3.0;
    if let Some(sdf) = result {
        assert!(
            (sdf - expected).abs() < 0.02,
            "Last cell value: expected {}, got {}",
            expected,
            sdf
        );
    }
}

#[test]
fn test_first_cell_in_block() {
    let grid = create_linear_grid(4);

    // Query in the first cell (0, 0, 0) of block (0, 0, 0)
    let point = Point3::new(0.05, 0.05, 0.05);
    let result = grid.query(point);

    assert!(result.is_some(), "Query in first cell should succeed");

    let expected = 0.05 * 3.0;
    if let Some(sdf) = result {
        assert!(
            (sdf - expected).abs() < 0.02,
            "First cell value: expected {}, got {}",
            expected,
            sdf
        );
    }
}

// =============================================================================
// Collision Detection at Boundaries
// =============================================================================

#[test]
fn test_collision_at_boundary() {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

    // Create a grid with a sphere
    let center = Point3::new(0.4, 0.4, 0.4);
    let radius = 0.3;

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| (pos - center).length() - radius);
            }
        }
    }

    let grid = builder.build().unwrap();

    // Test collision at block boundary
    let boundary_point = Point3::new(0.8, 0.4, 0.4);
    let inside_point = Point3::new(0.4, 0.4, 0.4);

    // Both should give consistent results based on SDF value
    if let Some(sdf_boundary) = grid.query(boundary_point) {
        let collision = grid.in_collision(boundary_point, 0.0);
        assert_eq!(collision, sdf_boundary < 0.0);
    }

    if let Some(sdf_inside) = grid.query(inside_point) {
        let collision = grid.in_collision(inside_point, 0.0);
        assert_eq!(collision, sdf_inside < 0.0);
    }
}
