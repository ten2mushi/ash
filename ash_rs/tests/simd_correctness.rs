//! SIMD correctness tests for ash_rs.
//!
//! These tests verify that SIMD implementations (NEON and AVX2) produce
//! results equivalent to the scalar implementation.
//!
//! Key focus areas:
//! - NEON vs scalar equivalence
//! - AVX2 vs scalar equivalence
//! - Boundary handling in SIMD chunks
//! - Remainder processing for non-aligned batch sizes
//! - FMA accuracy

use ash_core::{BlockCoord, Point3};
use ash_rs::{query_batch, GridBuilder, SparseDenseGrid, BatchResult};

// =============================================================================
// Test Grid Factories
// =============================================================================

/// Create a simple constant-value grid for basic tests
fn create_constant_grid(value: f32) -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_constant(coord, value);
            }
        }
    }

    builder.build().unwrap()
}

/// Create a sphere SDF grid
fn create_sphere_grid(center: Point3, radius: f32) -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(27);

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

/// Create a linear SDF grid (f = x + y + z)
fn create_linear_grid() -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(27);

    for bz in -1..=1 {
        for by in -1..=1 {
            for bx in -1..=1 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| pos.x + pos.y + pos.z);
            }
        }
    }

    builder.build().unwrap()
}

/// Create a grid with varying values for stress testing
fn create_varying_grid() -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(27);

    for bz in -1..=1 {
        for by in -1..=1 {
            for bx in -1..=1 {
                let coord = BlockCoord::new(bx, by, bz);
                // Use a more complex function to stress-test interpolation
                builder = builder.add_block_fn(coord, |pos| {
                    let x = pos.x;
                    let y = pos.y;
                    let z = pos.z;
                    (x * x + y * y + z * z).sqrt() + 0.1 * (10.0 * x).sin()
                });
            }
        }
    }

    builder.build().unwrap()
}

// =============================================================================
// Scalar Baseline Implementation for Comparison
// =============================================================================

/// Scalar batch query implementation for baseline comparison
fn scalar_batch_query(grid: &SparseDenseGrid, points: &[Point3]) -> BatchResult {
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

/// Compare two batch results with tolerance
fn results_match(a: &BatchResult, b: &BatchResult, tolerance: f32) -> Result<(), String> {
    if a.values.len() != b.values.len() {
        return Err(format!(
            "Length mismatch: {} vs {}",
            a.values.len(),
            b.values.len()
        ));
    }

    for i in 0..a.values.len() {
        if a.valid_mask[i] != b.valid_mask[i] {
            return Err(format!(
                "Validity mismatch at index {}: {} vs {}",
                i, a.valid_mask[i], b.valid_mask[i]
            ));
        }

        if a.valid_mask[i] {
            let diff = (a.values[i] - b.values[i]).abs();
            if diff > tolerance {
                return Err(format!(
                    "Value mismatch at index {}: {} vs {}, diff={}",
                    i, a.values[i], b.values[i], diff
                ));
            }
        }
    }

    Ok(())
}

// =============================================================================
// Basic SIMD vs Scalar Equivalence Tests
// =============================================================================

#[test]
fn test_batch_matches_scalar_empty() {
    let grid = create_constant_grid(0.5);
    let points: Vec<Point3> = vec![];

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    assert!(batch_result.values.is_empty());
    assert!(scalar_result.values.is_empty());
}

#[test]
fn test_batch_matches_scalar_single_point() {
    let grid = create_constant_grid(0.5);
    let points = vec![Point3::new(0.4, 0.4, 0.4)];

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Single point batch should match scalar");
}

#[test]
fn test_batch_matches_scalar_small_batch() {
    // Test batches smaller than SIMD width (4 for NEON, 8 for AVX2)
    let grid = create_linear_grid();

    for size in 1..16 {
        let points: Vec<Point3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Point3::new(t * 0.6, t * 0.6, t * 0.6)
            })
            .collect();

        let batch_result = query_batch(&grid, &points);
        let scalar_result = scalar_batch_query(&grid, &points);

        results_match(&batch_result, &scalar_result, 1e-5)
            .unwrap_or_else(|e| panic!("Batch size {} failed: {}", size, e));
    }
}

#[test]
fn test_batch_matches_scalar_neon_aligned() {
    // Test batch sizes that are multiples of NEON width (4)
    let grid = create_sphere_grid(Point3::new(0.4, 0.4, 0.4), 0.3);

    for multiplier in 1..=10 {
        let size = multiplier * 4;
        let points: Vec<Point3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Point3::new(t * 1.2, t * 1.2, t * 1.2)
            })
            .collect();

        let batch_result = query_batch(&grid, &points);
        let scalar_result = scalar_batch_query(&grid, &points);

        results_match(&batch_result, &scalar_result, 1e-5)
            .unwrap_or_else(|e| panic!("NEON-aligned batch size {} failed: {}", size, e));
    }
}

#[test]
fn test_batch_matches_scalar_avx2_aligned() {
    // Test batch sizes that are multiples of AVX2 width (8)
    let grid = create_sphere_grid(Point3::new(0.4, 0.4, 0.4), 0.3);

    for multiplier in 1..=5 {
        let size = multiplier * 8;
        let points: Vec<Point3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Point3::new(t * 1.2, t * 1.2, t * 1.2)
            })
            .collect();

        let batch_result = query_batch(&grid, &points);
        let scalar_result = scalar_batch_query(&grid, &points);

        results_match(&batch_result, &scalar_result, 1e-5)
            .unwrap_or_else(|e| panic!("AVX2-aligned batch size {} failed: {}", size, e));
    }
}

// =============================================================================
// Remainder Processing Tests
// =============================================================================

#[test]
fn test_batch_remainder_neon() {
    // Test batch sizes that have remainders after NEON chunks (4)
    let grid = create_linear_grid();

    for base in [8, 12, 16, 20] {
        for remainder in 1..4 {
            let size = base + remainder;
            let points: Vec<Point3> = (0..size)
                .map(|i| {
                    let t = i as f32 / size as f32;
                    Point3::new(t * 0.8, t * 0.8, t * 0.8)
                })
                .collect();

            let batch_result = query_batch(&grid, &points);
            let scalar_result = scalar_batch_query(&grid, &points);

            results_match(&batch_result, &scalar_result, 1e-5).unwrap_or_else(|e| {
                panic!("NEON remainder test (size={}) failed: {}", size, e)
            });
        }
    }
}

#[test]
fn test_batch_remainder_avx2() {
    // Test batch sizes that have remainders after AVX2 chunks (8)
    let grid = create_linear_grid();

    for base in [16, 24, 32] {
        for remainder in 1..8 {
            let size = base + remainder;
            let points: Vec<Point3> = (0..size)
                .map(|i| {
                    let t = i as f32 / size as f32;
                    Point3::new(t * 0.8, t * 0.8, t * 0.8)
                })
                .collect();

            let batch_result = query_batch(&grid, &points);
            let scalar_result = scalar_batch_query(&grid, &points);

            results_match(&batch_result, &scalar_result, 1e-5).unwrap_or_else(|e| {
                panic!("AVX2 remainder test (size={}) failed: {}", size, e)
            });
        }
    }
}

// =============================================================================
// Various Batch Sizes
// =============================================================================

#[test]
fn test_batch_matches_scalar_various_sizes() {
    let grid = create_varying_grid();

    // Test specific batch sizes that might reveal SIMD issues
    let sizes = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 256, 1000];

    for &size in &sizes {
        let points: Vec<Point3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                // Create a variety of positions
                Point3::new(
                    (t * 10.0).sin() * 0.5,
                    (t * 7.0).cos() * 0.5,
                    t * 0.8 - 0.4,
                )
            })
            .collect();

        let batch_result = query_batch(&grid, &points);
        let scalar_result = scalar_batch_query(&grid, &points);

        results_match(&batch_result, &scalar_result, 1e-5)
            .unwrap_or_else(|e| panic!("Batch size {} failed: {}", size, e));
    }
}

// =============================================================================
// Boundary Point Tests
// =============================================================================

#[test]
fn test_batch_boundary_points() {
    let grid = create_sphere_grid(Point3::new(0.4, 0.4, 0.4), 0.3);

    // Test points at various boundaries
    let points = vec![
        // Cell boundaries
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.1, 0.0, 0.0),
        Point3::new(0.2, 0.0, 0.0),
        // Block boundaries
        Point3::new(0.8, 0.0, 0.0),
        Point3::new(0.8, 0.8, 0.0),
        Point3::new(0.8, 0.8, 0.8),
        // Interior points
        Point3::new(0.4, 0.4, 0.4),
        Point3::new(0.5, 0.5, 0.5),
        // Points requiring cross-block interpolation
        Point3::new(0.79, 0.01, 0.01),
        Point3::new(0.81, 0.01, 0.01),
    ];

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    for (i, point) in points.iter().enumerate() {
        if batch_result.valid_mask[i] != scalar_result.valid_mask[i] {
            panic!(
                "Validity mismatch at point {:?}: batch={}, scalar={}",
                point, batch_result.valid_mask[i], scalar_result.valid_mask[i]
            );
        }

        if batch_result.valid_mask[i] {
            let diff = (batch_result.values[i] - scalar_result.values[i]).abs();
            if diff > 1e-5 {
                panic!(
                    "Value mismatch at point {:?}: batch={}, scalar={}, diff={}",
                    point, batch_result.values[i], scalar_result.values[i], diff
                );
            }
        }
    }
}

#[test]
fn test_batch_cross_block_interpolation() {
    // Create a grid where some queries require cross-block corner lookup
    let grid = create_linear_grid();

    // Points that span block boundaries (block_size = 0.8)
    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(0.75 + t * 0.1, 0.75 + t * 0.1, 0.75 + t * 0.1)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Cross-block interpolation should match scalar");
}

// =============================================================================
// Invalid Point Handling
// =============================================================================

#[test]
fn test_batch_invalid_points_mixed() {
    let grid = create_sphere_grid(Point3::new(0.4, 0.4, 0.4), 0.3);

    // Mix of valid and invalid (out of bounds) points
    let points = vec![
        Point3::new(0.4, 0.4, 0.4),   // Valid (inside sphere)
        Point3::new(10.0, 10.0, 10.0), // Invalid (out of bounds)
        Point3::new(0.5, 0.5, 0.5),   // Valid
        Point3::new(-10.0, 0.0, 0.0), // Invalid
        Point3::new(0.6, 0.6, 0.6),   // Valid
        Point3::new(100.0, 0.0, 0.0), // Invalid
        Point3::new(0.7, 0.4, 0.4),   // Valid
        Point3::new(0.0, 100.0, 0.0), // Invalid
    ];

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    for (i, point) in points.iter().enumerate() {
        assert_eq!(
            batch_result.valid_mask[i], scalar_result.valid_mask[i],
            "Validity mismatch at index {} ({:?}): batch={}, scalar={}",
            i, point, batch_result.valid_mask[i], scalar_result.valid_mask[i]
        );
    }
}

#[test]
fn test_batch_all_invalid_points() {
    let grid = create_sphere_grid(Point3::new(0.4, 0.4, 0.4), 0.3);

    // All points out of bounds
    let points: Vec<Point3> = (0..20)
        .map(|i| Point3::new(100.0 + i as f32, 100.0, 100.0))
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    // All should be invalid
    for i in 0..points.len() {
        assert!(!batch_result.valid_mask[i], "Point {} should be invalid", i);
        assert!(!scalar_result.valid_mask[i]);
    }
}

// =============================================================================
// FMA Accuracy Tests
// =============================================================================

#[test]
fn test_fma_accuracy_trilinear_weights() {
    // FMA operations can have slightly different rounding than separate mul+add
    // This test verifies the results are within acceptable tolerance

    let grid = create_varying_grid();

    // Generate many points to test FMA accuracy
    let points: Vec<Point3> = (0..1000)
        .map(|i| {
            let t = i as f32 / 1000.0;
            // Use values that might expose FMA differences
            Point3::new(t * 0.7999999, t * 0.7999999, t * 0.7999999)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    // Allow slightly more tolerance for FMA differences
    results_match(&batch_result, &scalar_result, 1e-4)
        .expect("FMA results should be within tolerance of scalar");
}

#[test]
fn test_fma_corner_case_values() {
    // Test with values that might cause FMA precision issues
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                let coord = BlockCoord::new(bx, by, bz);
                // Use values close to representable limits
                builder = builder.add_block_fn(coord, |pos| {
                    1e-7 * pos.x + 1e7 * pos.y + pos.z
                });
            }
        }
    }

    let grid = builder.build().unwrap();

    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(t * 1.4, t * 1.4, t * 1.4)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    // Check validity matches
    for i in 0..points.len() {
        assert_eq!(
            batch_result.valid_mask[i], scalar_result.valid_mask[i],
            "Validity mismatch at FMA corner case index {}",
            i
        );
    }
}

// =============================================================================
// Determinism Tests
// =============================================================================

#[test]
fn test_batch_deterministic() {
    let grid = create_varying_grid();

    let points: Vec<Point3> = (0..1000)
        .map(|i| {
            let t = i as f32 / 1000.0;
            Point3::new(t * 1.0, (t * 10.0).sin() * 0.5, (t * 7.0).cos() * 0.5)
        })
        .collect();

    // Run multiple times and verify results are identical
    let result1 = query_batch(&grid, &points);
    let result2 = query_batch(&grid, &points);
    let result3 = query_batch(&grid, &points);

    for i in 0..points.len() {
        assert_eq!(result1.valid_mask[i], result2.valid_mask[i]);
        assert_eq!(result1.valid_mask[i], result3.valid_mask[i]);

        if result1.valid_mask[i] {
            assert_eq!(
                result1.values[i], result2.values[i],
                "Non-deterministic result at index {}",
                i
            );
            assert_eq!(result1.values[i], result3.values[i]);
        }
    }
}

// =============================================================================
// Large Batch Tests
// =============================================================================

#[test]
fn test_batch_large_10000_points() {
    let grid = create_sphere_grid(Point3::new(0.4, 0.4, 0.4), 0.3);

    let points: Vec<Point3> = (0..10000)
        .map(|i| {
            let t = i as f32 / 10000.0;
            Point3::new(t * 1.4, t * 1.4, t * 1.4)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Large batch should match scalar");
}

#[test]
fn test_batch_stress_random_pattern() {
    // Use a deterministic "random" pattern for reproducibility
    let grid = create_varying_grid();

    let points: Vec<Point3> = (0..5000)
        .map(|i| {
            // Pseudo-random but deterministic
            let seed = i as f32;
            let x = ((seed * 1.1) % 1.0) * 1.6 - 0.8;
            let y = ((seed * 2.3) % 1.0) * 1.6 - 0.8;
            let z = ((seed * 3.7) % 1.0) * 1.6 - 0.8;
            Point3::new(x, y, z)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Random pattern batch should match scalar");
}

// =============================================================================
// Gradient Batch Tests
// =============================================================================

#[test]
fn test_batch_with_gradients_matches_scalar() {
    let grid = create_sphere_grid(Point3::new(0.4, 0.4, 0.4), 0.3);

    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(t * 1.4, t * 1.4, t * 1.4)
        })
        .collect();

    let batch_result = grid.query_batch_with_gradients(&points);

    // Compare with sequential gradient queries
    for (i, &point) in points.iter().enumerate() {
        let sequential = grid.query_with_gradient(point);

        match (batch_result.valid_mask[i], sequential) {
            (true, Some((value, grad))) => {
                let value_diff = (batch_result.values[i] - value).abs();
                assert!(
                    value_diff < 1e-5,
                    "Value mismatch at {}: batch={}, seq={}",
                    i,
                    batch_result.values[i],
                    value
                );

                for axis in 0..3 {
                    let grad_diff = (batch_result.gradients[i][axis] - grad[axis]).abs();
                    assert!(
                        grad_diff < 1e-5,
                        "Gradient mismatch at {} axis {}: batch={}, seq={}",
                        i,
                        axis,
                        batch_result.gradients[i][axis],
                        grad[axis]
                    );
                }
            }
            (false, None) => {}
            _ => panic!(
                "Validity mismatch at index {}: batch={}, seq={:?}",
                i, batch_result.valid_mask[i], sequential
            ),
        }
    }
}

// =============================================================================
// Edge Case Value Tests
// =============================================================================

#[test]
fn test_batch_negative_values() {
    // Grid with all negative values
    let grid = create_constant_grid(-0.5);

    let points: Vec<Point3> = (0..50)
        .map(|i| {
            let t = i as f32 / 50.0;
            Point3::new(t * 1.4, t * 1.4, t * 1.4)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Negative value batch should match scalar");
}

#[test]
fn test_batch_mixed_sign_values() {
    // Grid with values that cross zero
    let center = Point3::new(0.4, 0.4, 0.4);
    let radius = 0.5; // Large radius for more surface crossing
    let grid = create_sphere_grid(center, radius);

    let points: Vec<Point3> = (0..200)
        .map(|i| {
            let t = i as f32 / 200.0;
            Point3::new(t * 1.4, t * 1.4, t * 1.4)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Mixed sign value batch should match scalar");
}

#[test]
fn test_batch_near_zero_values() {
    // Create grid with values near zero
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_constant(coord, 1e-8);
            }
        }
    }

    let grid = builder.build().unwrap();

    let points: Vec<Point3> = (0..50)
        .map(|i| {
            let t = i as f32 / 50.0;
            Point3::new(t * 1.4, t * 1.4, t * 1.4)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    // Use relative tolerance for near-zero values
    for i in 0..points.len() {
        assert_eq!(
            batch_result.valid_mask[i], scalar_result.valid_mask[i],
            "Validity mismatch at index {}",
            i
        );

        if batch_result.valid_mask[i] {
            let diff = (batch_result.values[i] - scalar_result.values[i]).abs();
            let max_val = batch_result.values[i].abs().max(scalar_result.values[i].abs());
            let relative_diff = if max_val > 1e-10 {
                diff / max_val
            } else {
                diff
            };
            assert!(
                relative_diff < 1e-4,
                "Near-zero value mismatch at {}: batch={}, scalar={}",
                i,
                batch_result.values[i],
                scalar_result.values[i]
            );
        }
    }
}

// =============================================================================
// Clustered Query Tests (Spatial Locality)
// =============================================================================

#[test]
fn test_batch_clustered_queries() {
    let grid = create_varying_grid();

    // All queries in a small spatial region (should benefit from cache)
    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(0.4 + t * 0.1, 0.4 + t * 0.1, 0.4 + t * 0.1)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Clustered queries should match scalar");
}

#[test]
fn test_batch_scattered_queries() {
    let grid = create_varying_grid();

    // Queries scattered across the grid
    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let seed = i as f32;
            Point3::new(
                ((seed * 0.7) % 1.0) * 1.6 - 0.8,
                ((seed * 0.3) % 1.0) * 1.6 - 0.8,
                ((seed * 0.11) % 1.0) * 1.6 - 0.8,
            )
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Scattered queries should match scalar");
}

// =============================================================================
// Grid Configuration Tests
// =============================================================================

#[test]
fn test_batch_single_block_grid() {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(1);
    builder = builder.add_block_constant(BlockCoord::new(0, 0, 0), 0.5);
    let grid = builder.build().unwrap();

    let points: Vec<Point3> = (0..50)
        .map(|i| {
            let t = i as f32 / 50.0;
            Point3::new(t * 0.7, t * 0.7, t * 0.7)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Single block grid batch should match scalar");
}

#[test]
fn test_batch_large_grid() {
    // Create a larger grid (5x5x5 = 125 blocks)
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(125);

    for bz in 0..5 {
        for by in 0..5 {
            for bx in 0..5 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| pos.x + pos.y + pos.z);
            }
        }
    }

    let grid = builder.build().unwrap();

    let points: Vec<Point3> = (0..500)
        .map(|i| {
            let t = i as f32 / 500.0;
            Point3::new(t * 3.5, t * 3.5, t * 3.5)
        })
        .collect();

    let batch_result = query_batch(&grid, &points);
    let scalar_result = scalar_batch_query(&grid, &points);

    results_match(&batch_result, &scalar_result, 1e-5)
        .expect("Large grid batch should match scalar");
}
