//! Concurrent safety tests for ash_rs.
//!
//! These tests verify that the lock-free data structures work correctly
//! under concurrent access from multiple threads.

use ash_core::{BlockCoord, Point3};
use ash_rs::{GridBuilder, SparseDenseGrid};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// Test Grid Factories
// =============================================================================

/// Create a sphere grid for concurrent testing
fn create_sphere_grid() -> SparseDenseGrid {
    let center = Point3::new(0.4, 0.4, 0.4);
    let radius = 0.3;

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

/// Create a linear grid for concurrent testing
fn create_linear_grid() -> SparseDenseGrid {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(125);

    for bz in 0..5 {
        for by in 0..5 {
            for bx in 0..5 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| pos.x + pos.y + pos.z);
            }
        }
    }

    builder.build().unwrap()
}

// =============================================================================
// Concurrent Read Tests
// =============================================================================

#[test]
fn test_concurrent_reads_basic() {
    let grid = Arc::new(create_sphere_grid());

    let num_threads = 4;
    let queries_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                let mut results = Vec::with_capacity(queries_per_thread);
                for i in 0..queries_per_thread {
                    let t = (i + thread_id * queries_per_thread) as f32 / (num_threads * queries_per_thread) as f32;
                    let point = Point3::new(t * 1.4, t * 1.4, t * 1.4);
                    results.push(grid.query(point));
                }
                results
            })
        })
        .collect();

    // All threads should complete without panic
    for handle in handles {
        let results = handle.join().expect("Thread panicked");
        assert_eq!(results.len(), queries_per_thread);
    }
}

#[test]
fn test_concurrent_reads_many_threads() {
    let grid = Arc::new(create_linear_grid());

    let num_threads = 16;
    let queries_per_thread = 500;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let seed = (thread_id * 1000 + i) as f32;
                    let x = ((seed * 0.1) % 1.0) * 3.5;
                    let y = ((seed * 0.17) % 1.0) * 3.5;
                    let z = ((seed * 0.23) % 1.0) * 3.5;
                    let point = Point3::new(x, y, z);
                    let _ = grid.query(point);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_concurrent_reads_same_point() {
    let grid = Arc::new(create_sphere_grid());

    // All threads query the same point - tests for race conditions
    let point = Point3::new(0.4, 0.4, 0.4);
    let expected = grid.query(point);

    let num_threads = 8;
    let iterations = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for _ in 0..iterations {
                    let result = grid.query(point);
                    // All reads should return the same value
                    assert_eq!(
                        result.is_some(),
                        expected.is_some(),
                        "Inconsistent validity"
                    );
                    if let (Some(r), Some(e)) = (result, expected) {
                        assert!(
                            (r - e).abs() < 1e-6,
                            "Inconsistent value: {} vs {}",
                            r,
                            e
                        );
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_concurrent_reads_result_consistency() {
    let grid = Arc::new(create_linear_grid());

    let num_threads = 8;
    let iterations = 1000;

    // Generate test points
    let test_points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(t * 3.5, t * 3.5, t * 3.5)
        })
        .collect();

    // Pre-compute expected results
    let expected: Vec<Option<f32>> = test_points.iter().map(|p| grid.query(*p)).collect();

    let test_points = Arc::new(test_points);
    let expected = Arc::new(expected);

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let grid = Arc::clone(&grid);
            let points = Arc::clone(&test_points);
            let expected = Arc::clone(&expected);

            thread::spawn(move || {
                for _ in 0..iterations {
                    for (i, point) in points.iter().enumerate() {
                        let result = grid.query(*point);
                        match (result, expected[i]) {
                            (Some(r), Some(e)) => {
                                assert!(
                                    (r - e).abs() < 1e-5,
                                    "Inconsistent at {}: {} vs {}",
                                    i,
                                    r,
                                    e
                                );
                            }
                            (None, None) => {}
                            _ => panic!("Validity mismatch at index {}", i),
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Concurrent Batch Query Tests
// =============================================================================

#[test]
fn test_concurrent_batch_queries() {
    let grid = Arc::new(create_linear_grid());

    let num_threads = 8;
    let batches_per_thread = 50;
    let batch_size = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for batch in 0..batches_per_thread {
                    let points: Vec<Point3> = (0..batch_size)
                        .map(|i| {
                            let seed = (thread_id * 10000 + batch * 100 + i) as f32;
                            let t = seed / 100000.0;
                            Point3::new(t * 3.5, t * 3.5, t * 3.5)
                        })
                        .collect();

                    let result = grid.query_batch(&points);
                    assert_eq!(result.values.len(), batch_size);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_concurrent_batch_consistency() {
    let grid = Arc::new(create_sphere_grid());

    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(t * 1.4, t * 1.4, t * 1.4)
        })
        .collect();

    // Pre-compute expected batch result
    let expected = grid.query_batch(&points);
    let points = Arc::new(points);
    let expected = Arc::new(expected);

    let num_threads = 8;
    let iterations = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let grid = Arc::clone(&grid);
            let points = Arc::clone(&points);
            let expected = Arc::clone(&expected);

            thread::spawn(move || {
                for _ in 0..iterations {
                    let result = grid.query_batch(&points);

                    for i in 0..result.values.len() {
                        assert_eq!(
                            result.valid_mask[i], expected.valid_mask[i],
                            "Batch validity mismatch at {}",
                            i
                        );
                        if result.valid_mask[i] {
                            assert!(
                                (result.values[i] - expected.values[i]).abs() < 1e-5,
                                "Batch value mismatch at {}: {} vs {}",
                                i,
                                result.values[i],
                                expected.values[i]
                            );
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Concurrent Gradient Query Tests
// =============================================================================

#[test]
fn test_concurrent_gradient_queries() {
    let grid = Arc::new(create_sphere_grid());

    let num_threads = 8;
    let queries_per_thread = 500;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let seed = (thread_id * queries_per_thread + i) as f32;
                    let t = seed / (num_threads * queries_per_thread) as f32;
                    let point = Point3::new(t * 1.4, t * 1.4, t * 1.4);
                    let _ = grid.query_with_gradient(point);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Block Map Concurrent Access Tests
// =============================================================================

#[test]
fn test_concurrent_block_lookup() {
    let grid = Arc::new(create_linear_grid());

    let num_threads = 8;
    let lookups_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..lookups_per_thread {
                    let x = ((thread_id + i) % 5) as i32;
                    let y = ((thread_id + i * 2) % 5) as i32;
                    let z = ((thread_id + i * 3) % 5) as i32;
                    let coord = BlockCoord::new(x, y, z);
                    let _ = grid.has_block(coord);
                    let _ = grid.get_block_index(coord);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_concurrent_block_lookup_miss() {
    let grid = Arc::new(create_sphere_grid());

    let num_threads = 8;
    let lookups_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..lookups_per_thread {
                    // Lookup blocks that don't exist
                    let coord = BlockCoord::new(100 + i as i32, 100, 100);
                    assert!(!grid.has_block(coord));
                    assert!(grid.get_block_index(coord).is_none());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Mixed Operation Tests
// =============================================================================

#[test]
fn test_concurrent_mixed_operations() {
    let grid = Arc::new(create_linear_grid());

    let num_threads = 8;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..500 {
                    let seed = (thread_id * 500 + i) as f32;
                    let t = seed / 4000.0;
                    let point = Point3::new(t * 3.5, t * 3.5, t * 3.5);

                    match i % 4 {
                        0 => {
                            let _ = grid.query(point);
                        }
                        1 => {
                            let _ = grid.query_with_gradient(point);
                        }
                        2 => {
                            let _ = grid.in_collision(point, 0.0);
                        }
                        3 => {
                            let points = vec![point; 10];
                            let _ = grid.query_batch(&points);
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn test_stress_high_contention() {
    let grid = Arc::new(create_sphere_grid());

    // Many threads all querying the same small region
    let num_threads = 16;
    let queries_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    // All threads query points in the same small region
                    let t = (i % 10) as f32 / 100.0;
                    let offset = (thread_id as f32) * 0.001; // Tiny offset per thread
                    let point = Point3::new(0.4 + t + offset, 0.4 + t, 0.4 + t);
                    let _ = grid.query(point);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_stress_sustained_load() {
    let grid = Arc::new(create_linear_grid());

    let num_threads = 4;
    let duration = Duration::from_millis(500);

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                let start = Instant::now();
                let mut query_count = 0u64;

                while start.elapsed() < duration {
                    for _ in 0..100 {
                        let seed = query_count as f32 + thread_id as f32 * 1000000.0;
                        let t = (seed % 10000.0) / 10000.0;
                        let point = Point3::new(t * 3.5, t * 3.5, t * 3.5);
                        let _ = grid.query(point);
                        query_count += 1;
                    }
                }

                query_count
            })
        })
        .collect();

    let total_queries: u64 = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .sum();

    // Just verify a reasonable number of queries completed
    assert!(
        total_queries > 10000,
        "Expected at least 10000 queries, got {}",
        total_queries
    );
}

// =============================================================================
// Data Race Tests
// =============================================================================

#[test]
fn test_no_torn_reads() {
    // This test attempts to detect torn reads by checking that values are always valid f32
    let grid = Arc::new(create_linear_grid());

    let num_threads = 8;
    let iterations = 10000;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..iterations {
                    let seed = (thread_id * iterations + i) as f32;
                    let t = (seed % 10000.0) / 10000.0;
                    let point = Point3::new(t * 3.5, t * 3.5, t * 3.5);

                    if let Some(value) = grid.query(point) {
                        // Check that value is a valid f32 (not garbage from torn read)
                        assert!(
                            value.is_finite(),
                            "Got non-finite value {} at {:?}",
                            value,
                            point
                        );
                        assert!(
                            value.abs() < 1e10,
                            "Got unreasonable value {} at {:?}",
                            value,
                            point
                        );
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Collision Detection Concurrent Tests
// =============================================================================

#[test]
fn test_concurrent_collision_checks() {
    let grid = Arc::new(create_sphere_grid());

    let num_threads = 8;
    let checks_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for i in 0..checks_per_thread {
                    let seed = (thread_id * checks_per_thread + i) as f32;
                    let t = seed / (num_threads * checks_per_thread) as f32;
                    let point = Point3::new(t * 1.4, t * 1.4, t * 1.4);

                    // Check collision with various thresholds
                    let _ = grid.in_collision(point, 0.0);
                    let _ = grid.in_collision(point, 0.1);
                    let _ = grid.in_collision(point, -0.1);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_concurrent_batch_collision_checks() {
    let grid = Arc::new(create_sphere_grid());

    let num_threads = 4;
    let batches_per_thread = 100;
    let batch_size = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for batch in 0..batches_per_thread {
                    let points: Vec<Point3> = (0..batch_size)
                        .map(|i| {
                            let seed = (thread_id * 10000 + batch * 100 + i) as f32;
                            let t = seed / 100000.0;
                            Point3::new(t * 1.4, t * 1.4, t * 1.4)
                        })
                        .collect();

                    let collisions = grid.check_collisions_batch(&points, 0.0);
                    assert_eq!(collisions.len(), batch_size);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Mesh Extraction Concurrent Tests
// =============================================================================

#[test]
#[cfg(any(feature = "std", feature = "alloc"))]
fn test_concurrent_mesh_extraction() {
    let grid = Arc::new(create_sphere_grid());

    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                // Extract mesh from different blocks concurrently
                for bz in -1..=1 {
                    for by in -1..=1 {
                        for bx in -1..=1 {
                            let coord = BlockCoord::new(bx, by, bz);
                            if grid.has_block(coord) {
                                let triangles = grid.extract_block_mesh(coord, 0.0);
                                // Just verify we got some result
                                for tri in &triangles {
                                    for v in tri {
                                        assert!(v.x.is_finite());
                                        assert!(v.y.is_finite());
                                        assert!(v.z.is_finite());
                                    }
                                }
                            }
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Bounding Box Concurrent Tests
// =============================================================================

#[test]
fn test_concurrent_bounding_box() {
    let grid = Arc::new(create_linear_grid());

    let expected_bbox = grid.bounding_box();

    let num_threads = 8;
    let iterations = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                for _ in 0..iterations {
                    let bbox = grid.bounding_box();
                    assert_eq!(bbox.is_some(), expected_bbox.is_some());
                    if let (Some((min, max)), Some((exp_min, exp_max))) = (bbox, expected_bbox) {
                        assert!((min.x - exp_min.x).abs() < 1e-6);
                        assert!((min.y - exp_min.y).abs() < 1e-6);
                        assert!((min.z - exp_min.z).abs() < 1e-6);
                        assert!((max.x - exp_max.x).abs() < 1e-6);
                        assert!((max.y - exp_max.y).abs() < 1e-6);
                        assert!((max.z - exp_max.z).abs() < 1e-6);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// =============================================================================
// Block Iteration Concurrent Tests
// =============================================================================

#[test]
fn test_concurrent_block_iteration() {
    let grid = Arc::new(create_linear_grid());

    let expected_coords: Vec<BlockCoord> = grid.block_coords().collect();

    let num_threads = 8;
    let iterations = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let grid = Arc::clone(&grid);
            let expected = expected_coords.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    let coords: Vec<BlockCoord> = grid.block_coords().collect();
                    assert_eq!(coords.len(), expected.len());
                    // Note: order might differ due to iterator implementation
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}
