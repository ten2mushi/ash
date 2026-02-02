//! Integration tests for end-to-end scenarios.

use ash_core::{BlockCoord, Point3};
use ash_rs::{GridBuilder, SparseDenseGrid};
use std::sync::Arc;
use std::thread;

/// Create a warehouse-like environment grid
fn create_warehouse_grid() -> SparseDenseGrid {
    // Warehouse: 4m x 3m x 2m at 5cm resolution
    // Block size: 8 * 0.05 = 0.4m
    // Blocks needed: 10 x 8 x 1 (floor) + pillars above = ~130 blocks

    use std::collections::HashSet;

    let cell_size = 0.05;
    let grid_dim = 8u32;

    let mut builder = GridBuilder::new(grid_dim, cell_size).with_capacity(200);
    let mut added_blocks: HashSet<(i32, i32, i32)> = HashSet::new();

    // Create floor (z=0 plane) - 8*10 = 80 blocks
    for by in 0..8 {
        for bx in 0..10 {
            let coord = BlockCoord::new(bx, by, 0);
            added_blocks.insert((bx, by, 0));
            builder = builder.add_block_fn(coord, |pos| {
                // Distance to floor (z=0 plane)
                pos.z.abs()
            });
        }
    }

    // Add some obstacles (cylindrical pillars)
    let pillars = [
        (Point3::new(1.0, 1.0, 0.0), 0.2),
        (Point3::new(3.0, 1.5, 0.0), 0.15),
        (Point3::new(2.0, 2.5, 0.0), 0.25),
    ];

    // Add blocks around pillars (only z > 0 to avoid duplicates with floor)
    for (center, radius) in &pillars {
        let bx = (center.x / 0.4) as i32;
        let by = (center.y / 0.4) as i32;

        for dbz in 1..4 {
            // Start at z=1 to skip floor level
            for dby in -1..=1 {
                for dbx in -1..=1 {
                    let coord_tuple = (bx + dbx, by + dby, dbz);

                    // Skip if already added or approaching capacity
                    if added_blocks.contains(&coord_tuple) || builder.num_blocks() > 180 {
                        continue;
                    }

                    added_blocks.insert(coord_tuple);
                    let coord = BlockCoord::new(bx + dbx, by + dby, dbz);

                    let pillar_center = *center;
                    let pillar_radius = *radius;

                    builder = builder.add_block_fn(coord, move |pos| {
                        // Cylinder SDF (infinite height, then intersect with z bounds)
                        let dx = pos.x - pillar_center.x;
                        let dy = pos.y - pillar_center.y;
                        let cylinder = (dx * dx + dy * dy).sqrt() - pillar_radius;

                        // Intersect with z bounds [0, 1.5]
                        let top = pos.z - 1.5;
                        let bottom = -pos.z;

                        cylinder.max(top).max(bottom)
                    });
                }
            }
        }
    }

    builder.build().unwrap()
}

#[test]
fn test_warehouse_basic_queries() {
    let grid = create_warehouse_grid();

    // Query at floor level should be near zero
    let floor_point = Point3::new(0.5, 0.5, 0.0);
    if let Some(sdf) = grid.query(floor_point) {
        assert!(
            sdf.abs() < 0.1,
            "Floor query should be near zero: {}",
            sdf
        );
    }

    // Query above floor should be positive
    let above_floor = Point3::new(0.5, 0.5, 0.5);
    if let Some(sdf) = grid.query(above_floor) {
        assert!(sdf > 0.0, "Above floor should be positive: {}", sdf);
    }
}

#[test]
fn test_concurrent_reads() {
    let grid = Arc::new(create_warehouse_grid());

    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let grid = Arc::clone(&grid);
            thread::spawn(move || {
                let mut results = Vec::new();
                for i in 0..1000 {
                    let t = (i + thread_id * 1000) as f32 / 8000.0;
                    let point = Point3::new(t * 4.0, t * 3.0, t * 1.5);
                    results.push(grid.query(point));
                }
                results
            })
        })
        .collect();

    // All threads should complete without panic
    for handle in handles {
        let results = handle.join().unwrap();
        assert_eq!(results.len(), 1000);
    }
}

#[test]
fn test_collision_checking_path() {
    let grid = create_warehouse_grid();

    // Simulate a robot path
    let path: Vec<Point3> = (0..50)
        .map(|i| {
            let t = i as f32 / 50.0;
            Point3::new(t * 4.0, 1.5, 0.3) // Horizontal path at z=0.3
        })
        .collect();

    let robot_radius = 0.1;
    let collisions = grid.check_collisions_batch(&path, robot_radius);

    // Count collisions
    let collision_count = collisions.iter().filter(|&&c| c).count();

    // Path should mostly be collision-free (above floor)
    // But may intersect pillars depending on configuration
    println!(
        "Path collision check: {}/{} points in collision",
        collision_count,
        path.len()
    );
}

#[test]
fn test_gradient_for_motion_planning() {
    let grid = create_warehouse_grid();

    // Get gradient at a point - useful for potential field navigation
    let robot_pos = Point3::new(0.5, 0.5, 0.3);

    if let Some((sdf, gradient)) = grid.query_with_gradient(robot_pos) {
        println!("SDF at robot: {}", sdf);
        println!("Gradient: {:?}", gradient);

        // Gradient should point away from nearest surface
        let grad_mag = (gradient[0].powi(2) + gradient[1].powi(2) + gradient[2].powi(2)).sqrt();
        println!("Gradient magnitude: {}", grad_mag);

        // For valid SDF, gradient should be approximately unit length
        // (may not be exact due to discretization)
        assert!(grad_mag > 0.1, "Gradient should be non-zero");
    }
}

#[test]
fn test_raycast_for_sensing() {
    let grid = create_warehouse_grid();

    // Simulate a depth sensor ray
    let sensor_pos = Point3::new(0.5, 0.5, 1.0);
    let ray_direction = Point3::new(0.0, 0.0, -1.0); // Looking down

    if let Some((distance, hit_point)) = grid.raycast(sensor_pos, ray_direction, 2.0, 0.01) {
        println!("Raycast hit at distance {}", distance);
        println!("Hit point: {:?}", hit_point);

        // Should hit the floor around z=0
        assert!(hit_point.z < 0.2, "Should hit floor: {:?}", hit_point);
    }
}

#[test]
#[cfg(any(feature = "std", feature = "alloc"))]
fn test_mesh_extraction() {
    let center = Point3::new(0.4, 0.4, 0.4);
    let radius = 0.3;

    let grid = GridBuilder::new(8, 0.1)
        .with_capacity(8)
        .add_block_fn(BlockCoord::new(0, 0, 0), |pos| (pos - center).length() - radius)
        .build()
        .unwrap();

    let triangles = grid.extract_block_mesh(BlockCoord::new(0, 0, 0), 0.0);

    println!("Generated {} triangles", triangles.len());

    // Verify triangle validity
    for tri in &triangles {
        for v in tri {
            assert!(v.x.is_finite());
            assert!(v.y.is_finite());
            assert!(v.z.is_finite());
        }
    }

    // Compute mesh stats
    let stats = ash_rs::MeshStats::from_triangles(&triangles);
    println!("Surface area: {}", stats.surface_area);
    println!("Bounding box: {:?} to {:?}", stats.bbox_min, stats.bbox_max);
}

#[test]
fn test_batch_query_performance() {
    let grid = create_warehouse_grid();

    // Generate many random points
    let points: Vec<Point3> = (0..1000)
        .map(|i| {
            let t = i as f32 / 1000.0;
            let x = (t * 7.0 + i as f32 * 0.1).sin() * 2.0 + 2.0;
            let y = (t * 11.0 + i as f32 * 0.1).cos() * 1.5 + 1.5;
            let z = (t * 13.0).sin().abs() * 1.5;
            Point3::new(x, y, z)
        })
        .collect();

    let start = std::time::Instant::now();
    let results = grid.query_batch(&points);
    let batch_time = start.elapsed();

    let start = std::time::Instant::now();
    let _sequential: Vec<_> = points.iter().map(|p| grid.query(*p)).collect();
    let seq_time = start.elapsed();

    println!("Batch query (1000 points): {:?}", batch_time);
    println!("Sequential query (1000 points): {:?}", seq_time);
    println!("Valid results: {}/{}", results.num_valid(), results.len());

    // Note: With scalar fallback, batch may not be faster than sequential
    // Full SIMD implementation would show significant speedup
}

#[test]
fn test_bounding_box() {
    let grid = create_warehouse_grid();

    if let Some((min, max)) = grid.bounding_box() {
        println!("Grid bounds: {:?} to {:?}", min, max);

        // Bounds should be positive (warehouse starts at origin)
        assert!(min.x >= 0.0);
        assert!(min.y >= 0.0);
        assert!(max.x > min.x);
        assert!(max.y > min.y);
    }
}

#[test]
fn test_surface_normal_at_floor() {
    let grid = create_warehouse_grid();

    // Near the floor, normal should point up (+Z)
    let near_floor = Point3::new(0.5, 0.5, 0.02);

    if let Some(normal) = grid.surface_normal(near_floor) {
        println!("Floor normal: {:?}", normal);

        // Should be approximately (0, 0, 1) or (0, 0, -1) depending on SDF convention
        assert!(
            normal.z.abs() > 0.5,
            "Floor normal should be mostly vertical: {:?}",
            normal
        );
    }
}
