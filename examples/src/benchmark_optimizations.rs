//! Benchmark Suite for ASH Performance Optimizations
//!
//! This benchmark compares:
//! 1. SIMD vs scalar hash computation
//! 2. Batch vs sequential block lookups
//! 3. BVH vs brute-force distance queries
//! 4. Narrow band vs full bounding box import
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p ash_examples --bin benchmark_optimizations
//! ```

use std::time::Instant;

use ash_core::{BlockCoord, Point3};
use ash_io::{
    BlockMap, NarrowBandConfig, TriangleBvh, narrow_band_from_triangles,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("         ASH Performance Optimization Benchmarks");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // 1. Hash benchmarks
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 1. Hash Computation Benchmarks                              │");
    println!("└─────────────────────────────────────────────────────────────┘");
    bench_hash(1000);
    bench_hash(10000);
    println!();

    // 2. Batch lookup benchmarks
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 2. Block Lookup Benchmarks                                  │");
    println!("└─────────────────────────────────────────────────────────────┘");
    bench_lookup(100);
    bench_lookup(1000);
    println!();

    // 3. BVH benchmarks
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3. BVH Distance Query Benchmarks                            │");
    println!("└─────────────────────────────────────────────────────────────┘");
    bench_bvh(1000, 1000);
    bench_bvh(5000, 1000);
    bench_bvh(10000, 1000);
    println!();

    // 4. Narrow band benchmarks
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 4. Narrow Band Allocation Benchmarks                        │");
    println!("└─────────────────────────────────────────────────────────────┘");
    bench_narrow_band();
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Run with --release for accurate performance numbers.");
    println!("  Enable 'simd' feature for SIMD hash acceleration.");
    println!("═══════════════════════════════════════════════════════════════");
}

fn bench_hash(count: usize) {
    let coords: Vec<BlockCoord> = (0..count as i32)
        .map(|i| BlockCoord::new(i, i * 2, i * 3))
        .collect();

    // Scalar hash (using fnv1a_32 directly)
    let start = Instant::now();
    let mut scalar_sum: u64 = 0;
    for coord in &coords {
        scalar_sum = scalar_sum.wrapping_add(ash_core::fnv1a_32(*coord) as u64);
    }
    let scalar_time = start.elapsed();

    #[cfg(feature = "simd")]
    {
        use ash_core::hash_batch;

        let start = Instant::now();
        let hashes = hash_batch(&coords);
        let simd_sum: u64 = hashes.iter().map(|&h| h as u64).sum();
        let simd_time = start.elapsed();

        println!(
            "  Hash {} coords: scalar={:.2}μs, SIMD={:.2}μs, speedup={:.2}x",
            count,
            scalar_time.as_secs_f64() * 1_000_000.0,
            simd_time.as_secs_f64() * 1_000_000.0,
            scalar_time.as_secs_f64() / simd_time.as_secs_f64()
        );

        // Verify results match
        assert_eq!(scalar_sum, simd_sum, "Hash mismatch!");
    }

    #[cfg(not(feature = "simd"))]
    {
        println!(
            "  Hash {} coords: scalar={:.2}μs (SIMD not enabled)",
            count,
            scalar_time.as_secs_f64() * 1_000_000.0
        );
        let _ = scalar_sum; // Silence unused warning
    }
}

fn bench_lookup(count: usize) {
    // Create a block map with some entries
    let map = BlockMap::with_capacity(count * 2);
    let coords: Vec<BlockCoord> = (0..count as i32)
        .map(|i| BlockCoord::new(i, i * 2, i * 3))
        .collect();

    for (i, &coord) in coords.iter().enumerate() {
        let _ = map.insert(coord, i);
    }

    // Sequential lookup
    let start = Instant::now();
    let mut seq_found = 0;
    for coord in &coords {
        if map.get(*coord).is_some() {
            seq_found += 1;
        }
    }
    let seq_time = start.elapsed();

    // Batch lookup
    let start = Instant::now();
    let result = map.find_batch(&coords);
    let batch_found = result.found_count();
    let batch_time = start.elapsed();

    assert_eq!(seq_found, batch_found, "Lookup count mismatch!");

    println!(
        "  Lookup {} coords: sequential={:.2}μs, batch={:.2}μs, speedup={:.2}x",
        count,
        seq_time.as_secs_f64() * 1_000_000.0,
        batch_time.as_secs_f64() * 1_000_000.0,
        seq_time.as_secs_f64() / batch_time.as_secs_f64()
    );
}

fn bench_bvh(num_triangles: usize, num_queries: usize) {
    // Generate a random mesh
    let (vertices, triangles, _normals) = generate_mesh(num_triangles);

    // Build BVH
    let bvh_start = Instant::now();
    let bvh = TriangleBvh::build(&vertices, &triangles, 8);
    let bvh_build_time = bvh_start.elapsed();

    // Generate query points
    let query_points: Vec<Point3> = (0..num_queries)
        .map(|i| {
            let t = i as f32 / num_queries as f32;
            Point3::new(t * 10.0, (t * 7.0).sin() * 5.0, (t * 11.0).cos() * 5.0)
        })
        .collect();

    // Brute force queries
    let brute_start = Instant::now();
    let mut brute_sum: f64 = 0.0;
    for query in &query_points {
        let mut min_dist_sq = f32::MAX;
        for &[i0, i1, i2] in &triangles {
            let v0 = vertices[i0];
            let v1 = vertices[i1];
            let v2 = vertices[i2];
            let (closest, _) = closest_point_on_triangle(*query, v0, v1, v2);
            let dist_sq = (*query - closest).length_squared();
            min_dist_sq = min_dist_sq.min(dist_sq);
        }
        brute_sum += libm::sqrtf(min_dist_sq) as f64;
    }
    let brute_time = brute_start.elapsed();

    // BVH queries
    let bvh_start = Instant::now();
    let mut bvh_sum: f64 = 0.0;
    for query in &query_points {
        if let Some((_, _, _, dist_sq)) = bvh.nearest_triangle(&vertices, &triangles, *query) {
            bvh_sum += libm::sqrtf(dist_sq) as f64;
        }
    }
    let bvh_query_time = bvh_start.elapsed();

    // Verify results are close (may differ slightly due to numerical precision)
    let diff = (brute_sum - bvh_sum).abs() / brute_sum;
    assert!(diff < 0.01, "BVH/brute force mismatch: diff={:.4}", diff);

    println!(
        "  {} triangles, {} queries: brute={:.2}ms, BVH build={:.2}ms + query={:.2}ms, speedup={:.1}x",
        num_triangles,
        num_queries,
        brute_time.as_secs_f64() * 1000.0,
        bvh_build_time.as_secs_f64() * 1000.0,
        bvh_query_time.as_secs_f64() * 1000.0,
        brute_time.as_secs_f64() / bvh_query_time.as_secs_f64()
    );
}

fn bench_narrow_band() {
    // Create a sparse mesh (single large triangle in a big bounding box)
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(5.0, 10.0, 0.0),
    ];
    let triangles = vec![[0, 1, 2]];

    // Config: small blocks for fine resolution
    let config = NarrowBandConfig::new(0.5, 4, 1); // block_size = 2.0

    // Compute narrow band
    let start = Instant::now();
    let narrow_blocks = narrow_band_from_triangles(&vertices, &triangles, &config);
    let narrow_time = start.elapsed();

    // Compute full bounding box blocks
    let block_size = config.block_size();
    let bbox_min = Point3::new(0.0, 0.0, 0.0);
    let bbox_max = Point3::new(10.0, 10.0, 0.0);

    // Add padding (dilation = 1 block)
    let full_blocks_x = ((bbox_max.x - bbox_min.x) / block_size).ceil() as usize + 2;
    let full_blocks_y = ((bbox_max.y - bbox_min.y) / block_size).ceil() as usize + 2;
    let full_blocks_z = 3; // One block for z=0 + dilation
    let full_blocks_total = full_blocks_x * full_blocks_y * full_blocks_z;

    println!("  Sparse triangle (10x10 bbox, dilation=1):");
    println!("    Narrow band:  {} blocks", narrow_blocks.len());
    println!("    Full bbox:    {} blocks ({}x{}x{})", full_blocks_total, full_blocks_x, full_blocks_y, full_blocks_z);
    println!("    Reduction:    {:.1}%", 100.0 * (1.0 - narrow_blocks.len() as f64 / full_blocks_total as f64));
    println!("    Compute time: {:.2}μs", narrow_time.as_secs_f64() * 1_000_000.0);

    // Also test cube mesh for comparison
    let (cube_vertices, cube_triangles) = make_cube_mesh();
    let cube_config = NarrowBandConfig::new(0.05, 8, 1); // block_size = 0.4

    let start = Instant::now();
    let cube_narrow_blocks = narrow_band_from_triangles(&cube_vertices, &cube_triangles, &cube_config);
    let cube_time = start.elapsed();

    let cube_block_size = cube_config.block_size();
    let cube_full_x = (1.0 / cube_block_size).ceil() as usize + 2;
    let cube_full_total = cube_full_x.pow(3);

    println!();
    println!("  Unit cube (surface only, dilation=1):");
    println!("    Narrow band:  {} blocks", cube_narrow_blocks.len());
    println!("    Full bbox:    {} blocks ({}³)", cube_full_total, cube_full_x);
    println!("    Compute time: {:.2}μs", cube_time.as_secs_f64() * 1_000_000.0);
}

// Helper functions

fn generate_mesh(num_triangles: usize) -> (Vec<Point3>, Vec<[usize; 3]>, Vec<Point3>) {
    // Generate a simple mesh: random triangles in a 10x10x10 volume
    let mut vertices = Vec::with_capacity(num_triangles * 3);
    let mut triangles = Vec::with_capacity(num_triangles);

    for i in 0..num_triangles {
        let base_x = (i % 10) as f32;
        let base_y = ((i / 10) % 10) as f32;
        let base_z = (i / 100) as f32;

        let v_idx = vertices.len();
        vertices.push(Point3::new(base_x, base_y, base_z));
        vertices.push(Point3::new(base_x + 0.5, base_y, base_z));
        vertices.push(Point3::new(base_x + 0.25, base_y + 0.5, base_z));

        triangles.push([v_idx, v_idx + 1, v_idx + 2]);
    }

    // Compute normals
    let mut normals = vec![Point3::new(0.0, 0.0, 0.0); vertices.len()];
    for &[i0, i1, i2] in &triangles {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let n = e1.cross(e2);
        normals[i0] = normals[i0] + n;
        normals[i1] = normals[i1] + n;
        normals[i2] = normals[i2] + n;
    }
    for n in &mut normals {
        let len = n.length();
        if len > 1e-10 {
            *n = Point3::new(n.x / len, n.y / len, n.z / len);
        }
    }

    (vertices, triangles, normals)
}

fn make_cube_mesh() -> (Vec<Point3>, Vec<[usize; 3]>) {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
    ];

    let triangles = vec![
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 5, 1],
        [0, 4, 5],
        [3, 2, 6],
        [3, 6, 7],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ];

    (vertices, triangles)
}

fn closest_point_on_triangle(p: Point3, a: Point3, b: Point3, c: Point3) -> (Point3, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let point = a + ab * v;
        return (point, [1.0 - v, v, 0.0]);
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let point = a + ac * w;
        return (point, [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let point = b + (c - b) * w;
        return (point, [0.0, 1.0 - w, w]);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let point = a + ab * v + ac * w;
    (point, [1.0 - v - w, v, w])
}
