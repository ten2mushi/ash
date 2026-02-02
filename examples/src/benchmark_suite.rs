//! Comprehensive Benchmark Suite for ASH Ecosystem
//!
//! This benchmark measures all critical performance characteristics:
//!
//! 1. **Query Performance**: Single-point and batch queries
//! 2. **Gradient Computation**: Query with analytical gradients
//! 3. **I/O Performance**: Serialization and deserialization
//! 4. **Mesh Extraction**: Marching cubes performance
//! 5. **Memory Efficiency**: Memory usage analysis
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin benchmark_suite
//! cargo run --release --bin benchmark_suite -- --large  # Run with larger dataset
//! ```

use std::env;

use instant::Instant;

use ash_rs::{GridBuilder, SparseDenseGrid};
use ash_io::{save_grid, load_grid, compute_shared_size};
use ash_core::{BlockCoord, Point3};

/// Benchmark configuration
struct BenchConfig {
    /// Grid dimension (cells per block side)
    grid_dim: u32,
    /// Cell size in world units
    cell_size: f32,
    /// Number of blocks per side (total blocks = n³)
    blocks_per_side: i32,
    /// Number of random query points
    num_query_points: usize,
    /// Number of iterations for timing
    iterations: usize,
}

impl BenchConfig {
    fn small() -> Self {
        Self {
            grid_dim: 8,
            cell_size: 0.1,
            blocks_per_side: 4,
            num_query_points: 10_000,
            iterations: 100,
        }
    }

    fn large() -> Self {
        Self {
            grid_dim: 8,
            cell_size: 0.1,
            blocks_per_side: 10,
            num_query_points: 100_000,
            iterations: 50,
        }
    }

    fn total_blocks(&self) -> usize {
        let n = self.blocks_per_side as usize;
        n * n * n
    }

    fn cells_per_block(&self) -> usize {
        let d = self.grid_dim as usize;
        d * d * d
    }
}

/// Simple deterministic pseudo-random number generator
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

fn create_test_grid(config: &BenchConfig) -> SparseDenseGrid {
    let center = Point3::new(
        config.blocks_per_side as f32 * config.cell_size * config.grid_dim as f32 * 0.5,
        config.blocks_per_side as f32 * config.cell_size * config.grid_dim as f32 * 0.5,
        config.blocks_per_side as f32 * config.cell_size * config.grid_dim as f32 * 0.5,
    );
    let radius = center.x * 0.8;

    let mut builder = GridBuilder::new(config.grid_dim, config.cell_size)
        .with_capacity(config.total_blocks());

    for bz in 0..config.blocks_per_side {
        for by in 0..config.blocks_per_side {
            for bx in 0..config.blocks_per_side {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| {
                    (pos - center).length() - radius
                });
            }
        }
    }

    builder.build().unwrap()
}

fn generate_query_points(config: &BenchConfig, rng: &mut SimpleRng) -> Vec<Point3> {
    let extent = config.blocks_per_side as f32 * config.cell_size * config.grid_dim as f32;

    (0..config.num_query_points)
        .map(|_| {
            Point3::new(
                rng.next_f32() * extent,
                rng.next_f32() * extent,
                rng.next_f32() * extent,
            )
        })
        .collect()
}

fn benchmark_single_query(grid: &SparseDenseGrid, points: &[Point3], iterations: usize) -> (f64, f64) {
    let mut total_time = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        for point in points {
            let _ = grid.query(*point);
        }
        total_time += start.elapsed().as_secs_f64();
    }

    let avg_time = total_time / iterations as f64;
    let per_query_ns = (avg_time * 1e9) / points.len() as f64;

    (avg_time, per_query_ns)
}

fn benchmark_batch_query(grid: &SparseDenseGrid, points: &[Point3], iterations: usize) -> (f64, f64, usize) {
    let mut total_time = 0.0;
    let mut total_valid = 0usize;

    for _ in 0..iterations {
        let start = Instant::now();
        let results = grid.query_batch(points);
        total_time += start.elapsed().as_secs_f64();
        total_valid += results.num_valid();
    }

    let avg_time = total_time / iterations as f64;
    let per_point_ns = (avg_time * 1e9) / points.len() as f64;
    let avg_valid = total_valid / iterations;

    (avg_time, per_point_ns, avg_valid)
}

fn benchmark_gradient_query(grid: &SparseDenseGrid, points: &[Point3], iterations: usize) -> (f64, f64) {
    let mut total_time = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        for point in points {
            let _ = grid.query_with_gradient(*point);
        }
        total_time += start.elapsed().as_secs_f64();
    }

    let avg_time = total_time / iterations as f64;
    let per_query_ns = (avg_time * 1e9) / points.len() as f64;

    (avg_time, per_query_ns)
}

fn benchmark_serialization(grid: &SparseDenseGrid) -> (f64, f64, usize) {
    // Serialize
    let start = Instant::now();
    let mut buffer = Vec::new();
    save_grid(grid.inner(), &mut buffer).unwrap();
    let serialize_time = start.elapsed().as_secs_f64();

    let file_size = buffer.len();

    // Deserialize
    let start = Instant::now();
    let mut cursor = std::io::Cursor::new(&buffer);
    let _loaded: ash_io::InMemoryGrid<1> = load_grid(&mut cursor).unwrap();
    let deserialize_time = start.elapsed().as_secs_f64();

    (serialize_time, deserialize_time, file_size)
}

fn benchmark_mesh_extraction(grid: &SparseDenseGrid) -> (f64, usize) {
    let start = Instant::now();
    let triangles = grid.extract_mesh(0.0);
    let time = start.elapsed().as_secs_f64();

    (time, triangles.len())
}

fn benchmark_collision_check(grid: &SparseDenseGrid, points: &[Point3], iterations: usize) -> (f64, f64) {
    let mut total_time = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        for point in points {
            let _ = grid.in_collision(*point, 0.1);
        }
        total_time += start.elapsed().as_secs_f64();
    }

    let avg_time = total_time / iterations as f64;
    let per_check_ns = (avg_time * 1e9) / points.len() as f64;

    (avg_time, per_check_ns)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = if args.iter().any(|a| a == "--large") {
        println!("Running LARGE benchmark configuration...");
        BenchConfig::large()
    } else {
        println!("Running SMALL benchmark configuration (use --large for bigger dataset)");
        BenchConfig::small()
    };

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("              ASH Benchmark Suite");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Configuration
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Configuration                                               │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("  Grid dimension:    {}³ cells/block", config.grid_dim);
    println!("  Cell size:         {}", config.cell_size);
    println!("  Blocks:            {}³ = {}", config.blocks_per_side, config.total_blocks());
    println!("  Total cells:       {}", config.total_blocks() * config.cells_per_block());
    println!("  Query points:      {}", config.num_query_points);
    println!("  Iterations:        {}", config.iterations);
    println!();

    // Create grid
    println!("Creating test grid...");
    let start = Instant::now();
    let grid = create_test_grid(&config);
    let creation_time = start.elapsed();
    println!("  Created {} blocks in {:.3}s", grid.num_blocks(), creation_time.as_secs_f64());
    println!();

    // Generate query points
    let mut rng = SimpleRng::new(12345);
    let points = generate_query_points(&config, &mut rng);

    // =========================================================================
    // Query Benchmarks
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Query Performance                                           │");
    println!("└─────────────────────────────────────────────────────────────┘");

    // Single-point query
    let (single_time, single_ns) = benchmark_single_query(&grid, &points, config.iterations);
    println!("  Single-point query:");
    println!("    Total time:     {:.3}s for {} points", single_time, points.len());
    println!("    Per query:      {:.1} ns", single_ns);
    println!("    Target:         < 100 ns  {}", if single_ns < 100.0 { "✓" } else { "✗" });
    println!();

    // Batch query
    let (batch_time, batch_ns, valid_count) = benchmark_batch_query(&grid, &points, config.iterations);
    println!("  Batch query (SIMD):");
    println!("    Total time:     {:.3}s for {} points", batch_time, points.len());
    println!("    Per point:      {:.1} ns", batch_ns);
    println!("    Valid results:  {} / {}", valid_count, points.len());
    println!("    Speedup vs single: {:.2}x", single_ns / batch_ns);
    println!();

    // Gradient query
    let (grad_time, grad_ns) = benchmark_gradient_query(&grid, &points, config.iterations);
    println!("  Gradient query:");
    println!("    Total time:     {:.3}s for {} points", grad_time, points.len());
    println!("    Per query:      {:.1} ns", grad_ns);
    println!("    Target:         < 200 ns  {}", if grad_ns < 200.0 { "✓" } else { "✗" });
    println!();

    // Collision check
    let (coll_time, coll_ns) = benchmark_collision_check(&grid, &points, config.iterations);
    println!("  Collision check:");
    println!("    Total time:     {:.3}s for {} points", coll_time, points.len());
    println!("    Per check:      {:.1} ns", coll_ns);
    println!();

    // =========================================================================
    // I/O Benchmarks
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ I/O Performance                                             │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let (ser_time, deser_time, file_size) = benchmark_serialization(&grid);

    println!("  File size:        {} bytes ({:.2} MB)",
             file_size, file_size as f64 / 1_000_000.0);
    println!();
    println!("  Serialization:");
    println!("    Time:           {:.3}s", ser_time);
    println!("    Speed:          {:.2} MB/s", (file_size as f64 / 1_000_000.0) / ser_time);
    println!();
    println!("  Deserialization:");
    println!("    Time:           {:.3}s", deser_time);
    println!("    Speed:          {:.2} MB/s", (file_size as f64 / 1_000_000.0) / deser_time);
    println!();

    // =========================================================================
    // Mesh Extraction Benchmark
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Mesh Extraction Performance                                 │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let (mesh_time, tri_count) = benchmark_mesh_extraction(&grid);

    println!("  Triangles:        {}", tri_count);
    println!("  Extraction time:  {:.3}s", mesh_time);
    println!("  Throughput:       {:.0} triangles/sec", tri_count as f64 / mesh_time);
    println!();

    // =========================================================================
    // Memory Analysis
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Memory Analysis                                             │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let cells_per_block = config.cells_per_block();
    let bytes_per_block = cells_per_block * 4; // f32 = 4 bytes
    let total_data_bytes = grid.num_blocks() * bytes_per_block;

    println!("  Blocks:           {}", grid.num_blocks());
    println!("  Cells per block:  {}", cells_per_block);
    println!("  Bytes per block:  {} bytes ({:.1} KB)", bytes_per_block, bytes_per_block as f64 / 1024.0);
    println!("  Total data:       {} bytes ({:.2} MB)",
             total_data_bytes, total_data_bytes as f64 / 1_000_000.0);
    println!();

    // Shared memory size
    let shared_size = compute_shared_size::<1>(config.grid_dim, grid.num_blocks());
    println!("  Shared memory size (for {} blocks): {} bytes ({:.2} MB)",
             grid.num_blocks(), shared_size, shared_size as f64 / 1_000_000.0);
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("                       SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Performance Targets:");
    println!("    ┌──────────────────────┬─────────────┬─────────────┐");
    println!("    │ Metric               │ Target      │ Measured    │");
    println!("    ├──────────────────────┼─────────────┼─────────────┤");
    println!("    │ Single query         │ < 100 ns    │ {:.1} ns   │", single_ns);
    println!("    │ Batch query (1000)   │ < 50 μs     │ {:.1} μs   │",
             (batch_ns * 1000.0) / 1000.0);
    println!("    │ Gradient query       │ < 200 ns    │ {:.1} ns   │", grad_ns);
    println!("    │ File load (10 MB)    │ < 50 ms     │ {:.1} ms   │",
             deser_time * 1000.0 * (10_000_000.0 / file_size as f64));
    println!("    └──────────────────────┴─────────────┴─────────────┘");
    println!();
    println!("═══════════════════════════════════════════════════════════════");
}
