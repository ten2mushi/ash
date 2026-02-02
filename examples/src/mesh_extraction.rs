//! Mesh Extraction Benchmark
//!
//! This example demonstrates using ash_rs for parallel mesh extraction:
//! 1. Load an OBJ mesh
//! 2. Convert to SDF grid
//! 3. Use ash_rs parallel mesh extraction
//! 4. Compare with sequential extraction
//! 5. Export results
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin mesh_extraction -- input/gyroid.obj output/extracted.obj
//! ```

use std::env;
use std::path::Path;

use instant::Instant;

use ash_io::{
    parse_obj_file, import_obj_narrow_band, ObjImportConfig,
};
use ash_rs::{SparseDenseGrid, triangles_to_obj, MeshStats};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("          ASH Mesh Extraction Benchmark");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (input_path, output_path) = if args.len() >= 3 {
        (args[1].clone(), args[2].clone())
    } else {
        println!("Usage: {} <input.obj> <output.obj>", args[0]);
        println!();
        println!("Using default: input/landscape.obj → output/extracted.obj");
        (
            "input/landscape.obj".to_string(),
            "output/extracted.obj".to_string(),
        )
    };

    // Ensure output directory exists
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // =========================================================================
    // Step 1: Parse OBJ file
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Parse Input Mesh                                    │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mesh = match parse_obj_file(&input_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing OBJ file '{}': {}", input_path, e);
            std::process::exit(1);
        }
    };
    let parse_time = start.elapsed();

    let (bbox_min, bbox_max) = mesh.bounding_box();
    println!("  Input file:      {}", input_path);
    println!("  Vertices:        {}", mesh.vertices.len());
    println!("  Triangles:       {}", mesh.triangles.len());
    println!("  Bounding box:    ({:.2}, {:.2}, {:.2}) → ({:.2}, {:.2}, {:.2})",
             bbox_min.x, bbox_min.y, bbox_min.z,
             bbox_max.x, bbox_max.y, bbox_max.z);
    println!("  Parse time:      {:.3}s", parse_time.as_secs_f64());
    println!();

    // =========================================================================
    // Step 2: Convert to SDF grid
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Convert to SDF Grid                                 │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let mesh_extent = bbox_max - bbox_min;
    let max_extent = mesh_extent.x.max(mesh_extent.y).max(mesh_extent.z);

    // Higher resolution for mesh extraction benchmark
    let cell_size = max_extent / 150.0;
    let grid_dim = 8;

    let config = ObjImportConfig {
        cell_size,
        grid_dim,
        padding: cell_size * 3.0,
        max_blocks: 100_000,
    };

    println!("  Cell size:       {:.4}", config.cell_size);
    println!("  Grid dimension:  {}³", config.grid_dim);
    println!();

    println!("  Using optimized narrow band + BVH import");
    println!();

    let start = Instant::now();
    let (io_grid, import_stats) = match import_obj_narrow_band(&mesh, &config, true, Some(|done: usize, total: usize| {
        let progress = (done * 100) / total;
        if progress % 10 == 0 {
            print!("  Progress: {}%\r", progress);
        }
    })) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("\nError converting to SDF: {}", e);
            std::process::exit(1);
        }
    };
    let convert_time = start.elapsed();

    println!("  Progress: 100%                    ");
    println!("  Blocks:          {}", import_stats.blocks_allocated);
    println!("  Cells:           {}", import_stats.cells_processed);
    println!("  Convert time:    {:.3}s", convert_time.as_secs_f64());
    println!();

    // Convert InMemoryGrid<1> to SparseDenseGrid
    let grid = SparseDenseGrid::from_inner(io_grid);

    // =========================================================================
    // Step 3: Mesh Extraction Benchmarks
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Mesh Extraction Benchmarks                          │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // Sequential extraction (single-threaded)
    println!("  [Sequential Extraction]");
    let start = Instant::now();
    let mut triangles_seq = Vec::new();
    for coord in grid.block_coords() {
        let block_tris = grid.extract_block_mesh(coord, 0.0);
        triangles_seq.extend(block_tris);
    }
    let seq_time = start.elapsed();
    println!("    Triangles:     {}", triangles_seq.len());
    println!("    Time:          {:.3}s", seq_time.as_secs_f64());
    println!("    Throughput:    {:.0} triangles/sec",
             triangles_seq.len() as f64 / seq_time.as_secs_f64());
    println!();

    // Parallel extraction (using rayon via ash_rs)
    println!("  [Parallel Extraction]");
    let start = Instant::now();
    let triangles_par = grid.extract_mesh(0.0);
    let par_time = start.elapsed();
    println!("    Triangles:     {}", triangles_par.len());
    println!("    Time:          {:.3}s", par_time.as_secs_f64());
    println!("    Throughput:    {:.0} triangles/sec",
             triangles_par.len() as f64 / par_time.as_secs_f64());
    println!("    Speedup:       {:.2}x", seq_time.as_secs_f64() / par_time.as_secs_f64());
    println!();

    // Callback-based extraction (no allocation)
    println!("  [Callback Extraction (no alloc)]");
    let start = Instant::now();
    let mut callback_count = 0;
    grid.extract_mesh_callback(0.0, |_tri| {
        callback_count += 1;
    });
    let callback_time = start.elapsed();
    println!("    Triangles:     {}", callback_count);
    println!("    Time:          {:.3}s", callback_time.as_secs_f64());
    println!("    Throughput:    {:.0} triangles/sec",
             callback_count as f64 / callback_time.as_secs_f64());
    println!();

    // =========================================================================
    // Step 4: Mesh Statistics
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Mesh Statistics                                     │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let stats = MeshStats::from_triangles(&triangles_par);
    println!("  Triangle count:  {}", stats.triangle_count);
    println!("  Vertex count:    {}", stats.vertex_count);
    println!("  Surface area:    {:.4}", stats.surface_area);
    println!("  Bounding box:    ({:.2}, {:.2}, {:.2}) → ({:.2}, {:.2}, {:.2})",
             stats.bbox_min.x, stats.bbox_min.y, stats.bbox_min.z,
             stats.bbox_max.x, stats.bbox_max.y, stats.bbox_max.z);
    println!();

    // =========================================================================
    // Step 5: Export to OBJ
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Export to OBJ                                       │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let obj_content = triangles_to_obj(&triangles_par);
    let generation_time = start.elapsed();

    let start = Instant::now();
    if let Err(e) = std::fs::write(&output_path, &obj_content) {
        eprintln!("Error writing output file: {}", e);
        std::process::exit(1);
    }
    let write_time = start.elapsed();

    let file_size = obj_content.len();
    println!("  Output file:     {}", output_path);
    println!("  File size:       {} bytes ({:.2} MB)",
             file_size, file_size as f64 / 1_000_000.0);
    println!("  OBJ generation:  {:.3}s", generation_time.as_secs_f64());
    println!("  File write:      {:.3}s", write_time.as_secs_f64());
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");

    let total_time = parse_time + convert_time + par_time + generation_time + write_time;

    println!();
    println!("  Input → Output Pipeline:");
    println!("    {} vertices → {} blocks → {} triangles",
             mesh.vertices.len(), grid.num_blocks(), triangles_par.len());
    println!();
    println!("  Total time (parallel):  {:.3}s", total_time.as_secs_f64());
    println!();
    println!("  Mesh extraction comparison:");
    println!("    Sequential:    {:.3}s", seq_time.as_secs_f64());
    println!("    Parallel:      {:.3}s ({:.2}x speedup)",
             par_time.as_secs_f64(),
             seq_time.as_secs_f64() / par_time.as_secs_f64());
    println!("    Callback:      {:.3}s", callback_time.as_secs_f64());
    println!();
    println!("═══════════════════════════════════════════════════════════════");
}
