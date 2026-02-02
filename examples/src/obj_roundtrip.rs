//! OBJ → ASH → OBJ Roundtrip Example
//!
//! This example demonstrates:
//! 1. Loading an OBJ mesh file
//! 2. Converting it to an SDF grid (ASH format)
//! 3. Saving the grid to .ash file
//! 4. Loading the grid back
//! 5. Extracting the isosurface back to OBJ
//!
//! This validates the entire pipeline and provides benchmarks.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin obj_roundtrip -- input/landscape.obj output/roundtrip.obj
//! ```

use std::env;
use std::fs::File;
use std::io::{BufWriter, Cursor};
use std::path::Path;

use instant::Instant;

use ash_io::{
    parse_obj_file, import_obj_to_grid, import_obj_narrow_band, export_obj_with_stats,
    save_grid, load_grid, ObjImportConfig, ObjExportConfig,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("            ASH OBJ Roundtrip Showcase");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (input_path, output_path, ash_path) = if args.len() >= 3 {
        (
            args[1].clone(),
            args[2].clone(),
            args.get(3).cloned().unwrap_or_else(|| "output/roundtrip.ash".to_string()),
        )
    } else {
        println!("Usage: {} <input.obj> <output.obj> [intermediate.ash]", args[0]);
        println!();
        println!("Using default: input/landscape.obj → output/roundtrip.obj");
        (
            "input/landscape.obj".to_string(),
            "output/roundtrip.obj".to_string(),
            "output/roundtrip.ash".to_string(),
        )
    };

    // Ensure output directory exists
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Some(parent) = Path::new(&ash_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // =========================================================================
    // Step 1: Parse OBJ file
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Parse OBJ File                                      │");
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

    // Configure based on mesh size
    let mesh_extent = bbox_max - bbox_min;
    let max_extent = mesh_extent.x.max(mesh_extent.y).max(mesh_extent.z);

    // Choose cell size to get reasonable resolution
    let cell_size = max_extent / 100.0; // ~100 cells across largest dimension
    let grid_dim = 8;

    let config = ObjImportConfig {
        cell_size,
        grid_dim,
        padding: cell_size * 5.0,
        max_blocks: 50_000,
    };

    println!("  Cell size:       {:.4}", config.cell_size);
    println!("  Grid dimension:  {}³", config.grid_dim);
    println!("  Padding:         {:.4}", config.padding);
    println!("  Max blocks:      {}", config.max_blocks);
    println!();

    let start = Instant::now();
    let mut last_progress = 0;

    // Use optimized narrow band + BVH import for large meshes
    let use_optimized = mesh.triangles.len() > 1000;

    let (grid, import_stats) = if use_optimized {
        println!("  Using optimized narrow band + BVH import");
        match import_obj_narrow_band(&mesh, &config, true, Some(|done: usize, total: usize| {
            let progress = (done * 100) / total.max(1);
            if progress >= last_progress + 10 {
                print!("  Progress: {}%\r", progress);
                last_progress = progress;
            }
        })) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("\nError converting to SDF: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        match import_obj_to_grid(&mesh, &config, Some(|done: usize, total: usize| {
            let progress = (done * 100) / total.max(1);
            if progress >= last_progress + 10 {
                print!("  Progress: {}%\r", progress);
                last_progress = progress;
            }
        })) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("\nError converting to SDF: {}", e);
                std::process::exit(1);
            }
        }
    };
    let convert_time = start.elapsed();

    println!("  Progress: 100%                    ");
    println!("  Blocks allocated: {}", import_stats.blocks_allocated);
    println!("  Cells processed:  {}", import_stats.cells_processed);
    println!("  Convert time:     {:.3}s", convert_time.as_secs_f64());
    println!("  Throughput:       {:.2} cells/sec",
             import_stats.cells_processed as f64 / convert_time.as_secs_f64());
    println!();

    // =========================================================================
    // Step 3: Save to ASH file
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Save to ASH File                                    │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut buffer = Vec::new();
    if let Err(e) = save_grid(&grid, &mut buffer) {
        eprintln!("Error serializing grid: {}", e);
        std::process::exit(1);
    }
    let serialize_time = start.elapsed();

    let file_size = buffer.len();

    // Save to file
    if let Err(e) = std::fs::write(&ash_path, &buffer) {
        eprintln!("Error writing ASH file: {}", e);
        std::process::exit(1);
    }

    println!("  Output file:     {}", ash_path);
    println!("  File size:       {} bytes ({:.2} MB)",
             file_size, file_size as f64 / 1_000_000.0);
    println!("  Serialize time:  {:.3}s", serialize_time.as_secs_f64());
    println!("  Write speed:     {:.2} MB/s",
             (file_size as f64 / 1_000_000.0) / serialize_time.as_secs_f64());
    println!();

    // =========================================================================
    // Step 4: Load from ASH file
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Load from ASH File                                  │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut cursor = Cursor::new(&buffer);
    let loaded_grid = match load_grid::<1, _>(&mut cursor) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading grid: {}", e);
            std::process::exit(1);
        }
    };
    let load_time = start.elapsed();

    println!("  Blocks loaded:   {}", loaded_grid.num_blocks());
    println!("  Load time:       {:.3}s", load_time.as_secs_f64());
    println!("  Read speed:      {:.2} MB/s",
             (file_size as f64 / 1_000_000.0) / load_time.as_secs_f64());
    println!();

    // Verify data integrity
    assert_eq!(grid.num_blocks(), loaded_grid.num_blocks(),
               "Block count mismatch after roundtrip!");
    println!("  ✓ Data integrity verified");
    println!();

    // =========================================================================
    // Step 5: Extract mesh and export to OBJ
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Extract Mesh and Export to OBJ                      │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating output file: {}", e);
            std::process::exit(1);
        }
    };
    let mut writer = BufWriter::new(file);

    let export_stats = match export_obj_with_stats(&loaded_grid, &mut writer, ObjExportConfig::default()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error exporting OBJ: {}", e);
            std::process::exit(1);
        }
    };
    let export_time = start.elapsed();

    println!("  Output file:     {}", output_path);
    println!("  Vertices:        {}", export_stats.vertex_count);
    println!("  Triangles:       {}", export_stats.triangle_count);
    println!("  Export time:     {:.3}s", export_time.as_secs_f64());
    println!("  Throughput:      {:.2} triangles/sec",
             export_stats.triangle_count as f64 / export_time.as_secs_f64());
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    let total_time = parse_time + convert_time + serialize_time + load_time + export_time;
    println!("  Total time:      {:.3}s", total_time.as_secs_f64());
    println!();
    println!("  Breakdown:");
    println!("    Parse OBJ:     {:.3}s ({:.1}%)",
             parse_time.as_secs_f64(),
             (parse_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("    Convert SDF:   {:.3}s ({:.1}%)",
             convert_time.as_secs_f64(),
             (convert_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("    Serialize:     {:.3}s ({:.1}%)",
             serialize_time.as_secs_f64(),
             (serialize_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("    Load:          {:.3}s ({:.1}%)",
             load_time.as_secs_f64(),
             (load_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("    Export OBJ:    {:.3}s ({:.1}%)",
             export_time.as_secs_f64(),
             (export_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!();
    println!("═══════════════════════════════════════════════════════════════");
}
