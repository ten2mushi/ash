//! Example: Training a neural SDF from simulated depth camera data.
//!
//! This example demonstrates a complete robotics-style workflow:
//! 1. Simulate a depth camera capturing a scene from multiple viewpoints
//! 2. Convert depth images to point clouds
//! 3. Initialize the grid using TSDF fusion
//! 4. Train the neural SDF to refine the representation
//! 5. Export the trained grid to .ash and .obj formats
//!
//! # Usage
//!
//! ```bash
//! cargo run -p neural_ash --example denoising_tsdf --features examples
//! ```
//!
//! Output files are saved to `examples/output/`.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use burn::backend::{Autodiff, NdArray};

use neural_ash::{
    config::{DiffGridConfig, PointNetEncoderConfig, SdfDecoderConfig, SdfLossConfig, TrainingConfig},
    data::{generate_sphere_poses, DepthCameraSimulator},
    export::export_to_file,
    training::NeuralSdfTrainer,
    Point3,
};

type MyBackend = Autodiff<NdArray>;

/// Output directory for generated files.
const OUTPUT_DIR: &str = "examples/output";

fn main() {
    // Initialize logging
    env_logger::init();

    // Ensure output directory exists
    if let Err(e) = fs::create_dir_all(OUTPUT_DIR) {
        eprintln!("Warning: Could not create output directory: {}", e);
    }

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    println!("═══════════════════════════════════════════════════════════════");
    println!("          Neural SDF Training with Depth Camera");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // =========================================================================
    // Step 1: Configure the depth camera
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Configuring Depth Camera                           │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let mut camera = DepthCameraSimulator::new(128, 128, 60.0_f32.to_radians())
        .with_depth_range(0.1, 3.0)
        .with_noise(0.002);

    println!("  Resolution:      {}x{}", camera.resolution_x, camera.resolution_y);
    println!("  FOV:             {:.1}°", camera.fov_h.to_degrees());
    println!("  Depth range:     {:.2} - {:.2} m", camera.min_depth, camera.max_depth);
    println!("  Noise sigma:     {:.4} m", camera.noise_sigma);
    println!();

    // =========================================================================
    // Step 2: Generate synthetic scene (sphere)
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Creating Synthetic Scene                           │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let sphere_center = Point3::new(0.0, 0.0, 0.0);
    let sphere_radius = 0.3;

    // Signed distance function for sphere
    let scene_sdf = |p: Point3| -> Option<f32> {
        let dist = (p - sphere_center).length() - sphere_radius;
        Some(dist)
    };

    println!("  Shape:           Sphere");
    println!("  Center:          ({:.2}, {:.2}, {:.2})",
             sphere_center.x, sphere_center.y, sphere_center.z);
    println!("  Radius:          {:.2} m", sphere_radius);
    println!();

    // =========================================================================
    // Step 3: Render depth images from multiple viewpoints
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Rendering Depth Images                             │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let num_views = 16; // More views for full spherical coverage
    let camera_distance = 1.0;

    // Use Fibonacci sphere sampling for uniform coverage from all directions
    let poses = generate_sphere_poses(num_views, camera_distance, sphere_center);

    println!("  Number of views: {} (Fibonacci sphere)", num_views);
    println!("  Camera distance: {:.2} m", camera_distance);
    println!("  Coverage:        Full spherical (top, bottom, sides)");

    let mut frames = Vec::with_capacity(num_views);
    let mut total_pixels = 0;

    for (i, pose) in poses.iter().enumerate() {
        let depth = camera.render_depth(pose, &scene_sdf);
        let valid = depth.valid_count();
        total_pixels += valid;
        println!("  View {}: {} valid pixels", i + 1, valid);
        frames.push((pose.clone(), depth));
    }

    println!("  Total valid pixels: {}", total_pixels);
    println!();

    // =========================================================================
    // Step 4: Convert to point clouds
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Converting to Point Clouds                         │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let mut all_points = Vec::new();
    let mut all_normals = Vec::new();

    for (pose, depth) in &frames {
        let (points, normals) = camera.depth_to_points(depth, pose);
        all_points.extend(points);
        all_normals.extend(normals);
    }

    let (bbox_min, bbox_max) = compute_bounds(&all_points);
    println!("  Total points:    {}", all_points.len());
    println!("  Bounding box:    ({:.2}, {:.2}, {:.2}) → ({:.2}, {:.2}, {:.2})",
             bbox_min.x, bbox_min.y, bbox_min.z,
             bbox_max.x, bbox_max.y, bbox_max.z);
    println!();

    // =========================================================================
    // Step 5: Configure and create trainer
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Configuring Training                               │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let grid_config = DiffGridConfig::new(8, 0.03)
        .with_capacity(5000)
        .with_allocation_margin(0.1)
        .with_init_value(0.1); // Start with small positive SDF

    let loss_config = SdfLossConfig::new()
        .with_surface_weight(3.0)
        .with_free_space_weight(0.5)
        .with_eikonal_weight(0.1)
        .with_free_space_threshold(0.03);

    let config = TrainingConfig::new(
        grid_config,
        PointNetEncoderConfig::new(256),
        SdfDecoderConfig::new(256),
        loss_config,
    )
    .with_learning_rate(0.0001)  // Reduced from 0.01 - was causing divergence
    .with_surface_points_per_batch(512)
    .with_free_space_points_per_batch(256)
    .with_eikonal_points_per_batch(128);

    println!("  Grid dim:        {}³", config.grid.grid_dim);
    println!("  Cell size:       {:.3} m", config.grid.cell_size);
    println!("  Block size:      {:.3} m", config.grid.block_size());
    println!("  Capacity:        {} blocks", config.grid.capacity);
    println!("  Learning rate:   {}", config.learning_rate);
    println!("  Surface weight:  {}", config.loss.surface_weight);
    println!("  Eikonal weight:  {}", config.loss.eikonal_weight);
    println!();

    // Create trainer and initialize
    let mut trainer = NeuralSdfTrainer::<MyBackend>::new(config, &device);

    // =========================================================================
    // Step 6: Initialize grid using TSDF fusion
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: TSDF Fusion Initialization                         │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let truncation = 0.05; // 5cm truncation distance
    camera.fuse_to_grid(&frames, &mut trainer.grid, truncation);

    println!("  Truncation:      {:.3} m", truncation);
    println!("  Blocks allocated: {}", trainer.grid.num_blocks());
    println!();

    // =========================================================================
    // Step 7: Train to refine the SDF
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Training                                           │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let epochs = 200;
    let log_interval = 25;

    println!("  Epochs:          {}", epochs);
    println!("  Log interval:    {}", log_interval);
    println!();

    // Train using analytical gradients (use train_initialized since TSDF already set up the grid)
    let loss_history = trainer.train_initialized(&all_points, Some(&all_normals), epochs, log_interval);

    // Print loss history
    println!("  Training Progress:");
    for (epoch, loss) in &loss_history {
        let bar_len = (50.0 * (1.0 - (loss / loss_history[0].1).min(1.0))) as usize;
        let bar: String = "█".repeat(bar_len) + &"░".repeat(50 - bar_len);
        println!("    Epoch {:4}: loss = {:.6}  [{}]", epoch, loss, bar);
    }

    if let (Some(first), Some(last)) = (loss_history.first(), loss_history.last()) {
        let reduction = (first.1 - last.1) / first.1 * 100.0;
        println!();
        println!("  Initial loss:    {:.6}", first.1);
        println!("  Final loss:      {:.6}", last.1);
        println!("  Reduction:       {:.1}%", reduction);
    }
    println!();

    // =========================================================================
    // Step 8: Export results
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 8: Exporting Results                                  │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let grid = trainer.export_grid();

    // Export to .ash format
    let ash_path = Path::new(OUTPUT_DIR).join("trained_sphere.ash");
    println!("  Exporting .ash file...");
    match export_to_file(&grid, &ash_path) {
        Ok(()) => {
            let file_size = fs::metadata(&ash_path).map(|m| m.len()).unwrap_or(0);
            println!("    Path:          {}", ash_path.display());
            println!("    Blocks:        {}", grid.num_blocks());
            println!("    File size:     {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);
        }
        Err(e) => {
            eprintln!("    Error: Failed to export .ash file: {}", e);
        }
    }
    println!();

    // Export to .obj format (mesh extraction)
    let obj_path = Path::new(OUTPUT_DIR).join("trained_sphere.obj");
    println!("  Exporting .obj file...");
    match export_to_obj(&grid, &obj_path) {
        Ok(stats) => {
            let file_size = fs::metadata(&obj_path).map(|m| m.len()).unwrap_or(0);
            println!("    Path:          {}", obj_path.display());
            println!("    Vertices:      {}", stats.vertex_count);
            println!("    Triangles:     {}", stats.triangle_count);
            println!("    File size:     {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);
        }
        Err(e) => {
            eprintln!("    Error: Failed to export .obj file: {}", e);
        }
    }
    println!();

    // =========================================================================
    // Step 9: Verification queries
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Step 9: Verification Queries                               │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let test_points = [
        (Point3::new(0.0, 0.0, 0.0), "center", "< 0 (inside)"),
        (Point3::new(sphere_radius, 0.0, 0.0), "surface (+X)", "≈ 0"),
        (Point3::new(0.0, sphere_radius, 0.0), "surface (+Y)", "≈ 0"),
        (Point3::new(0.0, 0.0, sphere_radius), "surface (+Z)", "≈ 0"),
        (Point3::new(0.5, 0.0, 0.0), "outside", "> 0"),
        (Point3::new(-0.5, 0.0, 0.0), "outside (-X)", "> 0"),
    ];

    println!("  Point queries (expected SDF values for sphere r={:.2}):", sphere_radius);
    println!();

    for (p, location, expected) in &test_points {
        let ground_truth = p.length() - sphere_radius;
        if let Some([sdf]) = grid.query(*p) {
            let error = (sdf - ground_truth).abs();
            println!("  {:12} ({:+.2}, {:+.2}, {:+.2}): SDF = {:+.4}, GT = {:+.4}, err = {:.4}  [{}]",
                     location, p.x, p.y, p.z, sdf, ground_truth, error, expected);
        } else {
            println!("  {:12} ({:+.2}, {:+.2}, {:+.2}): N/A (unallocated)  [{}]",
                     location, p.x, p.y, p.z, expected);
        }
    }
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Training complete!");
    println!("  Output files: {}/trained_sphere.{{ash,obj}}", OUTPUT_DIR);
    println!("═══════════════════════════════════════════════════════════════");
}

/// Export an InMemoryGrid to an OBJ file using marching cubes mesh extraction.
fn export_to_obj(
    grid: &ash_io::InMemoryGrid<1>,
    path: &Path,
) -> Result<ash_io::MeshStats, Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let config = ash_io::ObjExportConfig::default();
    let stats = ash_io::export_obj_with_stats(grid, &mut writer, config)?;

    Ok(stats)
}

/// Compute the bounding box of points.
fn compute_bounds(points: &[Point3]) -> (Point3, Point3) {
    if points.is_empty() {
        return (Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
    }

    let mut min = points[0];
    let mut max = points[0];

    for p in points {
        min = min.min(*p);
        max = max.max(*p);
    }

    (min, max)
}
